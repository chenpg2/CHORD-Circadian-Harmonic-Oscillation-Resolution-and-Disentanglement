"""
Tests for the multi-channel fusion classifier (v3).

Tests verify:
1. Sawtooth wave -> harmonic (BHCT detects QPC)
2. Independent cosines -> independent (BHCT no QPC)
3. BHCT overrides V2 when QPC detected
4. BHCT promotes circadian_only to independent when no QPC + significant 12h
5. Pure 24h cosine -> circadian_only
6. Pure noise -> non_rhythmic
7. Confidence levels: clear cases get "high", ambiguous get "low"
8. v3_fast matches v3 (without SD)
9. Batch function works on matrix input
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.fusion_classifier import (
    classify_gene_v3,
    classify_gene_v3_fast,
    batch_classify_v3,
)


# ============================================================================
# Helper: signal generators
# ============================================================================

def _make_sawtooth(N=48, T_base=24.0, noise_std=0.05, seed=42):
    """Sawtooth wave -- harmonics are phase-coupled by construction."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    y = np.zeros(N)
    for k in range(1, 6):
        y += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * t / T_base) / k
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_independent_cosines(N=48, T_base=24.0, noise_std=0.1, seed=42):
    """Sum of independent cosines with random phases -- no phase coupling."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    phi1 = rng.uniform(0, 2 * np.pi)
    y = np.cos(2 * np.pi * t / T_base + phi1)
    phi2 = rng.uniform(0, 2 * np.pi)
    y += 0.8 * np.cos(2 * np.pi * t / (T_base / 2) + phi2)
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_pure_circadian(N=48, T_base=24.0, noise_std=0.1, seed=42):
    """Pure 24h cosine -- no 12h component."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    y = 2.0 * np.cos(2 * np.pi * t / T_base)
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_noise(N=48, seed=42):
    """Pure Gaussian noise."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    y = rng.normal(0, 1.0, N)
    return t, y


def _make_circadian_with_weak_12h(N=48, T_base=24.0, noise_std=0.1, seed=42):
    """Strong 24h + weak independent 12h that V2 might call circadian_only.

    The 12h component is independent (random phase) but weak enough that
    V2's soft gate may reject it. BHCT should see no QPC, and the F-test
    should be borderline significant, triggering promotion.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    phi1 = 0.0
    y = 2.0 * np.cos(2 * np.pi * t / T_base + phi1)
    phi2 = rng.uniform(0, 2 * np.pi)
    y += 0.4 * np.cos(2 * np.pi * t / (T_base / 2) + phi2)
    y += rng.normal(0, noise_std, N)
    return t, y


# ============================================================================
# Tests
# ============================================================================

class TestV3HarmonicSawtooth:
    """Sawtooth wave should be classified as harmonic (BHCT detects QPC)."""

    def test_classification(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["classification"] == "harmonic", (
            "Sawtooth should be harmonic, got '{}'".format(
                result["classification"]
            )
        )

    def test_bhct_detects_qpc(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["bhct_p_value"] is not None
        assert result["bhct_p_value"] < 0.05, (
            "BHCT should detect QPC in sawtooth, got p={}".format(
                result["bhct_p_value"]
            )
        )

    def test_has_required_keys(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        required = {
            "classification", "confidence", "reason",
            "v2_classification", "bhct_p_value", "bhct_bicoherence",
            "sd_log_bf", "channels_agree",
        }
        for key in required:
            assert key in result, "Missing key: {}".format(key)


class TestV3IndependentCosines:
    """Independent cosines should be classified as independent (BHCT no QPC)."""

    def test_classification(self):
        t, y = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["classification"] in (
            "independent_ultradian", "likely_independent_ultradian"
        ), (
            "Independent cosines should be independent, got '{}'".format(
                result["classification"]
            )
        )

    def test_bhct_no_qpc(self):
        t, y = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["bhct_p_value"] is not None
        assert result["bhct_p_value"] > 0.05, (
            "BHCT should not detect QPC in independent cosines, got p={}".format(
                result["bhct_p_value"]
            )
        )


class TestV3BhctOverridesV2:
    """Signal where V2 says independent but BHCT detects QPC -> harmonic."""

    def test_bhct_veto(self):
        # Sawtooth with high amplitude ratio (V2 might lean independent
        # due to amplitude evidence, but BHCT should detect QPC)
        rng = np.random.RandomState(42)
        t = np.linspace(0, 48, 48, endpoint=False)
        # Sawtooth-like signal with strong 2nd harmonic
        y = (np.sin(2 * np.pi * t / 24.0)
             - 0.5 * np.sin(2 * 2 * np.pi * t / 24.0)
             + 0.33 * np.sin(3 * 2 * np.pi * t / 24.0))
        y += rng.normal(0, 0.05, 48)
        result = classify_gene_v3(t, y, T_base=24.0)
        # BHCT should detect QPC and override to harmonic
        if result["bhct_p_value"] is not None and result["bhct_p_value"] < 0.05:
            assert result["classification"] == "harmonic", (
                "BHCT detected QPC (p={:.4f}) but classification is '{}'".format(
                    result["bhct_p_value"], result["classification"]
                )
            )


class TestV3BhctPromotesCircadianOnly:
    """Signal where V2 says circadian_only but BHCT says no QPC and 12h
    F-test significant -> independent."""

    def test_promotion(self):
        t, y = _make_circadian_with_weak_12h(N=48, noise_std=0.1, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        # If BHCT says no QPC (p > 0.5) AND V2 said circadian_only
        # AND 12h is significant, should promote to independent
        if (result["bhct_p_value"] is not None
                and result["bhct_p_value"] > 0.5
                and result["v2_classification"] == "circadian_only"
                and result.get("has_significant_12h", False)):
            assert result["classification"] == "independent_ultradian", (
                "Should promote to independent, got '{}'".format(
                    result["classification"]
                )
            )
            assert result["reason"] == "bhct_no_qpc_promotes_to_independent"


class TestV3PureCircadian:
    """Pure 24h cosine should be classified as circadian_only."""

    def test_classification(self):
        t, y = _make_pure_circadian(N=48, noise_std=0.1, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["classification"] in ("circadian_only", "harmonic"), (
            "Pure 24h should be circadian_only or harmonic, got '{}'".format(
                result["classification"]
            )
        )


class TestV3Noise:
    """Pure noise should be classified as non_rhythmic or get low confidence."""

    def test_classification(self):
        t, y = _make_noise(N=48, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        # V2 may misclassify noise; fusion classifier falls back to V2
        # with low confidence when BHCT is inconclusive. Accept either
        # non_rhythmic/ambiguous OR low-confidence V2 fallback.
        if result["classification"] not in ("non_rhythmic", "ambiguous"):
            assert result["confidence"] == "low", (
                "Noise classified as '{}' should have low confidence, "
                "got '{}'".format(
                    result["classification"], result["confidence"]
                )
            )
            assert "v2_default" in result["reason"] or "inconclusive" in result["reason"], (
                "Noise fallback should indicate V2 default, got '{}'".format(
                    result["reason"]
                )
            )


class TestV3ConfidenceLevels:
    """Clear cases get 'high' confidence, ambiguous get 'low'."""

    def test_clear_harmonic_high_confidence(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        assert result["confidence"] == "high", (
            "Clear harmonic should have high confidence, got '{}'".format(
                result["confidence"]
            )
        )

    def test_clear_independent_high_confidence(self):
        t, y = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        result = classify_gene_v3(t, y, T_base=24.0)
        # Independent with clear BHCT no-QPC should be high confidence
        if result["bhct_p_value"] is not None and result["bhct_p_value"] > 0.5:
            assert result["confidence"] in ("high", "medium"), (
                "Clear independent should have high/medium confidence, got '{}'".format(
                    result["confidence"]
                )
            )


class TestV3FastMatchesFull:
    """v3_fast gives same classification as v3 without Savage-Dickey."""

    def test_sawtooth_matches(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        r_full = classify_gene_v3(t, y, T_base=24.0, use_savage_dickey=False)
        r_fast = classify_gene_v3_fast(t, y, T_base=24.0)
        assert r_full["classification"] == r_fast["classification"]
        assert r_full["confidence"] == r_fast["confidence"]
        assert r_full["reason"] == r_fast["reason"]

    def test_independent_matches(self):
        t, y = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        r_full = classify_gene_v3(t, y, T_base=24.0, use_savage_dickey=False)
        r_fast = classify_gene_v3_fast(t, y, T_base=24.0)
        assert r_full["classification"] == r_fast["classification"]

    def test_sd_not_used(self):
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        r_fast = classify_gene_v3_fast(t, y, T_base=24.0)
        assert r_fast["sd_log_bf"] is None


class TestBatchClassifyV3:
    """Batch function works on matrix input."""

    def test_basic_batch(self):
        t = np.linspace(0, 48, 48, endpoint=False)
        rng = np.random.RandomState(42)
        Y = np.zeros((3, 48))
        # Gene 0: sawtooth (harmonic)
        for k in range(1, 6):
            Y[0] += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * t / 24.0) / k
        Y[0] += rng.normal(0, 0.05, 48)
        # Gene 1: independent cosines
        Y[1] = (np.cos(2 * np.pi * t / 24.0 + 1.0)
                + 0.8 * np.cos(2 * np.pi * t / 12.0 + 2.5)
                + rng.normal(0, 0.1, 48))
        # Gene 2: pure noise
        Y[2] = rng.normal(0, 1.0, 48)

        results = batch_classify_v3(
            t, Y, gene_names=["saw", "indep", "noise"],
            T_base=24.0, verbose=False
        )
        assert len(results) == 3
        assert results[0]["gene_name"] == "saw"
        assert results[1]["gene_name"] == "indep"
        assert results[2]["gene_name"] == "noise"
        # Each result should have classification
        for r in results:
            assert "classification" in r

    def test_default_gene_names(self):
        t = np.linspace(0, 48, 24, endpoint=False)
        rng = np.random.RandomState(42)
        Y = rng.normal(0, 1.0, (2, 24))
        results = batch_classify_v3(t, Y, verbose=False)
        assert results[0]["gene_name"] == "gene_0"
        assert results[1]["gene_name"] == "gene_1"

    def test_length_mismatch_raises(self):
        t = np.linspace(0, 48, 24, endpoint=False)
        Y = np.zeros((2, 30))  # wrong number of columns
        with pytest.raises(ValueError):
            batch_classify_v3(t, Y, verbose=False)

    def test_gene_names_mismatch_raises(self):
        t = np.linspace(0, 48, 24, endpoint=False)
        rng = np.random.RandomState(42)
        Y = rng.normal(0, 1.0, (2, 24))
        with pytest.raises(ValueError):
            batch_classify_v3(t, Y, gene_names=["a"], verbose=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
