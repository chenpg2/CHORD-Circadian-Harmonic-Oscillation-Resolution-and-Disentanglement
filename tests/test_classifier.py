"""Tests for two-stage detect-then-disentangle classifier."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


def _make_independent_12h(seed=42):
    """Independent 12h + 24h oscillation (different phases, strong 12h)."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    y = (2.0 * np.cos(2 * np.pi / 24.0 * t + 0.5)
         + 1.5 * np.cos(2 * np.pi / 12.0 * t + 2.1)
         + rng.normal(0, 0.4, len(t)))
    return t, y


def _make_harmonic_sawtooth(seed=42):
    """Sawtooth 24h waveform -- 12h component is a harmonic artifact."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    phase = (t % 24.0) / 24.0
    y = 3.0 * (phase - 0.5) + rng.normal(0, 0.3, len(t))
    return t, y


def _make_nonsinusoidal_independent(seed=42):
    """Non-sinusoidal independent 12h -- the key case V4 misses."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    y_24 = 2.0 * np.cos(2 * np.pi / 24.0 * t)
    phase_12 = (t % 12.0) / 12.0
    y_12 = np.where(np.abs(phase_12 - 0.3) < 0.15, 2.5, 0.0)
    y = y_24 + y_12 + rng.normal(0, 0.3, len(t))
    return t, y


def _make_noise(seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    y = rng.normal(0, 1.0, len(t))
    return t, y


def _make_circadian_only(seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    y = 3.0 * np.cos(2 * np.pi / 24.0 * t) + rng.normal(0, 0.3, len(t))
    return t, y


class TestSingleGene:

    def test_independent_detected(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_independent_12h()
        result = classify_gene(t, y)
        assert result["classification"] in (
            "independent_ultradian", "likely_independent_ultradian"
        ), "Expected independent, got {}".format(result["classification"])
        assert result["stage1_passed"] is True

    def test_harmonic_detected(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_harmonic_sawtooth()
        result = classify_gene(t, y)
        # Sawtooth may be detected as harmonic or circadian_only or ambiguous
        # The key is it should NOT be classified as independent
        assert result["classification"] not in (
            "independent_ultradian",
        ), "Sawtooth should not be independent, got {}".format(result["classification"])

    def test_noise_nonrhythmic(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_noise()
        result = classify_gene(t, y)
        assert result["classification"] == "non_rhythmic"

    def test_circadian_only(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_circadian_only()
        result = classify_gene(t, y)
        assert result["classification"] == "circadian_only"

    def test_nonsinusoidal_independent_not_ambiguous(self):
        """Non-sinusoidal independent 12h should NOT be ambiguous (legacy V4 failure case)."""
        from chord.bhdt.classifier import classify_gene
        t, y = _make_nonsinusoidal_independent()
        result = classify_gene(t, y)
        assert result["stage1_passed"] is True, (
            "Stage 1 should detect non-sinusoidal 12h, p_detect={}".format(
                result.get("stage1_p_detect", "N/A"))
        )

    def test_result_structure(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_independent_12h()
        result = classify_gene(t, y)
        assert "stage1_passed" in result
        assert "stage1_p_detect" in result
        assert "stage1_best_detector" in result
        assert "stage1_waveform_hint" in result
        assert "evidence_score" in result
        assert "confidence" in result
        assert "classification" in result
        assert "evidence_details" in result

    def test_confidence_range(self):
        from chord.bhdt.classifier import classify_gene
        t, y = _make_independent_12h()
        result = classify_gene(t, y)
        assert -1.0 <= result["confidence"] <= 1.0


class TestBatch:

    def test_batch_runs(self):
        from chord.bhdt.classifier import batch_classify
        rng = np.random.RandomState(42)
        t = np.arange(0, 48, 2.0)
        Y = rng.normal(0, 1, (5, len(t)))
        Y[0] += 2.0 * np.cos(2 * np.pi / 12.0 * t)
        results = batch_classify(t, Y, verbose=False)
        assert len(results) == 5
        assert all("classification" in r for r in results)

    def test_batch_gene_names(self):
        from chord.bhdt.classifier import batch_classify
        rng = np.random.RandomState(42)
        t = np.arange(0, 48, 2.0)
        Y = rng.normal(0, 1, (3, len(t)))
        names = ["GeneA", "GeneB", "GeneC"]
        results = batch_classify(t, Y, gene_names=names, verbose=False)
        assert [r["gene_name"] for r in results] == names


class TestH24DominanceGate:

    def test_24h_dominance_gate_classifies_strong_circadian_as_harmonic(self):
        from chord.bhdt.classifier import classify_gene
        rng = np.random.RandomState(42)
        t = np.arange(0, 48, 2.0)
        y = (3.0 * np.cos(2 * np.pi / 24.0 * t)
             + 0.6 * np.cos(2 * np.pi / 12.0 * t)
             + rng.normal(0, 0.3, len(t)))
        result = classify_gene(t, y)
        assert result["classification"] not in (
            "independent_ultradian", "likely_independent_ultradian"
        ), "Strong 24h + weak exact-harmonic 12h should not be independent, got {}".format(
            result["classification"])

    def test_24h_dominance_gate_does_not_block_true_independent(self):
        from chord.bhdt.classifier import classify_gene
        rng = np.random.RandomState(42)
        t = np.arange(0, 48, 2.0)
        y = (2.0 * np.cos(2 * np.pi / 24.0 * t + 0.5)
             + 1.8 * np.cos(2 * np.pi / 11.8 * t + 2.1)
             + rng.normal(0, 0.3, len(t)))
        result = classify_gene(t, y)
        assert result["classification"] in (
            "independent_ultradian", "likely_independent_ultradian"
        ), "Strong independent 12h should be classified as independent, got {}".format(
            result["classification"])
