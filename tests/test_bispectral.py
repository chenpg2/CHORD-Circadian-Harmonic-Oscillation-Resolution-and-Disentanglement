"""
Tests for the Bispectral Harmonic Coupling Test (BHCT).

Tests verify that:
1. Harmonic signals (sawtooth) show significant QPC (low p-value)
2. Independent cosines show no QPC (high p-value)
3. Peaked/asymmetric waveforms with strong harmonics show QPC
4. Pure noise shows no QPC
5. Evidence scores have correct signs
6. Small samples (N=12) don't crash

Power limitations (documented, not bugs):
    The multi-taper bicoherence test has good power for signals with
    strong multi-harmonic coupling (sawtooth: A_k/A_1 = 1/k for many k)
    but limited power for signals with only a weak 2nd harmonic
    (e.g., cos(wt) + 0.2*cos(2wt)). This is a fundamental limitation
    of bispectral analysis on short series (N < 50), not a deficiency
    of the implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.bispectral import (
    compute_bispectrum,
    bispectral_coupling_test,
    bhct_evidence,
)


# ============================================================================
# Helper: signal generators
# ============================================================================

def _make_sawtooth(N=48, T_base=24.0, noise_std=0.05, seed=42):
    """Sawtooth wave — harmonics are phase-coupled by construction.

    Fourier series: sum_k (-1)^(k+1) * sin(2*pi*k*t/T) / k
    Has strong harmonics at all integer multiples of the fundamental.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    y = np.zeros(N)
    for k in range(1, 6):
        y += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * t / T_base) / k
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_independent_cosines(N=48, T_base=24.0, noise_std=0.1, seed=42):
    """Sum of independent cosines with random phases — no phase coupling."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    phi1 = rng.uniform(0, 2 * np.pi)
    y = np.cos(2 * np.pi * t / T_base + phi1)
    phi2 = rng.uniform(0, 2 * np.pi)
    y += 0.5 * np.cos(2 * np.pi * t / (T_base / 2) + phi2)
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_peaked_wave(N=48, T_base=24.0, noise_std=0.05, seed=42):
    """Asymmetric peaked waveform with strong phase-coupled harmonics.

    Uses a modified sawtooth (partial Fourier series with 3 harmonics)
    which has strong, detectable phase coupling. This is representative
    of biological waveforms with sharp peaks and gradual troughs.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    phase = 2 * np.pi * t / T_base
    # Asymmetric peaked wave: fundamental + alternating-sign harmonics
    # sin(x) - 0.5*sin(2x) + 0.33*sin(3x) creates a peaked waveform
    y = (np.sin(phase)
         - 0.5 * np.sin(2 * phase)
         + 0.33 * np.sin(3 * phase))
    y += rng.normal(0, noise_std, N)
    return t, y


def _make_noise(N=48, seed=42):
    """Pure Gaussian noise."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 48, N, endpoint=False)
    y = rng.normal(0, 1.0, N)
    return t, y


# ============================================================================
# Tests
# ============================================================================

class TestBispectralCouplingTest:

    def test_harmonic_sawtooth_detected(self):
        """Sawtooth wave has phase-coupled harmonics — should yield significant bicoherence."""
        t, y = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        result = bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=199, seed=42)

        assert "bicoherence" in result
        assert "p_value" in result
        assert "bhct_score" in result
        # Sawtooth harmonics should show significant QPC
        assert result["p_value"] < 0.05, (
            f"Sawtooth should show significant QPC, got p={result['p_value']:.4f}"
        )
        # Score should be negative (harmonic evidence)
        assert result["bhct_score"] < 0

    def test_independent_cosines_not_detected(self):
        """Independent cosines should NOT show significant QPC."""
        t, y = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        result = bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=199, seed=42)

        # Independent oscillators should not show QPC
        assert result["p_value"] > 0.05, (
            f"Independent cosines should not show QPC, got p={result['p_value']:.4f}"
        )

    def test_peaked_wave_detected(self):
        """Strongly peaked/asymmetric waveform should show QPC."""
        t, y = _make_peaked_wave(N=48, noise_std=0.05, seed=42)
        result = bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=199, seed=42)

        # Strong peaked wave should show significant QPC
        assert result["p_value"] < 0.05, (
            f"Peaked wave should show QPC, got p={result['p_value']:.4f}"
        )
        assert result["bhct_score"] < 0

    def test_pure_noise_not_detected(self):
        """Pure Gaussian noise should not show significant QPC."""
        t, y = _make_noise(N=48, seed=42)
        result = bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=199, seed=42)

        # Noise should not show QPC
        assert result["p_value"] > 0.05, (
            f"Pure noise should not show QPC, got p={result['p_value']:.4f}"
        )

    def test_evidence_score_signs(self):
        """Harmonic signals get negative scores, independent signals get positive or zero scores."""
        # Harmonic signal (sawtooth)
        t_h, y_h = _make_sawtooth(N=48, noise_std=0.05, seed=42)
        ev_h = bhct_evidence(t_h, y_h, T_base=24.0, n_surrogates=199, seed=42)
        assert ev_h["score"] < 0, (
            f"Sawtooth should get negative score, got {ev_h['score']}"
        )
        assert "harmonic" in ev_h["label"]

        # Independent signal
        t_i, y_i = _make_independent_cosines(N=48, noise_std=0.1, seed=42)
        ev_i = bhct_evidence(t_i, y_i, T_base=24.0, n_surrogates=199, seed=42)
        assert ev_i["score"] >= 0, (
            f"Independent cosines should get non-negative score, got {ev_i['score']}"
        )

    def test_small_sample_robustness(self):
        """N=12 (minimal case) should not crash and give reasonable results."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 48, 12, endpoint=False)
        y = np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.1, 12)
        result = bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=99, seed=42)

        # Should not crash and return valid values
        assert 0.0 <= result["bicoherence"] <= 1.0
        assert 0.0 <= result["p_value"] <= 1.0
        assert isinstance(result["bhct_score"], int)
        assert result["n_surrogates"] == 99


class TestComputeBispectrum:

    def test_output_keys(self):
        """compute_bispectrum returns all expected keys."""
        t, y = _make_sawtooth(N=48, seed=42)
        result = compute_bispectrum(t, y, T_base=24.0)
        expected_keys = {"bispectrum_mag", "bicoherence", "bispectrum_real",
                         "bispectrum_imag", "n_tapers"}
        assert set(result.keys()) == expected_keys

    def test_bicoherence_bounded(self):
        """Bicoherence should always be in [0, 1]."""
        for seed in range(10):
            t, y = _make_noise(N=24, seed=seed)
            result = compute_bispectrum(t, y, T_base=24.0)
            assert 0.0 <= result["bicoherence"] <= 1.0

    def test_sawtooth_high_bicoherence(self):
        """Sawtooth should have high raw bicoherence (strong coupling)."""
        t, y = _make_sawtooth(N=48, noise_std=0.01, seed=42)
        result = compute_bispectrum(t, y, T_base=24.0)
        # Sawtooth has very strong phase coupling
        assert result["bicoherence"] > 0.3, (
            f"Sawtooth bicoherence ({result['bicoherence']:.4f}) should be > 0.3"
        )

    def test_short_series_uses_few_tapers(self):
        """For very short N, number of tapers should be limited."""
        t = np.linspace(0, 48, 8, endpoint=False)
        y = np.sin(2 * np.pi * t / 24.0)
        result = compute_bispectrum(t, y, T_base=24.0)
        assert result["n_tapers"] == 2  # min(4, max(2, 8//4)) = 2

    def test_longer_series_uses_more_tapers(self):
        """For longer N, more tapers should be used."""
        t = np.linspace(0, 96, 96, endpoint=False)
        y = np.sin(2 * np.pi * t / 24.0)
        result = compute_bispectrum(t, y, T_base=24.0)
        assert result["n_tapers"] == 4  # int(2*2.5) - 1 = 4


class TestBhctEvidence:

    def test_output_keys(self):
        """bhct_evidence returns all expected keys."""
        t, y = _make_sawtooth(N=48, seed=42)
        result = bhct_evidence(t, y, T_base=24.0, n_surrogates=49, seed=42)
        expected_keys = {"score", "p_value", "bicoherence", "label"}
        assert set(result.keys()) == expected_keys

    def test_reproducibility(self):
        """Same seed should give identical results."""
        t, y = _make_sawtooth(N=48, seed=42)
        r1 = bhct_evidence(t, y, T_base=24.0, n_surrogates=99, seed=123)
        r2 = bhct_evidence(t, y, T_base=24.0, n_surrogates=99, seed=123)
        assert r1["p_value"] == r2["p_value"]
        assert r1["score"] == r2["score"]
        assert r1["bicoherence"] == r2["bicoherence"]

    def test_score_range(self):
        """Evidence scores should be in the expected set."""
        valid_scores = {-3, -2, 0, 1, 2}
        for seed in range(5):
            t, y = _make_noise(N=48, seed=seed)
            result = bhct_evidence(t, y, T_base=24.0, n_surrogates=49, seed=seed)
            assert result["score"] in valid_scores, (
                f"Score {result['score']} not in valid set {valid_scores}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
