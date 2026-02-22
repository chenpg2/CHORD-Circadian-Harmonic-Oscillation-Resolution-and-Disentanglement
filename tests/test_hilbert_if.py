"""
Tests for Hilbert Instantaneous Frequency module.

Tests cover:
- Instantaneous frequency extraction from pure cosines
- IF coupling detection for harmonic vs independent signals
- Full VMD-Hilbert disentanglement pipeline
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.hilbert_if import (
    hilbert_instantaneous_frequency,
    instantaneous_frequency_coupling,
    vmd_hilbert_disentangle,
)


# ============================================================================
# Test: hilbert_instantaneous_frequency
# ============================================================================

class TestHilbertInstantaneousFrequency:
    def test_hilbert_if_pure_cosine(self):
        """Pure 24h cosine should have IF ≈ 1/24 cycles/hour."""
        t = np.linspace(0, 96, 500)  # long signal for stable IF
        y = np.cos(2 * np.pi * t / 24.0)
        if_vals = hilbert_instantaneous_frequency(t, y)

        assert if_vals.shape == t.shape
        # Trim edges (Hilbert edge effects)
        mid = if_vals[50:-50]
        expected = 1.0 / 24.0
        assert np.abs(np.median(mid) - expected) < 0.005, \
            f"Expected IF ≈ {expected:.4f}, got median {np.median(mid):.4f}"

    def test_hilbert_if_12h_cosine(self):
        """Pure 12h cosine should have IF ≈ 1/12 cycles/hour."""
        t = np.linspace(0, 96, 500)
        y = np.cos(2 * np.pi * t / 12.0)
        if_vals = hilbert_instantaneous_frequency(t, y)

        mid = if_vals[50:-50]
        expected = 1.0 / 12.0
        assert np.abs(np.median(mid) - expected) < 0.01, \
            f"Expected IF ≈ {expected:.4f}, got median {np.median(mid):.4f}"


# ============================================================================
# Test: instantaneous_frequency_coupling
# ============================================================================

class TestInstantaneousFrequencyCoupling:
    def test_coupling_harmonic_signal(self):
        """Exact harmonic (12h = 2nd harmonic of 24h) should show high
        IF correlation and is_harmonic=True."""
        t = np.linspace(0, 96, 500)
        phase_24 = 2 * np.pi * t / 24.0
        mode_24 = np.cos(phase_24)
        mode_12 = np.cos(2 * phase_24)  # exact 2nd harmonic

        result = instantaneous_frequency_coupling(t, mode_24, mode_12)

        assert "if_correlation" in result
        assert "if_ratio_mean" in result
        assert "if_ratio_std" in result
        assert "is_harmonic" in result
        assert "coupling_score" in result

        assert result["if_correlation"] > 0.6, \
            f"Expected high correlation for harmonic, got {result['if_correlation']:.3f}"
        assert result["is_harmonic"] is True
        assert result["coupling_score"] < 0, \
            f"Expected negative score for harmonic, got {result['coupling_score']}"

    def test_coupling_independent_signal(self):
        """Independent 11.5h oscillator should show low IF correlation."""
        t = np.linspace(0, 96, 500)
        mode_24 = np.cos(2 * np.pi * t / 24.0)
        # Frequency-modulated 11.5h signal — clearly not a harmonic
        mode_12 = np.cos(2 * np.pi * t / 11.5 + 0.3 * np.sin(2 * np.pi * t / 48.0))

        result = instantaneous_frequency_coupling(t, mode_24, mode_12)

        # Independent signal: correlation should be lower, not classified as harmonic
        assert result["is_harmonic"] is False, \
            f"Expected is_harmonic=False for independent signal"


# ============================================================================
# Test: vmd_hilbert_disentangle (full pipeline)
# ============================================================================

class TestVMDHilbertDisentangle:
    def test_vmd_hilbert_sawtooth_harmonic(self):
        """Sawtooth wave is a non-sinusoidal 24h signal — its 12h content
        is a harmonic, not an independent oscillator."""
        t = np.arange(0, 48, 2.0)  # 25 timepoints, typical data
        # Sawtooth via Fourier: fundamental + harmonics
        phase = 2 * np.pi * t / 24.0
        y = 5.0 + 2.0 * np.cos(phase) + 1.0 * np.cos(2 * phase) + 0.5 * np.cos(3 * phase)

        result = vmd_hilbert_disentangle(t, y, T_base=24.0)

        assert "classification_evidence" in result
        assert "if_correlation" in result
        assert "is_harmonic" in result
        assert "mode_24_amplitude" in result
        assert "mode_12_amplitude" in result
        assert "vmd_converged" in result

        # Sawtooth harmonics should be classified as harmonic
        assert result["is_harmonic"] is True, \
            f"Expected harmonic classification for sawtooth, got is_harmonic={result['is_harmonic']}"

    def test_vmd_hilbert_independent_12h(self):
        """Independent 11.5h oscillator should be classified as independent."""
        rng = np.random.RandomState(42)
        t = np.arange(0, 48, 2.0)
        y = (5.0
             + 2.0 * np.cos(2 * np.pi * t / 24.0)
             + 1.5 * np.cos(2 * np.pi * t / 11.5 + 1.0)
             + rng.normal(0, 0.2, len(t)))

        result = vmd_hilbert_disentangle(t, y, T_base=24.0)

        assert result["is_harmonic"] is False, \
            f"Expected independent classification, got is_harmonic={result['is_harmonic']}"

    def test_vmd_hilbert_peaked_waveform(self):
        """cos + cos^2 waveform: the 12h content is a harmonic."""
        t = np.arange(0, 48, 2.0)
        phase = 2 * np.pi * t / 24.0
        # cos^2(x) = 0.5 + 0.5*cos(2x), so this adds a 12h harmonic
        y = 5.0 + 2.0 * np.cos(phase) + 1.0 * np.cos(phase) ** 2

        result = vmd_hilbert_disentangle(t, y, T_base=24.0)

        assert result["is_harmonic"] is True, \
            f"Expected harmonic for peaked waveform, got is_harmonic={result['is_harmonic']}"

    def test_vmd_hilbert_pure_circadian(self):
        """Pure 24h cosine: 12h mode amplitude should be much smaller."""
        t = np.arange(0, 48, 2.0)
        y = 5.0 + 2.0 * np.cos(2 * np.pi * t / 24.0)

        result = vmd_hilbert_disentangle(t, y, T_base=24.0)

        assert result["mode_12_amplitude"] < result["mode_24_amplitude"], \
            (f"12h amplitude ({result['mode_12_amplitude']:.3f}) should be < "
             f"24h amplitude ({result['mode_24_amplitude']:.3f})")

    def test_vmd_hilbert_noise_only(self):
        """Pure noise should not crash the pipeline."""
        rng = np.random.RandomState(123)
        t = np.arange(0, 48, 2.0)
        y = rng.normal(5.0, 1.0, len(t))

        result = vmd_hilbert_disentangle(t, y, T_base=24.0)

        # Should return a valid dict without crashing
        assert isinstance(result, dict)
        assert "is_harmonic" in result
        assert "classification_evidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
