"""
Tests for VMD (Variational Mode Decomposition) module.

Tests cover:
- Mode separation of independent oscillators
- Harmonic signal decomposition
- VMD evidence scoring for independent vs harmonic signals
- Convergence behavior
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.vmd import vmd_decompose, vmd_evidence


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def independent_signal():
    """Two independent oscillators: 24h + 11.5h (clearly not a harmonic)."""
    rng = np.random.RandomState(42)
    t = np.arange(0, 48, 2.0)  # 24 timepoints, 2h sampling
    A_24, T_24, phi_24 = 2.0, 24.0, 0.3
    A_12, T_12, phi_12 = 1.5, 11.5, 1.0
    y = (5.0
         + A_24 * np.cos(2 * np.pi * t / T_24 - phi_24)
         + A_12 * np.cos(2 * np.pi * t / T_12 - phi_12)
         + rng.normal(0, 0.3, len(t)))
    return t, y


@pytest.fixture
def harmonic_signal():
    """Non-sinusoidal 24h signal: 12h component is a harmonic, not independent."""
    rng = np.random.RandomState(42)
    t = np.arange(0, 48, 2.0)
    # Peaked waveform: fundamental + exact 2nd harmonic (locked phase)
    A_24 = 2.0
    T_24 = 24.0
    phase = 2 * np.pi * t / T_24
    # 12h component is exactly half the period, phase-locked
    y = 5.0 + A_24 * np.cos(phase) + 0.5 * A_24 * np.cos(2 * phase) + rng.normal(0, 0.3, len(t))
    return t, y


# ============================================================================
# Test: vmd_decompose separates independent oscillators
# ============================================================================

class TestVMDDecompose:
    def test_vmd_separates_independent_oscillators(self, independent_signal):
        """Two independent oscillators (24h + 11.5h) should be separated
        with center periods close to their true values."""
        t, y = independent_signal
        result = vmd_decompose(t, y, K=3, alpha=2000,
                               init_periods=[24.0, 12.0, 8.0])

        assert "modes" in result
        assert "center_frequencies" in result
        assert "center_periods" in result
        assert "mode_energies" in result
        assert "mode_amplitudes" in result
        assert "n_iterations" in result

        # Shape checks
        assert result["modes"].shape[0] == 3
        assert result["modes"].shape[1] == len(t)

        periods = result["center_periods"]
        # One mode should be near 24h, another near 11.5h
        periods_sorted = sorted(periods, reverse=True)
        assert abs(periods_sorted[0] - 24.0) < 4.0, \
            f"Expected ~24h mode, got {periods_sorted[0]:.1f}h"
        assert abs(periods_sorted[1] - 11.5) < 3.0, \
            f"Expected ~11.5h mode, got {periods_sorted[1]:.1f}h"

    def test_vmd_harmonic_signal(self, harmonic_signal):
        """Non-sinusoidal 24h signal: the 12h mode should have lower energy
        relative to the 24h mode (it's a harmonic, not independent)."""
        t, y = harmonic_signal
        result = vmd_decompose(t, y, K=3, alpha=2000,
                               init_periods=[24.0, 12.0, 8.0])

        periods = result["center_periods"]
        energies = result["mode_energies"]

        # Find modes closest to 24h and 12h
        idx_24 = int(np.argmin([abs(p - 24.0) for p in periods]))
        idx_12 = int(np.argmin([abs(p - 12.0) for p in periods]))

        # 12h mode energy should be less than 24h mode energy
        # (it's a harmonic artifact, not an independent oscillator)
        ratio = energies[idx_12] / max(energies[idx_24], 1e-12)
        assert ratio < 0.8, \
            f"12h/24h energy ratio {ratio:.2f} too high for harmonic signal"

    def test_vmd_convergence(self, independent_signal):
        """VMD should converge within max_iter iterations."""
        t, y = independent_signal
        result = vmd_decompose(t, y, K=3, alpha=2000, max_iter=500)
        assert result["n_iterations"] < 500, \
            f"VMD did not converge: {result['n_iterations']} iterations"


# ============================================================================
# Test: vmd_evidence scoring
# ============================================================================

class TestVMDEvidence:
    def test_vmd_evidence_independent(self, independent_signal):
        """Independent signal (24h + 11.5h) should get positive vmd_score."""
        t, y = independent_signal
        result = vmd_evidence(t, y, T_base=24.0)

        assert "vmd_score" in result
        assert "energy_ratio_12_24" in result
        assert "T_12_vmd" in result
        assert "T_24_vmd" in result
        assert "freq_deviation" in result
        assert "harmonic_lock" in result
        assert "vmd_result" in result

        # Independent signal: should have positive evidence for independence
        assert result["vmd_score"] > 0, \
            f"Expected positive vmd_score for independent signal, got {result['vmd_score']}"

    def test_vmd_evidence_harmonic(self, harmonic_signal):
        """Harmonic signal should get negative or zero vmd_score."""
        t, y = harmonic_signal
        result = vmd_evidence(t, y, T_base=24.0)

        # Harmonic signal: should NOT get positive evidence for independence
        assert result["vmd_score"] <= 0, \
            f"Expected non-positive vmd_score for harmonic signal, got {result['vmd_score']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
