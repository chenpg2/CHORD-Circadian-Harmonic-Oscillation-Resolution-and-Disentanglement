"""
Tests for Spectral Independence Divergence (SID).

SID measures how much the observed spectral structure deviates from
what a purely harmonic (Fourier series) model would predict.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.sid import compute_sid, sid_evidence, sid_test


# ============================================================================
# Helpers
# ============================================================================

def _make_time(n_points=48, duration=48.0, seed=None):
    """Evenly spaced time points over `duration` hours."""
    return np.linspace(0, duration, n_points, endpoint=False)


def _sawtooth_like(t, T_base=24.0, K=3, noise_sigma=0.1, seed=42):
    """Sum of harmonics with 1/k amplitude decay — a single periodic waveform.

    This is the canonical harmonic signal: all energy at k*f_0 comes from
    a single shape function, so SID should be LOW.
    """
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi / T_base
    y = np.zeros_like(t)
    # Sawtooth-like: A_k = 1/k, phase = 0 for all k
    for k in range(1, K + 1):
        y += (1.0 / k) * np.cos(k * omega * t)
    y += noise_sigma * rng.standard_normal(len(t))
    return y


def _independent_signal(t, T_base=24.0, noise_sigma=0.1, seed=42):
    """24h + 11.5h oscillators with unrelated phases — independent sources.

    The 11.5h component is NOT a harmonic of 24h, so SID should be HIGH.
    """
    rng = np.random.default_rng(seed)
    omega_24 = 2 * np.pi / T_base
    omega_11p5 = 2 * np.pi / 11.5
    y = (1.0 * np.cos(omega_24 * t)
         + 0.8 * np.cos(omega_11p5 * t + 1.7))  # unrelated phase
    y += noise_sigma * rng.standard_normal(len(t))
    return y


def _weak_independent_signal(t, T_base=24.0, noise_sigma=0.1, seed=42):
    """24h + weak 12h with wrong phase — subtle independence.

    The 12h component has amplitude 0.3 * A_24 but its phase is shifted
    by pi/2 from what a harmonic model would predict.
    """
    rng = np.random.default_rng(seed)
    omega_24 = 2 * np.pi / T_base
    omega_12 = 2 * np.pi / 12.0
    # Harmonic model would predict phi_12 ≈ 0 if phi_24 = 0
    # We set phi_12 = pi/2 to create phase divergence
    y = (1.0 * np.cos(omega_24 * t)
         + 0.3 * np.cos(omega_12 * t + np.pi / 2))
    y += noise_sigma * rng.standard_normal(len(t))
    return y


# ============================================================================
# Tests for compute_sid
# ============================================================================

class TestComputeSid:

    def test_sid_low_for_harmonic_signal(self):
        """A sawtooth-like signal (sum of harmonics with decaying amplitudes)
        should have SID < 1.0 because it IS a single periodic waveform."""
        t = _make_time(n_points=48, duration=48.0)
        y = _sawtooth_like(t, seed=42)
        result = compute_sid(t, y, T_base=24.0, K_harmonics=3)

        assert "sid" in result
        assert "spectral_divergence" in result
        assert "phase_divergence" in result
        assert "lambda" in result
        assert "components" in result
        assert result["sid"] < 1.0, (
            f"SID={result['sid']:.3f} should be < 1.0 for harmonic signal"
        )

    def test_sid_high_for_independent_signal(self):
        """Independent 24h + 11.5h oscillators with unrelated phases
        should have SID > 0.5."""
        t = _make_time(n_points=48, duration=48.0)
        y = _independent_signal(t, seed=42)
        result = compute_sid(t, y, T_base=24.0, K_harmonics=3)

        assert result["sid"] > 0.5, (
            f"SID={result['sid']:.3f} should be > 0.5 for independent signal"
        )

    def test_sid_discriminates_weak_independent(self):
        """Weak 12h (A_12=0.3*A_24) with wrong phase should have
        detectable phase_divergence > 0.2."""
        t = _make_time(n_points=48, duration=48.0)
        y = _weak_independent_signal(t, seed=42)
        result = compute_sid(t, y, T_base=24.0, K_harmonics=3)

        assert result["phase_divergence"] > 0.2, (
            f"phase_divergence={result['phase_divergence']:.3f} should be > 0.2 "
            f"for weak independent signal with wrong phase"
        )


# ============================================================================
# Tests for sid_evidence
# ============================================================================

class TestSidEvidence:

    def test_sid_evidence_scoring(self):
        """Verify evidence scores match the defined thresholds."""
        t = _make_time(n_points=48, duration=48.0)

        # Harmonic signal → should get negative evidence (favours M0)
        y_harm = _sawtooth_like(t, seed=42)
        ev_harm = sid_evidence(t, y_harm, T_base=24.0, K_harmonics=3)
        assert "sid_score" in ev_harm
        assert "sid" in ev_harm
        assert "spectral_divergence" in ev_harm
        assert "phase_divergence" in ev_harm

        # Independent signal → should get positive evidence (favours M1)
        y_indep = _independent_signal(t, seed=42)
        ev_indep = sid_evidence(t, y_indep, T_base=24.0, K_harmonics=3)

        # The independent signal should score higher than the harmonic one
        assert ev_indep["sid_score"] > ev_harm["sid_score"], (
            f"Independent score ({ev_indep['sid_score']}) should exceed "
            f"harmonic score ({ev_harm['sid_score']})"
        )


# ============================================================================
# Tests for sid_test (bootstrap)
# ============================================================================

class TestSidTest:

    def test_sid_test_harmonic_not_significant(self):
        """Bootstrap test on harmonic signal should give p > 0.05."""
        t = _make_time(n_points=48, duration=48.0)
        y = _sawtooth_like(t, seed=42)
        result = sid_test(t, y, T_base=24.0, K_harmonics=3,
                          n_bootstrap=199, seed=123)

        assert "sid_observed" in result
        assert "p_value" in result
        assert "sid_null_mean" in result
        assert "sid_null_std" in result
        assert "significant" in result
        assert result["p_value"] > 0.05, (
            f"p={result['p_value']:.3f} should be > 0.05 for harmonic signal"
        )

    def test_sid_test_independent_significant(self):
        """Bootstrap test on independent signal should give p < 0.1."""
        t = _make_time(n_points=48, duration=48.0)
        y = _independent_signal(t, seed=42)
        result = sid_test(t, y, T_base=24.0, K_harmonics=3,
                          n_bootstrap=199, seed=123)

        assert result["p_value"] < 0.1, (
            f"p={result['p_value']:.3f} should be < 0.1 for independent signal"
        )
