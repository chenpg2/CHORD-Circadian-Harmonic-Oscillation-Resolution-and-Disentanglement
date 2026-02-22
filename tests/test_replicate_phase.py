"""Tests for Replicate Phase Consistency Test."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.replicate_phase import (
    _fit_phase,
    replicate_phase_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_time(n_points=48, duration=48.0):
    """Evenly spaced time points over ``duration`` hours."""
    return np.linspace(0, duration, n_points, endpoint=False)


# ---------------------------------------------------------------------------
# test_harmonic_replicates_consistent
# ---------------------------------------------------------------------------

class TestHarmonicReplicatesConsistent:
    def test_harmonic_signal_low_variance(self):
        """Harmonic signal (phi_12 = 2*phi_24) across 4 replicates
        should yield low circular variance and is_harmonic=True."""
        rng = np.random.RandomState(42)
        t = _make_time(48, 48.0)
        delta = 0.8  # fixed waveform-shape constant

        replicates = []
        for _ in range(4):
            phi_24 = rng.uniform(0, 2 * np.pi)
            phi_12 = 2.0 * phi_24 + delta  # exact harmonic relationship
            y = (
                3.0 * np.cos(2 * np.pi * t / 24.0 - phi_24)
                + 1.5 * np.cos(2 * np.pi * t / 12.0 - phi_12)
                + rng.normal(0, 0.1, size=len(t))
            )
            replicates.append(y)

        result = replicate_phase_consistency(t, replicates)

        assert result["circular_variance"] < 0.2
        assert result["is_harmonic"] is True
        assert result["consistency_score"] < 0


# ---------------------------------------------------------------------------
# test_independent_replicates_inconsistent
# ---------------------------------------------------------------------------

class TestIndependentReplicatesInconsistent:
    def test_independent_random_phases_high_variance(self):
        """Independent random phases across 20 replicates should yield
        high circular variance."""
        rng = np.random.RandomState(99)
        t = _make_time(48, 48.0)

        replicates = []
        for _ in range(20):
            phi_24 = rng.uniform(0, 2 * np.pi)
            phi_12 = rng.uniform(0, 2 * np.pi)  # independent
            y = (
                3.0 * np.cos(2 * np.pi * t / 24.0 - phi_24)
                + 1.5 * np.cos(2 * np.pi * t / 12.0 - phi_12)
                + rng.normal(0, 0.1, size=len(t))
            )
            replicates.append(y)

        result = replicate_phase_consistency(t, replicates)

        assert result["circular_variance"] > 0.4
        assert result["is_harmonic"] is False
        assert result["consistency_score"] > 0


# ---------------------------------------------------------------------------
# test_replicate_phase_returns_fields
# ---------------------------------------------------------------------------

class TestReplicatePhaseReturnsFields:
    def test_all_expected_keys_present(self):
        """Return dict must contain all documented keys."""
        rng = np.random.RandomState(7)
        t = _make_time(48, 48.0)
        replicates = [
            rng.normal(0, 1, size=len(t)) for _ in range(3)
        ]

        result = replicate_phase_consistency(t, replicates)

        expected_keys = {
            "circular_variance",
            "mean_resultant_length",
            "residual_phases",
            "is_harmonic",
            "consistency_score",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# test_single_replicate
# ---------------------------------------------------------------------------

class TestSingleReplicate:
    def test_single_replicate_defaults(self):
        """Edge case: 1 replicate should return safe defaults
        (V=0.5, score=0)."""
        rng = np.random.RandomState(11)
        t = _make_time(48, 48.0)
        replicates = [rng.normal(0, 1, size=len(t))]

        result = replicate_phase_consistency(t, replicates)

        assert result["circular_variance"] == pytest.approx(0.5)
        assert result["consistency_score"] == 0
        assert result["is_harmonic"] is False
        assert len(result["residual_phases"]) == 1


# ---------------------------------------------------------------------------
# test_two_replicates
# ---------------------------------------------------------------------------

class TestTwoReplicates:
    def test_two_replicates_harmonic(self):
        """Minimum useful case: 2 replicates with harmonic relationship."""
        rng = np.random.RandomState(55)
        t = _make_time(48, 48.0)
        delta = 1.0

        replicates = []
        for _ in range(2):
            phi_24 = rng.uniform(0, 2 * np.pi)
            phi_12 = 2.0 * phi_24 + delta
            y = (
                3.0 * np.cos(2 * np.pi * t / 24.0 - phi_24)
                + 1.5 * np.cos(2 * np.pi * t / 12.0 - phi_12)
                + rng.normal(0, 0.05, size=len(t))
            )
            replicates.append(y)

        result = replicate_phase_consistency(t, replicates)

        # With only 2 replicates the result should still be computable
        assert "circular_variance" in result
        assert "mean_resultant_length" in result
        assert len(result["residual_phases"]) == 2
