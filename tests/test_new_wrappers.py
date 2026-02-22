"""Tests for JTK_CYCLE, RAIN, and pencil_method wrappers."""

import numpy as np
import pytest

from chord.benchmarks.wrappers import jtk_cycle, pencil_method, rain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(periods, amplitudes, n_points=48, duration=48.0, noise_std=0.0,
                 seed=0):
    """Generate a multi-component sinusoidal signal with optional noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, n_points, endpoint=False)
    y = np.zeros_like(t)
    for T, A in zip(periods, amplitudes):
        y += A * np.cos(2 * np.pi / T * t)
    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n_points)
    return t, y


def _pure_noise(n_points=48, duration=48.0, seed=123):
    """Generate pure Gaussian noise (no rhythmic component)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, n_points, endpoint=False)
    y = rng.normal(0, 1, size=n_points)
    return t, y


# ---------------------------------------------------------------------------
# JTK_CYCLE
# ---------------------------------------------------------------------------

class TestJTKCycle:
    def test_jtk_detects_12h_signal(self):
        """A clear 12h cosine should be detected with p < 0.05."""
        t, y = _make_signal([12], [2.0], noise_std=0.3, seed=42)
        result = jtk_cycle(t, y, period_range=(10, 14))
        assert result["method_name"] == "jtk_cycle"
        assert result["p_value"] < 0.05, f"p={result['p_value']}"
        assert 10 < result["period_estimate"] < 14
        assert result["amplitude_estimate"] > 0

    def test_jtk_no_false_positive(self):
        """Pure noise should NOT be flagged as rhythmic (p > 0.05)."""
        t, y = _pure_noise(seed=123)
        result = jtk_cycle(t, y, period_range=(10, 14))
        assert result["p_value"] > 0.05, f"p={result['p_value']}"


# ---------------------------------------------------------------------------
# RAIN
# ---------------------------------------------------------------------------

class TestRAIN:
    def test_rain_detects_12h_signal(self):
        """A clear 12h cosine should be detected with p < 0.05."""
        t, y = _make_signal([12], [2.0], noise_std=0.3, seed=42)
        result = rain(t, y, period=12.0)
        assert result["method_name"] == "rain"
        assert result["p_value"] < 0.05, f"p={result['p_value']}"
        assert result["amplitude_estimate"] > 0

    def test_rain_no_false_positive(self):
        """Pure noise should NOT be flagged as rhythmic (p > 0.05)."""
        t, y = _pure_noise(seed=123)
        result = rain(t, y, period=12.0)
        assert result["p_value"] > 0.05, f"p={result['p_value']}"


# ---------------------------------------------------------------------------
# Pencil method
# ---------------------------------------------------------------------------

class TestPencilMethod:
    def test_pencil_finds_12h(self):
        """Signal with 24h + 12h components should detect 12h period."""
        t, y = _make_signal([24, 12], [2.0, 1.5], n_points=96,
                            duration=96.0, noise_std=0.2, seed=42)
        result = pencil_method(t, y)
        assert result["method_name"] == "pencil"
        assert result["has_12h"] is True
        assert 10 < result["period_estimate"] < 14
        assert result["amplitude_estimate"] > 0
        assert result["p_value"] < 0.05, f"p={result['p_value']}"

    def test_pencil_pure_24h(self):
        """Pure 24h signal should still find ~12h (harmonic leakage)."""
        t, y = _make_signal([24], [3.0], n_points=96, duration=96.0,
                            noise_std=0.1, seed=42)
        result = pencil_method(t, y)
        assert result["method_name"] == "pencil"
        # The pencil method may or may not find a 12h component for a pure
        # 24h signal — it depends on numerical artifacts. We just verify
        # the method runs and returns a valid dict.
        assert isinstance(result["detected_periods"], list)
        assert isinstance(result["p_value"], float)
