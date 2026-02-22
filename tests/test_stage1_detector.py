"""Tests for Stage 1 multi-method detection."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


def _make_sinusoidal_12h(seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    y = 2.0 * np.cos(2 * np.pi / 12.0 * t) + rng.normal(0, 0.5, len(t))
    return t, y


def _make_nonsinusoidal_12h(seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2.0)
    phase = (t % 12.0) / 12.0
    y = np.where(np.abs(phase - 0.3) < 0.1, 3.0, 0.0)
    y = y + rng.normal(0, 0.3, len(t))
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


class TestStage1Detector:

    def test_sinusoidal_12h_detected(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_sinusoidal_12h()
        result = stage1_detect(t, y)
        assert result["passed"], f"Sinusoidal 12h should pass Stage 1, p={result['p_detect']}"
        assert result["p_detect"] < 0.10

    def test_nonsinusoidal_12h_detected(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_nonsinusoidal_12h()
        result = stage1_detect(t, y)
        assert result["passed"], (
            f"Non-sinusoidal 12h should pass Stage 1, p={result['p_detect']}, "
            f"p_jtk={result['p_jtk']}, p_rain={result['p_rain']}"
        )

    def test_noise_not_detected(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_noise()
        result = stage1_detect(t, y)
        assert not result["passed"], f"Noise should not pass, p={result['p_detect']}"

    def test_circadian_only_classification(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_circadian_only()
        result = stage1_detect(t, y)
        assert result["stage1_class"] == "circadian_only", (
            f"Expected circadian_only, got {result['stage1_class']}"
        )

    def test_result_keys(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_sinusoidal_12h()
        result = stage1_detect(t, y)
        required_keys = [
            "p_detect", "p_ftest", "p_jtk", "p_rain", "p_harmreg",
            "p_24", "passed", "stage1_class", "best_detector",
            "detection_strength", "waveform_hint",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_custom_alpha(self):
        from chord.bhdt.detection.stage1_detector import stage1_detect
        t, y = _make_sinusoidal_12h()
        result = stage1_detect(t, y, alpha_detect=0.001)
        assert result["passed"] == (result["p_detect"] < 0.001)
