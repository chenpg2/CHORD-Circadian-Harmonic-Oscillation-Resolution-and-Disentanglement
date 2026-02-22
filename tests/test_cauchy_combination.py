"""Tests for Cauchy Combination Test (CCT)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestCauchyCombination:
    """Test CCT implementation."""

    def test_single_significant_pvalue(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        p_values = [0.001, 0.5, 0.8, 0.9]
        result = cauchy_combine(p_values)
        assert result < 0.01, f"CCT should be significant when one p is 0.001, got {result}"

    def test_all_nonsignificant(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        p_values = [0.5, 0.6, 0.7, 0.8]
        result = cauchy_combine(p_values)
        assert result > 0.1, f"CCT should be non-significant, got {result}"

    def test_all_significant(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        p_values = [0.01, 0.02, 0.03, 0.01]
        result = cauchy_combine(p_values)
        assert result < 0.02, f"CCT should be very significant, got {result}"

    def test_returns_float_in_01(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        result = cauchy_combine([0.05, 0.1, 0.5])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_with_weights(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        p_values = [0.01, 0.9]
        r1 = cauchy_combine(p_values, weights=[0.9, 0.1])
        r2 = cauchy_combine(p_values, weights=[0.1, 0.9])
        assert r1 < r2, f"Weighting significant p should give smaller result: {r1} vs {r2}"

    def test_empty_raises(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        with pytest.raises(ValueError):
            cauchy_combine([])

    def test_boundary_pvalues(self):
        from chord.bhdt.detection.cauchy_combination import cauchy_combine
        result = cauchy_combine([0.0, 0.5, 0.5])
        assert np.isfinite(result)
        assert result < 0.01
        result2 = cauchy_combine([1.0, 0.5, 0.5])
        assert np.isfinite(result2)
