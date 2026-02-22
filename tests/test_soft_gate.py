"""
Tests for the soft F-test gate (_classify_gene_v2) and permutation F-test.

Verifies that the v2 classifier:
  - Lets borderline 12h genes enter multi-evidence scoring instead of hard-gating
  - Still classifies truly absent 12h as circadian_only
  - permutation_f_test detects clear 12h signals
  - permutation_f_test does NOT detect 12h in pure 24h signals
  - classifier_version="v1" preserves backward compatibility
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.inference import bhdt_analytic, permutation_f_test, component_f_test
from chord.simulation.generator import pure_circadian, independent_superposition


# ============================================================================
# Helpers
# ============================================================================

def _make_borderline_12h(seed=42):
    """Create a signal with borderline 12h component (p_12 ~ 0.05-0.15).

    24h + moderate 12h with independent phase, enough to be biologically real
    but weak enough that a hard F-test gate at alpha=0.05 might miss it.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2, dtype=float)  # 24 points
    # Strong 24h + moderate 12h (A_12/A_24 ~ 0.5, enough to matter)
    y = (5.0
         + 2.0 * np.cos(2 * np.pi * t / 24.0)
         + 0.8 * np.cos(2 * np.pi * t / 12.0 - 1.2)  # independent phase
         + rng.normal(0, 0.7, len(t)))
    return t, y


def _make_truly_absent_12h(seed=42):
    """Create a pure 24h signal with no 12h component at all."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2, dtype=float)
    y = 5.0 + 2.0 * np.cos(2 * np.pi * t / 24.0) + rng.normal(0, 0.5, len(t))
    return t, y


def _make_clear_12h(seed=42):
    """Create a signal with a very clear independent 12h component."""
    rng = np.random.RandomState(seed)
    t = np.arange(0, 48, 2, dtype=float)
    y = (5.0
         + 2.0 * np.cos(2 * np.pi * t / 24.0)
         + 1.8 * np.cos(2 * np.pi * t / 12.0 - 0.8)
         + rng.normal(0, 0.4, len(t)))
    return t, y


# ============================================================================
# Test: Borderline 12h enters scoring (not hard-gated out)
# ============================================================================

class TestSoftGate:
    def test_borderline_12h_enters_scoring(self):
        """A gene with p_12 ~ 0.08 and high amp_ratio should NOT be
        classified as circadian_only by v2. The soft gate should let it
        through to multi-evidence scoring."""
        t, y = _make_borderline_12h(seed=42)
        result = bhdt_analytic(t, y, classifier_version="v2")
        # The key assertion: v2 should NOT hard-gate this as circadian_only
        assert result["classification"] != "circadian_only", (
            f"v2 classifier hard-gated a borderline 12h gene as circadian_only. "
            f"Classification: {result['classification']}"
        )
        # It should enter scoring and get some classification with 12h consideration
        assert result["classification"] in (
            "independent_ultradian",
            "likely_independent_ultradian",
            "ambiguous",
            "harmonic",
        )

    def test_truly_absent_12h_still_circadian_only(self):
        """A gene with p_12 > 0.5 and low amp_ratio should still be
        classified as circadian_only by v2."""
        t, y = _make_truly_absent_12h(seed=42)
        result = bhdt_analytic(t, y, classifier_version="v2")
        assert result["classification"] == "circadian_only", (
            f"v2 should classify truly absent 12h as circadian_only, "
            f"got: {result['classification']}"
        )


# ============================================================================
# Test: Permutation F-test
# ============================================================================

class TestPermutationFTest:
    def test_permutation_f_test_significant(self):
        """Clear 12h signal should be detected by permutation F-test."""
        t, y = _make_clear_12h(seed=42)
        periods = [24.0, 12.0, 8.0]
        result = permutation_f_test(t, y, periods, test_period_idx=1, n_perm=999, seed=42)
        assert "F_stat" in result
        assert "p_value" in result
        assert "significant" in result
        assert "n_perm" in result
        assert result["significant"] is True, (
            f"Permutation F-test failed to detect clear 12h signal. "
            f"p={result['p_value']:.4f}"
        )
        assert result["p_value"] < 0.05

    def test_permutation_f_test_not_significant(self):
        """Pure 24h signal should NOT have significant 12h in permutation F-test."""
        t, y = _make_truly_absent_12h(seed=42)
        periods = [24.0, 12.0, 8.0]
        result = permutation_f_test(t, y, periods, test_period_idx=1, n_perm=999, seed=42)
        assert result["significant"] is False, (
            f"Permutation F-test falsely detected 12h in pure 24h signal. "
            f"p={result['p_value']:.4f}"
        )
        assert result["p_value"] > 0.05


# ============================================================================
# Test: v1 backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    def test_v1_backward_compatible(self):
        """classifier_version='v1' should give the same results as the
        original _classify_gene function."""
        # Pure circadian → circadian_only in v1
        data = pure_circadian(seed=42)
        t, y = data["t"], data["y"]
        result_v1 = bhdt_analytic(t, y, classifier_version="v1")
        result_default_old = bhdt_analytic(t, y)  # default is now v2
        # v1 should still produce circadian_only for pure circadian
        assert result_v1["classification"] == "circadian_only", (
            f"v1 backward compat broken: expected circadian_only, "
            f"got {result_v1['classification']}"
        )

    def test_v1_noise_still_non_rhythmic(self):
        """v1 should still classify noise as non_rhythmic."""
        from chord.simulation.generator import pure_noise
        data = pure_noise(seed=42)
        t, y = data["t"], data["y"]
        result = bhdt_analytic(t, y, classifier_version="v1")
        assert result["classification"] == "non_rhythmic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
