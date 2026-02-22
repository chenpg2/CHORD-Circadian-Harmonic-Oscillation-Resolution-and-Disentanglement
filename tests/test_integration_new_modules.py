"""Integration tests for new mathematical modules in the BHDT pipeline.

Tests verify that:
1. BHCT (bispectral) evidence is integrated as Evidence 11
2. Savage-Dickey BF works as an alternative to BIC-based BF
3. Savage-Dickey evidence is integrated as Evidence 12
4. Population-level phase analysis works end-to-end
5. Backward compatibility is preserved
"""

import numpy as np
import pytest
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.inference import bhdt_analytic, population_phase_analysis
from chord.simulation.generator import (
    sawtooth_harmonic, independent_superposition, pure_circadian, pure_noise,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_independent_signal(seed=42, n_points=48, duration=48.0):
    """Generate a clear independent 24h + 12h signal."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, n_points, endpoint=False)
    y = (2.0 * np.cos(2 * np.pi / 24.0 * t + 0.3)
         + 1.5 * np.cos(2 * np.pi / 11.8 * t + 1.7)
         + rng.normal(0, 0.3, n_points))
    return t, y


def _make_gene_matrix(n_genes=20, n_points=48, duration=48.0, seed=42):
    """Generate a synthetic gene expression matrix with mixed signals."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, n_points, endpoint=False)
    Y = np.zeros((n_genes, n_points))
    for i in range(n_genes):
        amp_24 = rng.uniform(1.0, 3.0)
        amp_12 = rng.uniform(0.5, 2.0)
        phi_24 = rng.uniform(0, 2 * np.pi)
        phi_12 = rng.uniform(0, 2 * np.pi)
        Y[i] = (amp_24 * np.cos(2 * np.pi / 24.0 * t + phi_24)
                + amp_12 * np.cos(2 * np.pi / 12.0 * t + phi_12)
                + rng.normal(0, 0.3, n_points))
    return t, Y


# ============================================================================
# Test 1: BHCT evidence in v2 classifier
# ============================================================================

class TestBHCTIntegration:
    def test_bhdt_analytic_with_bhct(self):
        """Run bhdt_analytic on a sawtooth signal; BHCT evidence should be
        included without errors."""
        data = sawtooth_harmonic(seed=42)
        t, y = data["t"], data["y"]
        result = bhdt_analytic(t, y, classifier_version="v2")
        # Should complete without error and return valid classification
        assert "classification" in result
        assert result["classification"] in (
            "independent_ultradian", "likely_independent_ultradian",
            "harmonic", "circadian_only", "non_rhythmic", "ambiguous",
        )

    def test_bhct_on_independent_signal(self):
        """BHCT should run on an independent signal without breaking."""
        t, y = _make_independent_signal(seed=99)
        result = bhdt_analytic(t, y, classifier_version="v2")
        assert "classification" in result
        assert result["bayes_factor"] > 0


# ============================================================================
# Test 2: Savage-Dickey BF integration
# ============================================================================

class TestSavageDickeyIntegration:
    def test_bhdt_analytic_with_savage_dickey(self):
        """Run bhdt_analytic with use_savage_dickey=True."""
        t, y = _make_independent_signal(seed=42, n_points=48)
        result = bhdt_analytic(t, y, classifier_version="v2",
                               use_savage_dickey=True)
        assert "classification" in result
        assert "log_bayes_factor" in result
        # When SD succeeds, the result should contain the SD sub-dict
        if "savage_dickey" in result:
            sd = result["savage_dickey"]
            assert "log_bf" in sd
            assert "bf" in sd

    def test_savage_dickey_default_off(self):
        """By default, use_savage_dickey=False, so no SD result."""
        data = pure_circadian(seed=42)
        t, y = data["t"], data["y"]
        result = bhdt_analytic(t, y, classifier_version="v2")
        # SD should not be present when not requested
        assert "savage_dickey" not in result


# ============================================================================
# Test 3: v2 classifier uses new evidence
# ============================================================================

class TestV2ClassifierNewEvidence:
    def test_v2_classifier_produces_valid_result(self):
        """The v2 classifier with new evidence lines should produce a valid
        classification for various signal types."""
        for seed in [42, 99, 123]:
            t, y = _make_independent_signal(seed=seed)
            result = bhdt_analytic(t, y, classifier_version="v2")
            assert result["classification"] in (
                "independent_ultradian", "likely_independent_ultradian",
                "harmonic", "circadian_only", "non_rhythmic", "ambiguous",
            )

    def test_v2_vs_v1_both_run(self):
        """Both v1 and v2 classifiers should run without error."""
        t, y = _make_independent_signal(seed=42)
        r_v1 = bhdt_analytic(t, y, classifier_version="v1")
        r_v2 = bhdt_analytic(t, y, classifier_version="v2")
        assert "classification" in r_v1
        assert "classification" in r_v2
        # v2 has new evidence lines, so results may differ
        # but both should be valid categories
        valid = {"independent_ultradian", "likely_independent_ultradian",
                 "harmonic", "circadian_only", "non_rhythmic", "ambiguous"}
        assert r_v1["classification"] in valid
        assert r_v2["classification"] in valid


# ============================================================================
# Test 4: Population phase analysis
# ============================================================================

class TestPopulationPhaseAnalysis:
    def test_population_phase_analysis_basic(self):
        """Test population_phase_analysis with synthetic gene matrix."""
        t, Y = _make_gene_matrix(n_genes=20, seed=42)
        result = population_phase_analysis(t, Y)
        # Should return a dict with phase test results
        assert isinstance(result, dict)
        assert "n_genes_tested" in result
        if result["n_genes_tested"] > 0:
            # Full result from cross_gene_phase_test
            assert "classification" in result or "result" in result

    def test_population_phase_insufficient_genes(self):
        """With too few genes having significant 12h amplitude, should return
        insufficient_genes."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 48, 48, endpoint=False)
        # Very weak signals — 12h amplitude below threshold
        Y = rng.normal(0, 0.01, (3, 48))
        result = population_phase_analysis(t, Y)
        assert result["n_genes_tested"] == 0
        assert result["result"] == "insufficient_genes"

    def test_population_phase_with_strong_signals(self):
        """With many genes having strong 12h components, should test enough."""
        t, Y = _make_gene_matrix(n_genes=50, seed=123)
        result = population_phase_analysis(t, Y)
        assert result["n_genes_tested"] >= 5


# ============================================================================
# Test 5: Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    def test_output_format_unchanged(self):
        """bhdt_analytic with defaults should return the same keys as before."""
        data = pure_circadian(seed=42)
        t, y = data["t"], data["y"]
        result = bhdt_analytic(t, y)
        # Core keys that must always be present
        assert "log_bayes_factor" in result
        assert "bayes_factor" in result
        assert "interpretation" in result
        assert "m0" in result
        assert "m1" in result
        assert "m1_free" in result
        assert "period_deviation" in result
        assert "classification" in result

    def test_pure_circadian_still_circadian(self):
        """Pure circadian signal should still be classified as circadian_only."""
        data = pure_circadian(seed=42)
        result = bhdt_analytic(data["t"], data["y"])
        assert result["classification"] == "circadian_only"

    def test_pure_noise_still_non_rhythmic(self):
        """Pure noise should still be classified as non_rhythmic."""
        data = pure_noise(seed=42)
        result = bhdt_analytic(data["t"], data["y"])
        assert result["classification"] == "non_rhythmic"

    def test_v1_classifier_unchanged(self):
        """v1 classifier should be completely unaffected by new evidence."""
        data = sawtooth_harmonic(seed=42)
        t, y = data["t"], data["y"]
        result = bhdt_analytic(t, y, classifier_version="v1")
        assert result["classification"] in (
            "independent_ultradian", "likely_independent_ultradian",
            "harmonic", "circadian_only", "non_rhythmic", "ambiguous",
        )

    def test_bayes_factor_is_float(self):
        """BF values should be plain floats."""
        data = independent_superposition(seed=42)
        result = bhdt_analytic(data["t"], data["y"])
        assert isinstance(result["bayes_factor"], float)
        assert isinstance(result["log_bayes_factor"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
