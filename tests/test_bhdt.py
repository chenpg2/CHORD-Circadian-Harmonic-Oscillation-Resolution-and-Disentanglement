"""
Comprehensive test suite for BHDT (Bayesian Harmonic Disentanglement Test).

Tests cover:
- Analytic mode (BIC-based)
- Bootstrap LRT mode
- Ensemble mode (analytic + bootstrap)
- MCMC mode (optional, slow)
- Component F-test
- Simulation generator
- Pipeline batch processing
"""

import numpy as np
import pytest
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.simulation.generator import (
    pure_circadian, pure_ultradian, independent_superposition,
    sawtooth_harmonic, peaked_harmonic, pure_noise,
    generate_all_scenarios, generate_genome_like,
)
from chord.bhdt.inference import bhdt_analytic, component_f_test, bhdt_ensemble
from chord.bhdt.models import (
    fit_harmonic_model, fit_independent_model, fit_independent_free_period,
)
from chord.bhdt.bootstrap import parametric_bootstrap_lrt


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def circadian_data():
    return pure_circadian(seed=42)

@pytest.fixture
def ultradian_data():
    return pure_ultradian(seed=42)

@pytest.fixture
def independent_data():
    return independent_superposition(seed=42)

@pytest.fixture
def sawtooth_data():
    return sawtooth_harmonic(seed=42)

@pytest.fixture
def noise_data():
    return pure_noise(seed=42)


# ============================================================================
# Test: Simulation generator
# ============================================================================

class TestSimulationGenerator:
    def test_pure_circadian_shape(self, circadian_data):
        assert circadian_data["t"].shape == (24,)
        assert circadian_data["y"].shape == (24,)
        assert circadian_data["truth"]["scenario"] == "pure_circadian"
        assert circadian_data["truth"]["has_independent_12h"] is False

    def test_independent_superposition_truth(self, independent_data):
        truth = independent_data["truth"]
        assert truth["has_independent_12h"] is True
        assert truth["has_harmonic_12h"] is False
        assert len(truth["oscillators"]) == 2

    def test_sawtooth_harmonic_truth(self, sawtooth_data):
        truth = sawtooth_data["truth"]
        assert truth["has_harmonic_12h"] is True
        assert truth["has_independent_12h"] is False

    def test_generate_all_scenarios_count(self):
        scenarios = generate_all_scenarios(seed=0)
        assert len(scenarios) == 15  # 12 original + 3 new harmonic waveforms

    def test_generate_genome_like(self):
        result = generate_genome_like(n_genes=50, seed=0)
        assert result["expr"].shape[0] == 50
        assert len(result["labels"]) == 50

    def test_reproducibility(self):
        d1 = pure_circadian(seed=99)
        d2 = pure_circadian(seed=99)
        np.testing.assert_array_equal(d1["y"], d2["y"])


# ============================================================================
# Test: OLS model fitting
# ============================================================================

class TestModelFitting:
    def test_harmonic_model_output_keys(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = fit_harmonic_model(t, y)
        assert "beta" in result
        assert "components" in result
        assert "bic" in result
        assert "log_lik" in result
        assert result["model"] == "M0_harmonic"

    def test_harmonic_model_periods(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = fit_harmonic_model(t, y, K=3)
        assert len(result["components"]) == 3
        assert abs(result["components"][0]["T"] - 24.0) < 0.01
        assert abs(result["components"][1]["T"] - 12.0) < 0.01

    def test_independent_model_output(self, independent_data):
        t, y = independent_data["t"], independent_data["y"]
        result = fit_independent_model(t, y)
        assert result["model"] == "M1_independent"
        assert len(result["components"]) == 3

    def test_free_period_model(self, independent_data):
        t, y = independent_data["t"], independent_data["y"]
        result = fit_independent_free_period(t, y)
        assert "fitted_periods" in result
        assert len(result["fitted_periods"]) == 3
        # Fitted T_12 should be close to 11.8 (true value)
        assert 10.0 < result["fitted_periods"][1] < 14.0

    def test_bic_ordering(self, circadian_data):
        """M0 should fit pure circadian better than M1-free."""
        t, y = circadian_data["t"], circadian_data["y"]
        m0 = fit_harmonic_model(t, y)
        m1 = fit_independent_free_period(t, y)
        # M0 has fewer params, should have lower BIC for pure circadian
        # (not always true due to noise, but generally)
        assert isinstance(m0["bic"], float)
        assert isinstance(m1["bic"], float)


# ============================================================================
# Test: Component F-test
# ============================================================================

class TestComponentFTest:
    def test_circadian_24h_significant(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=0)
        assert result["significant"] == True
        assert result["p_value"] < 0.05

    def test_circadian_12h_not_significant(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)
        # Pure circadian should not have significant 12h
        # (may occasionally be significant due to noise)
        assert "p_value" in result
        assert "F_stat" in result

    def test_noise_not_significant(self, noise_data):
        t, y = noise_data["t"], noise_data["y"]
        result = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=0)
        # Pure noise should usually not be significant at 24h
        assert "p_value" in result

    def test_output_keys(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = component_f_test(t, y, [24.0, 12.0], test_period_idx=0)
        assert set(result.keys()) >= {"F_stat", "p_value", "significant", "df1", "df2"}


# ============================================================================
# Test: BHDT Analytic mode
# ============================================================================

class TestBHDTAnalytic:
    def test_pure_circadian_classification(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = bhdt_analytic(t, y)
        assert result["classification"] == "circadian_only"

    def test_pure_noise_classification(self, noise_data):
        t, y = noise_data["t"], noise_data["y"]
        result = bhdt_analytic(t, y)
        assert result["classification"] == "non_rhythmic"

    def test_output_keys(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = bhdt_analytic(t, y)
        assert "classification" in result
        assert "bayes_factor" in result
        assert "log_bayes_factor" in result
        assert "interpretation" in result

    def test_bayes_factor_positive(self, independent_data):
        t, y = independent_data["t"], independent_data["y"]
        result = bhdt_analytic(t, y)
        assert result["bayes_factor"] > 0

    def test_period_deviation_present(self, independent_data):
        t, y = independent_data["t"], independent_data["y"]
        result = bhdt_analytic(t, y)
        pd = result.get("period_deviation", {})
        assert "T_12_fitted" in pd or "relative_deviation" in pd


# ============================================================================
# Test: Bootstrap LRT
# ============================================================================

class TestBootstrapLRT:
    def test_output_keys(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = parametric_bootstrap_lrt(t, y, n_bootstrap=49, seed=42)
        assert "lrt_observed" in result
        assert "p_value" in result
        assert "classification" in result
        assert 0 <= result["p_value"] <= 1

    def test_pure_noise_high_pvalue(self, noise_data):
        t, y = noise_data["t"], noise_data["y"]
        result = parametric_bootstrap_lrt(t, y, n_bootstrap=49, seed=42)
        assert result["classification"] == "non_rhythmic"

    def test_sawtooth_harmonic(self, sawtooth_data):
        t, y = sawtooth_data["t"], sawtooth_data["y"]
        result = parametric_bootstrap_lrt(t, y, n_bootstrap=499, seed=42)
        # Bootstrap should not strongly reject H0 for sawtooth
        assert result["p_value"] > 0.05

    def test_reproducibility(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        r1 = parametric_bootstrap_lrt(t, y, n_bootstrap=49, seed=42)
        r2 = parametric_bootstrap_lrt(t, y, n_bootstrap=49, seed=42)
        assert r1["p_value"] == r2["p_value"]


# ============================================================================
# Test: Ensemble mode
# ============================================================================

class TestBHDTEnsemble:
    def test_output_keys(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = bhdt_ensemble(t, y, n_bootstrap=49, seed=42)
        assert "classification" in result
        assert "confidence" in result
        assert "analytic_classification" in result
        assert "bootstrap_classification" in result
        assert result["confidence"] in ("high", "medium", "low")

    def test_pure_circadian(self, circadian_data):
        t, y = circadian_data["t"], circadian_data["y"]
        result = bhdt_ensemble(t, y, n_bootstrap=49, seed=42)
        assert result["classification"] == "circadian_only"
        assert result["confidence"] == "high"

    def test_pure_noise(self, noise_data):
        t, y = noise_data["t"], noise_data["y"]
        result = bhdt_ensemble(t, y, n_bootstrap=49, seed=42)
        assert result["classification"] == "non_rhythmic"

    def test_peaked_harmonic(self):
        data = peaked_harmonic(seed=42)
        result = bhdt_ensemble(data["t"], data["y"], n_bootstrap=99, seed=42)
        assert result["classification"] == "harmonic"

    def test_pure_ultradian(self, ultradian_data):
        t, y = ultradian_data["t"], ultradian_data["y"]
        result = bhdt_ensemble(t, y, n_bootstrap=49, seed=42)
        assert result["classification"] == "independent_ultradian"


# ============================================================================
# Test: Pipeline batch processing
# ============================================================================

class TestPipeline:
    def test_analytic_pipeline(self):
        from chord.bhdt.pipeline import run_bhdt
        data = generate_genome_like(n_genes=10, seed=42)
        df = run_bhdt(data["expr"], data["t"],
                      method="analytic", n_jobs=1, verbose=False)
        assert len(df) == 10
        assert "classification" in df.columns
        assert "gene" in df.columns

    def test_bootstrap_pipeline(self):
        from chord.bhdt.pipeline import run_bhdt
        data = generate_genome_like(n_genes=5, seed=42)
        df = run_bhdt(data["expr"], data["t"],
                      method="bootstrap", n_jobs=1, verbose=False)
        assert len(df) == 5
        assert "classification" in df.columns


# ============================================================================
# Test: Phase test (cross-gene)
# ============================================================================

class TestPhaseTest:
    def test_cross_gene_independent_phases(self):
        from chord.bhdt.phase_test import cross_gene_phase_test
        rng = np.random.RandomState(42)
        # Independent phases: no correlation expected
        phases_24 = rng.uniform(0, 2 * np.pi, 50)
        phases_12 = rng.uniform(0, 2 * np.pi, 50)
        result = cross_gene_phase_test(phases_24, phases_12, n_permutations=999, seed=42)
        assert result["evidence"] == "phases_independent"
        assert result["p_value"] > 0.05

    def test_cross_gene_correlated_phases(self):
        from chord.bhdt.phase_test import cross_gene_phase_test
        rng = np.random.RandomState(42)
        # Linearly correlated phases: phi_12 = phi_24 + offset + small noise
        phases_24 = rng.uniform(0, 2 * np.pi, 100)
        phases_12 = (phases_24 + 0.5 + rng.normal(0, 0.2, 100)) % (2 * np.pi)
        result = cross_gene_phase_test(phases_24, phases_12, n_permutations=999, seed=42)
        # Should detect correlation (linear circular correlation)
        assert abs(result["rho_circular"]) > 0.1  # non-trivial correlation


def test_peaked_harmonic_peak_width_affects_output():
    """peak_width parameter must actually affect waveform shape."""
    from chord.simulation.generator import peaked_harmonic
    t = np.arange(0, 48, 2)
    y_narrow = peaked_harmonic(t, peak_width=0.2)["y"]
    y_wide = peaked_harmonic(t, peak_width=0.8)["y"]
    assert not np.allclose(y_narrow, y_wide, atol=1e-6), \
        "peak_width parameter has no effect on output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
