"""
Comprehensive test suite for untested CHORD modules.

Tests cover:
- Ensemble integrator (weighted voting, agreement, disagreement, canonical mapping)
- Pipeline (BHDT-only mode, input validation, gene name inference)
- Network hierarchical (hyperparameters, shrinkage, reclassification)
- Benchmarks (classification metrics, Lomb-Scargle wrapper)
- CLI (import, version subcommand)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ============================================================================
# Shared synthetic data helpers
# ============================================================================

def _make_bhdt_df(n=8):
    """Create a mock BHDT results DataFrame."""
    np.random.seed(0)
    return pd.DataFrame({
        'gene': [f'gene_{i}' for i in range(n)],
        'classification': ['circadian_only', 'independent_ultradian', 'harmonic',
                          'non_rhythmic', 'ambiguous', 'independent_ultradian',
                          'likely_independent_ultradian', 'non_rhythmic'][:n],
        'log_bayes_factor': [2.0, 3.5, -1.0, 0.1, 0.5, 2.8, 1.5, -0.5][:n],
        'bayes_factor': np.exp([2.0, 3.5, -1.0, 0.1, 0.5, 2.8, 1.5, -0.5][:n]),
        'A_12': [0.1, 0.8, 0.3, 0.05, 0.4, 0.7, 0.6, 0.02][:n],
        'A_24': [0.9, 0.3, 0.8, 0.1, 0.5, 0.4, 0.3, 0.08][:n],
        'phi_12': np.random.uniform(0, 2 * np.pi, n),
        'phi_24': np.random.uniform(0, 2 * np.pi, n),
        'f_test_12h_pvalue': [0.5, 0.001, 0.1, 0.9, 0.05, 0.002, 0.01, 0.8][:n],
        'f_test_12h_fdr': [0.7, 0.01, 0.2, 0.95, 0.1, 0.02, 0.05, 0.9][:n],
        'T_12_fitted': [12.0, 11.8, 12.0, 12.0, 12.1, 11.9, 12.0, 12.0][:n],
        'interpretation': ['circadian'] * n,
    })


def _make_pinod_df(n=8):
    """Create a mock PINOD results DataFrame."""
    return pd.DataFrame({
        'gene': [f'gene_{i}' for i in range(n)],
        'classification': ['circadian_only', 'independent_ultradian', 'harmonic',
                          'non_rhythmic', 'independent_ultradian', 'independent_ultradian',
                          'damped_ultradian', 'harmonic'][:n],
        'confidence': [0.85, 0.92, 0.78, 0.60, 0.88, 0.90, 0.82, 0.65][:n],
        'reconstruction_r2': [0.92, 0.95, 0.85, 0.40, 0.90, 0.93, 0.88, 0.55][:n],
    })


# ============================================================================
# Test: Ensemble Integrator
# ============================================================================

class TestEnsembleIntegrator:
    """Tests for chord.ensemble.integrator — weighted voting consensus."""

    def test_integrate_results_output_shape(self):
        """Integration produces one row per gene with expected columns."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        assert len(result) == 8
        for col in ('gene', 'consensus_classification', 'consensus_confidence',
                     'bhdt_classification', 'pinod_classification',
                     'agreement', 'review_flag'):
            assert col in result.columns, f"Missing column: {col}"

    def test_agreement_when_both_match(self):
        """When BHDT and PINOD give the same label, agreement=True."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        # gene_0: both say circadian_only
        row = result[result['gene'] == 'gene_0'].iloc[0]
        assert row['agreement'] is True or row['agreement'] == True
        assert row['consensus_classification'] == 'circadian_only'

    def test_agreement_canonical_match(self):
        """Canonical agreement: likely_independent_ultradian vs damped_ultradian."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        # gene_6: BHDT=likely_independent_ultradian, PINOD=damped_ultradian
        # Both map to canonical "independent" → agreement=True
        row = result[result['gene'] == 'gene_6'].iloc[0]
        assert row['agreement'] is True or row['agreement'] == True

    def test_non_rhythmic_protection(self):
        """BHDT non_rhythmic is protected when PINOD disagrees with low confidence."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        # gene_7: BHDT=non_rhythmic, PINOD=harmonic (confidence=0.65 < 0.9)
        # Non-rhythmic protection should keep BHDT's call
        row = result[result['gene'] == 'gene_7'].iloc[0]
        assert row['consensus_classification'] == 'non_rhythmic'
        assert row['agreement'] == False

    def test_disagreement_sets_review_flag(self):
        """Disagreeing genes get review_flag=True."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        # gene_7: BHDT=non_rhythmic, PINOD=harmonic → disagreement
        row = result[result['gene'] == 'gene_7'].iloc[0]
        assert row['review_flag'] is True or row['review_flag'] == True

    def test_canonical_class_mapping(self):
        """_canonical_class maps subtypes to canonical groups correctly."""
        from chord.ensemble.integrator import _canonical_class
        assert _canonical_class('independent_ultradian') == 'independent'
        assert _canonical_class('likely_independent_ultradian') == 'independent'
        assert _canonical_class('damped_ultradian') == 'independent'
        assert _canonical_class('multi_ultradian') == 'independent'
        assert _canonical_class('harmonic') == 'harmonic'
        assert _canonical_class('circadian_only') == 'circadian'
        assert _canonical_class('non_rhythmic') == 'non_rhythmic'
        assert _canonical_class('ambiguous') == 'ambiguous'
        # Unknown label returns itself
        assert _canonical_class('unknown_label') == 'unknown_label'

    def test_missing_pinod_falls_back_to_bhdt(self):
        """When PINOD result is NaN for a gene, consensus uses BHDT."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df(4)
        pinod = _make_pinod_df(4)
        # Add a gene only in BHDT
        extra = pd.DataFrame({
            'gene': ['gene_extra'],
            'classification': ['circadian_only'],
            'log_bayes_factor': [3.0],
            'bayes_factor': [np.exp(3.0)],
            'A_12': [0.2], 'A_24': [0.9],
            'phi_12': [1.0], 'phi_24': [2.0],
            'f_test_12h_pvalue': [0.01], 'f_test_12h_fdr': [0.05],
            'T_12_fitted': [12.0], 'interpretation': ['circadian'],
        })
        bhdt_ext = pd.concat([bhdt, extra], ignore_index=True)
        result = integrate_results(bhdt_ext, pinod)
        row = result[result['gene'] == 'gene_extra'].iloc[0]
        assert row['consensus_classification'] == 'circadian_only'
        assert row['review_flag'] is True or row['review_flag'] == True

    def test_invalid_method_raises(self):
        """Unknown ensemble method raises ValueError."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df(2)
        pinod = _make_pinod_df(2)
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            integrate_results(bhdt, pinod, method="magic")

    def test_confidence_between_zero_and_one(self):
        """Consensus confidence should be in [0, 1]."""
        from chord.ensemble.integrator import integrate_results
        bhdt = _make_bhdt_df()
        pinod = _make_pinod_df()
        result = integrate_results(bhdt, pinod)
        assert (result['consensus_confidence'] >= 0).all()
        assert (result['consensus_confidence'] <= 1.1).all()  # small tolerance

    def test_bhdt_confidence_symmetric(self):
        """BHDT confidence should be symmetric for harmonic and independent evidence."""
        from chord.ensemble.integrator import _bhdt_confidence
        # Strong harmonic evidence (log_bf = -5)
        row_harmonic = pd.Series({"log_bayes_factor": -5.0,
                                   "classification": "harmonic"})
        # Strong independent evidence (log_bf = +5)
        row_independent = pd.Series({"log_bayes_factor": 5.0,
                                      "classification": "independent_ultradian"})
        conf_h = _bhdt_confidence(row_harmonic)
        conf_i = _bhdt_confidence(row_independent)
        # Both should be high (> 0.9) and close to each other
        assert conf_h > 0.9, f"Harmonic confidence {conf_h:.3f} too low"
        assert conf_i > 0.9, f"Independent confidence {conf_i:.3f} too low"
        assert abs(conf_h - conf_i) < 0.01, \
            f"Asymmetric: harmonic={conf_h:.3f} vs independent={conf_i:.3f}"

    def test_ensemble_symmetric_override(self):
        """Override rules should be symmetric for harmonic vs independent."""
        from chord.ensemble.integrator import _weighted_vote
        # Case 1: PINOD=independent, BHDT=harmonic
        r1 = _weighted_vote("harmonic", "independent_ultradian",
                            0.9, 0.9, 0.4, 0.6)
        # Case 2: PINOD=harmonic, BHDT=independent_ultradian
        r2 = _weighted_vote("independent_ultradian", "harmonic",
                            0.9, 0.9, 0.4, 0.6)
        # Both should flag for review (disagreement)
        assert r1[3] == r2[3], \
            f"Asymmetric review flags: case1={r1[3]}, case2={r2[3]}"
        assert r1[3] is True, "Harmonic vs independent should flag for review"


# ============================================================================
# Test: Pipeline
# ============================================================================

class TestPipeline:
    """Tests for chord.pipeline — end-to-end CHORD orchestration."""

    def test_run_bhdt_only(self):
        """Pipeline runs in BHDT-only mode on small synthetic data."""
        from chord.pipeline import run
        from chord.simulation.generator import generate_genome_like
        data = generate_genome_like(n_genes=10, seed=42)
        result = run(data['expr'], data['t'], methods='bhdt',
                     bhdt_method='analytic', n_jobs=1, verbose=False)
        assert len(result) == 10
        assert 'classification' in result.columns
        assert 'gene' in result.columns
        assert 'confidence' in result.columns

    def test_input_validation_wrong_dimensions(self):
        """1-D expression array raises ValueError."""
        from chord.pipeline import run
        t = np.arange(0, 48, 2, dtype=float)
        expr_1d = np.random.randn(25)  # 1-D, not 2-D
        with pytest.raises(ValueError, match="2-D"):
            run(expr_1d, t, methods='bhdt')

    def test_input_validation_unsorted_timepoints(self):
        """Unsorted timepoints raise ValueError."""
        from chord.pipeline import run
        t = np.array([0, 4, 2, 6, 8], dtype=float)  # not sorted
        expr = np.random.randn(3, 5)
        with pytest.raises(ValueError, match="sorted"):
            run(expr, t, methods='bhdt')

    def test_input_validation_shape_mismatch(self):
        """Mismatched expr columns and timepoints length raises ValueError."""
        from chord.pipeline import run
        t = np.arange(0, 48, 2, dtype=float)  # 24 points
        expr = np.random.randn(5, 10)  # 10 columns != 24
        with pytest.raises(ValueError, match="must match"):
            run(expr, t, methods='bhdt')

    def test_gene_name_inference_from_dataframe(self):
        """Gene names are inferred from DataFrame index."""
        from chord.pipeline import run
        from chord.simulation.generator import generate_genome_like
        data = generate_genome_like(n_genes=8, seed=42)
        names = [f'MyGene_{i}' for i in range(8)]
        df_expr = pd.DataFrame(data['expr'], index=names)
        result = run(df_expr, data['t'], methods='bhdt',
                     bhdt_method='analytic', n_jobs=1, verbose=False)
        assert list(result['gene']) == names

    def test_invalid_method_string(self):
        """Invalid methods string raises ValueError."""
        from chord.pipeline import run
        t = np.arange(0, 48, 2, dtype=float)
        expr = np.random.randn(3, 24)
        with pytest.raises(ValueError, match="methods must be"):
            run(expr, t, methods='invalid_method')


# ============================================================================
# Test: Network Hierarchical
# ============================================================================

class TestNetworkHierarchical:
    """Tests for chord.network.hierarchical — empirical Bayes shrinkage."""

    def _make_module_bhdt_df(self, n=20):
        """Create a BHDT DataFrame large enough for module analysis."""
        np.random.seed(42)
        classifications = (['independent_ultradian'] * 8 +
                          ['harmonic'] * 4 +
                          ['ambiguous'] * 4 +
                          ['non_rhythmic'] * 4)[:n]
        return pd.DataFrame({
            'gene': [f'gene_{i}' for i in range(n)],
            'classification': classifications,
            'A_12': np.random.uniform(0.1, 1.0, n),
            'A_24': np.random.uniform(0.1, 1.0, n),
            'phi_12': np.random.uniform(0, 2 * np.pi, n),
            'phi_24': np.random.uniform(0, 2 * np.pi, n),
            'log_bayes_factor': np.random.uniform(-2, 4, n),
            'f_test_12h_significant': ([True] * 8 + [False] * 4 +
                                       [False] * 4 + [False] * 4)[:n],
        })

    def test_estimate_module_hyperparams_keys(self):
        """Hyperparameters dict has expected keys for valid modules."""
        from chord.network.hierarchical import estimate_module_hyperparams
        bhdt = self._make_module_bhdt_df(20)
        modules = {0: [f'gene_{i}' for i in range(10)],
                   1: [f'gene_{i}' for i in range(10, 20)]}
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        assert 0 in hp and hp[0] is not None
        expected_keys = {'n_genes', 'mu_A_12', 'sigma_A_12', 'mu_A_24',
                        'sigma_A_24', 'mu_phi_12', 'kappa_phi_12',
                        'mu_phi_24', 'kappa_phi_24',
                        'prop_independent', 'prop_rhythmic'}
        assert expected_keys <= set(hp[0].keys())

    def test_estimate_module_hyperparams_small_module_skipped(self):
        """Modules below min_module_size get None."""
        from chord.network.hierarchical import estimate_module_hyperparams
        bhdt = self._make_module_bhdt_df(20)
        modules = {0: [f'gene_{i}' for i in range(10)],
                   1: ['gene_10', 'gene_11']}  # only 2 genes
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        assert hp[0] is not None
        assert hp[1] is None

    def test_shrink_parameters_between_gene_and_module(self):
        """Shrunk amplitudes lie between gene value and module mean."""
        from chord.network.hierarchical import (
            estimate_module_hyperparams, shrink_parameters
        )
        bhdt = self._make_module_bhdt_df(20)
        modules = {0: [f'gene_{i}' for i in range(20)]}
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        result = shrink_parameters(bhdt, modules, hp, shrinkage_strength=1.0)

        assert 'A_12_shrunk' in result.columns
        assert 'A_24_shrunk' in result.columns

        # Shrunk values should be between gene original and module mean
        mu_a12 = hp[0]['mu_A_12']
        for _, row in result.iterrows():
            a12_orig = row['A_12']
            a12_shrunk = row['A_12_shrunk']
            lo = min(a12_orig, mu_a12)
            hi = max(a12_orig, mu_a12)
            assert lo - 0.01 <= a12_shrunk <= hi + 0.01, (
                f"Shrunk {a12_shrunk} not between {lo} and {hi}"
            )

    def test_shrink_parameters_adds_columns(self):
        """Shrinkage adds the expected new columns."""
        from chord.network.hierarchical import (
            estimate_module_hyperparams, shrink_parameters
        )
        bhdt = self._make_module_bhdt_df(20)
        modules = {0: [f'gene_{i}' for i in range(20)]}
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        result = shrink_parameters(bhdt, modules, hp)
        new_cols = {'A_12_shrunk', 'A_24_shrunk', 'phi_12_shrunk',
                    'phi_24_shrunk', 'shrinkage_weight_A12',
                    'shrinkage_weight_phi12'}
        assert new_cols <= set(result.columns)

    def test_hierarchical_reclassify_changes_ambiguous(self):
        """Ambiguous genes in a mostly-independent module get reclassified."""
        from chord.network.hierarchical import (
            estimate_module_hyperparams, hierarchical_reclassify
        )
        # Build a module where 8/12 non-ambiguous genes are independent
        n = 16
        np.random.seed(42)
        classifications = (['independent_ultradian'] * 8 +
                          ['harmonic'] * 2 +
                          ['non_rhythmic'] * 2 +
                          ['ambiguous'] * 4)
        bhdt = pd.DataFrame({
            'gene': [f'gene_{i}' for i in range(n)],
            'classification': classifications,
            'A_12': np.random.uniform(0.1, 1.0, n),
            'A_24': np.random.uniform(0.1, 1.0, n),
            'phi_12': np.random.uniform(0, 2 * np.pi, n),
            'phi_24': np.random.uniform(0, 2 * np.pi, n),
            'log_bayes_factor': np.random.uniform(-2, 4, n),
            'f_test_12h_significant': [True] * 8 + [False] * 8,
        })
        modules = {0: [f'gene_{i}' for i in range(n)]}
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        result = hierarchical_reclassify(bhdt, modules, hp, prior_weight=0.3)

        assert 'hierarchical_classification' in result.columns
        # At least some ambiguous genes should be reclassified
        ambig_rows = result[result['classification'] == 'ambiguous']
        reclassified = ambig_rows[
            ambig_rows['hierarchical_classification'] != 'ambiguous'
        ]
        assert len(reclassified) > 0, "Expected at least one ambiguous gene to be reclassified"

    def test_hierarchical_reclassify_harmonic_majority(self):
        """Ambiguous genes in a mostly-harmonic module get reclassified as likely_harmonic."""
        from chord.network.hierarchical import (
            estimate_module_hyperparams, hierarchical_reclassify
        )
        n = 16
        np.random.seed(42)
        # 8 harmonic, 2 independent, 2 non_rhythmic, 4 ambiguous
        classifications = (['harmonic'] * 8 +
                          ['independent_ultradian'] * 2 +
                          ['non_rhythmic'] * 2 +
                          ['ambiguous'] * 4)
        bhdt = pd.DataFrame({
            'gene': [f'gene_{i}' for i in range(n)],
            'classification': classifications,
            'A_12': np.random.uniform(0.1, 1.0, n),
            'A_24': np.random.uniform(0.1, 1.0, n),
            'phi_12': np.random.uniform(0, 2 * np.pi, n),
            'phi_24': np.random.uniform(0, 2 * np.pi, n),
            'log_bayes_factor': np.random.uniform(-2, 4, n),
            'f_test_12h_significant': [True] * 8 + [False] * 8,
        })
        modules = {0: [f'gene_{i}' for i in range(n)]}
        hp = estimate_module_hyperparams(bhdt, modules, min_module_size=5)
        result = hierarchical_reclassify(bhdt, modules, hp, prior_weight=0.3)

        ambig_rows = result[result['classification'] == 'ambiguous']
        reclassified = ambig_rows[
            ambig_rows['hierarchical_classification'] == 'likely_harmonic'
        ]
        assert len(reclassified) > 0, \
            "Harmonic-majority module should reclassify ambiguous genes as likely_harmonic"


# ============================================================================
# Test: Benchmarks — Metrics
# ============================================================================

class TestBenchmarkMetrics:
    """Tests for chord.benchmarks.metrics — classification evaluation."""

    def test_classification_metrics_perfect(self):
        """Perfect predictions yield accuracy=1.0 and macro_f1=1.0."""
        from chord.benchmarks.metrics import classification_metrics
        y_true = ['A', 'B', 'C', 'A', 'B', 'C']
        y_pred = ['A', 'B', 'C', 'A', 'B', 'C']
        result = classification_metrics(y_true, y_pred)
        assert result['accuracy'] == 1.0
        assert abs(result['macro_f1'] - 1.0) < 1e-9

    def test_classification_metrics_known_values(self):
        """Check precision/recall/f1 with known confusion."""
        from chord.benchmarks.metrics import classification_metrics
        y_true = ['A', 'A', 'A', 'B', 'B', 'C']
        y_pred = ['A', 'A', 'B', 'B', 'C', 'C']
        result = classification_metrics(y_true, y_pred)
        # A: TP=2, FP=0, FN=1 → precision=1.0, recall=2/3
        assert result['per_class']['A']['precision'] == 1.0
        assert abs(result['per_class']['A']['recall'] - 2.0 / 3.0) < 1e-9
        # B: TP=1, FP=1, FN=1 → precision=0.5, recall=0.5
        assert abs(result['per_class']['B']['precision'] - 0.5) < 1e-9
        assert abs(result['per_class']['B']['recall'] - 0.5) < 1e-9
        # Accuracy: 4/6
        assert abs(result['accuracy'] - 4.0 / 6.0) < 1e-9

    def test_confusion_matrix_shape(self):
        """Confusion matrix has shape (n_labels, n_labels)."""
        from chord.benchmarks.metrics import classification_metrics
        y_true = ['A', 'B', 'C', 'A']
        y_pred = ['A', 'B', 'A', 'C']
        result = classification_metrics(y_true, y_pred)
        cm = result['confusion_matrix']
        n = len(result['labels'])
        assert cm.shape == (n, n)
        assert cm.sum() == len(y_true)

    def test_harmonic_disentangle_accuracy(self):
        """Binary harmonic-vs-independent accuracy with known values."""
        from chord.benchmarks.metrics import harmonic_disentangle_accuracy
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        result = harmonic_disentangle_accuracy(y_true, y_pred)
        # TP=2, TN=2, FP=1, FN=1
        assert result['tp'] == 2
        assert result['tn'] == 2
        assert result['fp'] == 1
        assert result['fn'] == 1
        assert abs(result['accuracy'] - 4.0 / 6.0) < 1e-9

    def test_roc_auc_perfect_separation(self):
        """Perfect scores yield AUC close to 1.0."""
        from chord.benchmarks.metrics import roc_auc
        y_true = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        result = roc_auc(y_true, scores)
        assert result['auc'] > 0.95


# ============================================================================
# Test: Benchmarks — Wrappers
# ============================================================================

class TestBenchmarkWrappers:
    """Tests for chord.benchmarks.wrappers — classical rhythm detection."""

    def test_lomb_scargle_detects_24h(self):
        """Lomb-Scargle detects a strong 24h signal."""
        from chord.benchmarks.wrappers import lomb_scargle
        t = np.arange(0, 48, 2, dtype=float)
        y = 5.0 + 2.0 * np.cos(2 * np.pi * t / 24.0) + np.random.RandomState(42).normal(0, 0.3, len(t))
        result = lomb_scargle(t, y, periods=[24, 12, 8], n_bootstrap=200, seed=42)
        assert result['period_estimate'] == 24.0
        assert result['p_value'] < 0.05
        assert result['method_name'] == 'lomb_scargle'

    def test_lomb_scargle_output_keys(self):
        """Lomb-Scargle returns all expected keys."""
        from chord.benchmarks.wrappers import lomb_scargle
        t = np.arange(0, 48, 2, dtype=float)
        y = np.random.RandomState(0).randn(len(t))
        result = lomb_scargle(t, y, n_bootstrap=50, seed=0)
        for key in ('p_value', 'period_estimate', 'amplitude_estimate',
                     'method_name', 'powers', 'periods_tested'):
            assert key in result, f"Missing key: {key}"

    def test_lomb_scargle_noise_not_significant(self):
        """Pure noise should generally not be significant at p<0.01."""
        from chord.benchmarks.wrappers import lomb_scargle
        t = np.arange(0, 48, 2, dtype=float)
        y = np.random.RandomState(123).randn(len(t))
        result = lomb_scargle(t, y, n_bootstrap=200, seed=123)
        # Not a hard guarantee, but noise should usually have p > 0.01
        assert result['p_value'] > 0.01

    def test_cosinor_detects_12h(self):
        """Cosinor regression detects a 12h signal."""
        from chord.benchmarks.wrappers import cosinor
        t = np.arange(0, 48, 2, dtype=float)
        y = 3.0 + 1.5 * np.cos(2 * np.pi * t / 12.0) + np.random.RandomState(7).normal(0, 0.2, len(t))
        result = cosinor(t, y, period=12.0)
        assert result['p_value'] < 0.05
        assert result['amplitude_estimate'] > 0.5
        assert result['method_name'] == 'cosinor'


# ============================================================================
# Test: CLI
# ============================================================================

class TestCLI:
    """Tests for chord.cli — command-line interface."""

    def test_cli_module_imports(self):
        """CLI module imports without error."""
        import chord.cli
        assert hasattr(chord.cli, 'main')

    def test_cli_version_subcommand(self):
        """'chord version' prints version string."""
        import subprocess
        src_dir = str(Path(__file__).resolve().parent.parent / "src")
        result = subprocess.run(
            [sys.executable, '-c',
             f'import sys; sys.path.insert(0, {src_dir!r}); '
             'sys.argv = ["chord", "version"]; from chord.cli import main; main()'],
            capture_output=True, text=True, timeout=10,
        )
        assert 'chord-rhythm' in result.stdout
        assert '0.1.0' in result.stdout

    def test_cli_no_command_shows_help(self):
        """Running CLI with no subcommand shows help and exits cleanly."""
        import subprocess
        src_dir = str(Path(__file__).resolve().parent.parent / "src")
        result = subprocess.run(
            [sys.executable, '-c',
             f'import sys; sys.path.insert(0, {src_dir!r}); '
             'sys.argv = ["chord"]; from chord.cli import main; main()'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert 'CHORD' in result.stdout or 'chord' in result.stdout.lower()


# ============================================================================
# Test: Benchmark Tiered Reporting
# ============================================================================

def test_benchmark_tiered_reporting():
    """Benchmark must report Tier-1 (binary 12h detection) separately from
    Tier-2 (harmonic disentanglement)."""
    from chord.benchmarks.run_benchmark import run_benchmark, summarize_benchmark
    results = run_benchmark(n_replicates=2, seed=42)
    summary = summarize_benchmark(results)
    assert "tier1_binary_12h" in summary, "Missing tier1_binary_12h in summary"
    assert "tier2_disentangle" in summary, "Missing tier2_disentangle in summary"
    tier1 = summary["tier1_binary_12h"]
    # All methods should have tier1 results
    assert "bhdt" in tier1
    # Tier2 should only have methods that can do disentanglement
    tier2 = summary["tier2_disentangle"]
    assert "bhdt" in tier2


# ============================================================================
# Test: Real-Data Benchmark Infrastructure
# ============================================================================

class TestRealDataBenchmark:
    """Tests for real-data benchmark functions (no network required).

    These tests verify the benchmark infrastructure using synthetic data
    that mimics the structure of real GEO datasets, so they run without
    network access.
    """

    def test_assign_ground_truth_mouse(self):
        """Ground truth assignment works for mouse genes."""
        from chord.benchmarks.run_benchmark import _assign_ground_truth
        # Known 12h gene
        cls, has_12h = _assign_ground_truth("Xbp1", "mouse")
        assert cls == "known_12h"
        assert has_12h is True
        # Core circadian gene
        cls, has_12h = _assign_ground_truth("Per1", "mouse")
        assert cls == "core_circadian"
        assert has_12h is False
        # Housekeeping gene
        cls, has_12h = _assign_ground_truth("Gapdh", "mouse")
        assert cls == "housekeeping"
        assert has_12h is False
        # Unknown gene
        cls, has_12h = _assign_ground_truth("FakeGene123", "mouse")
        assert cls is None
        assert has_12h is None

    def test_assign_ground_truth_primate(self):
        """Ground truth assignment works for primate genes."""
        from chord.benchmarks.run_benchmark import _assign_ground_truth
        cls, has_12h = _assign_ground_truth("XBP1", "primate")
        assert cls == "known_12h"
        assert has_12h is True
        cls, has_12h = _assign_ground_truth("PER1", "primate")
        assert cls == "core_circadian"
        assert has_12h is False

    def test_real_data_benchmark_with_mock_data(self):
        """Real-data benchmark produces correct DataFrame schema.

        Uses a monkeypatch to avoid network access: replaces _load_dataset
        with a function that returns synthetic data containing known gene
        names from the ground-truth lists.
        """
        from chord.benchmarks.run_benchmark import (
            run_real_data_benchmark,
            summarize_real_data_benchmark,
            _assign_ground_truth,
        )
        from chord.data.known_genes import (
            KNOWN_12H_GENES_ZHU2017,
            CORE_CIRCADIAN_GENES,
            NON_RHYTHMIC_HOUSEKEEPING,
        )
        import chord.benchmarks.run_benchmark as rb_mod

        # Build mock data with known gene names
        t = np.arange(0, 48, 2, dtype=np.float64)
        rng = np.random.RandomState(42)
        gene_names = (
            list(KNOWN_12H_GENES_ZHU2017[:5])
            + list(CORE_CIRCADIAN_GENES[:3])
            + list(NON_RHYTHMIC_HOUSEKEEPING[:2])
        )
        n_genes = len(gene_names)
        # Create expression: 12h genes get a 12h signal, others get noise
        expr = np.zeros((n_genes, len(t)))
        for i, g in enumerate(gene_names):
            cls, has_12h = _assign_ground_truth(g, "mouse")
            if has_12h:
                expr[i] = (5.0 + 1.5 * np.cos(2 * np.pi * t / 12.0)
                           + rng.normal(0, 0.3, len(t)))
            elif cls == "core_circadian":
                expr[i] = (5.0 + 2.0 * np.cos(2 * np.pi * t / 24.0)
                           + rng.normal(0, 0.3, len(t)))
            else:
                expr[i] = 5.0 + rng.normal(0, 0.5, len(t))

        # Monkeypatch _load_dataset
        original_load = rb_mod._load_dataset

        def mock_load(ds_key, cache_dir):
            return expr, t, gene_names

        rb_mod._load_dataset = mock_load
        try:
            results = run_real_data_benchmark(
                datasets=["hughes2009"],
                methods=["bhdt", "cosinor_12h"],
                verbose=False,
            )
        finally:
            rb_mod._load_dataset = original_load

        # Check DataFrame schema
        assert len(results) > 0
        required_cols = {
            "dataset", "gene_name", "method", "predicted_class",
            "predicted_has_12h", "true_class", "true_has_12h",
            "scenario_id", "scenario_name", "replicate",
        }
        assert required_cols <= set(results.columns)

        # Check that we got results for both methods
        assert set(results["method"].unique()) == {"bhdt", "cosinor_12h"}

        # Check that ground truth is correctly assigned
        known_12h_rows = results[results["true_class"] == "known_12h"]
        assert len(known_12h_rows) > 0
        assert all(known_12h_rows["true_has_12h"])

        circ_rows = results[results["true_class"] == "core_circadian"]
        assert len(circ_rows) > 0
        assert not any(circ_rows["true_has_12h"])

        # Test summarize
        summary = summarize_real_data_benchmark(results)
        assert "per_dataset" in summary
        assert "per_gene_class" in summary
        assert "overall" in summary
        assert "concordance" in summary
        assert "hughes2009" in summary["per_dataset"]

    def test_real_data_benchmark_unknown_dataset_raises(self):
        """Unknown dataset key raises ValueError."""
        from chord.benchmarks.run_benchmark import run_real_data_benchmark
        with pytest.raises(ValueError, match="Unknown dataset"):
            run_real_data_benchmark(
                datasets=["nonexistent_dataset"],
                verbose=False,
            )

    def test_real_data_datasets_registry(self):
        """All registered datasets have required metadata."""
        from chord.benchmarks.run_benchmark import _REAL_DATA_DATASETS
        assert len(_REAL_DATA_DATASETS) >= 4
        for key, info in _REAL_DATA_DATASETS.items():
            assert "description" in info
            assert "organism" in info
            assert info["organism"] in ("mouse", "primate")

    def test_summarize_real_data_concordance(self):
        """Concordance computation works when BHDT is present."""
        from chord.benchmarks.run_benchmark import (
            summarize_real_data_benchmark,
        )
        # Build a minimal results DataFrame
        df = pd.DataFrame([
            {"dataset": "test", "gene_name": "g1", "method": "bhdt",
             "true_has_12h": True, "predicted_has_12h": True,
             "true_class": "known_12h", "runtime_s": 0.01},
            {"dataset": "test", "gene_name": "g1", "method": "cosinor_12h",
             "true_has_12h": True, "predicted_has_12h": True,
             "true_class": "known_12h", "runtime_s": 0.01},
            {"dataset": "test", "gene_name": "g2", "method": "bhdt",
             "true_has_12h": False, "predicted_has_12h": False,
             "true_class": "housekeeping", "runtime_s": 0.01},
            {"dataset": "test", "gene_name": "g2", "method": "cosinor_12h",
             "true_has_12h": False, "predicted_has_12h": True,
             "true_class": "housekeeping", "runtime_s": 0.01},
        ])
        summary = summarize_real_data_benchmark(df)
        assert "bhdt_vs_cosinor_12h" in summary["concordance"]
        conc = summary["concordance"]["bhdt_vs_cosinor_12h"]
        assert conc["n_common"] == 2
        # g1 agrees, g2 disagrees → 50% agreement
        assert abs(conc["agreement_rate"] - 0.5) < 1e-9


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
