"""
Tests for the Hierarchical Bayesian Classifier (HBC).
"""

import numpy as np
import pytest
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.hierarchical_classifier import (
    extract_features,
    HierarchicalBayesianClassifier,
    train_from_synthetic,
    hbc_classify,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_bhdt_result():
    """A minimal BHDT result dict with all expected keys."""
    return {
        "log_bayes_factor": 2.5,
        "bayes_factor": np.exp(2.5),
        "m0": {"rss": 10.0},
        "m1": {
            "components": [
                {"T": 24.0, "A": 2.0, "phi": 0.5},
                {"T": 12.0, "A": 1.5, "phi": 1.2},
                {"T": 8.0, "A": 0.3, "phi": 0.1},
            ]
        },
        "m1_free": {"rss": 7.0, "fitted_periods": [24.1, 11.8, 8.1]},
        "period_deviation": {
            "relative_deviation": 0.017,
            "deviates_from_harmonic": False,
        },
        "f_test_24": {"p_value": 0.001},
        "f_test_12": {"p_value": 0.01},
    }


@pytest.fixture
def empty_bhdt_result():
    """A BHDT result dict with missing keys."""
    return {}


@pytest.fixture
def simple_training_data():
    """Simple synthetic training data for 4 classes."""
    rng = np.random.RandomState(42)
    n_per_class = 30
    n_features = 9

    # Class 0: independent_ultradian -- high log_bf, high amp_ratio
    X0 = rng.randn(n_per_class, n_features) * 0.5
    X0[:, 0] += 5.0   # high log_bf
    X0[:, 1] += 0.9   # high amp_ratio
    X0[:, 3] += 0.4   # high residual improvement

    # Class 1: harmonic -- low log_bf, low amp_ratio
    X1 = rng.randn(n_per_class, n_features) * 0.5
    X1[:, 0] -= 3.0   # low log_bf
    X1[:, 1] += 0.2   # low amp_ratio
    X1[:, 3] += 0.05  # low residual improvement

    # Class 2: circadian_only -- moderate log_bf, very low amp_ratio
    X2 = rng.randn(n_per_class, n_features) * 0.5
    X2[:, 0] += 0.0
    X2[:, 1] += 0.05  # very low amp_ratio
    X2[:, 4] += 5.0   # strong 24h F-test
    X2[:, 5] += 0.5   # weak 12h F-test

    # Class 3: non_rhythmic -- everything near zero
    X3 = rng.randn(n_per_class, n_features) * 0.3
    X3[:, 4] += 0.2   # weak 24h
    X3[:, 5] += 0.2   # weak 12h

    X = np.vstack([X0, X1, X2, X3])
    labels = (
        [0] * n_per_class + [1] * n_per_class +
        [2] * n_per_class + [3] * n_per_class
    )
    return X, np.array(labels)


# ============================================================================
# Test 1: Feature vector shape
# ============================================================================

def test_extract_features_shape(minimal_bhdt_result):
    """Feature vector has correct shape (9,)."""
    feats = extract_features(minimal_bhdt_result)
    assert feats.shape == (9,)
    assert feats.dtype == np.float64


# ============================================================================
# Test 2: Missing keys filled with 0
# ============================================================================

def test_extract_features_handles_missing(empty_bhdt_result):
    """Missing keys in bhdt_result are filled with 0."""
    feats = extract_features(empty_bhdt_result)
    assert feats.shape == (9,)
    # All should be finite
    assert np.all(np.isfinite(feats))


# ============================================================================
# Test 3: Fit on simple data, predict returns valid probabilities
# ============================================================================

def test_classifier_fit_predict(simple_training_data):
    """Fit on simple data, predict returns valid probabilities."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    # Predict on a single sample
    proba = clf.predict_proba(X[0])
    assert proba.shape == (4,)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1.0 + 1e-10)

    # Predict class
    pred = clf.predict(X[0])
    assert pred in clf.class_names


# ============================================================================
# Test 4: Probabilities sum to 1
# ============================================================================

def test_probabilities_sum_to_one(simple_training_data):
    """P(c=k) sums to 1 for each sample."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    # Test on all training samples
    proba = clf.predict_proba(X)
    row_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ============================================================================
# Test 5: Classifier separates obvious cases
# ============================================================================

def test_classifier_separates_obvious_cases(simple_training_data):
    """Independent (high amp_ratio, high log_bf) vs harmonic correctly separated."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    # Create an obviously independent sample
    indep_sample = np.zeros(9)
    indep_sample[0] = 8.0   # very high log_bf
    indep_sample[1] = 1.2   # very high amp_ratio
    indep_sample[3] = 0.5   # high residual improvement
    pred_indep = clf.predict(indep_sample)
    assert pred_indep == "independent_ultradian", (
        "Expected independent_ultradian, got %s" % pred_indep
    )

    # Create an obviously harmonic sample
    harm_sample = np.zeros(9)
    harm_sample[0] = -5.0   # very low log_bf
    harm_sample[1] = 0.15   # low amp_ratio
    harm_sample[3] = 0.02   # low residual improvement
    pred_harm = clf.predict(harm_sample)
    assert pred_harm == "harmonic", (
        "Expected harmonic, got %s" % pred_harm
    )


# ============================================================================
# Test 6: Train from synthetic
# ============================================================================

def test_train_from_synthetic():
    """Synthetic training produces a working classifier."""
    t = np.arange(0, 48, 2, dtype=np.float64)
    clf = train_from_synthetic(t, n_genes_per_class=10, noise_sd=0.5, seed=123)

    assert isinstance(clf, HierarchicalBayesianClassifier)
    assert clf._fitted is True

    # Should be able to predict on a random feature vector
    feats = np.random.RandomState(0).randn(9)
    proba = clf.predict_proba(feats)
    assert proba.shape == (4,)
    np.testing.assert_allclose(np.sum(proba), 1.0, atol=1e-10)


# ============================================================================
# Test 7: hbc_classify convenience function
# ============================================================================

def test_hbc_classify_convenience(simple_training_data, minimal_bhdt_result):
    """Top-level function returns expected format."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    result = hbc_classify(minimal_bhdt_result, classifier=clf)

    assert "class" in result
    assert "probabilities" in result
    assert "confidence" in result
    assert "features" in result
    assert result["class"] in clf.class_names
    assert isinstance(result["probabilities"], dict)
    assert len(result["probabilities"]) == 4
    assert 0.0 <= result["confidence"] <= 1.0 + 1e-10
    assert result["features"].shape == (9,)

    # Probabilities should sum to 1
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-10


# ============================================================================
# Test 8: Confidence reflects certainty
# ============================================================================

def test_confidence_reflects_certainty(simple_training_data):
    """Clear cases have high confidence, ambiguous cases have low confidence."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    # Very clear independent case
    clear_indep = np.zeros(9)
    clear_indep[0] = 10.0   # extreme log_bf
    clear_indep[1] = 1.5    # extreme amp_ratio
    clear_indep[3] = 0.6    # extreme residual improvement
    proba_clear = clf.predict_proba(clear_indep)
    confidence_clear = float(np.max(proba_clear))

    # Ambiguous case: features near zero (class boundaries)
    ambiguous = np.zeros(9)
    proba_ambig = clf.predict_proba(ambiguous)
    confidence_ambig = float(np.max(proba_ambig))

    # Clear case should have higher confidence than ambiguous
    assert confidence_clear > confidence_ambig, (
        "Clear confidence %.3f should be > ambiguous confidence %.3f"
        % (confidence_clear, confidence_ambig)
    )
    # Clear case should be quite confident
    assert confidence_clear > 0.8, (
        "Clear case confidence %.3f should be > 0.8" % confidence_clear
    )


# ============================================================================
# Additional edge case tests
# ============================================================================

def test_classifier_not_fitted_raises():
    """Predicting before fitting raises RuntimeError."""
    clf = HierarchicalBayesianClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros(9))


def test_classifier_with_string_labels():
    """Classifier accepts string labels."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 9)
    labels = (
        ["independent_ultradian"] * 10 + ["harmonic"] * 10 +
        ["circadian_only"] * 10 + ["non_rhythmic"] * 10
    )
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)
    pred = clf.predict(X[0])
    assert pred in clf.class_names


def test_batch_predict(simple_training_data):
    """Batch prediction returns correct shapes."""
    X, labels = simple_training_data
    clf = HierarchicalBayesianClassifier()
    clf.fit(X, labels)

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 4)

    preds = clf.predict(X)
    assert len(preds) == len(X)
    for p in preds:
        assert p in clf.class_names
