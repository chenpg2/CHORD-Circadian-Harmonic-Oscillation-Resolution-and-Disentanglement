"""Data-driven calibration of BHDT multi-evidence scoring.

Trains a logistic regression on synthetic data to replace hand-tuned
discrete evidence scores with calibrated classification probabilities.

Usage:
    from chord.bhdt.calibration import get_calibrated_classifier
    clf = get_calibrated_classifier()  # trains on first call, caches
    # Then pass to bhdt_analytic or _classify_gene
"""

import numpy as np
import warnings
from functools import lru_cache


def extract_evidence_features(m0, m1, m1_free, log_bf, period_dev, f_test_24, f_test_12):
    """Extract continuous evidence features from BHDT model fits.
    
    Returns a 1D array of 7 features:
    [log_bf, amp_ratio, period_rel_deviation, rss_improvement,
     f_pvalue_24_log, f_pvalue_12_log, n_sig_components]
    """
    # Amplitude ratio
    m1_amps = {c["T"]: c["A"] for c in m1["components"]}
    a_24 = m1_amps.get(24.0, 1e-10)
    a_12 = m1_amps.get(12.0, 0)
    amp_ratio = a_12 / max(a_24, 1e-10)
    
    # Period deviation
    rel_dev = period_dev.get("relative_deviation", 0.0)
    
    # RSS improvement
    rss_m0 = m0["rss"]
    rss_m1f = m1_free["rss"]
    rss_improve = (rss_m0 - rss_m1f) / max(rss_m0, 1e-10) if rss_m0 > 0 else 0.0
    
    # F-test p-values (log-transformed)
    p24 = max(f_test_24.get("p_value", 1.0), 1e-300)
    p12 = max(f_test_12.get("p_value", 1.0), 1e-300)
    
    # Number of significant components
    n_sig = int(f_test_24.get("significant", False)) + int(f_test_12.get("significant", False))
    
    return np.array([
        log_bf, amp_ratio, rel_dev, rss_improve,
        -np.log10(p24), -np.log10(p12), float(n_sig)
    ])


def train_calibration_model(n_genes=1000, seed=42):
    """Train logistic regression on synthetic data for scoring calibration.
    
    Generates synthetic genes with known ground truth, extracts evidence
    features, and trains a multinomial logistic regression.
    
    Returns
    -------
    dict with 'model' (fitted LogisticRegression), 'cv_accuracy', 'feature_names'
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    from chord.simulation.generator import (
        pure_circadian, pure_ultradian, independent_superposition,
        sawtooth_harmonic, peaked_harmonic, pure_noise,
    )
    from chord.bhdt.models import fit_harmonic_model, fit_independent_model, fit_independent_free_period
    from chord.bhdt.inference import _bic_bayes_factor, _period_deviation_test, component_f_test
    
    t = np.arange(0, 48, 2, dtype=float)
    rng = np.random.default_rng(seed)
    
    features_list = []
    labels = []
    
    n_per_class = n_genes // 4
    
    # Generate training data across multiple noise levels
    for noise_mult in [0.7, 1.0, 1.5]:
        n_each = n_per_class // 3
        
        # Circadian only
        for i in range(n_each):
            s = rng.integers(0, 100000)
            data = pure_circadian(t, noise_sd=0.5*noise_mult, seed=s)
            feat = _extract_features_for_gene(t, data["y"])
            if feat is not None:
                features_list.append(feat)
                labels.append("circadian_only")
        
        # Independent 12h
        for i in range(n_each):
            s = rng.integers(0, 100000)
            data = independent_superposition(t, noise_sd=0.5*noise_mult, seed=s)
            feat = _extract_features_for_gene(t, data["y"])
            if feat is not None:
                features_list.append(feat)
                labels.append("independent_ultradian")
        
        # Harmonic 12h (sawtooth + peaked)
        for i in range(n_each // 2):
            s = rng.integers(0, 100000)
            data = sawtooth_harmonic(t, noise_sd=0.5*noise_mult, seed=s)
            feat = _extract_features_for_gene(t, data["y"])
            if feat is not None:
                features_list.append(feat)
                labels.append("harmonic")
        for i in range(n_each // 2):
            s = rng.integers(0, 100000)
            data = peaked_harmonic(t, noise_sd=0.5*noise_mult, seed=s)
            feat = _extract_features_for_gene(t, data["y"])
            if feat is not None:
                features_list.append(feat)
                labels.append("harmonic")
        
        # Non-rhythmic
        for i in range(n_each):
            s = rng.integers(0, 100000)
            data = pure_noise(t, noise_sd=0.5*noise_mult, seed=s)
            feat = _extract_features_for_gene(t, data["y"])
            if feat is not None:
                features_list.append(feat)
                labels.append("non_rhythmic")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression with balanced class weights
    clf = LogisticRegression(
        C=1.0, penalty="l2", multi_class="multinomial",
        max_iter=2000, class_weight="balanced", solver="lbfgs",
    )
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
    
    # Final fit on all data
    clf.fit(X_scaled, y)
    
    return {
        "model": clf,
        "scaler": scaler,
        "feature_names": ["log_bf", "amp_ratio", "period_rel_dev", "rss_improve",
                          "f_pval_24_log", "f_pval_12_log", "n_sig"],
        "cv_accuracy": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "n_training": len(y),
        "class_counts": {c: int((y == c).sum()) for c in np.unique(y)},
    }


def _extract_features_for_gene(t, y):
    """Helper: fit models and extract features for one gene."""
    try:
        from chord.bhdt.models import fit_harmonic_model, fit_independent_model, fit_independent_free_period
        from chord.bhdt.inference import _bic_bayes_factor, _period_deviation_test, component_f_test
        
        m0 = fit_harmonic_model(t, y)
        m1 = fit_independent_model(t, y)
        m1_free = fit_independent_free_period(t, y)
        log_bf = _bic_bayes_factor(m0["bic"], m1_free["bic"])
        period_dev = _period_deviation_test(m1_free["fitted_periods"])
        f24 = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=0)
        f12 = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)
        return extract_evidence_features(m0, m1, m1_free, log_bf, period_dev, f24, f12)
    except Exception:
        return None


_CACHED_MODEL = {}

def get_calibrated_classifier(force_retrain=False):
    """Get or train the calibrated classifier (cached singleton).
    
    Returns dict with 'model', 'scaler', etc. or None if training fails.
    """
    if not force_retrain and "clf" in _CACHED_MODEL:
        return _CACHED_MODEL["clf"]
    
    try:
        result = train_calibration_model(n_genes=800, seed=42)
        _CACHED_MODEL["clf"] = result
        return result
    except Exception as e:
        warnings.warn(f"Calibration training failed: {e}. Using hand-tuned scoring.")
        return None


def classify_calibrated(features, calibration=None):
    """Classify a gene using the calibrated model.
    
    Parameters
    ----------
    features : array of shape (7,)
        Output of extract_evidence_features()
    calibration : dict or None
        Output of get_calibrated_classifier(). If None, returns None.
    
    Returns
    -------
    tuple (classification_str, confidence_float) or None
    """
    if calibration is None:
        return None
    
    model = calibration["model"]
    scaler = calibration["scaler"]
    
    X = scaler.transform(features.reshape(1, -1))
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    best_idx = np.argmax(proba)
    
    return classes[best_idx], float(proba[best_idx])
