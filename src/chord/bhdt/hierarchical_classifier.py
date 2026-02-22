"""
Hierarchical Bayesian Classifier (HBC) for BHDT.

Replaces the ad hoc multi-evidence scoring system with a principled
Bayesian framework using Gaussian Discriminant Analysis (QDA with
per-class covariances).

Instead of hand-tuned integer scores and manual thresholds, the HBC:
  - Learns decision boundaries from data (synthetic or real)
  - Outputs posterior probabilities P(class | features) for all classes
  - Regularises covariances to handle near-singular cases
  - Can be retrained on real data when available

Mathematical background:
  For each gene g, given class c_g in {independent, harmonic, circadian_only, non_rhythmic}:
    P(features_g | c_g) = N(features_g; mu_c, Sigma_c)
  Classification via Bayes rule:
    P(c_g = k | features_g) = P(features_g | c_g = k) * pi_k
                               / sum_j P(features_g | c_g = j) * pi_j

Python 3.6 compatible.
"""

import numpy as np
from scipy.stats import multivariate_normal


# ============================================================================
# Feature extraction
# ============================================================================

def extract_features(bhdt_result):
    """Extract a fixed-length feature vector from a BHDT analysis result dict.

    Features (all continuous, no discretisation):
      0. log_bf        : log Bayes Factor (BIC or Savage-Dickey)
      1. amp_ratio     : A_12 / A_24
      2. period_deviation : |T_12_fitted - T_base/2| / (T_base/2)
      3. residual_improvement : (RSS_M0 - RSS_M1free) / RSS_M0
      4. f_test_p24    : -log10(p_value) for 24h F-test
      5. f_test_p12    : -log10(p_value) for 12h F-test
      6. phase_coupling : circular distance between phi_12 and 2*phi_24 (0 to pi)
      7. bhct_bicoherence : bicoherence value from BHCT (0 if not available)
      8. sd_log_bf     : Savage-Dickey log BF (0 if not available)

    Parameters
    ----------
    bhdt_result : dict
        Output from bhdt_analytic or similar.

    Returns
    -------
    numpy.ndarray of shape (9,)
        Feature vector. Missing values filled with 0.
    """
    features = np.zeros(9, dtype=np.float64)

    # 0. log Bayes Factor
    features[0] = float(bhdt_result.get("log_bayes_factor", 0.0))

    # 1. Amplitude ratio A_12 / A_24
    m1 = bhdt_result.get("m1", {})
    comps = m1.get("components", [])
    a_24 = 1e-10
    a_12 = 0.0
    for c in comps:
        T = c.get("T", 0)
        if abs(T - 24.0) < 2.0:
            a_24 = max(c.get("A", 1e-10), 1e-10)
        elif abs(T - 12.0) < 2.0:
            a_12 = c.get("A", 0.0)
    features[1] = a_12 / a_24

    # 2. Period deviation
    period_dev = bhdt_result.get("period_deviation", {})
    features[2] = float(period_dev.get("relative_deviation", 0.0))

    # 3. Residual improvement (RSS_M0 - RSS_M1free) / RSS_M0
    m0 = bhdt_result.get("m0", {})
    m1_free = bhdt_result.get("m1_free", {})
    rss_m0 = m0.get("rss", 0.0)
    rss_m1f = m1_free.get("rss", 0.0)
    if rss_m0 > 0:
        features[3] = (rss_m0 - rss_m1f) / rss_m0
    else:
        features[3] = 0.0

    # 4. F-test p24: -log10(p_value) for 24h
    # These may be stored in the result or need to be computed externally.
    # We look for f_test_24 / f_test_12 keys, or fall back to 0.
    f24 = bhdt_result.get("f_test_24", {})
    p24 = f24.get("p_value", 1.0)
    features[4] = -np.log10(max(p24, 1e-20))

    # 5. F-test p12: -log10(p_value) for 12h
    f12 = bhdt_result.get("f_test_12", {})
    p12 = f12.get("p_value", 1.0)
    features[5] = -np.log10(max(p12, 1e-20))

    # 6. Phase coupling: circular distance between phi_12 and 2*phi_24
    phi_24 = None
    phi_12 = None
    for c in comps:
        T = c.get("T", 0)
        if abs(T - 24.0) < 2.0:
            phi_24 = c.get("phi", None)
        elif abs(T - 12.0) < 2.0:
            phi_12 = c.get("phi", None)
    if phi_24 is not None and phi_12 is not None:
        expected = (2.0 * phi_24) % (2.0 * np.pi)
        actual = phi_12 % (2.0 * np.pi)
        diff = abs(actual - expected)
        diff = min(diff, 2.0 * np.pi - diff)
        features[6] = diff
    else:
        features[6] = 0.0

    # 7. BHCT bicoherence
    bhct = bhdt_result.get("bhct", {})
    features[7] = float(bhct.get("bicoherence", 0.0))

    # 8. Savage-Dickey log BF
    sd = bhdt_result.get("savage_dickey", {})
    features[8] = float(sd.get("log_bf", 0.0))

    # Replace any NaN/inf with 0
    features = np.where(np.isfinite(features), features, 0.0)

    return features


# ============================================================================
# Hierarchical Bayesian Classifier
# ============================================================================

class HierarchicalBayesianClassifier(object):
    """Gaussian Discriminant Analysis classifier for BHDT gene classification.

    Uses Quadratic Discriminant Analysis (per-class covariances) with
    Bayesian posterior probability output.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 4).
    class_names : list of str, optional
        Names for each class. Default:
        ['independent_ultradian', 'harmonic', 'circadian_only', 'non_rhythmic']
    """

    def __init__(self, n_classes=4, class_names=None):
        if class_names is None:
            class_names = [
                "independent_ultradian",
                "harmonic",
                "circadian_only",
                "non_rhythmic",
            ]
        if len(class_names) != n_classes:
            raise ValueError(
                "len(class_names)=%d != n_classes=%d" % (len(class_names), n_classes)
            )
        self.n_classes = n_classes
        self.class_names = list(class_names)
        self._fitted = False
        self._means = None       # list of (d,) arrays
        self._covs = None        # list of (d,d) arrays
        self._priors = None      # (n_classes,) array
        self._reg_lambda = 0.01  # covariance regularisation

    def fit(self, features_matrix, labels):
        """Train the classifier from labelled data.

        Parameters
        ----------
        features_matrix : array of shape (n_samples, n_features)
            Feature vectors.
        labels : array-like of shape (n_samples,)
            Class labels (integers 0..n_classes-1 or class name strings).

        Returns
        -------
        self
        """
        X = np.asarray(features_matrix, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("features_matrix must be 2-D, got %d-D" % X.ndim)
        n_samples, n_features = X.shape

        # Convert string labels to integer indices
        labels_arr = np.asarray(labels)
        if labels_arr.dtype.kind in ('U', 'S', 'O'):
            # String labels -- map to indices
            label_to_idx = {name: i for i, name in enumerate(self.class_names)}
            int_labels = np.array([label_to_idx[str(l)] for l in labels_arr])
        else:
            int_labels = np.asarray(labels_arr, dtype=int)

        self._means = []
        self._covs = []
        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        for k in range(self.n_classes):
            mask = (int_labels == k)
            n_k = np.sum(mask)
            self._priors[k] = float(n_k) / n_samples

            if n_k == 0:
                # No samples for this class -- use global mean/cov
                self._means.append(np.mean(X, axis=0))
                self._covs.append(
                    np.cov(X, rowvar=False)
                    + self._reg_lambda * np.eye(n_features)
                )
            elif n_k == 1:
                # Single sample -- use that sample as mean, global cov
                self._means.append(X[mask][0].copy())
                self._covs.append(
                    np.cov(X, rowvar=False)
                    + self._reg_lambda * np.eye(n_features)
                )
            else:
                X_k = X[mask]
                self._means.append(np.mean(X_k, axis=0))
                cov_k = np.cov(X_k, rowvar=False)
                # Regularise: Sigma_k + lambda * I
                cov_k = cov_k + self._reg_lambda * np.eye(n_features)
                self._covs.append(cov_k)

        self._fitted = True
        return self

    def predict_proba(self, features):
        """Return posterior probabilities P(c=k | features) for all classes.

        Parameters
        ----------
        features : array of shape (n_features,) or (n_samples, n_features)
            Feature vector(s).

        Returns
        -------
        numpy.ndarray of shape (n_classes,) or (n_samples, n_classes)
            Posterior probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Classifier has not been fitted yet.")

        features = np.asarray(features, dtype=np.float64)
        single = (features.ndim == 1)
        if single:
            features = features.reshape(1, -1)

        n_samples = features.shape[0]
        log_posteriors = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        for k in range(self.n_classes):
            if self._priors[k] <= 0:
                log_posteriors[:, k] = -np.inf
                continue
            try:
                rv = multivariate_normal(
                    mean=self._means[k], cov=self._covs[k], allow_singular=True
                )
                log_lik = rv.logpdf(features)
            except Exception:
                log_lik = np.full(n_samples, -1e10)
            log_posteriors[:, k] = log_lik + np.log(max(self._priors[k], 1e-20))

        # Normalise via log-sum-exp for numerical stability
        max_lp = np.max(log_posteriors, axis=1, keepdims=True)
        log_posteriors_shifted = log_posteriors - max_lp
        exp_lp = np.exp(log_posteriors_shifted)
        row_sums = np.sum(exp_lp, axis=1, keepdims=True)
        # Guard against zero sums
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        proba = exp_lp / row_sums

        if single:
            return proba[0]
        return proba

    def predict(self, features):
        """Return class with highest posterior probability.

        Parameters
        ----------
        features : array of shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        str or list of str
            Predicted class name(s).
        """
        proba = self.predict_proba(features)
        if proba.ndim == 1:
            idx = int(np.argmax(proba))
            return self.class_names[idx]
        else:
            indices = np.argmax(proba, axis=1)
            return [self.class_names[i] for i in indices]

    def classify_gene(self, bhdt_result):
        """Convenience: extract features, predict, return classification dict.

        Parameters
        ----------
        bhdt_result : dict
            Output from bhdt_analytic or similar.

        Returns
        -------
        dict with keys:
            class : str
                Predicted class name.
            probabilities : dict
                {class_name: probability} for all classes.
            confidence : float
                Max posterior probability (higher = more certain).
            features : numpy.ndarray
                Extracted feature vector.
        """
        feats = extract_features(bhdt_result)
        proba = self.predict_proba(feats)
        pred_idx = int(np.argmax(proba))
        prob_dict = {}
        for i, name in enumerate(self.class_names):
            prob_dict[name] = float(proba[i])
        return {
            "class": self.class_names[pred_idx],
            "probabilities": prob_dict,
            "confidence": float(proba[pred_idx]),
            "features": feats,
        }


# ============================================================================
# Training from synthetic data
# ============================================================================

def _label_from_truth(truth):
    """Map a simulation truth dict to one of the 4 HBC class labels."""
    scenario = truth.get("scenario", "")
    has_indep = truth.get("has_independent_12h", False)
    has_harm = truth.get("has_harmonic_12h", False)

    # Non-rhythmic scenarios
    if scenario in ("pure_noise", "trend_noise"):
        return "non_rhythmic"

    # Circadian-only (no 12h component)
    if scenario == "pure_circadian":
        return "circadian_only"

    # Harmonic scenarios
    if has_harm and not has_indep:
        return "harmonic"

    # Independent ultradian scenarios
    if has_indep:
        return "independent_ultradian"

    # Fallback
    return "non_rhythmic"


def train_from_synthetic(t, n_genes_per_class=100, noise_sd=0.5, seed=42):
    """Generate synthetic training data, run BHDT, and train the classifier.

    Parameters
    ----------
    t : array
        Time points in hours.
    n_genes_per_class : int
        Number of synthetic genes per class (default 100).
    noise_sd : float
        Noise standard deviation for synthetic data.
    seed : int
        Random seed.

    Returns
    -------
    HierarchicalBayesianClassifier
        A fitted classifier.
    """
    from chord.simulation.generator import (
        pure_circadian, pure_noise, independent_superposition,
        sawtooth_harmonic, peaked_harmonic, pure_ultradian,
    )
    from chord.bhdt.inference import bhdt_analytic, component_f_test

    rng = np.random.RandomState(seed)
    features_list = []
    labels_list = []

    # Generator functions for each class
    class_generators = {
        "independent_ultradian": lambda s: independent_superposition(
            t=t, noise_sd=noise_sd, seed=s,
            A_12=rng.uniform(0.8, 2.5), T_12=rng.uniform(11.0, 13.0),
            phi_12=rng.uniform(0, 2 * np.pi),
        ),
        "harmonic": lambda s: sawtooth_harmonic(
            t=t, noise_sd=noise_sd, seed=s,
            A=rng.uniform(1.0, 3.0),
        ),
        "circadian_only": lambda s: pure_circadian(
            t=t, noise_sd=noise_sd, seed=s,
            A=rng.uniform(1.0, 3.0),
        ),
        "non_rhythmic": lambda s: pure_noise(
            t=t, noise_sd=rng.uniform(0.5, 2.0), seed=s,
        ),
    }

    for class_name, gen_fn in class_generators.items():
        for i in range(n_genes_per_class):
            gene_seed = seed * 1000 + hash(class_name) % 10000 + i
            sim = gen_fn(gene_seed)

            # Run BHDT analytic
            try:
                result = bhdt_analytic(t, sim["y"], classifier_version="v1")
            except Exception:
                continue

            # Add F-test results to the result dict for feature extraction
            try:
                f24 = component_f_test(t, sim["y"], [24.0, 12.0, 8.0], test_period_idx=0)
                f12 = component_f_test(t, sim["y"], [24.0, 12.0, 8.0], test_period_idx=1)
                result["f_test_24"] = f24
                result["f_test_12"] = f12
            except Exception:
                pass

            feats = extract_features(result)
            features_list.append(feats)
            labels_list.append(class_name)

    features_matrix = np.array(features_list)
    clf = HierarchicalBayesianClassifier()
    clf.fit(features_matrix, labels_list)
    return clf


# ============================================================================
# Module-level cached classifier
# ============================================================================

_cached_classifier = None


def hbc_classify(bhdt_result, classifier=None):
    """Top-level convenience function for HBC classification.

    Parameters
    ----------
    bhdt_result : dict
        Output from bhdt_analytic or similar.
    classifier : HierarchicalBayesianClassifier, optional
        Pre-trained classifier. If None, trains from synthetic data (cached).

    Returns
    -------
    dict with keys: class, probabilities, confidence, features
    """
    global _cached_classifier

    if classifier is not None:
        return classifier.classify_gene(bhdt_result)

    if _cached_classifier is None:
        t = np.arange(0, 48, 2, dtype=np.float64)
        _cached_classifier = train_from_synthetic(t, n_genes_per_class=50, seed=42)

    return _cached_classifier.classify_gene(bhdt_result)
