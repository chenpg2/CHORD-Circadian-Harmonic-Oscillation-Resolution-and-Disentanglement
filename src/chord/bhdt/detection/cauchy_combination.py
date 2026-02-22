"""
Cauchy Combination Test (CCT) for p-value fusion.

Combines p-values from multiple (possibly correlated) tests into a single
p-value without requiring independence assumptions.

The CCT statistic is:
    T = sum_i w_i * tan((0.5 - p_i) * pi)
    p_combined = 0.5 - arctan(T / sum(w_i)) / pi

This is valid under arbitrary dependence structures among the input
p-values, making it ideal for combining F-test, JTK, RAIN, and harmonic
regression p-values which are all testing the same underlying 12h signal.

Reference:
    Liu, Y. & Xie, J. (2020). Cauchy combination test: a powerful test
    with analytic p-value calculation under arbitrary dependency structures.
    Journal of the American Statistical Association, 115(529), 393-402.
"""

import numpy as np


def cauchy_combine(p_values, weights=None, adaptive=True):
    """Combine p-values using the Cauchy Combination Test.

    Uses a truncated variant where p-values > 0.5 are treated as
    uninformative (set to 0.5 before transformation). This prevents
    non-significant tests from actively canceling significant ones,
    which is critical when some methods (e.g., JTK with Bonferroni)
    return p=1.0 while others detect the signal.

    When ``adaptive=True`` (default), methods with p >= 0.5 receive
    zero weight so that uninformative tests do not dilute the signal
    from informative ones.  This is critical for detecting harmonic
    12h components (e.g. peaked / pulse waveforms) where nonparametric
    methods (JTK, RAIN) return p=1.0 while parametric methods detect
    the signal.  Liu & Xie (2020) prove the CCT is valid for any
    non-negative weight vector, so zeroing uninformative weights is
    statistically sound.

    Parameters
    ----------
    p_values : list of float
        Individual p-values to combine. Each should be in (0, 1).
        Values at exactly 0 or 1 are clipped to (1e-15, 1-1e-15).
    weights : list of float, optional
        Non-negative weights for each p-value. Default: equal weights
        (subject to adaptive zeroing if ``adaptive=True``).
        Weights are normalized to sum to 1.
    adaptive : bool
        If True (default), zero the weight of any method with p >= 0.5
        before normalization.  Falls back to equal weights if all
        methods are uninformative.

    Returns
    -------
    float
        Combined p-value in [0, 1].

    Raises
    ------
    ValueError
        If p_values is empty or weights length doesn't match.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    if len(p_arr) == 0:
        raise ValueError("p_values must be non-empty.")

    # Clip boundary values to avoid inf from tan
    p_arr = np.clip(p_arr, 1e-15, 1.0 - 1e-15)

    # Identify informative methods (p < 0.5) BEFORE truncation
    informative = p_arr < 0.5

    # Truncation: cap p-values at 0.5 to prevent non-significant tests
    # from overwhelming significant ones. tan((0.5 - 0.5) * pi) = 0,
    # so p=0.5 contributes nothing to the statistic.
    p_arr = np.minimum(p_arr, 0.5)

    if weights is None:
        w = np.ones(len(p_arr))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(p_arr):
            raise ValueError(
                f"weights length ({len(w)}) != p_values length ({len(p_arr)})"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")

    # Adaptive weighting: zero out uninformative methods
    if adaptive and np.any(informative):
        w = w * informative.astype(np.float64)

    w_sum = np.sum(w)
    if w_sum <= 0:
        # All methods uninformative — return 0.5 (no evidence)
        return 0.5
    w = w / w_sum

    # CCT statistic: T = sum(w_i * tan((0.5 - p_i) * pi))
    T = np.sum(w * np.tan((0.5 - p_arr) * np.pi))

    # Combined p-value from standard Cauchy CDF
    p_combined = 0.5 - np.arctan(T) / np.pi

    # Clip to [0, 1] for numerical safety
    return float(np.clip(p_combined, 0.0, 1.0))
