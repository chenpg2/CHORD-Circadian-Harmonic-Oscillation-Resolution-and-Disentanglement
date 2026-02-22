"""
Parametric Bootstrap Likelihood Ratio Test for BHDT.

Solves Phase 1 Problem 1 (BIC too conservative at N=24) by computing
the exact null distribution of the LRT statistic via simulation.

Algorithm:
1. Fit M0 (harmonic) and M1-free (independent) to observed data
2. Compute observed LRT = -2 * (loglik_M0 - loglik_M1)
3. Simulate B datasets under M0 (null hypothesis)
4. For each simulated dataset, compute LRT_b
5. p-value = proportion of LRT_b >= LRT_observed

This is faster than MCMC (~1-2s/gene vs ~30s/gene) while still being
exact (no large-sample approximation like BIC).
"""

import numpy as np
from typing import Dict, Optional, Tuple

from chord.bhdt.models import (
    fit_harmonic_model, fit_independent_free_period,
    _cos_sin_design, _fit_ols
)


def parametric_bootstrap_lrt(t, y, T_base=24.0, K_harmonics=3,
                              n_bootstrap=999, seed=42,
                              period_inits=None, period_bounds=None):
    """Parametric bootstrap LRT for harmonic vs independent oscillators.

    Parameters
    ----------
    t : array
        Time points in hours.
    y : array
        Expression values.
    T_base : float
        Base circadian period.
    K_harmonics : int
        Number of harmonics for M0.
    n_bootstrap : int
        Number of bootstrap replicates (default 999 for p-value resolution).
    seed : int
        Random seed.
    period_inits, period_bounds : optional
        Passed to fit_independent_free_period.

    Returns
    -------
    dict with keys:
        lrt_observed, p_value, classification,
        m0_fit, m1_fit, lrt_null_distribution
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    N = len(t_arr)

    # Step 1: Fit both models to observed data
    m0 = fit_harmonic_model(t_arr, y_arr, T_base=T_base, K=K_harmonics)
    m1 = fit_independent_free_period(t_arr, y_arr,
                                      period_inits=period_inits,
                                      period_bounds=period_bounds)

    # Step 2: Observed LRT statistic
    # LRT = -2 * (loglik_M0 - loglik_M1) = 2 * (loglik_M1 - loglik_M0)
    lrt_obs = -2.0 * (m0["log_lik"] - m1["log_lik"])

    # Step 3: Simulate under M0 (null)
    rng = np.random.RandomState(seed)
    X_m0 = _cos_sin_design(t_arr, m0["periods"])
    y_hat_m0 = X_m0.dot(m0["beta"])
    sigma_m0 = np.sqrt(m0["sigma2"])

    lrt_null = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Generate data under M0
        y_sim = y_hat_m0 + rng.normal(0, sigma_m0, N)

        # Fit both models to simulated data
        m0_sim = fit_harmonic_model(t_arr, y_sim, T_base=T_base, K=K_harmonics)
        m1_sim = fit_independent_free_period(t_arr, y_sim,
                                              period_inits=period_inits,
                                              period_bounds=period_bounds)

        lrt_null[b] = -2.0 * (m0_sim["log_lik"] - m1_sim["log_lik"])

    # Step 4: p-value
    p_value = (np.sum(lrt_null >= lrt_obs) + 1) / (n_bootstrap + 1)

    # Step 5: Classification
    from chord.bhdt.inference import component_f_test
    f_test_24 = component_f_test(t_arr, y_arr, [24.0, 12.0, 8.0], test_period_idx=0)
    f_test_12 = component_f_test(t_arr, y_arr, [24.0, 12.0, 8.0], test_period_idx=1)

    classification = _classify_bootstrap(
        p_value, f_test_24["significant"], f_test_12["significant"],
        m1["fitted_periods"] if "fitted_periods" in m1 else None,
        T_base
    )

    return {
        "lrt_observed": float(lrt_obs),
        "p_value": float(p_value),
        "classification": classification,
        "m0_fit": m0,
        "m1_fit": m1,
        "lrt_null_mean": float(np.mean(lrt_null)),
        "lrt_null_std": float(np.std(lrt_null)),
        "n_bootstrap": n_bootstrap,
    }


def _classify_bootstrap(p_value, sig_24, sig_12, fitted_periods, T_base):
    """Classify gene based on bootstrap LRT p-value."""
    if not sig_24 and not sig_12:
        return "non_rhythmic"
    if sig_24 and not sig_12:
        return "circadian_only"
    if sig_12 and not sig_24:
        return "independent_ultradian"

    # Both significant — use bootstrap p-value
    if p_value < 0.01:
        # Strong rejection of H0 (harmonic) -> independent
        return "independent_ultradian"
    elif p_value < 0.05:
        return "likely_independent_ultradian"
    elif p_value > 0.5:
        return "harmonic"
    elif p_value > 0.1:
        return "harmonic"
    else:
        # Check period deviation
        if fitted_periods is not None and len(fitted_periods) >= 2:
            T_12_fit = fitted_periods[1]
            expected_12 = T_base / 2.0
            rel_dev = abs(T_12_fit - expected_12) / expected_12
            if rel_dev > 0.02:
                return "likely_independent_ultradian"
        return "ambiguous"
