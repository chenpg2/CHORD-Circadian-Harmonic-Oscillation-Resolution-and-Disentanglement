"""
BHDT: Bayesian Harmonic Disentanglement Test.

Core statistical models for distinguishing independent ultradian oscillations
from mathematical harmonics of circadian rhythms.

Two inference modes:
  - 'analytic': Fast closed-form BIC-based Bayes Factor (default, ~0.01s/gene)
  - 'mcmc': Full NumPyro MCMC with bridge sampling (~minutes/gene, optional)
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Harmonic regression fitting (shared by both models)
# ============================================================================

def _cos_sin_design(t, periods):
    """Build cos/sin design matrix for given periods.

    Parameters
    ----------
    t : array of shape (N,)
        Time points.
    periods : list of float
        Periods to include.

    Returns
    -------
    X : array of shape (N, 1 + 2*len(periods))
        Design matrix: [intercept, cos_1, sin_1, cos_2, sin_2, ...]
    """
    N = len(t)
    K = len(periods)
    X = np.ones((N, 1 + 2 * K))
    for j, T in enumerate(periods):
        omega = 2.0 * np.pi / T
        X[:, 1 + 2 * j] = np.cos(omega * t)
        X[:, 2 + 2 * j] = np.sin(omega * t)
    return X


def _fit_ols(X, y):
    """Ordinary least squares fit.

    Returns
    -------
    beta : array of shape (p,)
        Coefficients.
    rss : float
        Residual sum of squares.
    sigma2 : float
        Estimated noise variance (MLE).
    log_lik : float
        Maximised log-likelihood.
    """
    N = len(y)
    beta, rss_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X.dot(beta)
    rss = float(np.sum(residuals ** 2))
    sigma2 = max(rss / N, 1e-10)  # floor to prevent log(0)
    # Gaussian log-likelihood
    log_lik = -0.5 * N * np.log(2 * np.pi * sigma2) - 0.5 * N
    return beta, rss, sigma2, log_lik


def _extract_rhythm_params(beta, periods):
    """Extract amplitude, phase from cos/sin coefficients.

    Parameters
    ----------
    beta : array
        Coefficients from OLS: [intercept, a1, b1, a2, b2, ...]
    periods : list of float
        Corresponding periods.

    Returns
    -------
    list of dict with keys: T, A (amplitude), phi (phase in radians), a, b
    """
    params = []
    mesor = beta[0]
    for j, T in enumerate(periods):
        a = beta[1 + 2 * j]
        b = beta[2 + 2 * j]
        A = np.sqrt(a ** 2 + b ** 2)
        phi = np.arctan2(b, a)
        params.append({"T": T, "A": float(A), "phi": float(phi), "a": float(a), "b": float(b)})
    return mesor, params


# ============================================================================
# Model M0: Harmonic-constrained model
# ============================================================================

def fit_harmonic_model(t, y, T_base=24.0, K=3):
    """Fit M0: signal is a Fourier series of a single base period.

    y(t) = M + sum_{k=1}^{K} [a_k cos(k*omega*t) + b_k sin(k*omega*t)] + e

    The 12-h component is the k=2 harmonic of the 24-h base, constrained
    to have frequency exactly 2*omega (i.e. period = T_base/2).

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    T_base : float
        Base period (default 24 h).
    K : int
        Number of harmonics (default 3: 24h, 12h, 8h).

    Returns
    -------
    dict with keys: beta, mesor, components, rss, sigma2, log_lik, bic, n_params
    """
    periods = [T_base / k for k in range(1, K + 1)]
    X = _cos_sin_design(t, periods)
    beta, rss, sigma2, log_lik = _fit_ols(X, y)
    mesor, components = _extract_rhythm_params(beta, periods)
    n_params = len(beta) + 1  # +1 for sigma2
    N = len(y)
    bic = -2 * log_lik + n_params * np.log(N)
    aicc = _compute_aicc(log_lik, n_params, N)
    return {
        "beta": beta,
        "mesor": float(mesor),
        "components": components,
        "periods": periods,
        "rss": rss,
        "sigma2": sigma2,
        "log_lik": log_lik,
        "bic": bic,
        "aicc": aicc,
        "n_params": n_params,
        "model": "M0_harmonic",
    }


def _compute_aicc(log_lik, k, n):
    """Compute AICc (small-sample corrected AIC).

    AICc = -2*log_lik + 2*k + 2*k*(k+1)/(n-k-1)

    More appropriate than BIC when n/k < 40 (Burnham & Anderson 2002).
    At N=24 with k=8 (M1-free), the BIC penalty is 3*log(24)=9.5 for
    3 extra parameters, while AICc penalty is ~2*3 + correction ≈ 7.5,
    making it less conservative for detecting independent oscillators.

    When n - k - 1 <= 0 (too few observations for the correction term),
    falls back to BIC to avoid inf/nan values.
    """
    aic = -2.0 * log_lik + 2.0 * k
    if n - k - 1 > 0:
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    else:
        # Fall back to BIC when AICc correction is undefined
        aicc = -2.0 * log_lik + k * np.log(max(n, 1))
    return aicc


# ============================================================================
# Model M1: Independent oscillator model
# ============================================================================

def fit_independent_model(t, y, periods=None):
    """Fit M1: each period is an independent oscillator.

    y(t) = M + sum_k [A_k cos(2*pi*t/T_k - phi_k)] + e

    In the linear (fixed-period) version, this is also OLS but the periods
    are NOT constrained to be harmonics of each other.

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    periods : list of float, optional
        Independent periods to fit. Default [24, 12, 8].

    Returns
    -------
    dict (same structure as fit_harmonic_model)
    """
    if periods is None:
        periods = [24.0, 12.0, 8.0]
    X = _cos_sin_design(t, periods)
    beta, rss, sigma2, log_lik = _fit_ols(X, y)
    mesor, components = _extract_rhythm_params(beta, periods)
    n_params = len(beta) + 1
    N = len(y)
    bic = -2 * log_lik + n_params * np.log(N)
    aicc = _compute_aicc(log_lik, n_params, N)
    return {
        "beta": beta,
        "mesor": float(mesor),
        "components": components,
        "periods": periods,
        "rss": rss,
        "sigma2": sigma2,
        "log_lik": log_lik,
        "bic": bic,
        "aicc": aicc,
        "n_params": n_params,
        "model": "M1_independent",
    }


# ============================================================================
# Model M1-free: Independent oscillator with FREE period estimation
# ============================================================================

def fit_independent_free_period(t, y, period_inits=None, period_bounds=None):
    """Fit M1 with free (non-linear) period estimation.

    Unlike fit_independent_model which uses fixed periods, this version
    optimises the periods themselves via non-linear least squares.
    This is the key test: if the best-fit T_12 deviates significantly
    from T_24/2, it is evidence for an independent oscillator.

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    period_inits : list of float, optional
        Initial period guesses. Default [24.0, 12.0, 8.0].
    period_bounds : list of (lo, hi), optional
        Bounds for each period. Default [(20,28), (10,14), (6,10)].

    Returns
    -------
    dict with additional key 'fitted_periods'
    """
    if period_inits is None:
        period_inits = [24.0, 12.0, 8.0]
    if period_bounds is None:
        # Auto-generate ±~17% bounds around each initial period
        period_bounds = [(p * 0.83, p * 1.17) for p in period_inits]

    K = len(period_inits)

    # Grid resolutions per period (keyed by nearest canonical period)
    grid_steps = {24: 0.2, 12: 0.1, 8: 0.1}

    def _neg_ll_single(period_val, idx, current_periods):
        """Neg-log-lik when varying period `idx` while others are fixed."""
        periods = list(current_periods)
        periods[idx] = period_val
        X = _cos_sin_design(t, periods)
        _, rss, sigma2, log_lik = _fit_ols(X, y)
        return -log_lik

    # --- Coordinate descent: grid search + Brent per period, 2 sweeps ---
    fitted_periods = list(period_inits)
    for _sweep in range(2):
        for idx in range(K):
            lo, hi = period_bounds[idx]
            # Choose grid step based on nearest canonical period
            canonical = min(grid_steps.keys(), key=lambda c: abs(c - period_inits[idx]))
            step = grid_steps[canonical]
            grid = np.arange(lo, hi + step * 0.5, step)

            # Evaluate neg-log-lik at each grid point
            best_nll = np.inf
            best_p = fitted_periods[idx]
            for p in grid:
                nll = _neg_ll_single(p, idx, fitted_periods)
                if nll < best_nll:
                    best_nll = nll
                    best_p = p

            # Brent refinement around best grid point
            brent_lo = max(lo, best_p - 0.5 * step)
            brent_hi = min(hi, best_p + 0.5 * step)
            if brent_lo < brent_hi:
                res = minimize_scalar(
                    _neg_ll_single,
                    bounds=(brent_lo, brent_hi),
                    args=(idx, fitted_periods),
                    method="bounded",
                )
                fitted_periods[idx] = float(res.x)
            else:
                fitted_periods[idx] = float(best_p)

    # Re-fit with optimised periods
    X = _cos_sin_design(t, fitted_periods)
    beta, rss, sigma2, log_lik = _fit_ols(X, y)
    mesor, components = _extract_rhythm_params(beta, fitted_periods)
    # Update component periods to fitted values
    for comp, fp in zip(components, fitted_periods):
        comp["T"] = float(fp)

    n_params = len(beta) + 1 + K  # +K for the period parameters
    N = len(y)
    bic = -2 * log_lik + n_params * np.log(N)
    aicc = _compute_aicc(log_lik, n_params, N)

    return {
        "beta": beta,
        "mesor": float(mesor),
        "components": components,
        "periods": fitted_periods,
        "fitted_periods": fitted_periods,
        "rss": rss,
        "sigma2": sigma2,
        "log_lik": log_lik,
        "bic": bic,
        "aicc": aicc,
        "n_params": n_params,
        "model": "M1_free_period",
        "optimisation_success": True,
    }
