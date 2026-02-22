"""
Goodwin-Delay ODE Decomposition for CHORD.

Replaces the physically-motivated damped harmonic oscillator in PINOD with a
biologically-grounded Goodwin-type transcription-translation feedback loop model.

The Goodwin oscillator (Goodwin 1965) is the canonical model for biological
rhythm generation:

    dm/dt = alpha * K^n / (K^n + p^n) - delta_m * m   (mRNA)
    dp/dt = beta * m - delta_p * p                      (protein)

This module implements a coupled dual-oscillator variant where oscillator 1
represents the circadian (~24h) rhythm and oscillator 2 represents the
ultradian (~12h) rhythm, with coupling strength epsilon quantifying how much
oscillator 2 is driven by oscillator 1.

Requires: numpy, scipy (no PyTorch dependency).
Python 3.6 compatible.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import argrelextrema


def hill_function(x, K, n):
    """Hill repression function: K^n / (K^n + x^n).

    Parameters
    ----------
    x : float or array
        Input concentration (e.g., protein level).
    K : float
        Half-maximal repression constant. hill(K, K, n) = 0.5.
    n : float
        Hill coefficient controlling nonlinearity.

    Returns
    -------
    float or array
        Value in [0, 1]. Equals 1 when x=0, 0.5 when x=K, approaches 0 as x->inf.
    """
    x = np.asarray(x, dtype=np.float64)
    K = float(K)
    n = float(n)
    Kn = K ** n
    xn = np.abs(x) ** n  # abs for numerical safety
    return Kn / (Kn + xn)


def goodwin_dual_ode(t, state, params):
    """ODE right-hand side for the coupled dual Goodwin oscillator.

    State vector: [m1, p1, m2, p2]
        m1, p1: mRNA and protein for oscillator 1 (circadian, ~24h)
        m2, p2: mRNA and protein for oscillator 2 (ultradian, ~12h)

    Equations:
        dm1/dt = alpha1 * hill(p1, K1, n1) - delta_m1 * m1
        dp1/dt = beta1 * m1 - delta_p1 * p1
        dm2/dt = alpha2 * hill(p2, K2, n2) + epsilon * p1 - delta_m2 * m2
        dp2/dt = beta2 * m2 - delta_p2 * p2

    Parameters
    ----------
    t : float
        Current time (unused in autonomous system, kept for solve_ivp).
    state : array-like, shape (4,)
        [m1, p1, m2, p2].
    params : dict
        Must contain: alpha1, K1, n1, delta_m1, beta1, delta_p1,
                       alpha2, K2, n2, delta_m2, beta2, delta_p2, epsilon.

    Returns
    -------
    list of float
        [dm1/dt, dp1/dt, dm2/dt, dp2/dt].
    """
    m1, p1, m2, p2 = state

    # Oscillator 1 (circadian)
    dm1 = params['alpha1'] * hill_function(p1, params['K1'], params['n1']) - params['delta_m1'] * m1
    dp1 = params['beta1'] * m1 - params['delta_p1'] * p1

    # Oscillator 2 (ultradian) with coupling from oscillator 1
    dm2 = (params['alpha2'] * hill_function(p2, params['K2'], params['n2'])
           + params['epsilon'] * p1
           - params['delta_m2'] * m2)
    dp2 = params['beta2'] * m2 - params['delta_p2'] * p2

    return [dm1, dp1, dm2, dp2]


def simulate_goodwin(params, t_eval, y0=None):
    """Integrate the coupled dual Goodwin oscillator.

    Parameters
    ----------
    params : dict
        ODE parameters (see goodwin_dual_ode).
    t_eval : array-like
        Time points at which to evaluate the solution.
    y0 : array-like, shape (4,), optional
        Initial conditions [m1, p1, m2, p2].
        If None, uses biologically reasonable defaults.

    Returns
    -------
    dict or None
        Keys: 't', 'm1', 'p1', 'm2', 'p2' (all arrays of shape (len(t_eval),)).
        Returns None if the ODE solver fails.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)

    if y0 is None:
        # Reasonable defaults: moderate initial mRNA and protein levels
        y0 = [1.0, 1.0, 1.0, 1.0]

    t_span = (t_eval[0], t_eval[-1])

    def rhs(t, state):
        return goodwin_dual_ode(t, state, params)

    try:
        sol = solve_ivp(
            rhs, t_span, y0, t_eval=t_eval,
            method='LSODA', rtol=1e-8, atol=1e-10,
            max_step=0.5,
        )
        if not sol.success:
            return None
        return {
            't': sol.t,
            'm1': sol.y[0],
            'p1': sol.y[1],
            'm2': sol.y[2],
            'p2': sol.y[3],
        }
    except Exception:
        return None


def _estimate_period(signal, t):
    """Estimate dominant period from a signal using autocorrelation.

    Parameters
    ----------
    signal : array-like
        1D signal.
    t : array-like
        Corresponding time points (assumed uniformly spaced).

    Returns
    -------
    float
        Estimated period, or NaN if estimation fails.
    """
    signal = np.asarray(signal, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if len(signal) < 4:
        return float('nan')

    # Remove mean
    sig = signal - np.mean(signal)
    norm = np.sum(sig ** 2)
    if norm < 1e-15:
        return float('nan')

    # Autocorrelation via numpy correlate
    n = len(sig)
    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[n - 1:]  # keep non-negative lags
    autocorr = autocorr / autocorr[0]  # normalize

    # Find first peak after lag 0
    peaks = argrelextrema(autocorr, np.greater, order=1)[0]
    if len(peaks) == 0:
        return float('nan')

    # First peak corresponds to the dominant period
    dt = np.mean(np.diff(t))
    period = peaks[0] * dt
    return period


def _simulate_fast(params, t_eval, y0=None):
    """Fast ODE integration for use inside the optimizer loop.

    Uses RK45 with relaxed tolerances for speed.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if y0 is None:
        y0 = [1.0, 1.0, 1.0, 1.0]
    t_span = (t_eval[0], t_eval[-1])

    def rhs(t, state):
        return goodwin_dual_ode(t, state, params)

    try:
        sol = solve_ivp(
            rhs, t_span, y0, t_eval=t_eval,
            method='RK45', rtol=1e-5, atol=1e-7,
            max_step=2.0,
        )
        if not sol.success:
            return None
        return sol.y  # shape (4, len(t_eval))
    except Exception:
        return None


def fit_goodwin_to_gene(t, y, n_restarts=5, seed=42, maxiter=100, popsize=10):
    """Fit the coupled Goodwin model to observed gene expression.

    Uses scipy.optimize.differential_evolution for global optimization.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed gene expression values.
    n_restarts : int
        Number of restarts (seeds) for differential_evolution.
    seed : int
        Base random seed.
    maxiter : int
        Maximum iterations for differential_evolution.
    popsize : int
        Population size multiplier for differential_evolution.

    Returns
    -------
    dict or None
        Keys: 'params' (fitted ODE params), 'w1', 'w2', 'baseline',
              'reconstructed' (fitted signal), 'residuals',
              'coupling_strength' (epsilon), 'mse'.
        Returns None if fitting fails completely.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Parameter bounds: [alpha1, K1, n1, delta_m1, beta1, delta_p1,
    #                     alpha2, K2, n2, delta_m2, beta2, delta_p2,
    #                     epsilon, w1, w2, baseline]
    bounds = [
        (0.1, 50.0),    # alpha1
        (0.1, 20.0),    # K1
        (1.0, 10.0),    # n1
        (0.01, 2.0),    # delta_m1
        (0.1, 20.0),    # beta1
        (0.01, 2.0),    # delta_p1
        (0.1, 50.0),    # alpha2
        (0.1, 20.0),    # K2
        (1.0, 10.0),    # n2
        (0.01, 2.0),    # delta_m2
        (0.1, 20.0),    # beta2
        (0.01, 2.0),    # delta_p2
        (0.0, 10.0),    # epsilon
        (-5.0, 5.0),    # w1
        (-5.0, 5.0),    # w2
        (-10.0, 10.0),  # baseline
    ]

    def _params_from_vector(x):
        return {
            'alpha1': x[0], 'K1': x[1], 'n1': x[2],
            'delta_m1': x[3], 'beta1': x[4], 'delta_p1': x[5],
            'alpha2': x[6], 'K2': x[7], 'n2': x[8],
            'delta_m2': x[9], 'beta2': x[10], 'delta_p2': x[11],
            'epsilon': x[12],
        }

    def objective(x):
        params = _params_from_vector(x)
        w1, w2, baseline = x[13], x[14], x[15]

        sol_y = _simulate_fast(params, t)
        if sol_y is None:
            return 1e10  # penalty for solver failure

        reconstructed = w1 * sol_y[0] + w2 * sol_y[2] + baseline  # m1=row0, m2=row2

        # Check for NaN/Inf
        if not np.all(np.isfinite(reconstructed)):
            return 1e10

        mse = np.mean((y - reconstructed) ** 2)
        return mse

    best_result = None
    best_mse = float('inf')

    for i in range(n_restarts):
        try:
            res = differential_evolution(
                objective, bounds,
                seed=seed + i,
                maxiter=maxiter,
                tol=1e-4,
                polish=False,
                mutation=(0.5, 1.5),
                recombination=0.8,
                popsize=popsize,
            )
            if res.fun < best_mse:
                best_mse = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return None

    x = best_result.x
    fitted_params = _params_from_vector(x)
    w1, w2, baseline = x[13], x[14], x[15]

    sim = simulate_goodwin(fitted_params, t)
    if sim is None:
        return None

    reconstructed = w1 * sim['m1'] + w2 * sim['m2'] + baseline
    residuals = y - reconstructed

    return {
        'params': fitted_params,
        'w1': w1,
        'w2': w2,
        'baseline': baseline,
        'reconstructed': reconstructed,
        'residuals': residuals,
        'coupling_strength': fitted_params['epsilon'],
        'mse': best_mse,
        'sim': sim,
    }


def goodwin_decompose(t, y, **fit_kwargs):
    """High-level decomposition of gene expression using the Goodwin model.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed gene expression.
    **fit_kwargs
        Additional keyword arguments passed to fit_goodwin_to_gene
        (e.g., n_restarts, maxiter, popsize).

    Returns
    -------
    dict or None
        Keys: 'period_1', 'period_2', 'coupling_strength', 'coupling_ratio',
              'classification' ('independent', 'coupled', or 'ambiguous'),
              'component_1' (w1*m1), 'component_2' (w2*m2), 'baseline',
              'reconstructed', 'residuals', 'params', 'mse'.
        Returns None if fitting fails.
    """
    fit = fit_goodwin_to_gene(t, y, **fit_kwargs)
    if fit is None:
        return None

    sim = fit['sim']

    # Estimate periods from mRNA trajectories
    period_1 = _estimate_period(sim['m1'], sim['t'])
    period_2 = _estimate_period(sim['m2'], sim['t'])

    # Coupling ratio: epsilon / alpha2
    epsilon = fit['params']['epsilon']
    alpha2 = fit['params']['alpha2']
    coupling_ratio = epsilon / alpha2 if alpha2 > 1e-10 else float('inf')

    # Classification
    if coupling_ratio < 0.1:
        classification = 'independent'
    elif coupling_ratio > 0.5:
        classification = 'coupled'
    else:
        classification = 'ambiguous'

    component_1 = fit['w1'] * sim['m1']
    component_2 = fit['w2'] * sim['m2']

    return {
        'period_1': period_1,
        'period_2': period_2,
        'coupling_strength': epsilon,
        'coupling_ratio': coupling_ratio,
        'classification': classification,
        'component_1': component_1,
        'component_2': component_2,
        'baseline': fit['baseline'],
        'reconstructed': fit['reconstructed'],
        'residuals': fit['residuals'],
        'params': fit['params'],
        'w1': fit['w1'],
        'w2': fit['w2'],
        'mse': fit['mse'],
    }


def goodwin_evidence(t, y, **fit_kwargs):
    """Compute evidence score for BHDT integration.

    The coupling ratio epsilon/alpha2 indicates whether the 12h oscillator
    is independent (low ratio) or driven by the circadian clock (high ratio).

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed gene expression.
    **fit_kwargs
        Additional keyword arguments passed to fit_goodwin_to_gene.

    Returns
    -------
    dict
        Keys: 'score' (int in [-2, +3]), 'coupling_ratio' (float),
              'coupling_strength' (float), 'description' (str).
        Returns {'score': 0, 'coupling_ratio': None, ...} if fitting fails.
    """
    fit = fit_goodwin_to_gene(t, y, **fit_kwargs)
    if fit is None:
        return {
            'score': 0,
            'coupling_ratio': None,
            'coupling_strength': None,
            'description': 'Goodwin model fitting failed',
        }

    epsilon = fit['params']['epsilon']
    alpha2 = fit['params']['alpha2']
    ratio = epsilon / alpha2 if alpha2 > 1e-10 else float('inf')

    if ratio < 0.05:
        score = 3
        desc = 'Strong independent (Goodwin coupling ratio < 0.05)'
    elif ratio < 0.1:
        score = 2
        desc = 'Independent (Goodwin coupling ratio < 0.1)'
    elif ratio < 0.2:
        score = 1
        desc = 'Weakly independent (Goodwin coupling ratio < 0.2)'
    elif ratio > 0.5:
        score = -2
        desc = 'Strong coupled/harmonic (Goodwin coupling ratio > 0.5)'
    elif ratio > 0.3:
        score = -1
        desc = 'Weakly coupled (Goodwin coupling ratio > 0.3)'
    else:
        score = 0
        desc = 'Ambiguous (Goodwin coupling ratio 0.2-0.3)'

    return {
        'score': score,
        'coupling_ratio': ratio,
        'coupling_strength': epsilon,
        'description': desc,
    }
