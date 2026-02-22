"""
Savage-Dickey Density Ratio for exact Bayes Factor computation.

Computes BF for harmonic vs independent oscillator comparison
without the BIC approximation, using MCMC posterior sampling
and kernel density estimation.

Python 3.6 compatible -- no NumPyro dependency.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, lognorm, norm, halfnorm


# ============================================================================
# Log-likelihood for the free-period model M1
# ============================================================================

def _log_likelihood_m1(params, t, y):
    """Log-likelihood for the free-period two-oscillator model.

    Model: y(t) = mesor + A24*cos(2*pi*t/T24 - phi24)
                        + A12*cos(2*pi*t/T12 - phi12) + eps

    Parameters
    ----------
    params : array of length 8
        [mesor, A24, T24, phi24, A12, T12, phi12, log_sigma]
    t : array
        Time points.
    y : array
        Observed values.

    Returns
    -------
    float
        Log-likelihood value.
    """
    mesor, A24, T24, phi24, A12, T12, phi12, log_sigma = params
    sigma = np.exp(log_sigma)
    mu = (mesor
          + A24 * np.cos(2.0 * np.pi * t / T24 - phi24)
          + A12 * np.cos(2.0 * np.pi * t / T12 - phi12))
    N = len(y)
    ll = -0.5 * N * np.log(2.0 * np.pi) - N * log_sigma - 0.5 * np.sum((y - mu) ** 2) / (sigma ** 2)
    return ll


# ============================================================================
# Log-prior for M1 parameters
# ============================================================================

def _log_prior_m1(params, y_mean=0.0, y_std=1.0):
    """Log-prior for M1 parameters.

    Priors:
        T24  ~ LogNormal(log(24), 0.05)
        T12  ~ LogNormal(log(12), 0.1)
        A24  ~ HalfNormal(2.0)
        A12  ~ HalfNormal(2.0)
        phi24, phi12 ~ Uniform(0, 2*pi)
        mesor ~ Normal(y_mean, y_std)
        sigma ~ HalfNormal(y_std)  (parameterised via log_sigma)

    Parameters
    ----------
    params : array of length 8
        [mesor, A24, T24, phi24, A12, T12, phi12, log_sigma]
    y_mean : float
        Mean of observed data (for mesor prior).
    y_std : float
        Std of observed data (for mesor and sigma priors).

    Returns
    -------
    float
        Log-prior density.
    """
    mesor, A24, T24, phi24, A12, T12, phi12, log_sigma = params
    sigma = np.exp(log_sigma)

    lp = 0.0

    # T24 ~ LogNormal(log(24), 0.05)
    if T24 <= 0:
        return -np.inf
    lp += lognorm.logpdf(T24, s=0.05, scale=24.0)

    # T12 ~ LogNormal(log(12), 0.1)
    if T12 <= 0:
        return -np.inf
    lp += lognorm.logpdf(T12, s=0.1, scale=12.0)

    # A24 ~ HalfNormal(2.0)
    if A24 < 0:
        return -np.inf
    lp += halfnorm.logpdf(A24, scale=2.0)

    # A12 ~ HalfNormal(2.0)
    if A12 < 0:
        return -np.inf
    lp += halfnorm.logpdf(A12, scale=2.0)

    # phi24, phi12 ~ Uniform(0, 2*pi)
    if not (0 <= phi24 <= 2 * np.pi):
        return -np.inf
    if not (0 <= phi12 <= 2 * np.pi):
        return -np.inf
    lp += -np.log(2 * np.pi) * 2  # log(1/(2*pi)) for each

    # mesor ~ Normal(y_mean, y_std)
    lp += norm.logpdf(mesor, loc=y_mean, scale=max(y_std, 1e-6))

    # sigma ~ HalfNormal(y_std) -- Jacobian for log_sigma parameterisation
    if sigma <= 0:
        return -np.inf
    lp += halfnorm.logpdf(sigma, scale=max(y_std, 1e-6))
    lp += log_sigma  # Jacobian: d(sigma)/d(log_sigma) = sigma = exp(log_sigma)

    return lp


def _prior_density_T12(T12_val):
    """Evaluate the prior density of T12 at a given value.

    T12 ~ LogNormal(log(12), 0.1)

    Parameters
    ----------
    T12_val : float
        Value at which to evaluate the prior density.

    Returns
    -------
    float
        Prior density p(T12 = T12_val | M1).
    """
    return lognorm.pdf(T12_val, s=0.1, scale=12.0)


# ============================================================================
# Metropolis-Hastings sampler
# ============================================================================

def _init_from_ols(t, y):
    """Initialise MH parameters from OLS estimates.

    Returns
    -------
    params : array of length 8
        [mesor, A24, T24, phi24, A12, T12, phi12, log_sigma]
    """
    try:
        from chord.bhdt.models import fit_independent_free_period
        result = fit_independent_free_period(t, y)
        mesor = result["mesor"]
        comps = result["components"]
        sigma2 = result["sigma2"]

        # Extract component parameters
        A24 = comps[0]["A"]
        phi24 = comps[0]["phi"] % (2 * np.pi)
        T24 = comps[0]["T"]
        A12 = comps[1]["A"]
        phi12 = comps[1]["phi"] % (2 * np.pi)
        T12 = comps[1]["T"]

        log_sigma = np.log(max(np.sqrt(sigma2), 1e-6))
        return np.array([mesor, A24, T24, phi24, A12, T12, phi12, log_sigma])
    except Exception:
        # Fallback: simple initialisation
        mesor = np.mean(y)
        A24 = np.std(y)
        T24 = 24.0
        phi24 = 0.0
        A12 = np.std(y) * 0.5
        T12 = 12.0
        phi12 = 0.0
        log_sigma = np.log(max(np.std(y), 1e-6))
        return np.array([mesor, A24, T24, phi24, A12, T12, phi12, log_sigma])


def _metropolis_hastings(t, y, n_samples=5000, n_warmup=2000, seed=42):
    """Simple Metropolis-Hastings sampler for M1 posterior.

    Uses block updates with a multivariate normal proposal and
    adaptive step size during warmup.

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    n_samples : int
        Number of post-warmup samples to collect.
    n_warmup : int
        Number of warmup (burn-in) samples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'mesor', 'A24', 'T24', 'phi24', 'A12', 'T12', 'phi12',
              'log_sigma', 'accept_rate'
        Each value is an array of shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-10:
        y_std = 1.0

    # Initialise
    current = _init_from_ols(t, y)
    current_ll = _log_likelihood_m1(current, t, y)
    current_lp = _log_prior_m1(current, y_mean, y_std)
    current_log_post = current_ll + current_lp

    # Initial proposal scales
    # [mesor, A24, T24, phi24, A12, T12, phi12, log_sigma]
    prop_scale = np.array([
        y_std * 0.1,   # mesor
        0.1,           # A24
        0.2,           # T24
        0.1,           # phi24
        0.1,           # A12
        0.2,           # T12
        0.1,           # phi12
        0.1,           # log_sigma
    ])

    n_total = n_warmup + n_samples
    samples = np.zeros((n_total, 8))
    n_accept = 0

    # Adaptive parameters
    adapt_interval = 100
    target_rate = 0.234  # optimal for multivariate

    for i in range(n_total):
        # Propose
        proposal = current + rng.randn(8) * prop_scale
        # Wrap phases to [0, 2*pi]
        proposal[3] = proposal[3] % (2 * np.pi)
        proposal[6] = proposal[6] % (2 * np.pi)

        prop_ll = _log_likelihood_m1(proposal, t, y)
        prop_lp = _log_prior_m1(proposal, y_mean, y_std)
        prop_log_post = prop_ll + prop_lp

        # Accept/reject
        log_alpha = prop_log_post - current_log_post
        if np.isfinite(log_alpha) and np.log(rng.rand()) < log_alpha:
            current = proposal
            current_log_post = prop_log_post
            n_accept += 1

        samples[i] = current

        # Adapt step size during warmup
        if i < n_warmup and (i + 1) % adapt_interval == 0 and i > 0:
            recent_start = max(0, i - adapt_interval)
            recent_accept = 0
            for j in range(recent_start + 1, i + 1):
                if not np.allclose(samples[j], samples[j - 1]):
                    recent_accept += 1
            recent_rate = recent_accept / float(adapt_interval)
            if recent_rate < target_rate * 0.8:
                prop_scale *= 0.8
            elif recent_rate > target_rate * 1.2:
                prop_scale *= 1.2

    # Discard warmup
    post_samples = samples[n_warmup:]
    accept_rate = n_accept / float(n_total)

    names = ['mesor', 'A24', 'T24', 'phi24', 'A12', 'T12', 'phi12', 'log_sigma']
    result = {}
    for idx, name in enumerate(names):
        result[name] = post_samples[:, idx]
    result['accept_rate'] = accept_rate

    return result


# ============================================================================
# Savage-Dickey Bayes Factor
# ============================================================================

def savage_dickey_bf(t, y, T_base=24.0, n_samples=5000, n_warmup=2000, seed=42):
    """Compute exact Bayes Factor via Savage-Dickey Density Ratio.

    BF_01 = p(T12 = T_base/2 | y, M1) / p(T12 = T_base/2 | M1)

    BF > 1 favours harmonic (M0); BF < 1 favours independent (M1).

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    T_base : float
        Base circadian period (default 24.0).
    n_samples : int
        Number of MCMC samples (post-warmup).
    n_warmup : int
        Number of warmup samples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'log_bf', 'bf', 'posterior_density', 'prior_density',
              'T12_samples', 'accept_rate', 'method'
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T12_null = T_base / 2.0

    # Run MCMC
    posterior = _metropolis_hastings(t, y, n_samples=n_samples,
                                     n_warmup=n_warmup, seed=seed)
    T12_samples = posterior['T12']

    # Prior density at T12 = T_base/2
    prior_density = _prior_density_T12(T12_null)

    # Posterior density at T12 = T_base/2 via KDE
    try:
        # Filter out any extreme outliers that could break KDE
        T12_valid = T12_samples[np.isfinite(T12_samples)]
        if len(T12_valid) < 10:
            raise ValueError("Too few valid T12 samples")

        kde = gaussian_kde(T12_valid)
        posterior_density = float(kde(T12_null)[0])
    except Exception:
        # Fallback: use Gaussian approximation (Laplace)
        mu = np.mean(T12_samples)
        sigma = np.std(T12_samples)
        if sigma < 1e-10:
            sigma = 1e-10
        posterior_density = float(norm.pdf(T12_null, loc=mu, scale=sigma))

    # Compute BF
    if prior_density < 1e-300:
        # Prior density essentially zero -- can't compute ratio
        log_bf = 0.0
    elif posterior_density < 1e-300:
        log_bf = -30.0  # Strong evidence against harmonic
    else:
        log_bf = np.log(posterior_density) - np.log(prior_density)

    bf = np.exp(np.clip(log_bf, -30, 30))

    return {
        'log_bf': float(log_bf),
        'bf': float(bf),
        'posterior_density': float(posterior_density),
        'prior_density': float(prior_density),
        'T12_samples': T12_samples,
        'accept_rate': posterior['accept_rate'],
        'method': 'savage_dickey',
    }


# ============================================================================
# Evidence score wrapper
# ============================================================================

def savage_dickey_evidence(t, y, T_base=24.0, n_samples=5000, n_warmup=2000,
                           seed=42):
    """Compute evidence score from Savage-Dickey BF.

    Score mapping:
        log_bf > 3  -> -3  (strong harmonic)
        log_bf > 1  -> -2
        log_bf > 0  -> -1
        log_bf < -3 -> +3  (strong independent)
        log_bf < -1 -> +2
        log_bf < 0  -> +1
        else        ->  0  (inconclusive)

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    T_base : float
        Base circadian period.
    n_samples : int
        Number of MCMC samples.
    n_warmup : int
        Number of warmup samples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'score', 'log_bf', 'bf', 'label'
    """
    result = savage_dickey_bf(t, y, T_base=T_base, n_samples=n_samples,
                              n_warmup=n_warmup, seed=seed)
    log_bf = result['log_bf']

    if log_bf > 3:
        score = -3
        label = "strong_harmonic"
    elif log_bf > 1:
        score = -2
        label = "moderate_harmonic"
    elif log_bf > 0:
        score = -1
        label = "weak_harmonic"
    elif log_bf < -3:
        score = 3
        label = "strong_independent"
    elif log_bf < -1:
        score = 2
        label = "moderate_independent"
    elif log_bf < 0:
        score = 1
        label = "weak_independent"
    else:
        score = 0
        label = "inconclusive"

    return {
        'score': score,
        'log_bf': log_bf,
        'bf': result['bf'],
        'label': label,
    }
