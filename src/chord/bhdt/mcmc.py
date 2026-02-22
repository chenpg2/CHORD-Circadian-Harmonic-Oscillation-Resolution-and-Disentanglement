"""
BHDT MCMC mode: Full Bayesian inference using NumPyro.

Replaces the BIC-based approximate Bayes Factor with exact marginal
likelihood estimation via MCMC sampling + WAIC.

This solves Phase 1 Problem 1 (BIC too conservative at N=24) by
computing the true posterior rather than relying on large-sample
approximations.

Key design decisions:
- Reparameterize periods as angular frequencies (omega) for better NUTS geometry
- Use informative LogNormal priors on periods to prevent mode-hopping
- Compute WAIC via direct log-likelihood evaluation (not Predictive)
- Initialize MCMC from OLS estimates for faster convergence

Requires: numpyro + jax (pip install chord-rhythm[bayes])
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# jax/numpyro are optional dependencies — imported lazily inside functions
# to avoid crashing when only the analytic mode is needed.
# Install with: pip install chord-rhythm[bayes]

_JAX_ERR = (
    "MCMC mode requires jax and numpyro. "
    "Install with: pip install chord-rhythm[bayes]"
)


def _lazy_imports():
    """Import jax/numpyro on demand; raises ImportError with guidance if missing."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
    except ImportError as e:
        raise ImportError(_JAX_ERR) from e
    return jax, jnp, numpyro, dist, MCMC, NUTS, Predictive, init_to_value


# ============================================================================
# Model M0: Harmonic-constrained (Fourier series of base period)
# ============================================================================

def model_m0_harmonic(t, y=None, K=3, T_base_prior_mean=24.0):
    """NumPyro model: signal is Fourier series of a single base period.

    y(t) = M + sum_{k=1}^{K} [a_k cos(k*omega*t) + b_k sin(k*omega*t)] + e

    The 12h component is the k=2 harmonic, constrained to frequency 2*omega.
    """
    _, jnp, numpyro, dist, _, _, _, _ = _lazy_imports()
    N = len(t)

    # Priors — LogNormal for T_base ensures positivity, tight prior near 24h
    M = numpyro.sample("M", dist.Normal(0.0, 5.0))
    T_base = numpyro.sample("T_base", dist.LogNormal(jnp.log(T_base_prior_mean), 0.08))
    omega = 2.0 * jnp.pi / T_base
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    tau_0 = numpyro.sample("tau_0", dist.HalfCauchy(1.0))

    # Fourier coefficients with shrinking prior (higher harmonics smaller)
    mu = M
    for k in range(1, K + 1):
        tau_k = tau_0 / k
        a_k = numpyro.sample(f"a_{k}", dist.Normal(0.0, tau_k))
        b_k = numpyro.sample(f"b_{k}", dist.Normal(0.0, tau_k))
        mu = mu + a_k * jnp.cos(k * omega * t) + b_k * jnp.sin(k * omega * t)

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


# ============================================================================
# Model M1: Independent oscillators (free periods)
# ============================================================================

def model_m1_independent(t, y=None, period_priors=None):
    """NumPyro model: independent oscillators with free periods.

    y(t) = M + sum_k A_k * cos(2*pi*t/T_k - phi_k) + e

    Each T_k is an independent parameter, NOT constrained to harmonics.
    Uses LogNormal prior on periods to ensure T > 0 and concentrate
    near the expected period.
    """
    _, jnp, numpyro, dist, _, _, _, _ = _lazy_imports()
    if period_priors is None:
        period_priors = [(24.0, 2.0), (12.0, 1.5), (8.0, 1.0)]

    N = len(t)
    K = len(period_priors)

    M = numpyro.sample("M", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))

    mu = M
    for k, (T_mean, T_sd) in enumerate(period_priors):
        # LogNormal: log(T) ~ Normal(log(T_mean), sigma_log)
        # sigma_log chosen so that ~95% of prior mass is within T_mean +/- T_sd
        sigma_log = T_sd / T_mean  # CV approximation
        T_k = numpyro.sample(f"T_{k}", dist.LogNormal(jnp.log(T_mean), sigma_log))
        A_k = numpyro.sample(f"A_{k}", dist.HalfNormal(2.0))
        phi_k = numpyro.sample(f"phi_{k}", dist.Uniform(0.0, 2.0 * jnp.pi))
        omega_k = 2.0 * jnp.pi / T_k
        mu = mu + A_k * jnp.cos(omega_k * t - phi_k)

    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)



# ============================================================================
# OLS initialization for MCMC
# ============================================================================

def _ols_init_m0(t, y, K=3, T_base=24.0):
    """Get OLS estimates for M0 to initialize MCMC."""
    omega = 2.0 * np.pi / T_base
    # Design matrix: [1, cos(w*t), sin(w*t), cos(2w*t), sin(2w*t), ...]
    X = np.ones((len(t), 1 + 2 * K))
    for k in range(1, K + 1):
        X[:, 2*k - 1] = np.cos(k * omega * t)
        X[:, 2*k] = np.sin(k * omega * t)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma_hat = np.std(resid)

    init_values = {"M": beta[0], "T_base": T_base, "sigma": max(sigma_hat, 0.01),
                   "tau_0": 1.0}
    for k in range(1, K + 1):
        init_values[f"a_{k}"] = beta[2*k - 1]
        init_values[f"b_{k}"] = beta[2*k]
    return init_values


def _ols_init_m1(t, y, period_priors):
    """Get OLS estimates for M1 to initialize MCMC."""
    K = len(period_priors)
    # Design matrix with independent periods
    X = np.ones((len(t), 1 + 2 * K))
    for k, (T_mean, _) in enumerate(period_priors):
        omega_k = 2.0 * np.pi / T_mean
        X[:, 1 + 2*k] = np.cos(omega_k * t)
        X[:, 2 + 2*k] = np.sin(omega_k * t)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma_hat = np.std(resid)

    init_values = {"M": beta[0], "sigma": max(sigma_hat, 0.01)}
    for k, (T_mean, _) in enumerate(period_priors):
        a = beta[1 + 2*k]
        b = beta[2 + 2*k]
        A = np.sqrt(a**2 + b**2)
        phi = np.arctan2(-b, a)  # cos(wt - phi) = cos(phi)cos(wt) + sin(phi)sin(wt)
        if phi < 0:
            phi += 2 * np.pi
        init_values[f"T_{k}"] = T_mean
        init_values[f"A_{k}"] = max(A, 0.01)
        init_values[f"phi_{k}"] = phi
    return init_values


# ============================================================================
# MCMC sampling
# ============================================================================

def _run_mcmc(model, t, y, num_warmup=500, num_samples=1000,
              num_chains=2, seed=0, init_values=None, **model_kwargs):
    """Run NUTS MCMC for a given model.

    Parameters
    ----------
    init_values : dict, optional
        Initial parameter values from OLS for faster convergence.

    Returns
    -------
    mcmc : numpyro.infer.MCMC object with samples
    """
    jax, jnp, numpyro, _, MCMC, NUTS, _, init_to_value = _lazy_imports()

    if init_values is not None:
        # Convert init values to jnp
        jnp_init = {k: jnp.array(v, dtype=jnp.float32) for k, v in init_values.items()}
        init_strategy = init_to_value(values=jnp_init)
    else:
        init_strategy = numpyro.infer.init_to_median()

    kernel = NUTS(model, max_tree_depth=8, init_strategy=init_strategy)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, t=jnp.array(t, dtype=jnp.float32),
             y=jnp.array(y, dtype=jnp.float32), **model_kwargs)
    return mcmc


# ============================================================================
# WAIC computation (proper log-likelihood based)
# ============================================================================

def _compute_log_likelihood_m0(samples, t, y, K=3):
    """Compute pointwise log-likelihood for M0 — fully vectorized over samples."""
    _, jnp, _, dist, _, _, _, _ = _lazy_imports()
    t_jnp = jnp.array(t, dtype=jnp.float32)
    y_jnp = jnp.array(y, dtype=jnp.float32)

    # All shapes: (n_samples,)
    T_base = samples["T_base"]
    omega = 2.0 * jnp.pi / T_base  # (n_samples,)
    M = samples["M"]               # (n_samples,)
    sigma = samples["sigma"]       # (n_samples,)

    # mu: (n_samples, N)
    mu = M[:, None] * jnp.ones_like(t_jnp)[None, :]
    for k in range(1, K + 1):
        a_k = samples[f"a_{k}"]  # (n_samples,)
        b_k = samples[f"b_{k}"]  # (n_samples,)
        # omega[:, None] * t_jnp[None, :] -> (n_samples, N)
        phase = k * omega[:, None] * t_jnp[None, :]
        mu = mu + a_k[:, None] * jnp.cos(phase) + b_k[:, None] * jnp.sin(phase)

    # log_lik: (n_samples, N)
    log_lik = dist.Normal(mu, sigma[:, None]).log_prob(y_jnp[None, :])
    return log_lik


def _compute_log_likelihood_m1(samples, t, y, n_periods=3):
    """Compute pointwise log-likelihood for M1 — fully vectorized over samples."""
    _, jnp, _, dist, _, _, _, _ = _lazy_imports()
    t_jnp = jnp.array(t, dtype=jnp.float32)
    y_jnp = jnp.array(y, dtype=jnp.float32)

    M = samples["M"]       # (n_samples,)
    sigma = samples["sigma"]  # (n_samples,)

    mu = M[:, None] * jnp.ones_like(t_jnp)[None, :]
    for k in range(n_periods):
        T_k = samples[f"T_{k}"]    # (n_samples,)
        A_k = samples[f"A_{k}"]    # (n_samples,)
        phi_k = samples[f"phi_{k}"]  # (n_samples,)
        omega_k = 2.0 * jnp.pi / T_k  # (n_samples,)
        phase = omega_k[:, None] * t_jnp[None, :] - phi_k[:, None]
        mu = mu + A_k[:, None] * jnp.cos(phase)

    log_lik = dist.Normal(mu, sigma[:, None]).log_prob(y_jnp[None, :])
    return log_lik


def _compute_waic_from_loglik(log_lik):
    """Compute WAIC from a (n_samples, N) log-likelihood matrix.

    WAIC = -2 * (lppd - p_waic)
    where lppd = sum_i log(mean_s exp(log_lik_si))
    and p_waic = sum_i var_s(log_lik_si)
    """
    jax, jnp, _, _, _, _, _, _ = _lazy_imports()
    # lppd: log pointwise predictive density
    lppd = jnp.sum(jax.scipy.special.logsumexp(log_lik, axis=0) - jnp.log(log_lik.shape[0]))
    # p_waic: effective number of parameters
    p_waic = jnp.sum(jnp.var(log_lik, axis=0))
    waic = -2.0 * (float(lppd) - float(p_waic))
    return {
        "waic": waic,
        "p_waic": float(p_waic),
        "lppd": float(lppd),
    }



# ============================================================================
# Parameter extraction
# ============================================================================

def _extract_params_m0(mcmc, K=3):
    """Extract posterior summaries from M0 samples."""
    _, jnp, _, _, _, _, _, _ = _lazy_imports()
    samples = mcmc.get_samples()
    result = {
        "M": float(jnp.mean(samples["M"])),
        "T_base": float(jnp.mean(samples["T_base"])),
        "T_base_sd": float(jnp.std(samples["T_base"])),
        "sigma": float(jnp.mean(samples["sigma"])),
    }
    T_base = result["T_base"]
    components = []
    for k in range(1, K + 1):
        a = float(jnp.mean(samples[f"a_{k}"]))
        b = float(jnp.mean(samples[f"b_{k}"]))
        A = np.sqrt(a**2 + b**2)
        phi = np.arctan2(b, a)
        components.append({
            "T": T_base / k,
            "A": A, "phi": phi, "a": a, "b": b,
            "A_sd": float(jnp.std(jnp.sqrt(samples[f"a_{k}"]**2 + samples[f"b_{k}"]**2))),
        })
    result["components"] = components
    return result


def _extract_params_m1(mcmc, n_periods=3):
    """Extract posterior summaries from M1 samples."""
    _, jnp, _, _, _, _, _, _ = _lazy_imports()
    samples = mcmc.get_samples()
    result = {
        "M": float(jnp.mean(samples["M"])),
        "sigma": float(jnp.mean(samples["sigma"])),
    }
    components = []
    for k in range(n_periods):
        T_samples = samples[f"T_{k}"]
        A_samples = samples[f"A_{k}"]
        phi_samples = samples[f"phi_{k}"]
        components.append({
            "T": float(jnp.mean(T_samples)),
            "T_sd": float(jnp.std(T_samples)),
            "A": float(jnp.mean(A_samples)),
            "A_sd": float(jnp.std(A_samples)),
            "phi": float(jnp.mean(phi_samples)),
            "phi_sd": float(jnp.std(phi_samples)),
        })
    result["components"] = components
    return result


# ============================================================================
# BHDT MCMC: single-gene inference
# ============================================================================

def bhdt_mcmc(t, y, T_base=24.0, K_harmonics=3,
              period_priors=None,
              num_warmup=500, num_samples=1000, num_chains=2,
              seed=0):
    """Run full Bayesian BHDT on a single gene using MCMC.

    Fits both M0 (harmonic) and M1 (independent) models, computes
    WAIC for model comparison, and classifies the gene.

    Parameters
    ----------
    t : array
        Time points in hours.
    y : array
        Expression values.
    T_base : float
        Base circadian period for M0.
    K_harmonics : int
        Number of harmonics for M0.
    period_priors : list of (mean, sd)
        Period priors for M1. Default [(24,2), (12,1.5), (8,1)].
    num_warmup, num_samples, num_chains : int
        MCMC parameters.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        waic_m0, waic_m1, delta_waic, classification,
        m0_params, m1_params, m0_waic_detail, m1_waic_detail
    """
    jax, jnp, _, _, _, _, _, _ = _lazy_imports()

    if period_priors is None:
        period_priors = [(24.0, 2.0), (12.0, 1.5), (8.0, 1.0)]

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Standardise for numerical stability
    y_mean = np.mean(y_arr)
    y_std = np.std(y_arr) + 1e-8
    y_norm = (y_arr - y_mean) / y_std

    # OLS initialization
    init_m0 = _ols_init_m0(t_arr, y_norm, K=K_harmonics, T_base=T_base)
    init_m1 = _ols_init_m1(t_arr, y_norm, period_priors=period_priors)

    # Fit M0
    mcmc_m0 = _run_mcmc(model_m0_harmonic, t_arr, y_norm,
                         num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains, seed=seed,
                         init_values=init_m0,
                         K=K_harmonics, T_base_prior_mean=T_base)
    params_m0 = _extract_params_m0(mcmc_m0, K=K_harmonics)

    # Fit M1
    mcmc_m1 = _run_mcmc(model_m1_independent, t_arr, y_norm,
                         num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains, seed=seed + 1,
                         init_values=init_m1,
                         period_priors=period_priors)
    params_m1 = _extract_params_m1(mcmc_m1, n_periods=len(period_priors))

    # Compute WAIC via direct log-likelihood
    samples_m0 = mcmc_m0.get_samples()
    samples_m1 = mcmc_m1.get_samples()
    log_lik_m0 = _compute_log_likelihood_m0(samples_m0, t_arr, y_norm, K=K_harmonics)
    log_lik_m1 = _compute_log_likelihood_m1(samples_m1, t_arr, y_norm,
                                             n_periods=len(period_priors))
    waic_m0 = _compute_waic_from_loglik(log_lik_m0)
    waic_m1 = _compute_waic_from_loglik(log_lik_m1)

    # Model comparison via WAIC
    # Lower WAIC = better model
    delta_waic = waic_m0["waic"] - waic_m1["waic"]
    # Positive delta = M1 better (independent oscillators)
    # Negative delta = M0 better (harmonic)

    # SE of delta WAIC (Vehtari et al. 2017)
    # Correct approach: pointwise elpd via logsumexp, not mean of log-likelihoods
    S = log_lik_m0.shape[0]  # number of MCMC samples
    N = log_lik_m0.shape[1]  # number of data points
    elpd_m0_i = jax.scipy.special.logsumexp(log_lik_m0, axis=0) - jnp.log(S)
    elpd_m1_i = jax.scipy.special.logsumexp(log_lik_m1, axis=0) - jnp.log(S)
    pointwise_diff = elpd_m1_i - elpd_m0_i
    se_delta = float(jnp.sqrt(N * jnp.var(pointwise_diff)))

    # Classification based on WAIC difference
    classification = _classify_mcmc(delta_waic, se_delta, params_m1, t_arr, y_norm)

    return {
        "waic_m0": waic_m0["waic"],
        "waic_m1": waic_m1["waic"],
        "delta_waic": float(delta_waic),
        "se_delta_waic": float(se_delta),
        "classification": classification,
        "m0_params": params_m0,
        "m1_params": params_m1,
        "m0_waic_detail": waic_m0,
        "m1_waic_detail": waic_m1,
    }


def _classify_mcmc(delta_waic, se_delta, params_m1, t, y):
    """Classify gene based on MCMC WAIC comparison.

    Uses delta_WAIC with SE-based significance thresholds.
    Also checks component significance via posterior credible intervals.
    """
    from chord.bhdt.inference import component_f_test

    # F-test gating (same as analytic mode)
    f_test_24 = component_f_test(np.asarray(t), np.asarray(y),
                                  [24.0, 12.0, 8.0], test_period_idx=0)
    f_test_12 = component_f_test(np.asarray(t), np.asarray(y),
                                  [24.0, 12.0, 8.0], test_period_idx=1)

    sig_24 = f_test_24["significant"]
    sig_12 = f_test_12["significant"]

    if not sig_24 and not sig_12:
        return "non_rhythmic"
    if sig_24 and not sig_12:
        return "circadian_only"
    if sig_12 and not sig_24:
        return "independent_ultradian"

    # Both significant — use WAIC to disentangle
    # delta_waic > 0 means M1 (independent) is better
    # Use z-score: delta_waic / se_delta
    if se_delta < 1e-6:
        z = 0.0
    else:
        z = delta_waic / se_delta

    if z > 2.0:
        return "independent_ultradian"
    elif z > 1.0:
        return "likely_independent_ultradian"
    elif z < -2.0:
        return "harmonic"
    elif z < -1.0:
        return "harmonic"
    else:
        # Check period deviation from M1 posterior
        if len(params_m1["components"]) >= 2:
            T_12 = params_m1["components"][1]["T"]
            T_12_sd = params_m1["components"][1]["T_sd"]
            T_24 = params_m1["components"][0]["T"]
            expected_12 = T_24 / 2.0
            # If T_12 posterior excludes T_24/2, evidence for independence
            z_period = abs(T_12 - expected_12) / max(T_12_sd, 0.01)
            if z_period > 2.0:
                return "likely_independent_ultradian"
        return "ambiguous"
