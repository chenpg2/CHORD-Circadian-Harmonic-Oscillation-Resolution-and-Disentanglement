"""
Spectral Independence Divergence (SID).

An information-theoretic measure that quantifies how much the observed
spectral structure deviates from what a purely harmonic (Fourier series
of a single periodic waveform) model would predict.

SID combines:
  - Spectral energy divergence (KL divergence between observed and
    harmonic-predicted energy distributions)
  - Phase coherence divergence (how much observed phases violate the
    harmonic phase constraint)

For a purely periodic waveform, ALL spectral energy at harmonic
frequencies is constrained by a single shape function.  Independent
oscillators violate these constraints, yielding high SID.

Implementation approach
-----------------------
Rather than comparing M0 vs M1-free (which fails when the independent
component sits at an exact harmonic frequency), we compare the OBSERVED
spectral structure directly against harmonic constraints:

1. Fit OLS at harmonic periods [T, T/2, T/3, ...] to get observed
   amplitudes A_k and phases phi_k.
2. Fit M0 (harmonic model) to get the best-fit harmonic prediction
   for amplitude ratios.
3. Spectral divergence: KL divergence between observed and M0-predicted
   energy distributions.
4. Phase divergence: check whether phi_k is consistent with k * phi_1
   (the harmonic phase-locking constraint).  For a single periodic
   waveform whose fundamental has phase phi_1, higher harmonics satisfy
   phi_k = k * phi_1 + delta_k(shape).  We use M0's fitted phases as
   the harmonic prediction and measure deviation via
   1 - cos(phi_obs_k - phi_harm_k).  Since M0 and unconstrained OLS
   at the same periods give identical fits, we instead test the
   structural constraint phi_k ~ k * phi_1 directly.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from chord.bhdt.models import (
    fit_harmonic_model,
    _cos_sin_design,
    _fit_ols,
    _extract_rhythm_params,
)


# ============================================================================
# Core SID computation
# ============================================================================

def compute_sid(t, y, T_base=24.0, K_harmonics=3, lam=1.0):
    """Compute Spectral Independence Divergence.

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Observed values.
    T_base : float
        Base period (default 24 h).
    K_harmonics : int
        Number of harmonics to consider.
    lam : float
        Weight for phase divergence term (default 1.0).

    Returns
    -------
    dict with keys:
        sid : float -- total SID score
        spectral_divergence : float -- KL divergence of energy distributions
        phase_divergence : float -- weighted phase coherence divergence
        lambda : float -- weight used
        components : list of dict -- per-harmonic breakdown
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eps = 1e-10

    # --- Fit OLS at exact harmonic periods to get observed amplitudes/phases ---
    periods = [T_base / k for k in range(1, K_harmonics + 1)]
    X = _cos_sin_design(t, periods)
    beta, rss, sigma2, log_lik = _fit_ols(X, y)
    mesor, obs_components = _extract_rhythm_params(beta, periods)

    obs_A = np.array([c["A"] for c in obs_components])
    obs_phi = np.array([c["phi"] for c in obs_components])
    obs_energy = obs_A ** 2 + eps

    # --- Harmonic prediction for energy distribution ---
    # Fit M0 to get the harmonic model's predicted amplitudes.
    # M0 uses the same design matrix, so its amplitudes are identical to
    # the unconstrained OLS.  For spectral divergence we compare the
    # observed energy distribution against a "smooth decay" reference:
    # for a typical periodic waveform the energy decays roughly as 1/k^2.
    # We use this as the reference distribution q.
    ref_energy = np.array([1.0 / k ** 2 for k in range(1, K_harmonics + 1)])
    ref_energy = ref_energy + eps

    # Normalise to probability distributions
    p = obs_energy / obs_energy.sum()
    q = ref_energy / ref_energy.sum()

    # KL divergence: D_KL(p || q)
    d_kl = float(np.sum(p * np.log(p / q)))

    # --- Phase coherence divergence ---
    # Harmonic phase constraint: for a single periodic waveform, the
    # phase of the k-th harmonic is deterministically related to the
    # fundamental's phase.  The simplest constraint is phi_k = k * phi_1
    # (holds exactly for symmetric waveforms, approximately for peaked
    # biological waveforms).
    #
    # We measure deviation from this constraint for harmonics k >= 2,
    # weighted by their energy relative to each other (not the fundamental,
    # which has zero deviation by definition and would dilute the signal).
    phi_1 = obs_phi[0]

    # Energy weights among higher harmonics only (k >= 2)
    higher_energy = obs_energy[1:]  # k=2, k=3, ...
    if higher_energy.sum() > eps:
        w_higher = higher_energy / higher_energy.sum()
    else:
        w_higher = np.ones(K_harmonics - 1) / max(K_harmonics - 1, 1)

    phase_deviations = np.zeros(K_harmonics)
    per_harmonic_phase = np.zeros(K_harmonics)
    for k_idx in range(K_harmonics):
        k = k_idx + 1
        # Expected phase under harmonic constraint
        phi_expected = k * phi_1
        # Circular distance
        delta = obs_phi[k_idx] - phi_expected
        phase_deviations[k_idx] = delta
        if k_idx >= 1:  # only higher harmonics contribute
            per_harmonic_phase[k_idx] = w_higher[k_idx - 1] * (1.0 - np.cos(delta))

    d_phase = float(np.sum(per_harmonic_phase))

    # --- Total SID ---
    sid = d_kl + lam * d_phase

    # --- Per-harmonic breakdown ---
    components = []
    for k_idx in range(K_harmonics):
        k = k_idx + 1
        components.append({
            "harmonic": k,
            "T": float(periods[k_idx]),
            "A_observed": float(obs_A[k_idx]),
            "phi_observed": float(obs_phi[k_idx]),
            "phi_expected": float(k * phi_1),
            "energy_observed": float(obs_energy[k_idx]),
            "phase_diff": float(phase_deviations[k_idx]),
            "kl_contribution": float(p[k_idx] * np.log(p[k_idx] / q[k_idx])),
            "phase_contribution": float(per_harmonic_phase[k_idx]),
        })

    return {
        "sid": sid,
        "spectral_divergence": d_kl,
        "phase_divergence": d_phase,
        "lambda": lam,
        "components": components,
    }


# ============================================================================
# Evidence scoring
# ============================================================================

def sid_evidence(t, y, T_base=24.0, K_harmonics=3):
    """Convert SID into a BHDT evidence score.

    Thresholds:
        SID > 1.5  -> +2
        SID > 0.8  -> +1
        SID < 0.3  -> -2
        SID < 0.5  -> -1
        Phase divergence > 0.5 -> +1
        Phase divergence < 0.1 -> -1

    Parameters
    ----------
    t, y : arrays
        Time series data.
    T_base : float
        Base period.
    K_harmonics : int
        Number of harmonics.

    Returns
    -------
    dict with keys: sid_score, sid, spectral_divergence, phase_divergence
    """
    result = compute_sid(t, y, T_base=T_base, K_harmonics=K_harmonics)
    sid_val = result["sid"]
    phase_div = result["phase_divergence"]

    score = 0

    # SID thresholds (check from most extreme first)
    if sid_val > 1.5:
        score += 2
    elif sid_val > 0.8:
        score += 1
    elif sid_val < 0.3:
        score -= 2
    elif sid_val < 0.5:
        score -= 1

    # Phase divergence bonus
    if phase_div > 0.5:
        score += 1
    elif phase_div < 0.1:
        score -= 1

    return {
        "sid_score": score,
        "sid": sid_val,
        "spectral_divergence": result["spectral_divergence"],
        "phase_divergence": phase_div,
    }


# ============================================================================
# Bootstrap significance test
# ============================================================================

def sid_test(t, y, T_base=24.0, K_harmonics=3, n_bootstrap=499, seed=None):
    """Bootstrap test for SID significance.

    Simulates data under the harmonic null (M0) and computes the null
    distribution of SID to obtain a p-value.

    Parameters
    ----------
    t, y : arrays
        Time series data.
    T_base : float
        Base period.
    K_harmonics : int
        Number of harmonics.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        sid_observed, p_value, sid_null_mean, sid_null_std,
        significant, components
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Observed SID
    obs_result = compute_sid(t, y, T_base=T_base, K_harmonics=K_harmonics)
    sid_obs = obs_result["sid"]

    # Build a phase-constrained null model.
    # Standard M0 fits each harmonic's cos/sin independently, so its
    # phases can absorb an independent oscillator's phase.  For the null
    # we reconstruct a signal where phi_k = k * phi_1 (the harmonic
    # phase constraint), preserving the fitted amplitudes.
    m0 = fit_harmonic_model(t, y, T_base=T_base, K=K_harmonics)
    periods = m0["periods"]
    comps = m0["components"]
    phi_1 = comps[0]["phi"]
    omega_base = 2.0 * np.pi / T_base

    # Reconstruct y_hat with phase-locked harmonics
    y_hat = np.full(len(t), m0["mesor"])
    for k_idx, comp in enumerate(comps):
        k = k_idx + 1
        A_k = comp["A"]
        phi_k_constrained = k * phi_1
        y_hat = y_hat + A_k * np.cos(k * omega_base * t - phi_k_constrained)

    residuals = y - y_hat
    sigma_m0 = np.sqrt(np.mean(residuals ** 2))

    # Bootstrap: simulate under M0
    sid_null = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        noise = rng.normal(0, sigma_m0, len(t))
        y_sim = y_hat + noise
        try:
            sim_result = compute_sid(t, y_sim, T_base=T_base,
                                     K_harmonics=K_harmonics)
            sid_null[i] = sim_result["sid"]
        except Exception:
            sid_null[i] = 0.0

    # p-value (one-sided: how often null >= observed)
    p_value = float(np.sum(sid_null >= sid_obs) + 1) / (n_bootstrap + 1)

    return {
        "sid_observed": sid_obs,
        "p_value": p_value,
        "sid_null_mean": float(np.mean(sid_null)),
        "sid_null_std": float(np.std(sid_null)),
        "significant": p_value < 0.05,
        "components": obs_result["components"],
    }
