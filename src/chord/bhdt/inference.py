"""
BHDT inference engine: Bayes Factor computation for harmonic disentanglement.

Provides two modes:
  - analytic: BIC-based approximate Bayes Factor (fast, default)
  - mcmc: Full NumPyro MCMC (precise, requires chord-rhythm[bayes])
"""

import numpy as np
from typing import Dict, Optional, Any, List

from chord.bhdt.models import (
    fit_harmonic_model,
    fit_independent_model,
    fit_independent_free_period,
    _cos_sin_design,
)


# ============================================================================
# BIC-based approximate Bayes Factor (analytic mode)
# ============================================================================

def _bic_bayes_factor(bic_m0, bic_m1):
    """Approximate log Bayes Factor from BIC difference.

    log BF_10 ~ (BIC_M0 - BIC_M1) / 2

    A positive value favours M1 (independent oscillators).
    """
    return (bic_m0 - bic_m1) / 2.0


def _interpret_bf(log_bf):
    """Interpret natural log Bayes Factor using Kass & Raftery (1995) scale.

    Thresholds based on 2*ln(BF): 0-2 (not worth mentioning),
    2-6 (positive), 6-10 (strong), >10 (very strong).

    Returns
    -------
    str : interpretation label
    """
    bf = np.exp(log_bf)
    if bf > np.exp(5):          # 2*ln(BF) > 10: very strong
        return "very_strong_independent"
    elif bf > np.exp(3):        # 2*ln(BF) > 6: strong
        return "strong_independent"
    elif bf > np.exp(1):        # 2*ln(BF) > 2: positive
        return "moderate_independent"
    elif bf > 1:                # 2*ln(BF) > 0: barely worth mentioning
        return "weak_independent"
    elif bf > np.exp(-1):       # 2*ln(BF) > -2
        return "inconclusive"
    elif bf > np.exp(-3):       # 2*ln(BF) > -6
        return "moderate_harmonic"
    elif bf > np.exp(-5):       # 2*ln(BF) > -10
        return "strong_harmonic"
    else:
        return "very_strong_harmonic"


def _period_deviation_test(fitted_periods, T_base=24.0):
    """Test whether fitted 12-h period deviates from T_base/2.

    If the best-fit T_12 is significantly different from T_base/2,
    this is additional evidence for an independent oscillator.

    Returns
    -------
    dict with deviation statistics
    """
    expected_12h = T_base / 2.0
    # Find the period closest to 12h
    diffs = [abs(p - expected_12h) for p in fitted_periods]
    idx_12 = int(np.argmin(diffs))
    T_12_fitted = fitted_periods[idx_12]
    deviation = T_12_fitted - expected_12h
    rel_deviation = abs(deviation) / expected_12h

    return {
        "T_12_fitted": float(T_12_fitted),
        "T_12_expected": expected_12h,
        "deviation_hours": float(deviation),
        "relative_deviation": float(rel_deviation),
        "deviates_from_harmonic": rel_deviation > 0.02,  # >2% deviation
    }


# ============================================================================
# Single-gene BHDT (analytic mode)
# ============================================================================

def bhdt_analytic(t, y, T_base=24.0, K_harmonics=3, ultradian_periods=None,
                  classifier_version="v2", use_savage_dickey=False):
    """Run BHDT on a single gene using BIC-based Bayes Factor.

    Parameters
    ----------
    t : array
        Time points in hours.
    y : array
        Expression values.
    T_base : float
        Base circadian period (default 24).
    K_harmonics : int
        Number of harmonics for M0 (default 3: 24h, 12h, 8h).
    ultradian_periods : list of float, optional
        Periods for M1. Default [24, 12, 8].
    classifier_version : str
        "v1" for original hard-gate classifier, "v2" for soft-gate (default).

    Returns
    -------
    dict with keys:
        log_bayes_factor, bayes_factor, interpretation,
        m0 (harmonic model fit), m1 (independent model fit),
        m1_free (free-period model fit), period_deviation,
        classification
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("t and y must be 1-D arrays.")
    if len(t) != len(y):
        raise ValueError(
            f"t and y must have the same length, got {len(t)} and {len(y)}."
        )
    if len(t) < 6:
        raise ValueError(
            f"Need at least 6 observations (got {len(t)}). "
            "Minimum for a 3-parameter harmonic model is 2K+1 with K>=1."
        )
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values. Remove or impute them first.")
    if T_base <= 0:
        raise ValueError(f"T_base must be positive, got {T_base}.")
    if K_harmonics < 1:
        raise ValueError(f"K_harmonics must be >= 1, got {K_harmonics}.")

    if ultradian_periods is None:
        ultradian_periods = [24.0, 12.0, 8.0]

    # Fit models
    m0 = fit_harmonic_model(t, y, T_base=T_base, K=K_harmonics)
    m1 = fit_independent_model(t, y, periods=ultradian_periods)

    # Free-period fit: THIS is the key comparison.
    # M0 constrains periods to exact harmonics (T, T/2, T/3, ...)
    # M1-free allows each period to float independently.
    # The BIC difference between M0 and M1-free captures whether
    # freeing the periods improves the fit enough to justify the
    # extra parameters.
    m1_free = fit_independent_free_period(t, y, period_inits=ultradian_periods)
    if not m1_free.get("optimisation_success", True):
        import warnings
        warnings.warn(
            f"L-BFGS-B did not converge for free-period fit. "
            f"Fitted periods {m1_free['fitted_periods']} may be unreliable.",
            RuntimeWarning,
        )
    period_dev = _period_deviation_test(m1_free["fitted_periods"], T_base)

    # Primary BF: M0 (harmonic-constrained) vs M1-free (free periods)
    # Use AICc for small samples (Burnham & Anderson 2002)
    N = len(t)
    if N < 50 and "aicc" in m0 and "aicc" in m1_free:
        log_bf = (m0["aicc"] - m1_free["aicc"]) / 2.0
    else:
        log_bf = _bic_bayes_factor(m0["bic"], m1_free["bic"])

    # Guard: fall back to BIC if AICc produced non-finite log_bf
    if not np.isfinite(log_bf):
        log_bf = _bic_bayes_factor(m0["bic"], m1_free["bic"])

    # Optional: Use Savage-Dickey exact BF if requested
    savage_dickey_result = None
    if use_savage_dickey:
        try:
            from chord.bhdt.savage_dickey import savage_dickey_bf
            savage_dickey_result = savage_dickey_bf(t, y, T_base=T_base,
                                                    n_samples=2000, n_warmup=1000)
            # Use SD BF instead of BIC BF
            log_bf = savage_dickey_result['log_bf']
        except Exception:
            pass  # Fall back to BIC-based BF

    # Auto-trigger bootstrap when BIC/AICc is ambiguous (|log_bf| < 3)
    bootstrap_result = None
    if abs(log_bf) < 3.0:
        try:
            from chord.bhdt.bootstrap import parametric_bootstrap_lrt
            bootstrap_result = parametric_bootstrap_lrt(
                t, y, T_base=T_base, K_harmonics=K_harmonics,
                n_bootstrap=199, seed=hash(tuple(y[:5].tolist())) % (2**31),
                period_inits=ultradian_periods,
            )
        except Exception:
            pass  # bootstrap is supplementary, don't fail if it errors

    bf = np.exp(log_bf)
    interpretation = _interpret_bf(log_bf)

    # Combined classification — uses F-test for significance gating
    f_test_24 = component_f_test(t, y, ultradian_periods, test_period_idx=0)
    f_test_12 = component_f_test(t, y, ultradian_periods, test_period_idx=1)
    _bootstrap_pval = (bootstrap_result["p_value"]
                       if bootstrap_result is not None else None)

    # v2: auto-trigger permutation F-test for borderline p_12
    if classifier_version == "v2" and 0.02 <= f_test_12["p_value"] <= 0.15:
        try:
            perm_result = permutation_f_test(
                t, y, ultradian_periods, test_period_idx=1,
                n_perm=499, seed=hash(tuple(y[:5].tolist())) % (2**31),
            )
            # Use permutation p-value instead of parametric
            f_test_12 = dict(f_test_12)  # copy to avoid mutating
            f_test_12["p_value"] = perm_result["p_value"]
            f_test_12["significant"] = perm_result["significant"]
            f_test_12["permutation_used"] = True
        except Exception:
            pass  # fall back to parametric

    if classifier_version == "v2":
        classification = _classify_gene_v2(
            m0, m1, m1_free, log_bf, period_dev,
            f_test_24, f_test_12,
            bootstrap_pvalue=_bootstrap_pval,
            t=t, y=y,
        )
    else:
        classification = _classify_gene(
            m0, m1, m1_free, log_bf, period_dev,
            f_test_24, f_test_12,
            bootstrap_pvalue=_bootstrap_pval,
            t=t, y=y,
        )

    result = {
        "log_bayes_factor": float(log_bf),
        "bayes_factor": float(bf),
        "interpretation": interpretation,
        "m0": m0,
        "m1": m1,
        "m1_free": m1_free,
        "period_deviation": period_dev,
        "classification": classification,
    }
    if bootstrap_result is not None:
        result["bootstrap_pvalue"] = bootstrap_result["p_value"]
        result["bootstrap_lrt"] = bootstrap_result["lrt_observed"]
    if savage_dickey_result is not None:
        result["savage_dickey"] = savage_dickey_result
    return result


def _classify_gene(m0, m1, m1_free, log_bf, period_dev, f_test_24, f_test_12,
                   bootstrap_pvalue=None, t=None, y=None):
    """Classify a gene based on multi-evidence BHDT results.

    Uses up to six lines of evidence:
      1. F-test significance for 24h and 12h components (hard gate)
      2. BIC/AICc-based Bayes Factor (M0 vs M1-free)
      3. Period deviation (does fitted T_12 deviate from T_base/2?)
      4. Amplitude ratio (A_12/A_24 — high ratio unusual for harmonics)
      5. Residual improvement (does freeing periods substantially improve fit?)
      6. Bootstrap LRT p-value (optional, auto-triggered for ambiguous BF)

    The amplitude ratio is the strongest single indicator for independence:
    in true harmonic scenarios, A_12/A_24 is typically 0.2-0.4 (Fourier
    coefficients of non-sinusoidal waveforms). When A_12/A_24 > 0.5, the
    12h component is too strong to be a mere harmonic artifact.

    Categories:
      - 'independent_ultradian': strong evidence for independent 12h oscillator
      - 'likely_independent_ultradian': moderate evidence
      - 'harmonic': 12h signal is a harmonic of 24h
      - 'circadian_only': only 24h signal, no significant 12h
      - 'non_rhythmic': no significant oscillation
      - 'ambiguous': evidence is inconclusive
    """
    bf = np.exp(log_bf)
    sig_24 = f_test_24["significant"]
    sig_12 = f_test_12["significant"]

    # Hard gate: F-test significance
    if not sig_24 and not sig_12:
        return "non_rhythmic"
    if sig_24 and not sig_12:
        return "circadian_only"
    if sig_12 and not sig_24:
        return "independent_ultradian"

    # Both significant. Disentangle with multi-evidence scoring.
    # The scoring is SYMMETRIC: positive scores favour independent,
    # negative scores favour harmonic.  Each evidence line can push
    # in BOTH directions so that harmonic genes accumulate negative
    # evidence instead of landing in a dead zone.
    evidence_score = 0

    # --- Evidence 1: BIC-based Bayes Factor ---
    # BIC is conservative at N=24 but still informative directionally.
    if bf > 3:
        evidence_score += 2
    elif bf > 1:
        evidence_score += 1
    elif bf < 0.05:
        evidence_score -= 2  # strong: BIC strongly favors harmonic
    elif bf < 0.3:
        evidence_score -= 1
    elif bf < 1:
        evidence_score -= 1  # BF < 1 leans harmonic — no dead zone

    # --- Evidence 2: Period deviation ---
    # Deviation FROM exact harmonic → independent evidence.
    # NO deviation → harmonic evidence (period is exactly T/2).
    if period_dev["deviates_from_harmonic"]:
        evidence_score += 2
        if period_dev["relative_deviation"] > 0.05:
            evidence_score += 1
    else:
        # Period is consistent with being an exact harmonic of T_base
        evidence_score -= 1

    # --- Evidence 3: Amplitude ratio ---
    # In harmonic scenarios (sawtooth, peaked wave), A_12/A_24 is typically
    # 0.2-0.5 (Fourier coefficients of non-sinusoidal waveforms).
    # In independent superposition, it's typically 0.6-1.5+.
    # The scoring is now symmetric: harmonic-range ratios push negative.
    m1_amps = {c["T"]: c["A"] for c in m1["components"]}
    a_24 = m1_amps.get(24.0, 1e-10)
    a_12 = m1_amps.get(12.0, 0)
    amp_ratio = a_12 / max(a_24, 1e-10)
    if amp_ratio > 0.8:
        evidence_score += 3  # very strong: 12h nearly as strong as 24h
    elif amp_ratio > 0.65:
        evidence_score += 2  # strong: above typical harmonic range
    elif amp_ratio > 0.5:
        evidence_score += 1  # moderate: overlap zone
    elif amp_ratio < 0.25:
        evidence_score -= 2  # very weak 12h, strongly consistent with harmonic
    elif amp_ratio < 0.5:
        evidence_score -= 1  # harmonic-typical range (0.25-0.5)

    # --- Evidence 4: Residual improvement test ---
    residual_evidence = _residual_periodicity_test(m0, m1, m1_free)
    evidence_score += residual_evidence

    # --- Evidence 5 (optional): Bootstrap LRT p-value ---
    if bootstrap_pvalue is not None:
        if bootstrap_pvalue < 0.01:
            evidence_score += 2
        elif bootstrap_pvalue < 0.05:
            evidence_score += 1
        elif bootstrap_pvalue > 0.5:
            evidence_score -= 2
        elif bootstrap_pvalue > 0.3:
            evidence_score -= 1

    # --- Evidence 6: Phase coupling test ---
    # Harmonics have a fixed phase relationship: phi_12 ≈ 2*phi_24 (mod 2π)
    # Independent oscillators have no such constraint.
    phase_coupling = _phase_coupling_score(m1, m1_free=m1_free)
    evidence_score += phase_coupling

    # --- Evidence 7: Waveform asymmetry test ---
    # Non-sinusoidal 24h waveforms (which produce harmonic artifacts) tend
    # to have asymmetric rise/fall times.
    if t is not None and y is not None:
        asymmetry_evidence = _waveform_asymmetry_score(t, y, m0)
        evidence_score += asymmetry_evidence

    # --- Classification thresholds ---
    if evidence_score >= 4:
        return "independent_ultradian"
    elif evidence_score >= 2:
        return "likely_independent_ultradian"
    elif evidence_score <= -2:
        return "harmonic"
    else:
        return "ambiguous"


def _residual_periodicity_test(m0, m1, m1_free):
    """Test whether freeing periods improves fit over M0.

    Returns an evidence score: positive favours independent, 0 is inconclusive.
    No negative scores — M0 contains the 12h component for both harmonic and
    independent genes, so low improvement is inconclusive, not harmonic evidence.
    """
    rss_m0 = m0["rss"]
    rss_m1f = m1_free["rss"]

    if rss_m0 <= 0:
        return 0

    frac_improvement = (rss_m0 - rss_m1f) / rss_m0

    if frac_improvement > 0.35:
        return 2
    elif frac_improvement > 0.25:
        return 1
    return 0


def _phase_coupling_score(m1, m1_free=None):
    """Score based on phase coupling between 24h and 12h components.

    Uses z-score approach: angular distance from harmonic prediction
    normalized by phase estimation uncertainty (SE = sigma / (A * sqrt(N/2))).
    This adapts to sample size and amplitude — at low SNR or small N,
    the test correctly abstains instead of making noisy calls.

    For harmonics: phi_12 ≈ 2*phi_24 (mod 2π) or phi_12 ≈ 2*phi_24 + π
    For independent oscillators: phi_12 and phi_24 are unrelated.

    Returns: negative score (harmonic evidence) or positive (independent).
    """
    source = m1_free if m1_free is not None else m1
    comps = source.get("components", [])
    if len(comps) < 2:
        return 0

    if m1_free is None:
        return 0

    phi_24, phi_12, a_12 = None, None, None
    for c in comps:
        T = c.get("T", 0)
        if abs(T - 24.0) < 4.0 and abs(T - 24.0) < abs(T - 12.0):
            phi_24 = c.get("phi", None)
        elif abs(T - 12.0) < 4.0:
            phi_12 = c.get("phi", None)
            a_12 = c.get("A", 0)

    if phi_24 is None or phi_12 is None:
        return 0

    # Circular distance to harmonic prediction (and shifted by pi)
    expected = (2 * phi_24) % (2 * np.pi)
    actual = phi_12 % (2 * np.pi)

    diff1 = min(abs(actual - expected), 2 * np.pi - abs(actual - expected))
    expected_shifted = (expected + np.pi) % (2 * np.pi)
    diff2 = min(abs(actual - expected_shifted),
                2 * np.pi - abs(actual - expected_shifted))
    min_diff = min(diff1, diff2)

    # Phase SE: sigma_phi = sigma_noise / (A_12 * sqrt(N/2))
    N = source.get("N", 24)
    rss = source.get("rss", 1.0)
    n_params = len(comps) * 3 + 1
    sigma = np.sqrt(rss / max(N - n_params, 1))
    phase_se = sigma / max(a_12 * np.sqrt(N / 2.0), 1e-10)

    # If phase SE > 45°, estimate is too noisy to be informative
    if phase_se > np.pi / 4:
        return 0

    z = min_diff / max(phase_se, 1e-10)

    # Only trust positive (independence) evidence when the fitted period is
    # close to 12h. Large period deviations shift the phase estimate, making
    # the comparison to 2*phi_24 unreliable.
    fitted_T_12 = None
    for c in comps:
        if abs(c.get("T", 0) - 12.0) < 4.0:
            fitted_T_12 = c.get("T", 12.0)
    period_ok = fitted_T_12 is not None and abs(fitted_T_12 - 12.0) / 12.0 < 0.05

    if z > 3.0 and period_ok:
        return 1
    elif z < 1.0:
        return -1
    return 0


def _waveform_asymmetry_score(t, y, m0):
    """Score based on 24h waveform asymmetry.

    Non-sinusoidal 24h waveforms (sawtooth, peaked, square) produce harmonic
    artifacts. These waveforms typically have asymmetric rise/fall times.
    High asymmetry + good M0 fit → harmonic evidence.

    Returns: negative score (harmonic evidence) or 0.
    """
    T_base = 24.0
    phase = (np.asarray(t) % T_base) / T_base
    y_arr = np.asarray(y, dtype=float)

    order = np.argsort(phase)
    phase_sorted = phase[order]
    y_sorted = y_arr[order]

    if len(y_sorted) < 6:
        return 0

    peak_idx = np.argmax(y_sorted)
    peak_phase = phase_sorted[peak_idx]

    # Asymmetry: deviation from nearest symmetric peak position (0.0 or 0.5)
    nearest_sym = round(peak_phase * 2) / 2
    asymmetry = abs(peak_phase - nearest_sym)

    # Rise/fall time ratio
    trough_idx = np.argmin(y_sorted)
    trough_phase = phase_sorted[trough_idx]
    rise_time = (peak_phase - trough_phase) % 1.0
    fall_time = 1.0 - rise_time
    if fall_time > 0 and rise_time > 0:
        rf_ratio = min(rise_time, fall_time) / max(rise_time, fall_time)
    else:
        rf_ratio = 1.0

    # High asymmetry AND good M0 fit → harmonic evidence
    m0_r2 = 1.0 - m0["rss"] / max(np.var(y_arr) * len(y_arr), 1e-10)
    if (asymmetry > 0.1 or rf_ratio < 0.5) and m0_r2 > 0.5:
        return -1  # asymmetric waveform well-fit by M0 → harmonic
    return 0


def _harmonic_coherence_score(t, y, T_base=24.0):
    """Test whether 12h component follows sawtooth-like 1/n amplitude decay.

    For a non-sinusoidal periodic waveform with period T, the Fourier series
    has harmonics at T/n with amplitudes decaying as 1/n (sawtooth) or faster.
    This test checks whether the observed amplitude ratios A_12/A_24, A_8/A_24,
    A_6/A_24 are consistent with the 1/n prediction.

    Uses a chi-squared goodness-of-fit statistic comparing observed amplitudes
    at 12h, 8h, and 6h against the sawtooth prediction A_n = A_24/n.

    Gates (all must pass for the test to fire):
      - A_24 must be detectable (> 3× noise amplitude)
      - A_12/A_24 must be in harmonic range [0.2, 0.8]
      - A_8 must be detectable (> 2× noise amplitude)

    Returns
    -------
    dict with keys:
        score : float
            Negative (harmonic evidence) if consistent with sawtooth, 0 otherwise.
            -2.0 for strong match (chi2 < 4), -1.0 for moderate (chi2 < 8).
        chi2 : float
            Chi-squared statistic (lower = more sawtooth-like).
        gated_in : bool
            Whether the signal passed all gates.

    References
    ----------
    Aarts 2021, IEEE Signal Processing Magazine 38(5):86-95 (form factor).
    Sheppard et al. 2011, Phys Rev E 83:016206 (harmonic detection).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = len(t)

    omega_24 = 2 * np.pi / T_base
    omega_12 = 2 * np.pi / (T_base / 2.0)
    omega_8 = 2 * np.pi / (T_base / 3.0)
    omega_6 = 2 * np.pi / (T_base / 4.0)

    # Fit all four harmonics simultaneously
    X = np.column_stack([
        np.ones(N),
        np.cos(omega_24 * t), np.sin(omega_24 * t),
        np.cos(omega_12 * t), np.sin(omega_12 * t),
        np.cos(omega_8 * t), np.sin(omega_8 * t),
        np.cos(omega_6 * t), np.sin(omega_6 * t),
    ])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X.dot(beta)
    sigma2 = float(np.sum(residuals ** 2)) / max(N - 9, 1)
    noise_amp = np.sqrt(2.0 * sigma2 / max(N, 1))

    A_24 = np.sqrt(beta[1] ** 2 + beta[2] ** 2)
    A_12 = np.sqrt(beta[3] ** 2 + beta[4] ** 2)
    A_8 = np.sqrt(beta[5] ** 2 + beta[6] ** 2)
    A_6 = np.sqrt(beta[7] ** 2 + beta[8] ** 2)

    amp_ratio = A_12 / max(A_24, 1e-10)

    # Gates
    gate_24_strong = A_24 > 3.0 * noise_amp
    gate_ratio = 0.2 < amp_ratio < 0.8
    gate_8h = A_8 > 1.5 * noise_amp

    if not (gate_24_strong and gate_ratio and gate_8h):
        return {"score": 0.0, "chi2": float("inf"), "gated_in": False}

    # Chi-squared test: observed vs sawtooth prediction (A_n = A_24/n)
    expected = [A_24 / 2.0, A_24 / 3.0, A_24 / 4.0]
    observed = [A_12, A_8, A_6]
    var_A = 2.0 * sigma2 / max(N, 1)

    chi2_val = sum(
        (o - e) ** 2 / max(var_A, 1e-10)
        for o, e in zip(observed, expected)
    )

    # Asymmetric scoring: only harmonic direction
    if chi2_val < 4.0:
        score = -2.0
    elif chi2_val < 8.0:
        score = -1.0
    else:
        score = 0.0

    return {"score": score, "chi2": float(chi2_val), "gated_in": True}


def _classify_gene_v2(m0, m1, m1_free, log_bf, period_dev, f_test_24, f_test_12,
                       bootstrap_pvalue=None, t=None, y=None):
    """Classify a gene using soft F-test gating (v2 classifier).

    Replaces the binary sig_24/sig_12 hard gate in _classify_gene with
    continuous p-value weights, allowing borderline 12h genes to enter
    multi-evidence scoring instead of being immediately discarded.

    Weight formula: w = min(-log10(max(p, 1e-20)) / 1.3, 1.0)
      - p=0.05 → w ≈ 1.0  (fully weighted)
      - p=0.10 → w ≈ 0.77
      - p=0.50 → w ≈ 0.23
      - p=0.90 → w ≈ 0.035

    Hard gate only when BOTH p-values > 0.5 → non_rhythmic.
    Soft gate: 24h present + 12h truly absent (p_12 > 0.5 AND amp_ratio < 0.15)
    → circadian_only.
    Otherwise: enter multi-evidence scoring with evidence weighted by w_12.

    Categories (same as v1):
      - 'independent_ultradian': strong evidence for independent 12h oscillator
      - 'likely_independent_ultradian': moderate evidence
      - 'harmonic': 12h signal is a harmonic of 24h
      - 'circadian_only': only 24h signal, no significant 12h
      - 'non_rhythmic': no significant oscillation
      - 'ambiguous': evidence is inconclusive
    """
    bf = np.exp(log_bf)
    p_24 = f_test_24["p_value"]
    p_12 = f_test_12["p_value"]

    # Continuous weights from p-values
    def _p_to_weight(p):
        return min(-np.log10(max(p, 1e-20)) / 1.3, 1.0)

    w_24 = _p_to_weight(p_24)
    w_12 = _p_to_weight(p_12)

    # Hard gate: BOTH p-values > 0.5 (both weights < ~0.23) → non_rhythmic
    if p_24 > 0.5 and p_12 > 0.5:
        return "non_rhythmic"

    # Compute amplitude ratio for soft gate check
    m1_amps = {c["T"]: c["A"] for c in m1["components"]}
    a_24 = m1_amps.get(24.0, 1e-10)
    a_12 = m1_amps.get(12.0, 0)
    amp_ratio = a_12 / max(a_24, 1e-10)

    # Soft gate: 24h present + 12h truly absent → circadian_only
    # Requires BOTH: high p_12 (no statistical signal) AND low amp_ratio
    if p_12 > 0.5 and amp_ratio < 0.15:
        return "circadian_only"

    # Fix 1: Strengthened soft gate for strong circadian genes with harmonic
    # artifacts. When 24h is very strong AND 12h amplitude ratio is in the
    # harmonic range, classify as harmonic regardless of 12h p-value.
    if p_24 < 0.001 and amp_ratio < 0.3:
        return "harmonic"

    # 12h present but no 24h at all → independent ultradian
    if p_24 > 0.5 and p_12 < 0.05:
        return "independent_ultradian"

    # --- Multi-evidence scoring (float, weighted by w_12) ---
    evidence_score = 0.0

    # --- Fix 2: 24h dominance evidence ---
    # When 24h is extremely significant and 12h amplitude is low relative to
    # 24h, this is characteristic of strong circadian genes with harmonic
    # artifacts. Push evidence toward harmonic.
    if p_24 < 1e-6 and amp_ratio < 0.4:
        evidence_score -= 2.0
    # Weaker penalty for mid-range ratios with very strong 24h
    elif p_24 < 1e-4 and amp_ratio < 0.7:
        evidence_score -= 1.5

    # --- Fix 3: Hard floor for non-rhythmic genes ---
    # If BOTH p-values are weak (> 0.1), any positive evidence from VMD/SID
    # is likely spurious. Cap evidence so these can only reach "ambiguous".
    _both_weak = (p_24 > 0.1 and p_12 > 0.1)

    # --- Evidence 1: BIC-based Bayes Factor ---
    if bf > 3:
        evidence_score += 2.0 * w_12
    elif bf > 1:
        evidence_score += 1.0 * w_12
    elif bf < 0.05:
        evidence_score -= 2.0 * w_12
    elif bf < 0.3:
        evidence_score -= 1.0 * w_12
    elif bf < 1:
        evidence_score -= 1.0 * w_12

    # --- Evidence 2: Period deviation ---
    if period_dev["deviates_from_harmonic"]:
        evidence_score += 2.0 * w_12
        if period_dev["relative_deviation"] > 0.05:
            evidence_score += 1.0 * w_12
    else:
        evidence_score -= 1.0 * w_12

    # --- Evidence 3: Amplitude ratio ---
    if amp_ratio > 0.8:
        evidence_score += 3.0 * w_12
    elif amp_ratio > 0.65:
        evidence_score += 2.0 * w_12
    elif amp_ratio > 0.5:
        evidence_score += 1.0 * w_12
    elif amp_ratio < 0.25:
        evidence_score -= 2.0 * w_12
    elif amp_ratio < 0.5:
        evidence_score -= 1.0 * w_12

    # --- Evidence 4: Residual improvement test ---
    residual_evidence = _residual_periodicity_test(m0, m1, m1_free)
    evidence_score += residual_evidence * w_12

    # --- Evidence 5 (optional): Bootstrap LRT p-value ---
    if bootstrap_pvalue is not None:
        if bootstrap_pvalue < 0.01:
            evidence_score += 2.0 * w_12
        elif bootstrap_pvalue < 0.05:
            evidence_score += 1.0 * w_12
        elif bootstrap_pvalue > 0.5:
            evidence_score -= 2.0 * w_12
        elif bootstrap_pvalue > 0.3:
            evidence_score -= 1.0 * w_12

    # --- Evidence 6: Phase coupling test ---
    phase_coupling = _phase_coupling_score(m1, m1_free=m1_free)
    evidence_score += phase_coupling * w_12

    # --- Evidence 7: Waveform asymmetry test ---
    if t is not None and y is not None:
        asymmetry_evidence = _waveform_asymmetry_score(t, y, m0)
        evidence_score += asymmetry_evidence * w_12

    # --- Evidence 8: F-test strength bonus ---
    if p_12 < 0.01:
        evidence_score += 1.5
    elif p_12 < 0.05:
        evidence_score += 0.5

    # --- Evidence 9: VMD energy/frequency analysis ---
    if t is not None and y is not None:
        try:
            from chord.bhdt.vmd import vmd_evidence
            vmd_ev = vmd_evidence(np.asarray(t), np.asarray(y), T_base=24.0)
            evidence_score += vmd_ev["vmd_score"] * w_12
        except Exception:
            pass  # VMD is supplementary

    # --- Evidence 10: Spectral Independence Divergence ---
    if t is not None and y is not None:
        try:
            from chord.bhdt.sid import sid_evidence
            sid_ev = sid_evidence(np.asarray(t), np.asarray(y), T_base=24.0)
            evidence_score += sid_ev["sid_score"] * w_12
        except Exception:
            pass  # SID is supplementary

    # --- Evidence 11: Bispectral Harmonic Coupling Test ---
    if t is not None and y is not None:
        try:
            from chord.bhdt.bispectral import bhct_evidence
            bhct_ev = bhct_evidence(np.asarray(t), np.asarray(y), T_base=T_base)
            evidence_score += bhct_ev["score"] * w_12
        except Exception:
            pass  # BHCT is supplementary

    # --- Evidence 12: Savage-Dickey exact Bayes Factor ---
    if t is not None and y is not None:
        try:
            from chord.bhdt.savage_dickey import savage_dickey_evidence
            sd_ev = savage_dickey_evidence(np.asarray(t), np.asarray(y), T_base=T_base)
            evidence_score += sd_ev["score"] * w_12
        except Exception:
            pass  # Savage-Dickey is supplementary

    # --- Fix 3 (cont.): Apply hard floor for non-rhythmic genes ---
    # If both p-values are weak, cap evidence to prevent spurious calls.
    if _both_weak:
        evidence_score = min(evidence_score, 1.0)

    # --- Classification thresholds (Fix 4: raised for real data) ---
    if evidence_score >= 4.5:
        return "independent_ultradian"
    elif evidence_score >= 2.5:
        return "likely_independent_ultradian"
    elif evidence_score <= -2.0:
        return "harmonic"

    # Fallback: 24h present, 12h weak, negative score → circadian_only
    if w_24 > 0.5 and w_12 < 0.5 and evidence_score < 0:
        return "circadian_only"

    return "ambiguous"


# ============================================================================
# F-test for individual component significance
# ============================================================================

def component_f_test(t, y, periods, test_period_idx=1, alpha=0.05):
    """F-test for significance of a specific oscillatory component.

    Compares full model (all periods) vs reduced model (without test period).

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Expression values.
    periods : list of float
        All periods in the full model.
    test_period_idx : int
        Index of the period to test (default 1 = second period, typically 12h).
    alpha : float
        Significance level.

    Returns
    -------
    dict with F_stat, p_value, significant
    """
    from scipy.stats import f as f_dist

    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(t) != len(y):
        raise ValueError(
            f"t and y must have the same length, got {len(t)} and {len(y)}."
        )
    if not periods:
        raise ValueError("periods must be a non-empty list.")
    if not (0 <= test_period_idx < len(periods)):
        raise ValueError(
            f"test_period_idx={test_period_idx} is out of range for "
            f"periods of length {len(periods)}."
        )

    N = len(y)

    # Full model
    X_full = _cos_sin_design(t, periods)
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    rss_full = float(np.sum((y - X_full.dot(beta_full)) ** 2))
    p_full = X_full.shape[1]

    # Reduced model (without the test period)
    reduced_periods = [p for i, p in enumerate(periods) if i != test_period_idx]
    X_red = _cos_sin_design(t, reduced_periods)
    beta_red = np.linalg.lstsq(X_red, y, rcond=None)[0]
    rss_red = float(np.sum((y - X_red.dot(beta_red)) ** 2))
    p_red = X_red.shape[1]

    # F statistic
    df1 = p_full - p_red  # should be 2 (cos + sin)
    df2 = N - p_full
    if df2 <= 0 or rss_full < 0:
        # Insufficient degrees of freedom or invalid RSS — cannot test
        return {"F_stat": float('nan'), "p_value": float('nan'),
                "significant": False, "df1": df1, "df2": df2}
    if rss_full == 0:
        # Perfect fit in full model — component is significant
        return {"F_stat": np.inf, "p_value": 0.0,
                "significant": True, "df1": df1, "df2": df2}

    F_stat = ((rss_red - rss_full) / df1) / (rss_full / df2)
    if F_stat < 0:
        # Numerical issue: reduced model fits better than full model
        F_stat = 0.0
        p_value = 1.0
    else:
        p_value = 1.0 - f_dist.cdf(F_stat, df1, df2)

    return {
        "F_stat": float(F_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "df1": df1,
        "df2": df2,
    }


def permutation_f_test(t, y, periods, test_period_idx=1, alpha=0.05,
                       n_perm=999, seed=None):
    """Freedman-Lane permutation F-test for a specific oscillatory component.

    Uses the Freedman-Lane procedure: fit the reduced model, permute its
    residuals, add them back to the reduced-model fitted values, then
    recompute the F-statistic.  This gives an exact (non-parametric) null
    distribution for the F-test.

    Parameters
    ----------
    t : array
        Time points.
    y : array
        Expression values.
    periods : list of float
        All periods in the full model.
    test_period_idx : int
        Index of the period to test (default 1 = 12h).
    alpha : float
        Significance level.
    n_perm : int
        Number of permutations (default 999).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with F_stat, p_value, significant, n_perm
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(y)
    rng = np.random.RandomState(seed)

    # Full model
    X_full = _cos_sin_design(t, periods)
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    rss_full = float(np.sum((y - X_full.dot(beta_full)) ** 2))
    p_full = X_full.shape[1]

    # Reduced model (without the test period)
    reduced_periods = [p for i, p in enumerate(periods) if i != test_period_idx]
    X_red = _cos_sin_design(t, reduced_periods)
    beta_red = np.linalg.lstsq(X_red, y, rcond=None)[0]
    y_hat_red = X_red.dot(beta_red)
    resid_red = y - y_hat_red
    rss_red = float(np.sum(resid_red ** 2))
    p_red = X_red.shape[1]

    df1 = p_full - p_red
    df2 = N - p_full
    if df2 <= 0 or rss_full <= 0:
        return {"F_stat": float('nan'), "p_value": float('nan'),
                "significant": False, "n_perm": n_perm}

    # Observed F-statistic
    F_obs = ((rss_red - rss_full) / df1) / (rss_full / df2)
    if F_obs < 0:
        F_obs = 0.0

    # Freedman-Lane permutation
    n_ge = 0
    for _ in range(n_perm):
        perm_resid = rng.permutation(resid_red)
        y_perm = y_hat_red + perm_resid

        beta_full_p = np.linalg.lstsq(X_full, y_perm, rcond=None)[0]
        rss_full_p = float(np.sum((y_perm - X_full.dot(beta_full_p)) ** 2))

        beta_red_p = np.linalg.lstsq(X_red, y_perm, rcond=None)[0]
        rss_red_p = float(np.sum((y_perm - X_red.dot(beta_red_p)) ** 2))

        if rss_full_p <= 0:
            continue
        F_perm = ((rss_red_p - rss_full_p) / df1) / (rss_full_p / df2)
        if F_perm >= F_obs:
            n_ge += 1

    p_value = (n_ge + 1) / (n_perm + 1)

    return {
        "F_stat": float(F_obs),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "n_perm": n_perm,
    }


# ============================================================================
# Ensemble inference: analytic + bootstrap
# ============================================================================

def bhdt_ensemble(t, y, T_base=24.0, K_harmonics=3, n_bootstrap=499, seed=42):
    """Ensemble BHDT combining analytic BF and bootstrap LRT.

    Runs both methods and combines evidence for more robust classification.
    Faster than MCMC (~3s/gene) while being more accurate than either alone.

    Parameters
    ----------
    t, y : arrays
        Time series data.
    T_base : float
        Base circadian period.
    K_harmonics : int
        Number of harmonics.
    n_bootstrap : int
        Bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict with combined classification and evidence from both methods.
    """
    from chord.bhdt.bootstrap import parametric_bootstrap_lrt

    # Run both methods
    r_analytic = bhdt_analytic(t, y, T_base=T_base, K_harmonics=K_harmonics)
    r_bootstrap = parametric_bootstrap_lrt(t, y, T_base=T_base,
                                            K_harmonics=K_harmonics,
                                            n_bootstrap=n_bootstrap, seed=seed)

    cls_a = r_analytic["classification"]
    cls_b = r_bootstrap["classification"]

    # Agreement: both methods agree -> high confidence
    if cls_a == cls_b:
        classification = cls_a
        confidence = "high"
    elif _is_compatible(cls_a, cls_b):
        # Compatible (e.g., one says "independent", other says "likely_independent")
        classification = _resolve_compatible(cls_a, cls_b)
        confidence = "medium"
    else:
        # Disagreement: use evidence-weighted resolution
        # Extract amplitude ratio from M1 fit
        m1_comps = r_analytic.get("m1", {}).get("components", [])
        amp_ratio = 0.0
        if len(m1_comps) >= 2:
            a24 = max(m1_comps[0].get("A", 0), 1e-10)
            a12 = m1_comps[1].get("A", 0)
            amp_ratio = a12 / a24

        classification = _resolve_disagreement(
            cls_a, cls_b,
            r_analytic["bayes_factor"],
            r_bootstrap["p_value"],
            r_analytic.get("period_deviation", {}),
            amp_ratio,
        )
        confidence = "low"

    return {
        "classification": classification,
        "confidence": confidence,
        "analytic_classification": cls_a,
        "bootstrap_classification": cls_b,
        "bayes_factor": r_analytic["bayes_factor"],
        "log_bayes_factor": r_analytic["log_bayes_factor"],
        "bootstrap_p_value": r_bootstrap["p_value"],
        "bootstrap_lrt": r_bootstrap["lrt_observed"],
        "interpretation": r_analytic["interpretation"],
        "m0": r_analytic.get("m0"),
        "m1": r_analytic.get("m1"),
        "m1_free": r_analytic.get("m1_free"),
        "period_deviation": r_analytic.get("period_deviation"),
        "f_test_12h": component_f_test(
            np.asarray(t), np.asarray(y), [24.0, 12.0, 8.0], test_period_idx=1
        ),
    }


def _is_compatible(cls_a, cls_b):
    """Check if two classifications are compatible (same direction)."""
    groups = {
        "independent": {"independent_ultradian", "likely_independent_ultradian"},
        "harmonic": {"harmonic"},
        "non_rhythmic": {"non_rhythmic"},
        "circadian": {"circadian_only"},
        "ambiguous": {"ambiguous"},
    }
    for group_classes in groups.values():
        if cls_a in group_classes and cls_b in group_classes:
            return True
    # Check cross-group compatibility
    if cls_a in groups.get("independent", set()) and cls_b in groups.get("independent", set()):
        return True
    return False


def _resolve_compatible(cls_a, cls_b):
    """Resolve compatible but not identical classifications."""
    # Prefer the more specific one
    if cls_a.startswith("likely_"):
        return cls_b
    if cls_b.startswith("likely_"):
        return cls_a
    return cls_a


def _resolve_disagreement(cls_a, cls_b, bf, p_value, period_dev, amp_ratio=0.0):
    """Resolve disagreement between analytic and bootstrap.

    Uses a multi-evidence approach:
    1. Bootstrap p-value (proper null distribution)
    2. Analytic BF (fast but BIC-biased at small N)
    3. Period deviation (frequency-domain evidence)
    4. Amplitude ratio A_12/A_24 (high ratio unusual for harmonics)
    """
    if cls_a == "non_rhythmic" or cls_b == "non_rhythmic":
        return "non_rhythmic"
    # When one says circadian_only but the other says harmonic, check BF:
    # a very low BF (strong harmonic evidence) means the 12h component exists
    # as a harmonic artifact even if the F-test missed it.
    if cls_a == "circadian_only" or cls_b == "circadian_only":
        other = cls_b if cls_a == "circadian_only" else cls_a
        if other == "harmonic" and bf < 0.1:
            return "harmonic"
        return "circadian_only"

    # Both think there's a 12h signal — disagree on harmonic vs independent
    has_period_dev = (period_dev and period_dev.get("deviates_from_harmonic")
                      and period_dev.get("relative_deviation", 0) > 0.02)
    high_amp_ratio = amp_ratio > 0.7  # 12h > 70% of 24h is unusual for harmonics

    # Count converging evidence for independence
    indep_evidence = 0
    if has_period_dev:
        indep_evidence += 1
    if high_amp_ratio:
        indep_evidence += 1
    if bf > 1:
        indep_evidence += 1

    if p_value < 0.01:
        return "independent_ultradian"
    elif p_value < 0.05:
        return "likely_independent_ultradian"
    elif bf > 10:
        return "independent_ultradian"
    elif bf > 3 and has_period_dev:
        return "independent_ultradian"
    elif bf > 3:
        return "likely_independent_ultradian"
    elif indep_evidence >= 3 and p_value < 0.3:
        # All three non-bootstrap evidence sources agree: independent
        return "likely_independent_ultradian"
    elif indep_evidence >= 2 and bf > 1 and p_value < 0.25:
        # Two evidence sources + BF leans independent + bootstrap borderline
        return "likely_independent_ultradian"
    elif p_value > 0.1 and bf < 1 and not (has_period_dev and high_amp_ratio):
        # Bootstrap not significant, BF favors harmonic, no strong counter-evidence
        return "harmonic"
    elif p_value > 0.3 and not (has_period_dev and high_amp_ratio):
        return "harmonic"
    elif has_period_dev and high_amp_ratio:
        # Period deviation + high amplitude ratio = converging non-bootstrap evidence
        return "likely_independent_ultradian" if indep_evidence >= 2 else "ambiguous"
    elif has_period_dev and bf > 1:
        return "ambiguous"
    else:
        return "ambiguous"


# ============================================================================
# Population-level cross-gene phase analysis
# ============================================================================

def population_phase_analysis(t, Y_matrix, T_base=24.0):
    """Run cross-gene phase distribution test on a gene expression matrix.

    Parameters
    ----------
    t : array
        Time points.
    Y_matrix : array of shape (n_genes, n_timepoints)
        Expression matrix.
    T_base : float
        Base circadian period.

    Returns
    -------
    dict with cross-gene phase test results
    """
    from chord.bhdt.cross_gene_phase import batch_extract_phases, cross_gene_phase_test
    phases = batch_extract_phases(np.asarray(t), np.asarray(Y_matrix))
    phi_24 = phases["phi_24"]
    phi_12 = phases["phi_12"]
    amp_12 = phases["amp_12"]
    # Filter to genes with significant 12h amplitude
    mask = amp_12 > 0.1 * np.std(Y_matrix)
    if np.sum(mask) < 5:
        return {"n_genes_tested": 0, "result": "insufficient_genes"}
    result = cross_gene_phase_test(phi_24[mask], phi_12[mask])
    result["n_genes_tested"] = int(np.sum(mask))
    return result
