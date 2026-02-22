"""
Multi-channel fusion classifier for CHORD (v3).

Replaces the ad hoc scoring system in BHDT v2 with a multi-channel
fusion architecture where each module is an independent decision channel:

    Channel 1: BHDT V2 frequency analysis -> classification + evidence_score
    Channel 2: BHCT bicoherence           -> is_harmonic (p < 0.05), is_independent (p > 0.2)
    Channel 3: Savage-Dickey BF            -> log_bf (optional, slow)
                        |
                 Fusion Decision
                        |
              Final class + confidence

Key design decisions:
    - BHCT has veto power for harmonic detection (QPC is a physical mechanism)
    - BHCT promotes circadian_only to independent when no QPC + significant 12h F-test
    - Savage-Dickey is a tiebreaker, not a primary channel
    - has_significant_12h threshold is p < 0.1 (relaxed to catch borderline cases)

Python 3.6 compatible.
"""

import numpy as np
from typing import Dict, List, Optional, Any


def _get_12h_f_test_p(bhdt_result, t, y, T_base=24.0):
    """Extract or compute the 12h F-test p-value.

    The bhdt_analytic result does not directly expose the F-test p-value,
    so we compute it from the raw data using component_f_test.

    Parameters
    ----------
    bhdt_result : dict
        Result from bhdt_analytic.
    t : array
        Time points.
    y : array
        Expression values.
    T_base : float
        Base period.

    Returns
    -------
    dict
        F-test result dict with keys F_stat, p_value, significant.
    """
    from chord.bhdt.inference import component_f_test
    ultradian_periods = [T_base, T_base / 2.0, T_base / 3.0]
    return component_f_test(
        np.asarray(t), np.asarray(y), ultradian_periods, test_period_idx=1
    )


def classify_gene_v3(t, y, T_base=24.0, use_savage_dickey=False):
    """Multi-channel fusion classifier (v3).

    Combines three independent decision channels:
      1. BHDT V2 frequency analysis
      2. BHCT bispectral coupling test
      3. Savage-Dickey Bayes Factor (optional)

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base circadian period (default 24.0).
    use_savage_dickey : bool
        Whether to run Savage-Dickey as a tiebreaker (default False).

    Returns
    -------
    dict
        classification : str
            Final class: 'harmonic', 'independent_ultradian',
            'likely_independent_ultradian', 'circadian_only',
            'non_rhythmic', 'ambiguous'.
        confidence : str
            'high', 'medium', or 'low'.
        reason : str
            Explanation of the fusion decision.
        v2_classification : str
            Original V2 result.
        bhct_p_value : float or None
            BHCT p-value (None if BHCT failed).
        bhct_bicoherence : float or None
            Bicoherence value (None if BHCT failed).
        sd_log_bf : float or None
            Savage-Dickey log BF (None if not used).
        channels_agree : bool
            Whether all channels point in the same direction.
        Plus all original BHDT result fields.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # ---- Step 1: Run BHDT V2 analysis ----
    from chord.bhdt.inference import bhdt_analytic
    bhdt_result = bhdt_analytic(t, y, T_base=T_base, classifier_version="v2")
    v2_class = bhdt_result["classification"]

    # ---- Step 2: Run BHCT standalone ----
    bhct_p = None
    bhct_bic = None
    bhct_ran = False
    try:
        from chord.bhdt.bispectral import bispectral_coupling_test
        bhct_result = bispectral_coupling_test(t, y, T_base=T_base,
                                                n_surrogates=199, seed=42)
        bhct_p = bhct_result["p_value"]
        bhct_bic = bhct_result["bicoherence"]
        bhct_ran = True
    except Exception:
        pass  # BHCT failed, fall back to V2

    # ---- Step 3: (Optional) Run Savage-Dickey ----
    sd_log_bf = None
    sd_available = False
    if use_savage_dickey:
        try:
            from chord.bhdt.savage_dickey import savage_dickey_bf
            sd_result = savage_dickey_bf(t, y, T_base=T_base,
                                         n_samples=2000, n_warmup=1000)
            sd_log_bf = sd_result["log_bf"]
            sd_available = True
        except Exception:
            pass

    # ---- Step 4: Get 12h F-test significance ----
    f_test_12 = _get_12h_f_test_p(bhdt_result, t, y, T_base)
    p_12 = f_test_12["p_value"]
    # Relaxed threshold: p < 0.1 to catch borderline cases
    has_significant_12h = p_12 < 0.1

    # ---- Step 5: Fusion decision logic ----
    if not bhct_ran:
        # BHCT failed — fall back to V2
        final_class = v2_class
        confidence = "low"
        reason = "bhct_failed_v2_default"
    elif bhct_p < 0.01:
        # Priority 1: Strong QPC detected -> this IS a harmonic
        final_class = "harmonic"
        confidence = "high"
        reason = "bispectral_qpc_detected"
    elif bhct_p < 0.05:
        # Moderate QPC -> lean harmonic, but check other channels
        if v2_class in ("harmonic", "circadian_only"):
            final_class = v2_class
            confidence = "high"
            reason = "bhct_confirms_v2_harmonic"
        else:
            # BHCT says harmonic but V2 says independent -- conflict
            if sd_available and sd_log_bf is not None and sd_log_bf > 0:
                final_class = "harmonic"
                confidence = "medium"
                reason = "bhct_and_sd_agree_harmonic"
            elif sd_available and sd_log_bf is not None and sd_log_bf < -1:
                final_class = v2_class  # SD supports independent
                confidence = "low"
                reason = "bhct_harmonic_but_sd_independent"
            else:
                final_class = "harmonic"
                confidence = "medium"
                reason = "bhct_moderate_qpc"
    elif bhct_p > 0.5:
        # Priority 2: No QPC at all -> strong independent evidence
        if v2_class in ("independent_ultradian", "likely_independent_ultradian"):
            final_class = v2_class
            confidence = "high"
            reason = "bhct_confirms_v2_independent"
        elif v2_class == "circadian_only":
            # BHCT says no QPC but V2 says circadian_only
            # Check if there is actually a 12h signal
            if has_significant_12h:
                final_class = "independent_ultradian"
                confidence = "medium"
                reason = "bhct_no_qpc_promotes_to_independent"
            else:
                final_class = "circadian_only"
                confidence = "medium"
                reason = "no_12h_signal"
        else:
            final_class = v2_class
            confidence = "medium"
            reason = "bhct_no_qpc_v2_default"
    else:
        # Priority 3: BHCT inconclusive -- fall back to V2
        final_class = v2_class
        confidence = "low"
        reason = "bhct_inconclusive_v2_default"

    # ---- Determine channel agreement ----
    channels_agree = _check_channel_agreement(
        v2_class, bhct_p, sd_log_bf, bhct_ran, sd_available
    )

    # ---- Build result dict ----
    result = dict(bhdt_result)  # copy all original BHDT fields
    result.update({
        "classification": final_class,
        "confidence": confidence,
        "reason": reason,
        "v2_classification": v2_class,
        "bhct_p_value": bhct_p,
        "bhct_bicoherence": bhct_bic,
        "sd_log_bf": sd_log_bf,
        "channels_agree": channels_agree,
        "f_test_12h_p": p_12,
        "has_significant_12h": has_significant_12h,
    })
    return result


def _check_channel_agreement(v2_class, bhct_p, sd_log_bf, bhct_ran, sd_available):
    """Check whether all available channels agree on direction.

    Parameters
    ----------
    v2_class : str
        V2 classification.
    bhct_p : float or None
        BHCT p-value.
    sd_log_bf : float or None
        Savage-Dickey log BF.
    bhct_ran : bool
        Whether BHCT ran successfully.
    sd_available : bool
        Whether Savage-Dickey result is available.

    Returns
    -------
    bool
        True if all available channels agree.
    """
    v2_harmonic = v2_class in ("harmonic",)
    v2_independent = v2_class in ("independent_ultradian",
                                   "likely_independent_ultradian")

    if not bhct_ran:
        return True  # only one channel, trivially agrees

    bhct_harmonic = bhct_p is not None and bhct_p < 0.05
    bhct_independent = bhct_p is not None and bhct_p > 0.2

    if sd_available and sd_log_bf is not None:
        sd_harmonic = sd_log_bf > 0
        sd_independent = sd_log_bf < -1
        # All three channels
        if v2_harmonic and bhct_harmonic and sd_harmonic:
            return True
        if v2_independent and bhct_independent and sd_independent:
            return True
        return False
    else:
        # Two channels
        if v2_harmonic and bhct_harmonic:
            return True
        if v2_independent and bhct_independent:
            return True
        # V2 non_rhythmic/circadian_only + BHCT inconclusive
        if v2_class in ("non_rhythmic", "circadian_only") and not bhct_harmonic:
            return True
        return False


def classify_gene_v3_fast(t, y, T_base=24.0):
    """Multi-channel fusion classifier without Savage-Dickey (fast mode).

    Same as classify_gene_v3 but never runs Savage-Dickey.
    This is the default for large-scale analysis.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base circadian period (default 24.0).

    Returns
    -------
    dict
        Same as classify_gene_v3.
    """
    return classify_gene_v3(t, y, T_base=T_base, use_savage_dickey=False)


def batch_classify_v3(t, Y_matrix, gene_names=None, T_base=24.0,
                       use_savage_dickey=False, verbose=True):
    """Run v3 fusion classifier on all genes in a matrix.

    Parameters
    ----------
    t : array-like
        Time points in hours (shared across genes).
    Y_matrix : array-like
        Expression matrix of shape (n_genes, n_timepoints).
    gene_names : list of str, optional
        Gene names. If None, uses 'gene_0', 'gene_1', etc.
    T_base : float
        Base circadian period (default 24.0).
    use_savage_dickey : bool
        Whether to run Savage-Dickey (default False).
    verbose : bool
        Print progress (default True).

    Returns
    -------
    list of dict
        One result dict per gene, each with a 'gene_name' key added.
    """
    Y = np.asarray(Y_matrix, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    n_genes, n_tp = Y.shape

    if len(t) != n_tp:
        raise ValueError(
            "t length ({}) != Y_matrix columns ({})".format(len(t), n_tp)
        )

    if gene_names is None:
        gene_names = ["gene_{}".format(i) for i in range(n_genes)]
    if len(gene_names) != n_genes:
        raise ValueError(
            "gene_names length ({}) != Y_matrix rows ({})".format(
                len(gene_names), n_genes
            )
        )

    results = []
    for i in range(n_genes):
        if verbose:
            print("[{}/{}] {}".format(i + 1, n_genes, gene_names[i]))
        try:
            r = classify_gene_v3(t, Y[i], T_base=T_base,
                                  use_savage_dickey=use_savage_dickey)
        except Exception as e:
            r = {
                "classification": "error",
                "confidence": "none",
                "reason": str(e),
                "v2_classification": "error",
                "bhct_p_value": None,
                "bhct_bicoherence": None,
                "sd_log_bf": None,
                "channels_agree": False,
            }
        r["gene_name"] = gene_names[i]
        results.append(r)

    return results
