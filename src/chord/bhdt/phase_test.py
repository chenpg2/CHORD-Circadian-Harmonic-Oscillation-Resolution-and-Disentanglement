"""
Phase and Amplitude Independence Tests for BHDT.

Two complementary approaches for distinguishing harmonics from
independent oscillators when frequency-based methods fail:

1. Single-gene: Amplitude ratio test
   - Harmonics of common waveforms have predictable A_k/A_1 ratios
   - Independent oscillators have arbitrary amplitude ratios
   - Uses bootstrap to test if observed ratio is consistent with
     common harmonic waveform families

2. Cross-gene: Circular phase correlation test
   - If 12h oscillations are harmonics, phi_12 correlates with phi_24
     across genes (same waveform shape -> same phase relationship)
   - If independent, phi_12 and phi_24 are uncorrelated on the circle
   - Uses Fisher & Lee (1983) circular correlation + permutation test
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from chord.bhdt.models import fit_harmonic_model, fit_independent_model


# ============================================================================
# Single-gene: Amplitude ratio consistency test
# ============================================================================

def amplitude_ratio_test(t, y, T_base=24.0, K_harmonics=3,
                          n_bootstrap=999, seed=42):
    """Test if the 12h/24h amplitude ratio is consistent with harmonic waveforms.

    Common harmonic waveforms have specific A_2/A_1 ratios:
    - Sawtooth: A_k/A_1 = 1/k (A_2/A_1 = 0.5)
    - Square wave: A_k/A_1 = 1/k for odd k, 0 for even k
    - Peaked cosine: A_k/A_1 decreases monotonically

    Independent oscillators can have any A_2/A_1 ratio, including
    values > 1 (12h stronger than 24h).

    The test: under H0 (harmonic), the observed A_2/A_1 should be
    consistent with the noise-perturbed harmonic model. Under H1,
    A_2/A_1 may be unusually large.

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
        Bootstrap replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict with amplitude_ratio, p_value, evidence
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Fit harmonic model
    m0 = fit_harmonic_model(t_arr, y_arr, T_base=T_base, K=K_harmonics)

    # Observed amplitude ratio
    A_24 = m0["components"][0]["A"]
    A_12 = m0["components"][1]["A"]
    if A_24 < 1e-10:
        ratio_obs = float("inf")
    else:
        ratio_obs = A_12 / A_24

    # Bootstrap under M0: what amplitude ratios are expected?
    rng = np.random.RandomState(seed)
    from chord.bhdt.models import _cos_sin_design
    X_m0 = _cos_sin_design(t_arr, m0["periods"])
    y_hat = X_m0.dot(m0["beta"])
    sigma = np.sqrt(m0["sigma2"])

    ratios_null = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        y_sim = y_hat + rng.normal(0, sigma, len(t_arr))
        m0_sim = fit_harmonic_model(t_arr, y_sim, T_base=T_base, K=K_harmonics)
        A_24_sim = m0_sim["components"][0]["A"]
        A_12_sim = m0_sim["components"][1]["A"]
        if A_24_sim < 1e-10:
            ratios_null[b] = 0.0
        else:
            ratios_null[b] = A_12_sim / A_24_sim

    # Two-sided p-value
    p_value = (np.sum(np.abs(ratios_null - np.median(ratios_null)) >=
                       np.abs(ratio_obs - np.median(ratios_null))) + 1) / (n_bootstrap + 1)

    if ratio_obs > 0.8 and p_value < 0.05:
        evidence = "independent_strong_12h"
    elif p_value < 0.05:
        evidence = "amplitude_anomalous"
    else:
        evidence = "amplitude_consistent_harmonic"

    return {
        "amplitude_ratio": float(ratio_obs),
        "A_24": float(A_24),
        "A_12": float(A_12),
        "p_value": float(p_value),
        "evidence": evidence,
        "null_median_ratio": float(np.median(ratios_null)),
        "null_std_ratio": float(np.std(ratios_null)),
    }


# ============================================================================
# Cross-gene: Circular phase correlation test
# ============================================================================

def cross_gene_phase_test(phases_24, phases_12, n_permutations=9999, seed=42):
    """Test phase independence across a population of genes.

    If 12h oscillations are harmonics of 24h, then phi_12 should be
    predictable from phi_24 (correlated). If independent, they should
    be uncorrelated on the circle.

    Uses circular correlation coefficient (Fisher & Lee, 1983).

    Parameters
    ----------
    phases_24 : array of shape (n_genes,)
        24h phases in radians.
    phases_12 : array of shape (n_genes,)
        12h phases in radians.
    n_permutations : int
        Permutation test replicates.
    seed : int
        Random seed.

    Returns
    -------
    dict with rho_circular, p_value, evidence
    """
    phases_24 = np.asarray(phases_24)
    phases_12 = np.asarray(phases_12)
    n = len(phases_24)
    assert len(phases_12) == n, "phases_24 and phases_12 must have same length"

    # Circular correlation (Fisher & Lee 1983)
    rho_obs = _circular_correlation(phases_24, phases_12)

    # Permutation test — streaming: only keep a running count of how many
    # permutation statistics exceed the observed, instead of storing all.
    rng = np.random.RandomState(seed)
    abs_rho_obs = np.abs(rho_obs)
    exceed_count = 0
    for i in range(n_permutations):
        perm = rng.permutation(n)
        rho_perm = _circular_correlation(phases_24, phases_12[perm])
        if np.abs(rho_perm) >= abs_rho_obs:
            exceed_count += 1

    p_value = (exceed_count + 1) / (n_permutations + 1)

    if p_value < 0.01:
        evidence = "phases_correlated_harmonic"
    elif p_value < 0.05:
        evidence = "phases_weakly_correlated"
    else:
        evidence = "phases_independent"

    return {
        "rho_circular": float(rho_obs),
        "p_value": float(p_value),
        "evidence": evidence,
        "n_genes": n,
    }



def _circular_correlation(alpha, beta):
    """Circular correlation coefficient (Fisher & Lee 1983).

    Chunked implementation to avoid O(n^2) memory for large n.
    Processes pairwise sin differences in row-chunks, accumulating
    the sums needed for the numerator and denominator.
    """
    n = len(alpha)
    # For small n, the full broadcast is fine and faster
    if n <= 5000:
        sin_a = np.sin(alpha[:, None] - alpha[None, :])
        sin_b = np.sin(beta[:, None] - beta[None, :])
        numerator = np.sum(sin_a * sin_b)
        denominator = np.sqrt(np.sum(sin_a**2) * np.sum(sin_b**2))
        if denominator < 1e-10:
            return 0.0
        return numerator / denominator

    # Chunked computation for large n
    chunk_size = max(1, min(2000, 5000 * 5000 // n))
    num_sum = 0.0
    ss_a = 0.0
    ss_b = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Chunk of rows [start:end] against all columns
        sin_a_chunk = np.sin(alpha[start:end, None] - alpha[None, :])
        sin_b_chunk = np.sin(beta[start:end, None] - beta[None, :])
        num_sum += np.sum(sin_a_chunk * sin_b_chunk)
        ss_a += np.sum(sin_a_chunk**2)
        ss_b += np.sum(sin_b_chunk**2)
    denominator = np.sqrt(ss_a * ss_b)
    if denominator < 1e-10:
        return 0.0
    return num_sum / denominator
