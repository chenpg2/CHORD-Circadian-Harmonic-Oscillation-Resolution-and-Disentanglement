"""
PINOD damping analysis: detect and quantify oscillation decay.

Compares damping rates (gamma) across conditions (e.g., young vs old)
to identify oscillators that are "dying" during aging.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats


def compare_damping(
    gammas_group1: np.ndarray,
    gammas_group2: np.ndarray,
    gene_names: Optional[List[str]] = None,
    group_names: Tuple[str, str] = ("young", "old"),
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare damping rates between two groups for each oscillator.

    Parameters
    ----------
    gammas_group1 : (n_genes, K) damping rates for group 1
    gammas_group2 : (n_genes, K) damping rates for group 2
    gene_names : list of str
    group_names : tuple of str
    alpha : significance level

    Returns
    -------
    dict with per-oscillator comparison statistics
    """
    n_genes, K = gammas_group1.shape
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    results = []
    for k in range(K):
        g1 = gammas_group1[:, k]
        g2 = gammas_group2[:, k]

        # Wilcoxon rank-sum test (non-parametric)
        stat_w, p_w = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        # Effect size: rank-biserial correlation
        n1, n2 = len(g1), len(g2)
        r_rb = 1 - 2 * stat_w / (n1 * n2)

        # Log fold change of median damping
        median_g1 = np.median(g1)
        median_g2 = np.median(g2)
        log2fc = np.log2(max(median_g2, 1e-10) / max(median_g1, 1e-10))

        results.append({
            "oscillator_index": k,
            "median_group1": float(median_g1),
            "median_group2": float(median_g2),
            "log2fc_damping": float(log2fc),
            "wilcoxon_stat": float(stat_w),
            "p_value": float(p_w),
            "significant": p_w < alpha,
            "rank_biserial_r": float(r_rb),
            "interpretation": _interpret_damping_change(log2fc, p_w, alpha, group_names),
        })

    return {
        "comparisons": results,
        "group_names": group_names,
        "n_genes": n_genes,
        "n_oscillators": K,
    }


def _interpret_damping_change(
    log2fc: float, p_value: float, alpha: float, group_names: Tuple[str, str]
) -> str:
    """Interpret damping rate change between groups."""
    if p_value >= alpha:
        return "no_significant_change"
    if log2fc > 0.5:
        return f"increased_damping_in_{group_names[1]}"
    elif log2fc < -0.5:
        return f"decreased_damping_in_{group_names[1]}"
    else:
        return "marginal_change"


def detect_dying_oscillators(
    analysis_results_g1: List[Dict],
    analysis_results_g2: List[Dict],
    oscillator_index: int = 1,
    gamma_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Identify genes where a specific oscillator is "dying" (gamma increases).

    A "dying" oscillator: active in group1 (low gamma) but suppressed in
    group2 (high gamma), suggesting loss of independent oscillation.

    Parameters
    ----------
    analysis_results_g1 : list of extract_oscillator_params results (group 1)
    analysis_results_g2 : list of extract_oscillator_params results (group 2)
    oscillator_index : int — which oscillator to check (1 = 12h typically)
    gamma_threshold : float — gamma above this = "dying"

    Returns
    -------
    dict with dying/stable/gained gene lists
    """
    dying = []
    stable = []
    gained = []

    for i, (r1, r2) in enumerate(zip(analysis_results_g1, analysis_results_g2)):
        osc1 = r1["oscillators"][oscillator_index]
        osc2 = r2["oscillators"][oscillator_index]

        active_g1 = osc1["active"] and osc1["gamma"] < gamma_threshold
        active_g2 = osc2["active"] and osc2["gamma"] < gamma_threshold

        gene_info = {
            "index": i,
            "gamma_g1": osc1["gamma"],
            "gamma_g2": osc2["gamma"],
            "amp_g1": osc1["amplitude_rms"],
            "amp_g2": osc2["amplitude_rms"],
        }

        if active_g1 and not active_g2:
            dying.append(gene_info)
        elif not active_g1 and active_g2:
            gained.append(gene_info)
        else:
            stable.append(gene_info)

    return {
        "dying": dying,
        "stable": stable,
        "gained": gained,
        "n_dying": len(dying),
        "n_stable": len(stable),
        "n_gained": len(gained),
        "oscillator_index": oscillator_index,
    }


def half_life(gamma: float) -> float:
    """Convert damping rate to half-life in hours.

    t_half = ln(2) / gamma
    """
    if gamma <= 0:
        return float("inf")
    return np.log(2) / gamma


def damping_summary(gammas: np.ndarray, periods: np.ndarray) -> Dict[str, Any]:
    """Summarise damping statistics for a set of genes.

    Parameters
    ----------
    gammas : (n_genes, K)
    periods : (n_genes, K) or (K,)

    Returns
    -------
    dict with summary statistics per oscillator
    """
    if periods.ndim == 1:
        periods = np.tile(periods, (gammas.shape[0], 1))

    K = gammas.shape[1]
    summary = []
    for k in range(K):
        g = gammas[:, k]
        T = periods[:, k]
        hl = np.array([half_life(gi) for gi in g])

        # Quality factor Q = omega / (2 * gamma)
        omega = 2 * np.pi / T
        Q = omega / (2 * g + 1e-10)

        summary.append({
            "oscillator_index": k,
            "median_period": float(np.median(T)),
            "median_gamma": float(np.median(g)),
            "mean_gamma": float(np.mean(g)),
            "std_gamma": float(np.std(g)),
            "median_half_life": float(np.median(hl[np.isfinite(hl)])) if np.any(np.isfinite(hl)) else float("inf"),
            "median_Q_factor": float(np.median(Q)),
            "frac_underdamped": float(np.mean(Q > 0.5)),
        })

    return {"oscillator_summary": summary, "n_genes": gammas.shape[0]}
