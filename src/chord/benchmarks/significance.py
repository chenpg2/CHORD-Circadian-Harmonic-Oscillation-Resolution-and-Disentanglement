"""
Statistical significance tests for CHORD benchmark comparisons.

Provides McNemar, DeLong, bootstrap CI, and permutation tests for
comparing CHORD against classical circadian rhythm detection methods.
Designed for publication-quality reporting (Nature Aging).

All implementations use numpy/scipy/pandas only (no sklearn).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test for comparing two classifiers on paired data.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    pred_a : array-like
        Predictions from method A.
    pred_b : array-like
        Predictions from method B.

    Returns
    -------
    dict
        chi2 : float – test statistic (NaN when exact test is used).
        p_value : float – two-sided p-value.
        better : str – 'a', 'b', or 'neither'.
        n_discordant : int – b + c (off-diagonal counts).
        exact : bool – True if exact binomial test was used.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b: A correct, B wrong  |  c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    n_discordant = b + c

    if n_discordant < 25:
        # Exact binomial test (two-sided)
        p_value = float(sp_stats.binom_test(b, n_discordant, 0.5))
        chi2 = float("nan")
        exact = True
    else:
        # Chi-squared approximation with continuity correction
        chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1.0 - sp_stats.chi2.cdf(chi2, df=1))
        exact = False

    if b > c:
        better = "a"
    elif c > b:
        better = "b"
    else:
        better = "neither"

    return {
        "chi2": chi2,
        "p_value": p_value,
        "better": better,
        "n_discordant": n_discordant,
        "exact": exact,
    }


# ---------------------------------------------------------------------------
# DeLong test for comparing two correlated AUCs
# ---------------------------------------------------------------------------

def _placement_values(y_true_binary, scores):
    """Compute placement values for the DeLong test.

    For each positive sample, V10 is the fraction of negatives with
    strictly lower score plus half the fraction with equal score.
    Analogously for V01.

    Returns
    -------
    V10 : ndarray of shape (n_pos,)
    V01 : ndarray of shape (n_neg,)
    """
    y = np.asarray(y_true_binary, dtype=int)
    s = np.asarray(scores, dtype=float)

    pos_scores = s[y == 1]
    neg_scores = s[y == 0]

    m = len(pos_scores)
    n = len(neg_scores)

    # V10[i] = (1/n) * sum_j [ I(X_j < Y_i) + 0.5 * I(X_j == Y_i) ]
    V10 = np.empty(m)
    for i in range(m):
        V10[i] = (np.sum(neg_scores < pos_scores[i])
                  + 0.5 * np.sum(neg_scores == pos_scores[i])) / n

    # V01[j] = (1/m) * sum_i [ I(Y_i < X_j) + 0.5 * I(Y_i == X_j) ]
    V01 = np.empty(n)
    for j in range(n):
        V01[j] = (np.sum(pos_scores < neg_scores[j])
                  + 0.5 * np.sum(pos_scores == neg_scores[j])) / m

    return V10, V01


def delong_test(y_true_binary, scores_a, scores_b):
    """DeLong test for comparing two correlated ROC AUCs.

    Implements the method of DeLong, DeLong & Clarke-Pearson (1988)
    for paired comparison of AUCs estimated on the same sample.

    Parameters
    ----------
    y_true_binary : array-like of {0, 1}
        True binary labels.
    scores_a : array-like of float
        Predicted scores from method A.
    scores_b : array-like of float
        Predicted scores from method B.

    Returns
    -------
    dict
        z_stat : float – z-statistic.
        p_value : float – two-sided p-value.
        auc_a : float – AUC of method A.
        auc_b : float – AUC of method B.
        auc_diff : float – AUC_A - AUC_B.
    """
    y = np.asarray(y_true_binary, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)

    m = int(np.sum(y == 1))  # positives
    n = int(np.sum(y == 0))  # negatives

    # Placement values for each method
    V10_a, V01_a = _placement_values(y, sa)
    V10_b, V01_b = _placement_values(y, sb)

    # AUC = mean of V10
    auc_a = float(np.mean(V10_a))
    auc_b = float(np.mean(V10_b))

    # Covariance matrix of (AUC_a, AUC_b)
    # S10 is the m x 2 matrix of placement values for positives
    S10 = np.column_stack([V10_a, V10_b])  # (m, 2)
    S01 = np.column_stack([V01_a, V01_b])  # (n, 2)

    # Covariance among positives and negatives
    cov10 = np.cov(S10, rowvar=False, ddof=1) if m > 1 else np.zeros((2, 2))
    cov01 = np.cov(S01, rowvar=False, ddof=1) if n > 1 else np.zeros((2, 2))

    # Combined variance of AUC difference
    # Var(AUC_a - AUC_b) = L^T (S10/m + S01/n) L  where L = [1, -1]
    S = cov10 / m + cov01 / n
    L = np.array([1.0, -1.0])
    var_diff = float(L @ S @ L)

    auc_diff = auc_a - auc_b

    if var_diff <= 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = float(auc_diff / np.sqrt(var_diff))
        p_value = float(2.0 * sp_stats.norm.sf(abs(z_stat)))

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": auc_diff,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=10000, ci=0.95,
                 seed=42):
    """Bootstrap confidence interval for any metric function.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95 %).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (point_estimate, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.RandomState(seed)
    point_estimate = float(metric_fn(y_true, y_pred))

    boot_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

    return (point_estimate, ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(y_true, pred_a, pred_b, metric_fn,
                     n_permutations=10000, seed=42):
    """Two-sided permutation test for comparing two methods.

    Under the null hypothesis the two sets of predictions are
    exchangeable.  At each permutation, for every sample we randomly
    swap pred_a[i] and pred_b[i], then recompute the metric difference.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    pred_a, pred_b : array-like
        Predictions from method A and B.
    metric_fn : callable
        ``metric_fn(y_true, y_pred) -> float``.
    n_permutations : int
        Number of random permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        observed_diff : float – metric(A) - metric(B).
        p_value : float – two-sided permutation p-value.
        null_distribution : ndarray of shape (n_permutations,).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)

    observed_a = metric_fn(y_true, pred_a)
    observed_b = metric_fn(y_true, pred_b)
    observed_diff = float(observed_a - observed_b)

    rng = np.random.RandomState(seed)
    null_dist = np.empty(n_permutations)

    for i in range(n_permutations):
        swap = rng.randint(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        null_dist[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    # Two-sided p-value (include observed in count for conservative estimate)
    p_value = float((np.sum(np.abs(null_dist) >= abs(observed_diff)) + 1)
                    / (n_permutations + 1))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "null_distribution": null_dist,
    }


# ---------------------------------------------------------------------------
# Compare all methods
# ---------------------------------------------------------------------------

def _accuracy(y_true, y_pred):
    """Simple accuracy helper."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def compare_all_methods(results_df, reference_method="chord_ensemble",
                        n_bootstrap=10000, ci=0.95, seed=42,
                        bonferroni=True):
    """Compare a reference method against all others.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain a ``'y_true'`` column and one column per method
        holding that method's predictions.
    reference_method : str
        Column name of the reference method.
    n_bootstrap : int
        Bootstrap resamples for CI.
    ci : float
        Confidence level.
    seed : int
        Random seed.
    bonferroni : bool
        If True, apply Bonferroni correction to McNemar p-values.

    Returns
    -------
    pd.DataFrame
        Columns: method, accuracy, accuracy_ci_lower, accuracy_ci_upper,
        mcnemar_chi2, mcnemar_p, mcnemar_p_corrected, better_than_reference.
    """
    y_true = results_df["y_true"].values
    methods = [c for c in results_df.columns if c != "y_true"]

    if reference_method not in methods:
        raise ValueError(
            f"Reference method '{reference_method}' not found in columns."
        )

    ref_pred = results_df[reference_method].values
    other_methods = [m for m in methods if m != reference_method]
    n_comparisons = len(other_methods)

    rows = []

    # Reference method row
    pt, lo, hi = bootstrap_ci(y_true, ref_pred, _accuracy,
                              n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    rows.append({
        "method": reference_method,
        "accuracy": pt,
        "accuracy_ci_lower": lo,
        "accuracy_ci_upper": hi,
        "mcnemar_chi2": float("nan"),
        "mcnemar_p": float("nan"),
        "mcnemar_p_corrected": float("nan"),
        "better_than_reference": "reference",
    })

    for m in other_methods:
        m_pred = results_df[m].values
        pt_m, lo_m, hi_m = bootstrap_ci(y_true, m_pred, _accuracy,
                                        n_bootstrap=n_bootstrap, ci=ci,
                                        seed=seed)
        mc = mcnemar_test(y_true, ref_pred, m_pred)

        p_corr = min(mc["p_value"] * n_comparisons, 1.0) if bonferroni else mc["p_value"]

        better = "no"
        if p_corr < 0.05:
            better = "yes" if mc["better"] == "a" else "no"

        rows.append({
            "method": m,
            "accuracy": pt_m,
            "accuracy_ci_lower": lo_m,
            "accuracy_ci_upper": hi_m,
            "mcnemar_chi2": mc["chi2"],
            "mcnemar_p": mc["p_value"],
            "mcnemar_p_corrected": p_corr,
            "better_than_reference": better,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Publication-ready formatting
# ---------------------------------------------------------------------------

def _p_value_stars(p):
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_significance_table(comparison_df, fmt="markdown"):
    """Format comparison results as a publication-ready table.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`compare_all_methods`.
    fmt : str
        ``'markdown'`` or ``'latex'``.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []

    if fmt == "latex":
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(
            r"Method & Accuracy (95\% CI) & McNemar $p$ & Sig. & Better \\"
        )
        lines.append(r"\midrule")

        for _, row in comparison_df.iterrows():
            acc_str = f"{row['accuracy']:.3f} [{row['accuracy_ci_lower']:.3f}, {row['accuracy_ci_upper']:.3f}]"
            if np.isnan(row["mcnemar_p"]):
                p_str = "---"
            else:
                p_str = f"{row['mcnemar_p_corrected']:.2e}"
            stars = _p_value_stars(row["mcnemar_p_corrected"])
            better = row["better_than_reference"]
            lines.append(
                f"  {row['method']} & {acc_str} & {p_str} & {stars} & {better} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    else:  # markdown
        lines.append(
            "| Method | Accuracy (95% CI) | McNemar p | Sig. | Better than ref. |"
        )
        lines.append(
            "|--------|-------------------|-----------|------|------------------|"
        )

        for _, row in comparison_df.iterrows():
            acc_str = (
                f"{row['accuracy']:.3f} "
                f"[{row['accuracy_ci_lower']:.3f}, "
                f"{row['accuracy_ci_upper']:.3f}]"
            )
            if np.isnan(row["mcnemar_p"]):
                p_str = "---"
            else:
                p_str = f"{row['mcnemar_p_corrected']:.2e}"
            stars = _p_value_stars(row["mcnemar_p_corrected"])
            better = row["better_than_reference"]
            lines.append(
                f"| {row['method']} | {acc_str} | {p_str} | {stars} | {better} |"
            )

    return "\n".join(lines)
