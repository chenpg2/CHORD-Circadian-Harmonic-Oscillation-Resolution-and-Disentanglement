"""
Chi-square Population Independence Test.

Tests whether 12h rhythm detection is statistically independent of 24h
rhythm detection across a population of genes, following the approach
described in Zhu et al. (2024).

Logic
-----
If 12h rhythms are harmonics of 24h oscillations, then detecting a 12h
rhythm should be statistically *dependent* on detecting a 24h rhythm
(12h genes would be a subset of 24h genes).  If 12h rhythms arise from
independent oscillators, the two detections should be statistically
independent.

We build a 2x2 contingency table:

              24h+    24h-
    12h+       a       b
    12h-       c       d

and test H0: 12h detection is independent of 24h detection using either
a chi-square test (when expected cell counts >= 5) or Fisher's exact
test (when any expected count < 5).

The observed/expected ratio for the "both+" cell (a) quantifies the
degree of co-detection:
    O/E >> 1  ->  harmonic (co-detection far exceeds independence)
    O/E ~ 1   ->  independent
"""

import numpy as np
from scipy import stats


def chi_square_independence_test(is_24h, is_12h):
    """Test independence of 12h and 24h rhythm detection across genes.

    Parameters
    ----------
    is_24h : array-like of bool
        Whether each gene has significant 24h rhythm.
    is_12h : array-like of bool
        Whether each gene has significant 12h rhythm.

    Returns
    -------
    dict with keys:
        p_value : float
        chi2_statistic : float
        observed_expected_ratio : float (O/E for the both+ cell)
        contingency_table : ndarray (2, 2)
        n_both : int
        n_24h_only : int
        n_12h_only : int
        n_neither : int
        interpretation : str ('harmonic_dependent', 'independent', 'inconclusive')
    """
    is_24h = np.asarray(is_24h, dtype=bool)
    is_12h = np.asarray(is_12h, dtype=bool)

    if len(is_24h) != len(is_12h):
        raise ValueError(
            "is_24h and is_12h must have the same length, "
            "got {} and {}".format(len(is_24h), len(is_12h))
        )

    n = len(is_24h)

    # Build contingency table
    #              24h+    24h-
    #   12h+        a       b
    #   12h-        c       d
    a = int(np.sum(is_12h & is_24h))       # both+
    b = int(np.sum(is_12h & ~is_24h))      # 12h+ only
    c = int(np.sum(~is_12h & is_24h))      # 24h+ only
    d = int(np.sum(~is_12h & ~is_24h))     # neither

    table = np.array([[a, b], [c, d]])

    # Expected count for the "both+" cell under independence
    row_12h_pos = a + b
    col_24h_pos = a + c
    if n > 0:
        expected_a = float(row_12h_pos * col_24h_pos) / n
    else:
        expected_a = 0.0

    # Observed / Expected ratio
    if expected_a > 0:
        oe_ratio = float(a) / expected_a
    else:
        # All genes in one category — O/E is undefined; use 1.0
        oe_ratio = 1.0

    # Decide test method: Fisher exact when any expected cell count < 5
    # Compute all expected counts
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    use_fisher = False
    if n > 0:
        for r in range(2):
            for cc in range(2):
                exp_cell = float(row_sums[r] * col_sums[cc]) / n
                if exp_cell < 5.0:
                    use_fisher = True
                    break
            if use_fisher:
                break
    else:
        use_fisher = True

    # Also use Fisher if any marginal is zero (chi2 would be degenerate)
    if 0 in row_sums or 0 in col_sums:
        use_fisher = True

    if use_fisher:
        # Fisher exact test
        odds_ratio, p_value = stats.fisher_exact(table)
        # Approximate chi2 statistic from the table for reporting
        # (may be nan for degenerate tables)
        if n > 0 and not (0 in row_sums or 0 in col_sums):
            chi2_stat = float(stats.chi2_contingency(table, correction=False)[0])
        else:
            chi2_stat = 0.0
    else:
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(
            table, correction=False
        )
        chi2_stat = float(chi2_stat)
        p_value = float(p_value)

    # Interpretation
    if p_value < 0.05 and oe_ratio > 1.5:
        interpretation = "harmonic_dependent"
    elif p_value > 0.05 or (0.5 < oe_ratio < 1.5):
        interpretation = "independent"
    else:
        interpretation = "inconclusive"

    return {
        "p_value": float(p_value),
        "chi2_statistic": float(chi2_stat),
        "observed_expected_ratio": float(oe_ratio),
        "contingency_table": table,
        "n_both": a,
        "n_24h_only": c,
        "n_12h_only": b,
        "n_neither": d,
        "interpretation": interpretation,
    }
