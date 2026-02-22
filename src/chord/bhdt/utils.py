"""
Shared utility functions for BHDT module.
"""

import numpy as np


def bh_fdr_correction(pvalues):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values.

    Returns
    -------
    fdr : np.ndarray
        Adjusted p-values (same length as input). NaN inputs stay NaN.
    """
    n = len(pvalues)
    valid = ~np.isnan(pvalues)
    fdr = np.full(n, np.nan)
    if not np.any(valid):
        return fdr
    pv = pvalues[valid]
    order = np.argsort(pv)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(pv) + 1)
    adjusted = pv * len(pv) / ranks
    # Enforce monotonicity
    adjusted_sorted = adjusted[order]
    for i in range(len(adjusted_sorted) - 2, -1, -1):
        adjusted_sorted[i + 1] = min(adjusted_sorted[i + 1], 1.0)
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    adjusted = np.minimum(adjusted, 1.0)
    fdr[valid] = adjusted
    return fdr
