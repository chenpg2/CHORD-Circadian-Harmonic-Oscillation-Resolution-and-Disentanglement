"""
Top-level rhythm detection API.

Usage:
    import chord
    result = chord.detect(expr_matrix, timepoints, period=[12, 24, 8])
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union

from chord.bhdt.models import fit_independent_model, _cos_sin_design, _fit_ols
from chord.bhdt.inference import component_f_test
from chord.bhdt.utils import bh_fdr_correction


def detect(
    expr,
    timepoints,
    period=None,
    alpha=0.05,
    n_jobs=1,
    verbose=True,
):
    """Detect rhythmic genes at specified periods.

    A lightweight entry point that performs cosinor regression + F-test
    for each candidate period. For harmonic disentanglement, use
    ``chord.disentangle()`` instead.

    Parameters
    ----------
    expr : pd.DataFrame or np.ndarray
        Rows = genes, columns = samples (time-ordered).
    timepoints : array-like
        Time points in hours.
    period : float or list of float, optional
        Periods to test. Default [24, 12].
    alpha : float
        Significance threshold for F-test.
    n_jobs : int
        Parallel jobs.
    verbose : bool
        Progress bar.

    Returns
    -------
    pd.DataFrame with columns: gene, A_{T}, phi_{T}, p_{T}, fdr_{T}, ...
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    t = np.asarray(timepoints, dtype=np.float64)
    if period is None:
        period = [24.0, 12.0]
    if isinstance(period, (int, float)):
        period = [float(period)]
    period = [float(p) for p in period]

    if isinstance(expr, pd.DataFrame):
        gene_names = list(expr.index)
        data = expr.values
    else:
        data = np.asarray(expr)
        gene_names = ["gene_%d" % i for i in range(data.shape[0])]

    n_genes = data.shape[0]

    def _fit_one(i):
        y = data[i]
        row = {}
        m = fit_independent_model(t, y, periods=period)
        for comp in m["components"]:
            T = comp["T"]
            tag = "%.0f" % T
            row["A_" + tag] = comp["A"]
            row["phi_" + tag] = comp["phi"]
        # F-test for each period
        for j, T in enumerate(period):
            ft = component_f_test(t, y, period, test_period_idx=j, alpha=alpha)
            tag = "%.0f" % T
            row["p_" + tag] = ft["p_value"]
            row["sig_" + tag] = ft["significant"]
        row["mesor"] = m["mesor"]
        return row

    if verbose:
        print("CHORD detect: %d genes, periods=%s" % (n_genes, period))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(i)
        for i in tqdm(range(n_genes), disable=not verbose, desc="detect")
    )

    df = pd.DataFrame(results, index=gene_names)
    df.index.name = "gene"

    # BH correction for each period
    for T in period:
        tag = "%.0f" % T
        pcol = "p_" + tag
        if pcol in df.columns:
            df["fdr_" + tag] = bh_fdr_correction(df[pcol].values)

    return df
