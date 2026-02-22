"""
Genome-wide BHDT pipeline: batch processing with parallel execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from joblib import Parallel, delayed
from tqdm import tqdm

from chord.bhdt.inference import bhdt_analytic, component_f_test
from chord.bhdt.utils import bh_fdr_correction


def run_bhdt(
    expr,
    timepoints,
    method="analytic",
    periods=None,
    T_base=24.0,
    K_harmonics=3,
    n_jobs=1,
    verbose=True,
):
    """Run BHDT on a gene expression matrix.

    Parameters
    ----------
    expr : pd.DataFrame or np.ndarray
        Gene expression matrix. Rows = genes, columns = samples (time-ordered).
        If DataFrame, index is used as gene names.
    timepoints : array-like
        Time points in hours corresponding to columns.
    method : str
        'analytic' (fast, BIC-based) or 'mcmc' (precise, requires numpyro).
    periods : list of float, optional
        Ultradian periods to test. Default [24, 12, 8].
    T_base : float
        Base circadian period.
    K_harmonics : int
        Number of harmonics for M0.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.DataFrame with one row per gene and columns:
        gene, log_bayes_factor, bayes_factor, interpretation, classification,
        T_12_fitted, T_12_deviation, A_24, A_12, A_8, phi_24, phi_12, phi_8,
        f_test_12h_pvalue, f_test_12h_significant
    """
    t = np.asarray(timepoints, dtype=np.float64)
    if periods is None:
        periods = [24.0, 12.0, 8.0]

    # --- Input validation ---
    if t.ndim != 1:
        raise ValueError(f"timepoints must be 1-D, got shape {t.shape}.")

    # Handle input types
    if isinstance(expr, pd.DataFrame):
        gene_names = list(expr.index)
        data = expr.values
    else:
        data = np.asarray(expr)
        gene_names = ["gene_%d" % i for i in range(data.shape[0])]

    if data.ndim != 2:
        raise ValueError(
            f"expr must be 2-D (genes x timepoints), got {data.ndim}-D array."
        )
    if data.shape[1] != len(t):
        raise ValueError(
            f"Number of columns in expr ({data.shape[1]}) must match "
            f"len(timepoints) ({len(t)})."
        )

    n_genes = data.shape[0]

    if method == "analytic":
        _run_fn = _run_single_analytic
    elif method == "mcmc":
        _run_fn = _run_single_mcmc
    elif method == "bootstrap":
        _run_fn = _run_single_bootstrap
    elif method == "ensemble":
        _run_fn = _run_single_ensemble
    else:
        raise ValueError("method must be 'analytic', 'mcmc', 'bootstrap', or 'ensemble', got '%s'" % method)

    # Parallel execution
    if verbose:
        print("CHORD BHDT: running %s mode on %d genes with %d jobs..." % (method, n_genes, n_jobs))

    # Build the job list eagerly so tqdm tracks actual job completion, not
    # just iterator consumption (which joblib does instantly).
    jobs = [delayed(_run_fn)(t, data[i], periods, T_base, K_harmonics)
            for i in range(n_genes)]
    results = []
    with tqdm(total=n_genes, disable=not verbose, desc="BHDT") as pbar:
        # Use batch_size=1 so the callback fires per-gene
        for out in Parallel(n_jobs=n_jobs)(jobs):
            results.append(out)
            pbar.update(1)

    # Assemble results DataFrame
    rows = []
    for i, r in enumerate(results):
        row = {
            "gene": gene_names[i],
            "log_bayes_factor": r["log_bayes_factor"],
            "bayes_factor": r["bayes_factor"],
            "interpretation": r["interpretation"],
            "classification": r["classification"],
        }
        # Period deviation
        pd_info = r.get("period_deviation", {})
        row["T_12_fitted"] = pd_info.get("T_12_fitted", np.nan)
        row["T_12_deviation"] = pd_info.get("deviation_hours", np.nan)

        # Component amplitudes and phases from M1
        m1 = r.get("m1", {})
        for comp in m1.get("components", []):
            T = comp["T"]
            if abs(T - 24) < 2:
                row["A_24"] = comp["A"]
                row["phi_24"] = comp["phi"]
            elif abs(T - 12) < 2:
                row["A_12"] = comp["A"]
                row["phi_12"] = comp["phi"]
            elif abs(T - 8) < 2:
                row["A_8"] = comp["A"]
                row["phi_8"] = comp["phi"]

        # F-test for 12h component
        ft = r.get("f_test_12h", {})
        row["f_test_12h_pvalue"] = ft.get("p_value", np.nan)
        row["f_test_12h_significant"] = ft.get("significant", False)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Multiple testing correction (BH)
    if "f_test_12h_pvalue" in df.columns:
        df["f_test_12h_fdr"] = bh_fdr_correction(df["f_test_12h_pvalue"].values)

    return df


def _run_single_analytic(t, y, periods, T_base, K_harmonics):
    """Run BHDT analytic mode on a single gene."""
    result = bhdt_analytic(t, y, T_base=T_base, K_harmonics=K_harmonics,
                           ultradian_periods=periods)
    # Add F-test for 12h
    ft = component_f_test(t, y, periods, test_period_idx=1)
    result["f_test_12h"] = ft
    return result


def _run_single_mcmc(t, y, periods, T_base, K_harmonics):
    """Run BHDT MCMC mode on a single gene (requires numpyro)."""
    try:
        from chord.bhdt.mcmc import bhdt_mcmc
    except ImportError:
        raise ImportError(
            "MCMC mode requires numpyro. Install with: pip install chord-rhythm[bayes]"
        )
    # Convert periods list to period_priors format: [(T, sd), ...]
    period_priors = [(T, T * 0.1) for T in periods]
    result = bhdt_mcmc(t, y, T_base=T_base, K_harmonics=K_harmonics,
                        period_priors=period_priors)
    # Normalize output keys to match analytic format
    # MCMC mode does not compute a true Bayes Factor; expose native metric instead
    result["log_bayes_factor"] = np.nan
    result["bayes_factor"] = np.nan
    result["delta_waic"] = result.get("delta_waic", 0.0)
    result["se_delta_waic"] = result.get("se_delta_waic", np.nan)
    result["interpretation"] = result["classification"]
    # Add F-test
    from chord.bhdt.inference import component_f_test
    ft = component_f_test(t, y, periods, test_period_idx=1)
    result["f_test_12h"] = ft
    # Add period deviation info
    if result.get("m1_params") and result["m1_params"].get("components"):
        comps = result["m1_params"]["components"]
        if len(comps) >= 2:
            T_12_fit = comps[1]["T"]
            expected_12 = T_base / 2.0
            deviation = T_12_fit - expected_12
            rel_deviation = abs(deviation) / expected_12
            result["period_deviation"] = {
                "T_12_fitted": T_12_fit,
                "deviation_hours": abs(deviation),
                "relative_deviation": float(rel_deviation),
                "deviates_from_harmonic": rel_deviation > 0.02,
            }
    # Map m1_params to m1 format for pipeline compatibility
    if result.get("m1_params"):
        result["m1"] = result["m1_params"]
    return result


def _run_single_bootstrap(t, y, periods, T_base, K_harmonics):
    """Run BHDT bootstrap LRT mode on a single gene."""
    from chord.bhdt.bootstrap import parametric_bootstrap_lrt
    from chord.bhdt.inference import component_f_test

    result = parametric_bootstrap_lrt(t, y, T_base=T_base, K_harmonics=K_harmonics,
                                       n_bootstrap=499)
    # Normalize output keys
    # Bootstrap mode does not compute a true Bayes Factor; expose native metric instead
    result["log_bayes_factor"] = np.nan
    result["bayes_factor"] = np.nan
    result["bootstrap_p_value"] = result.get("p_value", np.nan)
    result["interpretation"] = result["classification"]
    # F-test
    ft = component_f_test(t, y, periods, test_period_idx=1)
    result["f_test_12h"] = ft
    # Period deviation
    m1 = result.get("m1_fit", {})
    if "fitted_periods" in m1:
        fp = m1["fitted_periods"]
        if len(fp) >= 2:
            expected_12 = T_base / 2.0
            deviation = fp[1] - expected_12
            rel_deviation = abs(deviation) / expected_12
            result["period_deviation"] = {
                "T_12_fitted": fp[1],
                "deviation_hours": abs(deviation),
                "relative_deviation": float(rel_deviation),
                "deviates_from_harmonic": rel_deviation > 0.02,
            }
    # Map m1_fit components to m1 format
    if m1.get("components"):
        result["m1"] = m1
    return result


def _run_single_ensemble(t, y, periods, T_base, K_harmonics):
    """Run BHDT ensemble mode (analytic + bootstrap) on a single gene."""
    from chord.bhdt.inference import bhdt_ensemble
    result = bhdt_ensemble(t, y, T_base=T_base, K_harmonics=K_harmonics,
                            n_bootstrap=499)
    # Ensure all keys expected by the DataFrame assembly are present with
    # safe defaults, matching the defensive pattern of the other runners.
    result.setdefault("log_bayes_factor", np.nan)
    result.setdefault("bayes_factor", np.nan)
    result.setdefault("interpretation", result.get("classification", "ambiguous"))
    result.setdefault("period_deviation", {})
    result.setdefault("m1", {})
    result.setdefault("f_test_12h", {})
    return result
