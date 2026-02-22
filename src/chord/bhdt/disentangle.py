"""
Top-level harmonic disentanglement API.

Usage:
    import chord
    result = chord.disentangle(expr_matrix, timepoints, method='analytic')
"""

from chord.bhdt.pipeline import run_bhdt


def disentangle(
    expr,
    timepoints,
    method="analytic",
    periods=None,
    T_base=24.0,
    K_harmonics=3,
    n_jobs=1,
    verbose=True,
):
    """Bayesian Harmonic Disentanglement Test (BHDT).

    Distinguishes independent ultradian oscillations from mathematical
    harmonics of circadian rhythms using Bayes Factor model comparison.

    Parameters
    ----------
    expr : pd.DataFrame or np.ndarray
        Rows = genes, columns = samples (time-ordered).
    timepoints : array-like
        Time points in hours.
    method : str
        'analytic' (fast, BIC-based) or 'mcmc' (precise, requires numpyro).
    periods : list of float, optional
        Ultradian periods to test. Default [24, 12, 8].
    T_base : float
        Base circadian period (default 24).
    K_harmonics : int
        Number of harmonics for the constrained model.
    n_jobs : int
        Parallel jobs (-1 = all cores).
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.DataFrame with columns:
        gene, log_bayes_factor, bayes_factor, interpretation, classification,
        T_12_fitted, T_12_deviation, A_24, A_12, A_8, phi_24, phi_12, phi_8,
        f_test_12h_pvalue, f_test_12h_fdr, f_test_12h_significant
    """
    return run_bhdt(
        expr=expr,
        timepoints=timepoints,
        method=method,
        periods=periods,
        T_base=T_base,
        K_harmonics=K_harmonics,
        n_jobs=n_jobs,
        verbose=verbose,
    )
