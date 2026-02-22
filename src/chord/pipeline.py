"""
End-to-end CHORD pipeline.

Orchestrates BHDT (Bayesian Harmonic Disentanglement Test),
PINOD (Physics-Informed Neural ODE Decomposition), and
ensemble integration into a single ``chord.run()`` call.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def run(
    expr: Union[np.ndarray, pd.DataFrame],
    timepoints: Union[np.ndarray, Sequence[float]],
    gene_names: Optional[List[str]] = None,
    methods: str = "auto",
    # BHDT params
    bhdt_method: str = "analytic",
    # PINOD params
    pinod_epochs: int = 300,
    pinod_solver: str = "rk4",
    pinod_device: str = "cpu",
    pinod_lambda_phys: float = 0.01,
    pinod_lambda_sparse: float = 0.01,
    # Ensemble params
    ensemble_weights: Optional[Tuple[float, float]] = None,
    # General
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the full CHORD rhythm-detection pipeline.

    Parameters
    ----------
    expr : (n_genes, n_timepoints) array or DataFrame
        Gene expression matrix.  Rows = genes, columns = time-ordered samples.
        If a DataFrame, the index is used as gene names.
    timepoints : (n_timepoints,) array
        Sampling time points in hours.
    gene_names : list of str, optional
        Gene identifiers.  Inferred from *expr* index when it is a DataFrame.
    methods : str
        Which detection back-ends to run:

        * ``'bhdt'``  — BHDT only (fast, no PyTorch needed).
        * ``'pinod'`` — PINOD only (requires PyTorch).
        * ``'both'``  — Run both and ensemble.
        * ``'auto'``  — Try to import ``torch``; if available use ``'both'``,
          otherwise fall back to ``'bhdt'``.
    bhdt_method : str
        BHDT inference mode: ``'analytic'`` (fast, BIC-based),
        ``'bootstrap'`` (parametric bootstrap LRT, ~3s/gene),
        ``'ensemble'`` (analytic + bootstrap, recommended), or
        ``'mcmc'`` (full Bayesian, requires numpyro, ~30s/gene).
    pinod_epochs : int
        Maximum training epochs for PINOD.
    pinod_solver : str
        ODE solver for PINOD (e.g. ``'rk4'``, ``'dopri5'``).
    pinod_device : str
        PyTorch device for PINOD (``'cpu'`` or ``'cuda'``).
    pinod_lambda_phys : float
        Physics-regularisation weight for PINOD.
    pinod_lambda_sparse : float
        Sparsity-regularisation weight for PINOD.
    ensemble_weights : tuple of (float, float), optional
        ``(w_bhdt, w_pinod)`` voting weights.  Default ``(0.4, 0.6)``.
    n_jobs : int
        Number of parallel workers for BHDT (``-1`` = all cores).
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame
        One row per gene with columns ``gene``, ``classification``,
        ``confidence``, plus all prefixed columns from BHDT / PINOD results.
    """
    timepoints = np.asarray(timepoints, dtype=np.float64)

    # --- Input validation ---
    if timepoints.ndim != 1:
        raise ValueError(
            f"timepoints must be 1-D, got shape {timepoints.shape}."
        )
    if not np.all(timepoints[:-1] <= timepoints[1:]):
        raise ValueError("timepoints must be sorted in non-decreasing order.")

    # --- resolve gene names from DataFrame index --------------------------
    if isinstance(expr, pd.DataFrame):
        if gene_names is None:
            gene_names = expr.index.tolist()
        expr_arr = expr.values
    else:
        expr_arr = np.asarray(expr, dtype=np.float64)
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(expr_arr.shape[0])]

    if expr_arr.ndim != 2:
        raise ValueError(
            f"expr must be 2-D (genes x timepoints), got {expr_arr.ndim}-D array."
        )
    if expr_arr.shape[1] != len(timepoints):
        raise ValueError(
            f"Number of columns in expr ({expr_arr.shape[1]}) must match "
            f"len(timepoints) ({len(timepoints)})."
        )

    # --- resolve method selection -----------------------------------------
    methods = _resolve_methods(methods, verbose=verbose)

    run_bhdt = methods in ("bhdt", "both")
    run_pinod = methods in ("pinod", "both")

    bhdt_df: Optional[pd.DataFrame] = None
    pinod_df: Optional[pd.DataFrame] = None

    # --- BHDT -------------------------------------------------------------
    if run_bhdt:
        if verbose:
            print("[CHORD] Running BHDT (%s) ..." % bhdt_method)
        from chord.bhdt.pipeline import run_bhdt as _run_bhdt

        bhdt_df = _run_bhdt(
            expr_arr,
            timepoints,
            method=bhdt_method,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # Attach gene names (run_bhdt may generate default names)
        if gene_names is not None and len(gene_names) == len(bhdt_df):
            bhdt_df["gene"] = gene_names

    # --- PINOD ------------------------------------------------------------
    if run_pinod:
        if verbose:
            print("[CHORD] Running PINOD ...")
        from chord.pinod.decompose import decompose as _decompose

        pinod_df = _decompose(
            expr_arr,
            timepoints,
            gene_names=gene_names,
            n_epochs=pinod_epochs,
            lambda_phys=pinod_lambda_phys,
            lambda_sparse=pinod_lambda_sparse,
            device=pinod_device,
            verbose=verbose,
        )

    # --- Ensemble / single-method output ----------------------------------
    if bhdt_df is not None and pinod_df is not None:
        if verbose:
            print("[CHORD] Integrating BHDT + PINOD via ensemble ...")
        from chord.ensemble.integrator import integrate_results

        weights_dict = None
        if ensemble_weights is not None:
            weights_dict = {"bhdt": ensemble_weights[0], "pinod": ensemble_weights[1]}

        result = integrate_results(bhdt_df, pinod_df, weights=weights_dict)

        # Normalise output columns
        result = _normalise_output(result, source="ensemble")
        return result

    if bhdt_df is not None:
        return _normalise_output(bhdt_df, source="bhdt")

    if pinod_df is not None:
        return _normalise_output(pinod_df, source="pinod")

    raise RuntimeError("No detection method was executed — this should not happen.")


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _resolve_methods(methods: str, verbose: bool = False) -> str:
    """Resolve ``'auto'`` to ``'both'`` or ``'bhdt'`` based on torch availability."""
    methods = methods.lower().strip()
    if methods not in ("auto", "bhdt", "pinod", "both"):
        raise ValueError(
            f"methods must be 'auto', 'bhdt', 'pinod', or 'both', got {methods!r}"
        )
    if methods != "auto":
        return methods

    try:
        import torch  # noqa: F401
        if verbose:
            print("[CHORD] PyTorch available — using both BHDT + PINOD.")
        return "both"
    except ImportError:
        if verbose:
            print("[CHORD] PyTorch not found — falling back to BHDT only.")
        return "bhdt"


def _normalise_output(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure the returned DataFrame always has ``gene``, ``classification``,
    and ``confidence`` as its first three columns, regardless of which
    back-end produced the result.
    """
    out = df.copy()

    # Map ensemble column names to the canonical ones
    if "consensus_classification" in out.columns and "classification" not in out.columns:
        out = out.rename(columns={"consensus_classification": "classification"})
    if "consensus_confidence" in out.columns and "confidence" not in out.columns:
        out = out.rename(columns={"consensus_confidence": "confidence"})

    # BHDT-only results may lack a confidence column
    if "confidence" not in out.columns:
        if "log_bayes_factor" in out.columns:
            lbf = out["log_bayes_factor"].astype(float)
            out["confidence"] = 1.0 / (1.0 + np.exp(-lbf))
        else:
            out["confidence"] = np.nan

    # Prefix non-canonical columns with the source when running single-method
    if source in ("bhdt", "pinod"):
        prefix = f"{source}_"
        rename_map = {}
        skip = {"gene", "classification", "confidence"}
        for col in out.columns:
            if col not in skip and not col.startswith(prefix):
                rename_map[col] = f"{prefix}{col}"
        if rename_map:
            out = out.rename(columns=rename_map)

    # Reorder so gene, classification, confidence come first
    priority = ["gene", "classification", "confidence"]
    existing = [c for c in priority if c in out.columns]
    rest = [c for c in out.columns if c not in existing]
    out = out[existing + rest]

    return out
