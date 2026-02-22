"""
PINOD top-level API: chord.decompose()

Provides a simple interface for Physics-Informed Neural ODE decomposition
of gene expression time series into independent oscillatory components.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union


def decompose(
    expr: Union[np.ndarray, pd.DataFrame],
    timepoints: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_oscillators: int = 3,
    period_inits: Optional[List[float]] = None,
    n_epochs: int = 500,
    lr: float = 1e-3,
    lambda_phys: float = 0.01,
    lambda_sparse: float = 0.005,
    lambda_period: float = 0.1,
    patience: int = 50,
    decoder_type: str = "linear",
    device: str = "cpu",
    verbose: bool = False,
) -> pd.DataFrame:
    """Decompose gene expression into oscillatory components using PINOD.

    This is the main entry point for the PINOD module. For each gene,
    it trains a Physics-Informed Neural ODE model that decomposes the
    signal into K independent damped harmonic oscillators.

    Parameters
    ----------
    expr : (n_genes, n_timepoints) array or DataFrame
        Gene expression matrix. Rows = genes, columns = time points.
    timepoints : (n_timepoints,) array
        Time points in hours.
    gene_names : list of str, optional
        Gene names. Inferred from DataFrame index if available.
    n_oscillators : int
        Number of oscillators (K). Default 3 for {24h, 12h, 8h}.
    period_inits : list of float
        Initial period guesses in hours. Default [24, 12, 8].
    n_epochs : int
        Maximum training epochs per gene.
    lr : float
        Learning rate.
    lambda_phys : float
        Physics regularisation weight.
    lambda_sparse : float
        Sparsity regularisation weight.
    lambda_period : float
        Period prior regularisation weight.
    patience : int
        Early stopping patience.
    decoder_type : str
        'linear' (interpretable) or 'mlp' (flexible).
    device : str
        'cpu' or 'cuda'.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame with columns:
        gene, classification, confidence, evidence,
        T_0, gamma_0, amp_0, ..., T_K, gamma_K, amp_K,
        reconstruction_r2, best_loss
    """
    from chord.pinod.networks import PINODSingleGene
    from chord.pinod.trainer import train_single_gene
    from chord.pinod.analysis import extract_oscillator_params, classify_gene_pinod

    if period_inits is None:
        period_inits = [24.0, 12.0, 8.0]

    # Handle DataFrame input
    if isinstance(expr, pd.DataFrame):
        if gene_names is None:
            gene_names = expr.index.tolist()
        expr = expr.values

    n_genes, n_tp = expr.shape
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    rows = []
    for i in range(n_genes):
        if verbose and (i % 10 == 0 or i == n_genes - 1):
            print(f"  PINOD: gene {i+1}/{n_genes} ({gene_names[i]})")

        # Create fresh model for each gene
        model = PINODSingleGene(
            n_timepoints=n_tp,
            n_oscillators=n_oscillators,
            period_inits=period_inits,
            decoder_type=decoder_type,
        )

        # Train
        train_result = train_single_gene(
            model=model,
            expr=expr[i],
            timepoints=timepoints,
            n_epochs=n_epochs,
            lr=lr,
            lambda_phys=lambda_phys,
            lambda_sparse=lambda_sparse,
            lambda_period=lambda_period,
            patience=patience,
            device=device,
            verbose=False,
        )

        # Analyse
        analysis = extract_oscillator_params(
            train_result["model"], expr[i], timepoints, device=device
        )
        classification = classify_gene_pinod(analysis)

        # Build result row
        row = {
            "gene": gene_names[i],
            "classification": classification["classification"],
            "confidence": classification["confidence"],
            "evidence": classification["evidence"],
            "n_active_oscillators": analysis["n_active"],
            "reconstruction_r2": analysis["reconstruction_r2"],
            "reconstruction_mse": analysis["reconstruction_mse"],
            "best_loss": train_result["parameters"]["best_loss"],
            "best_epoch": train_result["parameters"]["best_epoch"],
        }

        # Per-oscillator parameters
        for osc in analysis["oscillators"]:
            k = osc["index"]
            row[f"T_{k}"] = osc["period"]
            row[f"gamma_{k}"] = osc["gamma"]
            row[f"amp_{k}"] = osc["amplitude_rms"]
            row[f"phase_{k}"] = osc["phase_rad"]
            row[f"energy_{k}"] = osc["mean_energy"]
            row[f"active_{k}"] = osc["active"]
            row[f"correction_norm_{k}"] = osc["correction_norm"]

        rows.append(row)

    return pd.DataFrame(rows)
