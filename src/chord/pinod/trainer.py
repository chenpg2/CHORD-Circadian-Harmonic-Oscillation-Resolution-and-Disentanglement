"""
PINOD training loop and loss functions.

Handles single-gene and batch training with:
  - MSE reconstruction loss
  - Physics regularisation (penalise nonlinear corrections)
  - Sparsity regularisation (L1 on oscillator amplitudes)
  - Period prior regularisation (soft constraint on learned periods)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class PINODLoss(nn.Module):
    """Combined loss for PINOD training.

    L = MSE(x, x_hat) + lambda_phys * Physics_Reg
                       + lambda_sparse * Sparsity_Reg
                       + lambda_period * Period_Reg

    Parameters
    ----------
    lambda_phys : float
        Weight for physics regularisation (nonlinear correction penalty).
    lambda_sparse : float
        Weight for sparsity regularisation (L1 on amplitudes).
    lambda_period : float
        Weight for period prior regularisation.
    period_priors : list of (mean, std)
        Gaussian priors on periods. E.g. [(24, 2), (12, 1.5), (8, 1)].
    """

    def __init__(
        self,
        lambda_phys: float = 0.01,
        lambda_sparse: float = 0.001,
        lambda_period: float = 0.1,
        period_priors: Optional[List[Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.lambda_sparse = lambda_sparse
        self.lambda_period = lambda_period
        self.period_priors = period_priors or [(24.0, 2.0), (12.0, 1.5), (8.0, 1.0)]

    def forward(
        self,
        expr: torch.Tensor,
        expr_hat: torch.Tensor,
        model: nn.Module,
        amplitudes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        expr : (batch, T) ground truth
        expr_hat : (batch, T) reconstruction
        model : PINODSingleGene — for accessing ODE parameters
        amplitudes : (batch, K) oscillator amplitudes

        Returns
        -------
        total_loss : scalar tensor
        loss_dict : dict of individual loss components (for logging)
        """
        # Reconstruction loss
        mse = nn.functional.mse_loss(expr_hat, expr)

        # Physics regularisation: penalise nonlinear correction magnitudes
        phys_reg = model.ode.physics_regularisation()

        # Sparsity: L1 on amplitudes
        sparse_reg = amplitudes.abs().mean()

        # Period prior: Gaussian penalty on learned periods
        periods = model.ode.periods
        period_reg = torch.tensor(0.0, device=expr.device)
        for k, (mu, sigma) in enumerate(self.period_priors):
            if k < len(periods):
                period_reg = period_reg + ((periods[k] - mu) / sigma) ** 2
        period_reg = period_reg / max(len(self.period_priors), 1)

        total = (
            mse
            + self.lambda_phys * phys_reg
            + self.lambda_sparse * sparse_reg
            + self.lambda_period * period_reg
        )

        loss_dict = {
            "total": total.item(),
            "mse": mse.item(),
            "phys_reg": phys_reg.item(),
            "sparse_reg": sparse_reg.item(),
            "period_reg": period_reg.item(),
        }
        return total, loss_dict


def train_single_gene(
    model: nn.Module,
    expr: np.ndarray,
    timepoints: np.ndarray,
    n_epochs: int = 500,
    lr: float = 1e-3,
    lambda_phys: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_period: float = 0.1,
    period_priors: Optional[List[Tuple[float, float]]] = None,
    patience: int = 50,
    min_delta: float = 1e-6,
    solver: str = "dopri5",
    verbose: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train PINOD on a single gene.

    Parameters
    ----------
    model : PINODSingleGene
    expr : (N,) or (1, N) expression values
    timepoints : (N,) time points in hours
    n_epochs : int
    lr : float
    lambda_phys, lambda_sparse, lambda_period : float
    period_priors : list of (mean, std)
    patience : int — early stopping patience
    min_delta : float — minimum improvement for early stopping
    solver : str — ODE solver
    verbose : bool
    device : str

    Returns
    -------
    dict with keys: model, history, best_epoch, parameters
    """
    model = model.to(device)
    model.train()

    # Prepare data
    if expr.ndim == 1:
        expr = expr.reshape(1, -1)
    expr_t = torch.tensor(expr, dtype=torch.float32, device=device)
    t_eval = torch.tensor(timepoints, dtype=torch.float32, device=device)

    # Normalise expression for stable training
    expr_mean = expr_t.mean()
    expr_std = expr_t.std() + 1e-8
    expr_norm = (expr_t - expr_mean) / expr_std

    loss_fn = PINODLoss(lambda_phys, lambda_sparse, lambda_period, period_priors)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=patience // 3, factor=0.5, min_lr=1e-6
    )

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        optimiser.zero_grad()

        try:
            expr_hat, trajectories, amplitudes = model(
                expr_norm, t_eval, solver=solver
            )
        except Exception as e:
            if verbose:
                print(f"  Epoch {epoch}: ODE solver failed ({e}), reducing lr")
            for pg in optimiser.param_groups:
                pg["lr"] *= 0.5
            continue

        loss, loss_dict = loss_fn(expr_norm, expr_hat, model, amplitudes)

        if torch.isnan(loss) or torch.isinf(loss):
            if verbose:
                print(f"  Epoch {epoch}: NaN/Inf loss, reducing lr")
            for pg in optimiser.param_groups:
                pg["lr"] *= 0.5
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step(loss.item())

        history.append(loss_dict)

        # Early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and epoch % 100 == 0:
            periods = model.ode.periods.detach().cpu().numpy()
            gammas = model.ode.gamma.detach().cpu().numpy()
            print(
                f"  Epoch {epoch}: loss={loss.item():.6f} "
                f"mse={loss_dict['mse']:.6f} "
                f"periods={np.round(periods, 2)} "
                f"gammas={np.round(gammas, 4)}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    # Extract final parameters
    params = model.get_parameters_dict()
    params["expr_mean"] = float(expr_mean.item())
    params["expr_std"] = float(expr_std.item())
    params["best_loss"] = best_loss
    params["best_epoch"] = best_epoch
    params["n_epochs_run"] = len(history)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "parameters": params,
    }


def train_batch(
    expr_matrix: np.ndarray,
    timepoints: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_oscillators: int = 3,
    period_inits: Optional[List[float]] = None,
    n_epochs: int = 500,
    lr: float = 1e-3,
    lambda_phys: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_period: float = 0.1,
    patience: int = 50,
    decoder_type: str = "linear",
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train PINOD on a batch of genes (each gene gets its own model).

    Parameters
    ----------
    expr_matrix : (n_genes, n_timepoints)
    timepoints : (n_timepoints,)
    gene_names : list of str, optional
    n_oscillators : int
    period_inits : list of float
    n_epochs : int
    lr : float
    lambda_phys, lambda_sparse, lambda_period : float
    patience : int
    decoder_type : str
    device : str
    verbose : bool

    Returns
    -------
    dict with keys:
        results : list of per-gene result dicts
        summary : DataFrame-like list of dicts with key parameters
    """
    from chord.pinod.networks import PINODSingleGene

    if period_inits is None:
        period_inits = [24.0, 12.0, 8.0]

    n_genes, n_tp = expr_matrix.shape
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    results = []
    summary = []

    for i in range(n_genes):
        if verbose:
            print(f"Gene {i+1}/{n_genes}: {gene_names[i]}")

        model = PINODSingleGene(
            n_timepoints=n_tp,
            n_oscillators=n_oscillators,
            period_inits=period_inits,
            decoder_type=decoder_type,
        )

        result = train_single_gene(
            model=model,
            expr=expr_matrix[i],
            timepoints=timepoints,
            n_epochs=n_epochs,
            lr=lr,
            lambda_phys=lambda_phys,
            lambda_sparse=lambda_sparse,
            lambda_period=lambda_period,
            patience=patience,
            device=device,
            verbose=verbose,
        )

        result["gene_name"] = gene_names[i]
        results.append(result)

        # Summary row
        params = result["parameters"]
        row = {
            "gene": gene_names[i],
            "best_loss": params["best_loss"],
            "best_epoch": params["best_epoch"],
        }
        for k, T in enumerate(params["periods"]):
            row[f"T_{k}"] = T
            row[f"gamma_{k}"] = params["gammas"][k]
            if "mixing_weights" in params:
                row[f"weight_{k}"] = params["mixing_weights"][k]
        summary.append(row)

    return {"results": results, "summary": summary}
