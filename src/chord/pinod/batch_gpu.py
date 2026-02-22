"""
GPU-accelerated batch training for PINOD.

Solves the performance bottleneck of sequential single-gene training
(~38s/gene on CPU = 32 days for 20K genes) by batching B genes into a
single forward pass with vectorised RK4 integration.

Target: 256 genes/batch on 16GB VRAM -> 20K genes in ~40-160s.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Batched nonlinear correction
# ---------------------------------------------------------------------------

class BatchNonlinearCorrection(nn.Module):
    """Vectorised nonlinear correction for B genes x K oscillators.

    Instead of K separate MLPs per gene, we use a single set of weight
    tensors with an explicit batch (gene) dimension.  Each gene gets
    independent parameters but the matmuls are fused into one GEMM.

    Parameters
    ----------
    n_oscillators : int
        Number of oscillators K.
    hidden_dim : int
        Hidden width of the two-layer correction MLP.
    batch_size : int
        Number of genes B processed in parallel.
    """

    def __init__(self, n_oscillators: int, hidden_dim: int, batch_size: int):
        super().__init__()
        K = n_oscillators
        input_dim = 2 * K + 1  # full state + time

        # (B, hidden, input_dim) — one weight matrix per gene
        self.w1 = nn.Parameter(torch.randn(batch_size, hidden_dim, input_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(batch_size, hidden_dim, hidden_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        # Output: one correction per oscillator -> (B, K)
        self.w3 = nn.Parameter(torch.randn(batch_size, K, hidden_dim) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(batch_size, K))

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, 2*K)
        t     : scalar tensor

        Returns
        -------
        corrections : (B, K)
        """
        B = state.shape[0]
        t_expand = t.expand(B, 1)
        x = torch.cat([state, t_expand], dim=-1)  # (B, 2K+1)

        # Layer 1: (B, hidden) = bmm((B,hidden,in), (B,in,1)).squeeze + bias
        x = torch.bmm(self.w1, x.unsqueeze(-1)).squeeze(-1) + self.b1
        x = torch.tanh(x)
        # Layer 2
        x = torch.bmm(self.w2, x.unsqueeze(-1)).squeeze(-1) + self.b2
        x = torch.tanh(x)
        # Layer 3 -> (B, K)
        x = torch.bmm(self.w3, x.unsqueeze(-1)).squeeze(-1) + self.b3
        return x


# ---------------------------------------------------------------------------
# Batched ODE right-hand side
# ---------------------------------------------------------------------------

class BatchODE(nn.Module):
    """Vectorised ODE system for B genes simultaneously.

    Each gene has independent gamma_k, omega_k, and correction MLP
    parameters, but the integration is fully vectorised.

    State layout per gene: [z_1, v_1, z_2, v_2, ..., z_K, v_K]
    """

    def __init__(self, n_oscillators: int, hidden_dim: int, batch_size: int,
                 period_inits: Optional[List[float]] = None,
                 gamma_init: float = 0.01):
        super().__init__()
        self.K = n_oscillators
        self.B = batch_size

        if period_inits is None:
            period_inits = [24.0, 12.0, 8.0]

        omega_inits = [2.0 * np.pi / T for T in period_inits]

        # Per-gene, per-oscillator parameters: (B, K)
        self.log_omega = nn.Parameter(
            torch.tensor(omega_inits, dtype=torch.float32)
            .log().unsqueeze(0).expand(batch_size, -1).clone()
        )
        self.log_gamma = nn.Parameter(
            torch.full((batch_size, n_oscillators),
                       np.log(max(gamma_init, 1e-6)), dtype=torch.float32)
        )

        self.corrections = BatchNonlinearCorrection(
            n_oscillators, hidden_dim, batch_size
        )

    @property
    def omega(self) -> torch.Tensor:
        return torch.exp(self.log_omega)  # (B, K)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)  # (B, K)

    @property
    def periods(self) -> torch.Tensor:
        return 2.0 * np.pi / self.omega  # (B, K)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ODE RHS: dstate/dt for all B genes at once.

        Parameters
        ----------
        t     : scalar tensor
        state : (B, 2*K)

        Returns
        -------
        dstate : (B, 2*K)
        """
        K = self.K
        omega = self.omega  # (B, K)
        gamma = self.gamma  # (B, K)

        # Separate displacements and velocities
        z = state[:, 0::2]  # (B, K)
        v = state[:, 1::2]  # (B, K)

        # Linear oscillator dynamics
        dz = v
        dv = -2.0 * gamma * v - omega ** 2 * z

        # Nonlinear corrections: (B, K)
        corr = self.corrections(state, t)
        dv = dv + corr

        # Interleave back: [dz_1, dv_1, dz_2, dv_2, ...]
        dstate = torch.stack([dz, dv], dim=-1).reshape(state.shape)
        return dstate

    def physics_reg(self) -> torch.Tensor:
        """L2 norm of correction parameters, summed over all genes."""
        reg = torch.tensor(0.0, device=self.log_omega.device)
        for p in self.corrections.parameters():
            reg = reg + p.pow(2).sum()
        return reg


# ---------------------------------------------------------------------------
# Vectorised RK4 integrator
# ---------------------------------------------------------------------------

def rk4_integrate(ode_fn, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Fixed-step RK4 integration, fully vectorised over batch dim.

    Parameters
    ----------
    ode_fn : callable(t, y) -> dy
    y0     : (B, D) initial state
    t      : (T,) time grid (must be sorted)

    Returns
    -------
    trajectory : (T, B, D)
    """
    T = t.shape[0]
    ys = [y0]
    y = y0
    for i in range(T - 1):
        dt = t[i + 1] - t[i]
        t_i = t[i]
        k1 = ode_fn(t_i, y)
        k2 = ode_fn(t_i + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = ode_fn(t_i + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = ode_fn(t_i + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ys.append(y)
    return torch.stack(ys, dim=0)  # (T, B, D)


# ---------------------------------------------------------------------------
# Batch PINOD model
# ---------------------------------------------------------------------------

class BatchPINODModel(nn.Module):
    """Batch PINOD model for simultaneous training of B genes.

    All parameters are stored with an explicit gene dimension so that a
    single forward pass processes B genes through encode -> RK4 -> decode.

    Parameters
    ----------
    n_timepoints : int
        Number of time points per gene.
    n_oscillators : int
        Number of oscillators K (default 3 for 24h/12h/8h).
    hidden_dim : int
        Hidden width for encoder and correction MLPs.
    batch_size : int
        Number of genes B in this batch.
    period_inits : list of float
        Initial period guesses in hours.
    gamma_init : float
        Initial damping rate.
    device : str or torch.device
        Target device.
    """

    def __init__(
        self,
        n_timepoints: int,
        n_oscillators: int = 3,
        hidden_dim: int = 32,
        batch_size: int = 256,
        period_inits: Optional[List[float]] = None,
        gamma_init: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.K = n_oscillators
        self.B = batch_size
        self.n_timepoints = n_timepoints
        state_dim = 2 * n_oscillators

        if period_inits is None:
            period_inits = [24.0, 12.0, 8.0]

        # --- Batched encoder: (B, n_tp) -> (B, 2K) ---
        # Stored as explicit (B, out, in) weight tensors for bmm
        self.enc_w1 = nn.Parameter(
            torch.randn(batch_size, hidden_dim, n_timepoints) * (2.0 / n_timepoints) ** 0.5
        )
        self.enc_b1 = nn.Parameter(torch.zeros(batch_size, hidden_dim))
        self.enc_w2 = nn.Parameter(
            torch.randn(batch_size, state_dim, hidden_dim) * (2.0 / hidden_dim) ** 0.5
        )
        self.enc_b2 = nn.Parameter(torch.zeros(batch_size, state_dim))

        # --- Batched ODE ---
        self.ode = BatchODE(
            n_oscillators, hidden_dim=16, batch_size=batch_size,
            period_inits=period_inits, gamma_init=gamma_init,
        )

        # --- Batched linear decoder: z_k -> expression ---
        # weights (B, K): per-gene mixing coefficients
        self.dec_weights = nn.Parameter(torch.ones(batch_size, n_oscillators))
        self.dec_bias = nn.Parameter(torch.zeros(batch_size))

    def encode(self, expr: torch.Tensor) -> torch.Tensor:
        """Encode expression to initial ODE state.

        Parameters
        ----------
        expr : (B, n_timepoints)

        Returns
        -------
        z0 : (B, 2*K)
        """
        # Layer 1
        h = torch.bmm(self.enc_w1, expr.unsqueeze(-1)).squeeze(-1) + self.enc_b1
        h = torch.nn.functional.gelu(h)
        # Layer 2
        z0 = torch.bmm(self.enc_w2, h.unsqueeze(-1)).squeeze(-1) + self.enc_b2
        return z0

    def decode(self, displacements: torch.Tensor) -> torch.Tensor:
        """Decode oscillator displacements to expression.

        Parameters
        ----------
        displacements : (T, B, K)

        Returns
        -------
        expr_hat : (B, T)
        """
        # (T, B, K) * (1, B, K) -> sum over K -> (T, B)
        w = self.dec_weights.unsqueeze(0)  # (1, B, K)
        out = (displacements * w).sum(dim=-1)  # (T, B)
        out = out + self.dec_bias.unsqueeze(0)  # (T, B)
        return out.transpose(0, 1)  # (B, T)

    def forward(self, expr: torch.Tensor, t: torch.Tensor):
        """Full forward pass for a batch of genes.

        Parameters
        ----------
        expr : (B, n_timepoints)
        t    : (n_timepoints,)

        Returns
        -------
        expr_hat    : (B, n_timepoints) reconstructed expression
        trajectories: (T, B, 2*K) full ODE state
        amplitudes  : (B, K) RMS amplitudes per oscillator
        """
        z0 = self.encode(expr)                          # (B, 2K)
        traj = rk4_integrate(self.ode, z0, t)           # (T, B, 2K)
        displacements = traj[:, :, 0::2]                # (T, B, K)
        expr_hat = self.decode(displacements)            # (B, T)
        amplitudes = torch.sqrt((displacements ** 2).mean(dim=0))  # (B, K)
        return expr_hat, traj, amplitudes


# ---------------------------------------------------------------------------
# Loss computation (vectorised over gene batch)
# ---------------------------------------------------------------------------

def batch_loss(
    expr: torch.Tensor,
    expr_hat: torch.Tensor,
    model: BatchPINODModel,
    amplitudes: torch.Tensor,
    lambda_phys: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_period: float = 0.1,
    period_priors: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss for a batch of B genes.

    Parameters
    ----------
    expr      : (B, T) ground truth (normalised)
    expr_hat  : (B, T) reconstruction
    model     : BatchPINODModel
    amplitudes: (B, K)

    Returns
    -------
    total_loss : scalar
    loss_dict  : component breakdown
    """
    if period_priors is None:
        period_priors = [(24.0, 2.0), (12.0, 1.5), (8.0, 1.0)]

    # MSE averaged over genes and timepoints
    mse = nn.functional.mse_loss(expr_hat, expr)

    # Physics regularisation
    phys_reg = model.ode.physics_reg()

    # Sparsity: L1 on amplitudes
    sparse_reg = amplitudes.abs().mean()

    # Period prior: Gaussian penalty on learned periods (B, K)
    periods = model.ode.periods  # (B, K)
    period_reg = torch.tensor(0.0, device=expr.device)
    for k, (mu, sigma) in enumerate(period_priors):
        if k < periods.shape[1]:
            period_reg = period_reg + ((periods[:, k] - mu) / sigma).pow(2).mean()
    period_reg = period_reg / max(len(period_priors), 1)

    total = (
        mse
        + lambda_phys * phys_reg
        + lambda_sparse * sparse_reg
        + lambda_period * period_reg
    )

    loss_dict = {
        "total": total.item(),
        "mse": mse.item(),
        "phys_reg": phys_reg.item(),
        "sparse_reg": sparse_reg.item(),
        "period_reg": period_reg.item(),
    }
    return total, loss_dict


# ---------------------------------------------------------------------------
# Parameter extraction (vectorised)
# ---------------------------------------------------------------------------

def extract_batch_params(
    model: BatchPINODModel,
    expr_batch: torch.Tensor,
    t: torch.Tensor,
    gene_names: List[str],
    expr_means: torch.Tensor,
    expr_stds: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Extract per-gene oscillator parameters from a trained batch model.

    Mirrors the logic of ``chord.pinod.analysis.extract_oscillator_params``
    but operates on the full batch at once.

    Parameters
    ----------
    model       : trained BatchPINODModel
    expr_batch  : (B, T) normalised expression
    t           : (T,) time grid
    gene_names  : length-B list of gene names
    expr_means  : (B,) per-gene expression means
    expr_stds   : (B,) per-gene expression stds

    Returns
    -------
    List of dicts, one per gene, with oscillator parameters.
    """
    model.eval()
    B = expr_batch.shape[0]
    K = model.K

    with torch.no_grad():
        expr_hat, traj, amplitudes = model(expr_batch, t)
        # Denormalise
        expr_hat_denorm = expr_hat * expr_stds.unsqueeze(1) + expr_means.unsqueeze(1)

        displacements = traj[:, :, 0::2]  # (T, B, K)
        velocities = traj[:, :, 1::2]     # (T, B, K)

        periods = model.ode.periods.cpu().numpy()   # (B, K)
        gammas = model.ode.gamma.cpu().numpy()       # (B, K)
        dec_w = model.dec_weights.abs().cpu().numpy()  # (B, K)

        disp_np = displacements.cpu().numpy()  # (T, B, K)
        vel_np = velocities.cpu().numpy()

        # RMS amplitudes per oscillator: (B, K)
        amp_rms = np.sqrt((disp_np ** 2).mean(axis=0))
        amp_peak = np.max(np.abs(disp_np), axis=0)

        # Effective amplitude = rms * mixing weight
        eff_amp = amp_rms * dec_w  # (B, K)

        # Variance explained per oscillator
        osc_signals = disp_np * dec_w[np.newaxis, :, :]  # (T, B, K)
        osc_var = np.var(osc_signals, axis=0)  # (B, K)

        # Reconstruction quality
        expr_orig = (expr_batch * expr_stds.unsqueeze(1) + expr_means.unsqueeze(1)).cpu().numpy()
        expr_hat_np = expr_hat_denorm.cpu().numpy()
        mse_per_gene = ((expr_hat_np - expr_orig) ** 2).mean(axis=1)  # (B,)
        var_per_gene = np.var(expr_orig, axis=1)  # (B,)
        r2_per_gene = 1.0 - mse_per_gene / np.maximum(var_per_gene, 1e-10)

        t_np = t.cpu().numpy()

    results = []
    for i in range(B):
        total_var = max(float(var_per_gene[i]), 1e-10)
        oscillators = []
        for k in range(K):
            d_k = disp_np[:, i, k]
            peak_idx = int(np.argmax(d_k))
            T_k = float(periods[i, k])
            if amp_peak[i, k] > 1e-8:
                phase_hours = float(t_np[peak_idx] % T_k)
                phase_rad = float(2 * np.pi * phase_hours / T_k)
            else:
                phase_hours = phase_rad = 0.0

            oscillators.append({
                "index": k,
                "period": T_k,
                "gamma": float(gammas[i, k]),
                "amplitude_rms": float(amp_rms[i, k]),
                "amplitude_peak": float(amp_peak[i, k]),
                "effective_amplitude": float(eff_amp[i, k]),
                "mixing_weight": float(dec_w[i, k]),
                "phase_rad": phase_rad,
                "phase_hours": phase_hours,
                "var_explained": float(osc_var[i, k]) / total_var,
            })

        # Mark active oscillators (>15% of max effective amplitude)
        eff_list = [o["effective_amplitude"] for o in oscillators]
        max_eff = max(eff_list) if eff_list else 1.0
        for o in oscillators:
            o["active"] = o["effective_amplitude"] > 0.15 * max_eff

        results.append({
            "gene": gene_names[i],
            "oscillators": oscillators,
            "reconstruction_mse": float(mse_per_gene[i]),
            "reconstruction_r2": float(r2_per_gene[i]),
            "n_active": sum(1 for o in oscillators if o["active"]),
            "active_periods": [o["period"] for o in oscillators if o["active"]],
        })

    return results


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def _select_device(device: str) -> Tuple[torch.device, int]:
    """Resolve device string and pick a sensible default batch size."""
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU with batch_size=32")
        return torch.device("cpu"), 32
    return torch.device(device), 256


def train_batch_gpu(
    expr_matrix: np.ndarray,
    timepoints: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_oscillators: int = 3,
    period_inits: Optional[List[float]] = None,
    n_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.01,
    lambda_phys: float = 0.01,
    lambda_sparse: float = 0.001,
    lambda_period: float = 0.1,
    period_priors: Optional[List[Tuple[float, float]]] = None,
    patience: int = 50,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train PINOD on all genes using GPU-batched forward passes.

    Instead of training one gene at a time, this function groups genes
    into mini-batches of ``batch_size`` and trains each batch as a single
    model with independent per-gene parameters but shared structure.

    Parameters
    ----------
    expr_matrix : (n_genes, n_timepoints) numpy array
        Expression matrix (genes x timepoints).
    timepoints : (n_timepoints,) numpy array
        Time grid in hours.
    gene_names : list of str, optional
        Gene identifiers.  Auto-generated if None.
    n_oscillators : int
        Number of oscillators K (default 3).
    period_inits : list of float
        Initial period guesses in hours.
    n_epochs : int
        Maximum training epochs per batch.
    batch_size : int
        Genes per GPU batch.
    lr : float
        Learning rate.
    lambda_phys, lambda_sparse, lambda_period : float
        Regularisation weights.
    period_priors : list of (mean, std)
        Gaussian priors on periods.
    patience : int
        Early-stopping patience (epochs without improvement).
    device : str
        ``'cuda'`` or ``'cpu'``.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        results  : list of per-gene parameter dicts
        summary  : list of flat dicts suitable for pd.DataFrame
        elapsed  : total wall-clock seconds
    """
    dev, default_bs = _select_device(device)
    if device == "cuda" and not torch.cuda.is_available():
        batch_size = default_bs

    n_genes, n_tp = expr_matrix.shape
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]
    if period_inits is None:
        period_inits = [24.0, 12.0, 8.0]

    t_tensor = torch.tensor(timepoints, dtype=torch.float32, device=dev)

    all_results: List[Dict[str, Any]] = []
    n_batches = int(np.ceil(n_genes / batch_size))
    t_start = time.time()

    for b_idx in range(n_batches):
        lo = b_idx * batch_size
        hi = min(lo + batch_size, n_genes)
        B_actual = hi - lo
        names_b = gene_names[lo:hi]

        # Prepare expression batch
        expr_b = torch.tensor(
            expr_matrix[lo:hi], dtype=torch.float32, device=dev
        )  # (B_actual, T)

        # Per-gene normalisation
        expr_means = expr_b.mean(dim=1)          # (B_actual,)
        expr_stds = expr_b.std(dim=1) + 1e-8     # (B_actual,)
        expr_norm = (expr_b - expr_means.unsqueeze(1)) / expr_stds.unsqueeze(1)

        # Pad to full batch_size if last batch is smaller
        if B_actual < batch_size:
            pad = batch_size - B_actual
            expr_norm = torch.cat([
                expr_norm,
                torch.zeros(pad, n_tp, device=dev),
            ], dim=0)

        # Build model for this batch
        model = BatchPINODModel(
            n_timepoints=n_tp,
            n_oscillators=n_oscillators,
            hidden_dim=32,
            batch_size=batch_size,
            period_inits=period_inits,
            device=str(dev),
        ).to(dev)

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=patience // 3, factor=0.5, min_lr=1e-6
        )

        best_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(n_epochs):
            model.train()
            optimiser.zero_grad()

            try:
                expr_hat, traj, amps = model(expr_norm, t_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(
                        f"OOM at batch {b_idx}, epoch {epoch}. "
                        "Consider reducing batch_size."
                    )
                    torch.cuda.empty_cache()
                    break
                raise

            loss, loss_dict = batch_loss(
                expr_norm[:batch_size], expr_hat, model, amps,
                lambda_phys=lambda_phys,
                lambda_sparse=lambda_sparse,
                lambda_period=lambda_period,
                period_priors=period_priors,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                for pg in optimiser.param_groups:
                    pg["lr"] *= 0.5
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step(loss.item())

            # Early stopping
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Pad means/stds to match batch_size for extraction
        if B_actual < batch_size:
            expr_means = torch.cat([
                expr_means,
                torch.zeros(batch_size - B_actual, device=dev),
            ])
            expr_stds = torch.cat([
                expr_stds,
                torch.ones(batch_size - B_actual, device=dev),
            ])
            names_b = names_b + [f"_pad_{j}" for j in range(batch_size - B_actual)]

        batch_results = extract_batch_params(
            model, expr_norm, t_tensor, names_b, expr_means, expr_stds,
        )

        # Keep only real genes (drop padding)
        all_results.extend(batch_results[:B_actual])

        if verbose:
            elapsed = time.time() - t_start
            genes_done = hi
            rate = elapsed / genes_done if genes_done > 0 else 0
            eta = rate * (n_genes - genes_done)
            print(
                f"Batch {b_idx+1}/{n_batches} | "
                f"genes {lo}-{hi-1} | loss={best_loss:.6f} | "
                f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s"
            )

    elapsed_total = time.time() - t_start

    # Build flat summary for DataFrame construction
    summary = []
    for r in all_results:
        row = {
            "gene": r["gene"],
            "reconstruction_mse": r["reconstruction_mse"],
            "reconstruction_r2": r["reconstruction_r2"],
            "n_active": r["n_active"],
        }
        for osc in r["oscillators"]:
            k = osc["index"]
            row[f"period_{k}"] = osc["period"]
            row[f"gamma_{k}"] = osc["gamma"]
            row[f"amplitude_{k}"] = osc["effective_amplitude"]
            row[f"phase_hours_{k}"] = osc["phase_hours"]
            row[f"var_explained_{k}"] = osc["var_explained"]
            row[f"mixing_weight_{k}"] = osc["mixing_weight"]
        summary.append(row)

    if verbose:
        print(
            f"Done: {n_genes} genes in {elapsed_total:.1f}s "
            f"({elapsed_total/max(n_genes,1)*1000:.1f}ms/gene)"
        )

    return {
        "results": all_results,
        "summary": summary,
        "elapsed": elapsed_total,
    }
