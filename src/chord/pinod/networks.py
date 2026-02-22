"""
PINOD network architectures.

Provides specialized encoder/decoder variants and a batch-level model
that processes multiple genes simultaneously with shared oscillator priors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from chord.pinod.oscillator import PhysicsInformedODE


class GeneEncoder(nn.Module):
    """Encode a single gene's expression time series into ODE initial state.

    Supports optional batch normalisation and dropout for regularisation
    on short time series (N~25).
    """

    def __init__(
        self,
        n_timepoints: int,
        n_oscillators: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        state_dim = 2 * n_oscillators
        self.net = nn.Sequential(
            nn.Linear(n_timepoints, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_timepoints) -> (batch, 2*K)"""
        return self.net(x)


class GeneDecoder(nn.Module):
    """Decode ODE displacements to expression values.

    Maps K oscillator displacements at each time point to a scalar
    expression value. Learns per-oscillator mixing weights.
    """

    def __init__(self, n_oscillators: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_oscillators, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        """displacements: (..., K) -> (..., 1)"""
        return self.net(displacements)


class LinearDecoder(nn.Module):
    """Simple linear mixing of oscillator displacements.

    More interpretable than MLP decoder: output = sum_k w_k * z_k + bias.
    The weights w_k directly represent oscillator contribution strengths.
    """

    def __init__(self, n_oscillators: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_oscillators))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        """displacements: (..., K) -> (..., 1)"""
        return (displacements * self.weights).sum(dim=-1, keepdim=True) + self.bias

    def get_mixing_weights(self) -> torch.Tensor:
        """Return absolute mixing weights (interpretable as amplitudes)."""
        return self.weights.abs().detach()


class PINODSingleGene(nn.Module):
    """PINOD model for a single gene: Encoder -> ODE -> Decoder.

    This is the core unit. For batch processing of many genes,
    use PINODBatch which wraps this with shared ODE parameters.

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    n_oscillators : int
        Number of oscillators (K).
    period_inits : list of float
        Initial period guesses in hours.
    gamma_init : float
        Initial damping rate.
    encoder_hidden : int
        Encoder hidden dimension.
    ode_hidden : int
        ODE correction MLP hidden dimension.
    decoder_type : str
        'linear' for interpretable linear mixing, 'mlp' for nonlinear.
    decoder_hidden : int
        Decoder hidden dimension (only for 'mlp').
    learn_periods : bool
        Whether to optimise periods.
    dropout : float
        Encoder dropout rate.
    """

    def __init__(
        self,
        n_timepoints: int = 24,
        n_oscillators: int = 3,
        period_inits: Optional[List[float]] = None,
        gamma_init: float = 0.01,
        encoder_hidden: int = 32,
        ode_hidden: int = 16,
        decoder_type: str = "linear",
        decoder_hidden: int = 32,
        learn_periods: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if period_inits is None:
            period_inits = [24.0, 12.0, 8.0]

        self.K = n_oscillators
        self.n_timepoints = n_timepoints

        self.encoder = GeneEncoder(
            n_timepoints, n_oscillators, encoder_hidden, dropout
        )
        self.ode = PhysicsInformedODE(
            n_oscillators, period_inits, gamma_init, ode_hidden, learn_periods
        )
        if decoder_type == "linear":
            self.decoder = LinearDecoder(n_oscillators)
        else:
            self.decoder = GeneDecoder(n_oscillators, decoder_hidden)

        self.baseline = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        expr: torch.Tensor,
        t_eval: torch.Tensor,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        expr : (batch, n_timepoints)
        t_eval : (n_timepoints,)

        Returns
        -------
        expr_hat : (batch, n_timepoints) — reconstructed expression
        trajectories : (T, batch, 2*K) — full ODE state
        amplitudes : (batch, K) — per-oscillator RMS amplitudes
        """
        from torchdiffeq import odeint

        z0 = self.encoder(expr)

        trajectories = odeint(
            self.ode, z0, t_eval,
            method=solver, rtol=rtol, atol=atol,
        )

        displacements = trajectories[:, :, 0::2]  # (T, batch, K)
        expr_hat = self.decoder(displacements).squeeze(-1)  # (T, batch)
        expr_hat = expr_hat.transpose(0, 1) + self.baseline  # (batch, T)

        amplitudes = torch.sqrt((displacements ** 2).mean(dim=0))  # (batch, K)

        return expr_hat, trajectories, amplitudes

    def get_parameters_dict(self) -> dict:
        """Extract interpretable parameters."""
        result = self.ode.get_parameters_dict() if hasattr(self.ode, 'get_parameters_dict') else {}
        with torch.no_grad():
            result["periods"] = self.ode.periods.cpu().numpy().tolist()
            result["gammas"] = self.ode.gamma.cpu().numpy().tolist()
            result["baseline"] = float(self.baseline.item())
            if isinstance(self.decoder, LinearDecoder):
                result["mixing_weights"] = self.decoder.get_mixing_weights().cpu().numpy().tolist()
        return result
