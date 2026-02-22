"""
PINOD: Physics-Informed Neural Oscillator Decomposition.

Core ODE module implementing damped harmonic oscillators with learnable
nonlinear correction terms. Each oscillator follows:

    d²x_k/dt² + 2*gamma_k * dx_k/dt + omega_k² * x_k = f_theta_k(z, t)

where gamma_k is the damping rate, omega_k is the angular frequency,
and f_theta_k is a small MLP learning nonlinear corrections.

The system is reformulated as a first-order ODE for torchdiffeq:
    dz_k/dt = v_k
    dv_k/dt = -2*gamma_k*v_k - omega_k²*z_k + f_theta_k(z, t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class NonlinearCorrection(nn.Module):
    """Small MLP learning nonlinear correction f_theta_k for one oscillator.

    Inputs: full latent state z (all oscillators) + time t.
    Output: scalar correction force for this oscillator.
    """

    def __init__(self, n_oscillators: int, hidden_dim: int = 16):
        super().__init__()
        # Input: 2*K (position + velocity for each oscillator) + 1 (time)
        input_dim = 2 * n_oscillators + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize near zero so the model starts close to linear oscillator
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (batch, 2*K) — [z_1, v_1, z_2, v_2, ...]
        t : scalar or (batch,) — current time

        Returns
        -------
        (batch, 1) correction force
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(state.shape[0])
        inp = torch.cat([state, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


class PhysicsInformedODE(nn.Module):
    """Multi-oscillator ODE system with physics priors.

    State vector layout: [z_1, v_1, z_2, v_2, ..., z_K, v_K]
    where z_k is displacement and v_k is velocity of oscillator k.

    Parameters
    ----------
    n_oscillators : int
        Number of oscillators (K). Default 3 for {24h, 12h, 8h}.
    period_inits : list of float
        Initial period guesses in hours. Default [24, 12, 8].
    gamma_init : float
        Initial damping rate for all oscillators.
    hidden_dim : int
        Hidden dimension of nonlinear correction MLPs.
    learn_periods : bool
        Whether to learn omega_k (True) or keep fixed (False).
    """

    def __init__(
        self,
        n_oscillators: int = 3,
        period_inits: Optional[List[float]] = None,
        gamma_init: float = 0.01,
        hidden_dim: int = 16,
        learn_periods: bool = True,
    ):
        super().__init__()
        self.K = n_oscillators

        if period_inits is None:
            period_inits = [24.0, 12.0, 8.0]
        assert len(period_inits) == n_oscillators

        # Parameterise omega via log to ensure positivity
        omega_inits = [2.0 * np.pi / T for T in period_inits]
        self.log_omega = nn.Parameter(
            torch.tensor([np.log(w) for w in omega_inits], dtype=torch.float32)
        )
        if not learn_periods:
            self.log_omega.requires_grad_(False)

        # Damping rates (log-parameterised for positivity)
        self.log_gamma = nn.Parameter(
            torch.full((n_oscillators,), np.log(max(gamma_init, 1e-6)),
                        dtype=torch.float32)
        )

        # Nonlinear correction MLPs — one per oscillator
        self.corrections = nn.ModuleList([
            NonlinearCorrection(n_oscillators, hidden_dim)
            for _ in range(n_oscillators)
        ])

    @property
    def omega(self) -> torch.Tensor:
        """Angular frequencies (rad/h)."""
        return torch.exp(self.log_omega)

    @property
    def gamma(self) -> torch.Tensor:
        """Damping rates (1/h)."""
        return torch.exp(self.log_gamma)

    @property
    def periods(self) -> torch.Tensor:
        """Periods in hours."""
        return 2.0 * np.pi / self.omega

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side: dstate/dt.

        Parameters
        ----------
        t : scalar tensor — current time
        state : (batch, 2*K) — [z_1, v_1, z_2, v_2, ...]

        Returns
        -------
        dstate_dt : (batch, 2*K)
        """
        omega = self.omega
        gamma = self.gamma
        K = self.K

        dstate = torch.zeros_like(state)
        for k in range(K):
            z_k = state[:, 2 * k]       # displacement
            v_k = state[:, 2 * k + 1]   # velocity

            # dz_k/dt = v_k
            dstate[:, 2 * k] = v_k

            # dv_k/dt = -2*gamma_k*v_k - omega_k^2*z_k + f_theta_k(state, t)
            linear_force = -2.0 * gamma[k] * v_k - omega[k] ** 2 * z_k
            correction = self.corrections[k](state, t).squeeze(-1)
            dstate[:, 2 * k + 1] = linear_force + correction

        return dstate

    def physics_regularisation(self) -> torch.Tensor:
        """L2 norm of all nonlinear correction parameters.

        Encourages the model to explain data with linear oscillators
        before resorting to nonlinear corrections.
        """
        reg = torch.tensor(0.0, device=self.log_omega.device)
        for corr in self.corrections:
            for p in corr.parameters():
                reg = reg + p.pow(2).sum()
        return reg

    def sparsity_regularisation(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """L1 penalty on oscillator amplitudes to encourage parsimony.

        Parameters
        ----------
        amplitudes : (K,) or (batch, K) — estimated amplitudes per oscillator
        """
        return amplitudes.abs().sum()


class PINODModel(nn.Module):
    """Complete PINOD model: Encoder -> ODE -> Decoder.

    Encodes a gene expression time series into initial ODE state,
    integrates the physics-informed ODE, and decodes back to expression.

    Parameters
    ----------
    n_timepoints : int
        Number of time points in the input.
    n_oscillators : int
        Number of oscillators.
    period_inits : list of float
        Initial period guesses.
    encoder_hidden : int
        Encoder hidden dimension.
    decoder_hidden : int
        Decoder hidden dimension.
    ode_hidden : int
        ODE nonlinear correction hidden dimension.
    gamma_init : float
        Initial damping rate.
    learn_periods : bool
        Whether to learn periods.
    """

    def __init__(
        self,
        n_timepoints: int = 24,
        n_oscillators: int = 3,
        period_inits: Optional[List[float]] = None,
        encoder_hidden: int = 32,
        decoder_hidden: int = 32,
        ode_hidden: int = 16,
        gamma_init: float = 0.01,
        learn_periods: bool = True,
    ):
        super().__init__()
        self.K = n_oscillators
        self.n_timepoints = n_timepoints
        state_dim = 2 * n_oscillators  # z_k + v_k for each

        # Encoder: expression vector -> initial ODE state
        self.encoder = nn.Sequential(
            nn.Linear(n_timepoints, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, state_dim),
        )

        # Physics-informed ODE
        self.ode = PhysicsInformedODE(
            n_oscillators=n_oscillators,
            period_inits=period_inits,
            gamma_init=gamma_init,
            hidden_dim=ode_hidden,
            learn_periods=learn_periods,
        )

        # Decoder: ODE state at each time -> expression value
        # Maps K displacements to a scalar expression value
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )

        # Learnable baseline (mesor)
        self.baseline = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        expr: torch.Tensor,
        t_eval: torch.Tensor,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode -> integrate ODE -> decode.

        Parameters
        ----------
        expr : (batch, n_timepoints) — input expression
        t_eval : (n_timepoints,) — time points for ODE integration
        solver : str — ODE solver name
        rtol, atol : float — solver tolerances

        Returns
        -------
        expr_hat : (batch, n_timepoints) — reconstructed expression
        trajectories : (n_timepoints, batch, 2*K) — full ODE trajectories
        """
        from torchdiffeq import odeint

        batch_size = expr.shape[0]

        # Encode initial state
        z0 = self.encoder(expr)  # (batch, 2*K)

        # Integrate ODE
        # odeint expects (T, batch, state_dim)
        trajectories = odeint(
            self.ode, z0, t_eval,
            method=solver, rtol=rtol, atol=atol,
        )  # (T, batch, 2*K)

        # Extract displacements (skip velocities)
        displacements = trajectories[:, :, 0::2]  # (T, batch, K)

        # Decode each time point
        expr_hat = self.decoder(displacements).squeeze(-1)  # (T, batch)
        expr_hat = expr_hat.transpose(0, 1) + self.baseline  # (batch, T)

        return expr_hat, trajectories

    def get_oscillator_amplitudes(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Estimate amplitude of each oscillator from trajectories.

        Parameters
        ----------
        trajectories : (T, batch, 2*K)

        Returns
        -------
        amplitudes : (batch, K) — RMS amplitude per oscillator
        """
        displacements = trajectories[:, :, 0::2]  # (T, batch, K)
        # RMS amplitude
        return torch.sqrt((displacements ** 2).mean(dim=0))

    def get_parameters_dict(self) -> dict:
        """Extract interpretable oscillator parameters."""
        with torch.no_grad():
            periods = self.ode.periods.cpu().numpy()
            gammas = self.ode.gamma.cpu().numpy()
            omegas = self.ode.omega.cpu().numpy()
        return {
            "periods": periods.tolist(),
            "gammas": gammas.tolist(),
            "omegas": omegas.tolist(),
            "baseline": float(self.baseline.item()),
            "n_oscillators": self.K,
        }
