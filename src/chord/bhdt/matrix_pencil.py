"""Matrix-pencil (Hua-Sarkar / LS-ESPRIT) decomposition of a sampled signal into
damped complex exponentials.

The pole solve is ordinary least-squares ESPRIT (pseudo-inverse of the shifted
right-singular subspace), not the total-least-squares variant. DC is removed before
fitting, so the modes describe the OSCILLATORY (mean-subtracted) part of the signal.

This is the estimator Zhu et al. 2017 used to recover per-component (frequency,
decay, complex amplitude) WITHOUT a pre-specified period window (principle P-0010),
enabling the mode-vector orthogonality criterion for harmonic-vs-independent
(P-0034). It replaces the period-fixed OLS front end for Stage-2 features.

Model:  y[n] ≈ Σ_i  a_i * z_i^n,   z_i = exp((−γ_i + j·2π f_i)·dt)
returning per mode: frequency f_i (cycles/time), decay γ_i, complex amplitude a_i.

Caveat (P-0035): noise-sensitive at small N; cap the model order and treat outputs
as estimates to be stabilised across replicates, not as exact truths.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

__all__ = ["Mode", "matrix_pencil"]


@dataclass(frozen=True)
class Mode:
    frequency: float      # cycles per unit time (>= 0 for the canonical conjugate)
    period: float         # 1 / frequency (inf if frequency ~ 0)
    decay: float          # gamma; >0 decaying, <0 growing
    amplitude: float      # |a_i| (real magnitude of the conjugate pair)
    phase: float          # arg(a_i) in radians


def matrix_pencil(y: np.ndarray, dt: float, model_order: int,
                  pencil_L: int | None = None) -> List[Mode]:
    """Decompose uniformly-sampled real y (step dt) into <= model_order modes.

    Returns physical (positive-frequency) modes sorted by descending amplitude.
    """
    if model_order < 1:
        raise ValueError(f"model_order must be >= 1, got {model_order}")
    if not (dt > 0):
        raise ValueError(f"dt must be > 0, got {dt}")
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 4:
        raise ValueError("matrix_pencil needs at least 4 samples")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite (no NaN/inf)")
    y = y - y.mean()  # remove DC; the pencil models the oscillatory part
    L = pencil_L if pencil_L is not None else n // 3
    L = int(np.clip(L, model_order, n - model_order - 1))
    if L < 2:
        raise ValueError("signal too short for the requested model order")

    # Hankel matrix Y: (n-L) x (L+1)
    hankel = np.array([y[i:i + L + 1] for i in range(n - L)])
    # SVD; keep the M dominant right-singular vectors
    _, s, vh = np.linalg.svd(hankel, full_matrices=False)
    m = int(min(model_order, np.sum(s > 1e-9 * s[0]) if s[0] > 0 else 1, vh.shape[0]))
    m = max(m, 1)
    v = vh[:m, :].conj().T              # (L+1) x m
    v1 = v[:-1, :]                       # L x m
    v2 = v[1:, :]                        # L x m
    # poles = eigenvalues of the pencil pinv(V1) V2
    z = np.linalg.eigvals(np.linalg.pinv(v1) @ v2)
    z = z[np.abs(z) > 0]

    # complex amplitudes via Vandermonde least squares  y[k] = Σ a_i z_i^k
    k = np.arange(n)
    vander = z[None, :] ** k[:, None]    # n x len(z)
    a, *_ = np.linalg.lstsq(vander, y, rcond=None)

    modes: List[Mode] = []
    seen = set()
    for zi, ai in zip(z, a):
        f = float(np.angle(zi) / (2 * np.pi * dt))   # cycles/time (signed)
        if f < -1e-9:
            continue  # keep only the non-negative member of each conjugate pair
        gamma = float(-np.log(max(abs(zi), 1e-300)) / dt)
        key = round(abs(f), 6)
        if key in seen:
            continue
        seen.add(key)
        period = float(1.0 / f) if abs(f) > 1e-9 else float("inf")
        # Double the amplitude only for a genuine conjugate pair, i.e. strictly
        # between DC and Nyquist. DC (f~0) and Nyquist (f~1/2dt) poles are real and
        # self-conjugate, so they are NOT doubled.
        nyq = 0.5 / dt
        is_pair = 1e-9 < f < nyq - 1e-9
        amp = float(2.0 * abs(ai)) if is_pair else float(abs(ai))
        modes.append(Mode(frequency=abs(f), period=period, decay=gamma,
                          amplitude=amp, phase=float(np.angle(ai))))
    modes.sort(key=lambda mm: mm.amplitude, reverse=True)
    return modes
