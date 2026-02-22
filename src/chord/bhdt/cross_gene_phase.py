"""
Cross-Gene Phase Distribution Test (CGPDT).

Tests whether 12h phases are coupled to 24h phases across a population of
genes, providing group-level evidence for harmonic vs independent classification.

Key insight:
  - If 12h components are harmonics of 24h, then phi_12 = 2*phi_24 + delta
    where delta is constant across genes.  The residual psi = phi_12 - 2*phi_24
    is CONCENTRATED around a single direction.
  - If 12h components are independent oscillators, psi is UNIFORMLY distributed
    on [0, 2*pi).

The Rayleigh test on the residual phases discriminates these two scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from chord.bhdt.models import _cos_sin_design, _fit_ols


# ---------------------------------------------------------------------------
# Phase extraction
# ---------------------------------------------------------------------------

def extract_phases(
    t: np.ndarray,
    y: np.ndarray,
    periods: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Extract phase and amplitude estimates for each period via cosinor regression.

    Parameters
    ----------
    t : array of shape (N,)
        Time points.
    y : array of shape (N,)
        Observed values.
    periods : list of float, optional
        Periods to fit (default [24.0, 12.0]).

    Returns
    -------
    dict with keys:
        'phases'     : dict mapping period -> phase in [0, 2*pi)
        'amplitudes' : dict mapping period -> amplitude (>= 0)
        'mesor'      : float, intercept
    """
    if periods is None:
        periods = [24.0, 12.0]

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    X = _cos_sin_design(t, periods)
    beta, rss, sigma2, log_lik = _fit_ols(X, y)

    phases = {}
    amplitudes = {}
    mesor = float(beta[0])

    for j, T in enumerate(periods):
        a = beta[1 + 2 * j]
        b = beta[2 + 2 * j]
        A = np.sqrt(a**2 + b**2)
        # arctan2(b, a) gives phase in (-pi, pi]; shift to [0, 2*pi)
        phi = np.arctan2(b, a) % (2.0 * np.pi)
        phases[T] = float(phi)
        amplitudes[T] = float(A)

    return {"phases": phases, "amplitudes": amplitudes, "mesor": mesor}


# ---------------------------------------------------------------------------
# Rayleigh test for circular uniformity
# ---------------------------------------------------------------------------

def rayleigh_test(angles: np.ndarray) -> Dict[str, float]:
    """Rayleigh test for circular uniformity.

    Parameters
    ----------
    angles : array of shape (n,)
        Circular observations in radians.

    Returns
    -------
    dict with keys:
        'R'              : mean resultant length in [0, 1]
        'Z'              : test statistic n * R^2
        'p_value'        : p-value (small => reject uniformity)
        'mean_direction' : circular mean direction in [0, 2*pi)
        'n'              : sample size
    """
    angles = np.asarray(angles, dtype=float)
    n = len(angles)

    if n == 0:
        return {"R": np.nan, "Z": np.nan, "p_value": np.nan,
                "mean_direction": np.nan, "n": 0}

    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    Z = n * R**2
    mean_dir = np.arctan2(S, C) % (2.0 * np.pi)

    # p-value approximation (Mardia & Jupp 2000)
    if n >= 10:
        # Higher-order correction
        p = np.exp(-Z) * (
            1.0
            + (2.0 * Z - Z**2) / (4.0 * n)
            - (24.0 * Z - 132.0 * Z**2 + 76.0 * Z**3 - 9.0 * Z**4)
            / (288.0 * n**2)
        )
    else:
        # Simple exponential approximation for small n
        p = np.exp(-Z)

    # Clamp to [0, 1]
    p = float(np.clip(p, 0.0, 1.0))

    return {"R": float(R), "Z": float(Z), "p_value": p,
            "mean_direction": float(mean_dir), "n": n}


# ---------------------------------------------------------------------------
# Cross-gene phase distribution test
# ---------------------------------------------------------------------------

def cross_gene_phase_test(
    phi_24_array: np.ndarray,
    phi_12_array: np.ndarray,
) -> Dict[str, object]:
    """Test whether 12h phases are coupled to 24h phases across genes.

    Computes residual phases psi_g = (phi_12_g - 2*phi_24_g) mod 2*pi
    and runs a Rayleigh test on the residuals.

    Parameters
    ----------
    phi_24_array : array of shape (n_genes,)
        24h phase estimates per gene (radians).
    phi_12_array : array of shape (n_genes,)
        12h phase estimates per gene (radians).

    Returns
    -------
    dict with keys:
        'residual_phases' : array of residual phases psi
        'rayleigh'        : dict from rayleigh_test
        'classification'  : 'harmonic' | 'independent' | 'inconclusive'
        'evidence_score'  : int, negative = harmonic, positive = independent
        'n_genes'         : int
    """
    phi_24 = np.asarray(phi_24_array, dtype=float)
    phi_12 = np.asarray(phi_12_array, dtype=float)
    n = len(phi_24)

    if n != len(phi_12):
        raise ValueError("phi_24_array and phi_12_array must have the same length")

    if n < 5:
        return {
            "residual_phases": np.array([]),
            "rayleigh": {"R": np.nan, "Z": np.nan, "p_value": np.nan,
                         "mean_direction": np.nan, "n": n},
            "classification": "inconclusive",
            "evidence_score": 0,
            "n_genes": n,
        }

    # Residual phase: psi = (phi_12 - 2*phi_24) mod 2*pi
    psi = (phi_12 - 2.0 * phi_24) % (2.0 * np.pi)

    ray = rayleigh_test(psi)
    score = _evidence_score_from_p(ray["p_value"])

    if ray["p_value"] < 0.05:
        classification = "harmonic"
    elif ray["p_value"] > 0.2:
        classification = "independent"
    else:
        classification = "inconclusive"

    return {
        "residual_phases": psi,
        "rayleigh": ray,
        "classification": classification,
        "evidence_score": score,
        "n_genes": n,
    }


# ---------------------------------------------------------------------------
# Evidence score wrapper
# ---------------------------------------------------------------------------

def _evidence_score_from_p(p_rayleigh: float) -> int:
    """Map Rayleigh p-value to an integer evidence score.

    Negative scores favour harmonic, positive favour independent.
    """
    if np.isnan(p_rayleigh):
        return 0
    if p_rayleigh < 0.001:
        return -3  # very strong harmonic
    if p_rayleigh < 0.01:
        return -2
    if p_rayleigh < 0.05:
        return -1
    if p_rayleigh > 0.5:
        return 2   # strong independent
    if p_rayleigh > 0.2:
        return 1
    return 0


def cross_gene_phase_evidence(
    phi_24_array: np.ndarray,
    phi_12_array: np.ndarray,
) -> Dict[str, object]:
    """Convenience wrapper returning the evidence score and classification.

    Parameters
    ----------
    phi_24_array, phi_12_array : arrays of shape (n_genes,)
        Phase estimates per gene (radians).

    Returns
    -------
    dict with keys:
        'evidence_score'  : int
        'classification'  : str
        'p_value'         : float
        'R'               : float (mean resultant length)
        'mean_direction'  : float
        'n_genes'         : int
    """
    result = cross_gene_phase_test(phi_24_array, phi_12_array)
    ray = result["rayleigh"]
    return {
        "evidence_score": result["evidence_score"],
        "classification": result["classification"],
        "p_value": ray["p_value"],
        "R": ray["R"],
        "mean_direction": ray["mean_direction"],
        "n_genes": result["n_genes"],
    }


# ---------------------------------------------------------------------------
# Batch phase extraction
# ---------------------------------------------------------------------------

def batch_extract_phases(
    t: np.ndarray,
    Y_matrix: np.ndarray,
    periods: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """Extract phases for all genes in a matrix.

    Parameters
    ----------
    t : array of shape (N_timepoints,)
        Time points.
    Y_matrix : array of shape (n_genes, N_timepoints)
        Expression matrix (genes x timepoints).
    periods : list of float, optional
        Periods to fit (default [24.0, 12.0]).

    Returns
    -------
    dict with keys:
        'phi_24'  : array of shape (n_genes,), 24h phases
        'phi_12'  : array of shape (n_genes,), 12h phases
        'amp_24'  : array of shape (n_genes,), 24h amplitudes
        'amp_12'  : array of shape (n_genes,), 12h amplitudes
    """
    if periods is None:
        periods = [24.0, 12.0]

    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y_matrix, dtype=float)

    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n_genes, n_tp = Y.shape
    if n_tp != len(t):
        raise ValueError(
            f"Y_matrix has {n_tp} columns but t has {len(t)} elements"
        )

    # Vectorised: build design matrix once, solve for all genes at once
    X = _cos_sin_design(t, periods)  # (N, 1 + 2*K)
    # Solve X @ B = Y^T  =>  B = (X^T X)^{-1} X^T Y^T
    # B has shape (1+2K, n_genes)
    B, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)

    result = {}
    for j, T in enumerate(periods):
        a = B[1 + 2 * j, :]  # (n_genes,)
        b = B[2 + 2 * j, :]  # (n_genes,)
        amp = np.sqrt(a**2 + b**2)
        phi = np.arctan2(b, a) % (2.0 * np.pi)

        key_phi = f"phi_{int(T)}" if T == int(T) else f"phi_{T}"
        key_amp = f"amp_{int(T)}" if T == int(T) else f"amp_{T}"
        result[key_phi] = phi
        result[key_amp] = amp

    return result
