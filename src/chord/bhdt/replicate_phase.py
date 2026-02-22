"""
Replicate Phase Consistency Test.

Tests whether the 12h-to-24h phase relationship is consistent across
biological replicates, providing evidence for harmonic vs independent
classification at the single-gene level.

Key insight:
  - Harmonic: phi_12 = 2*phi_24 + const  (waveform shape is fixed)
    => residual phase psi = phi_12 - 2*phi_24 is consistent across
       replicates (low circular variance).
  - Independent: phi_12 is independent of phi_24
    => residual phase psi varies randomly (high circular variance).

Circular statistics on the residual phases discriminate these scenarios.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Phase fitting
# ---------------------------------------------------------------------------

def _fit_phase(t, y, period):
    """Fit phase of a single sinusoidal component.

    Uses projection onto cos/sin basis:
        y ~ a*cos(2*pi*t/T) + b*sin(2*pi*t/T) + c

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    period : float
        Period of the component to fit.

    Returns
    -------
    float
        Phase in radians [0, 2*pi).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    omega = 2.0 * np.pi / period
    cos_comp = np.cos(omega * t)
    sin_comp = np.sin(omega * t)

    # Design matrix: [intercept, cos, sin]
    X = np.column_stack([np.ones(len(t)), cos_comp, sin_comp])
    # OLS solve
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    a = beta[1]  # cos coefficient
    b = beta[2]  # sin coefficient

    # phase = arctan2(-sin_comp, cos_comp) mod 2*pi
    # Convention: y ~ A*cos(omega*t - phi)
    #   = A*cos(phi)*cos(omega*t) + A*sin(phi)*sin(omega*t)
    # So a = A*cos(phi), b = A*sin(phi) => phi = arctan2(b, a)
    phase = np.arctan2(b, a) % (2.0 * np.pi)
    return float(phase)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def replicate_phase_consistency(t, replicates, T_base=24.0,
                                variance_threshold=0.4):
    """Test phase consistency across biological replicates.

    For each replicate, fits the 24h and 12h phase components, computes
    the residual phase psi = (phi_12 - 2*phi_24) mod 2*pi, then evaluates
    circular statistics on the collection of residual phases.

    Parameters
    ----------
    t : array-like
        Time points in hours (shared across replicates).
    replicates : list of array-like
        Expression values for each replicate.
    T_base : float
        Base circadian period (default 24.0).
    variance_threshold : float
        Circular variance threshold for harmonic classification.

    Returns
    -------
    dict with keys:
        circular_variance : float (0 = consistent, 1 = random)
        mean_resultant_length : float (R value)
        residual_phases : list of float
        is_harmonic : bool
        consistency_score : float (negative = harmonic, positive = independent)
            V < 0.2 -> -2, V < threshold -> -1, V > 0.7 -> +2, V > 0.5 -> +1,
            else -> 0
    """
    t = np.asarray(t, dtype=float)
    n_rep = len(replicates)

    T_half = T_base / 2.0

    # --- Edge case: single replicate ---
    if n_rep < 2:
        psi_list = []
        for rep in replicates:
            y = np.asarray(rep, dtype=float)
            phi_24 = _fit_phase(t, y, T_base)
            phi_12 = _fit_phase(t, y, T_half)
            psi = (phi_12 - 2.0 * phi_24) % (2.0 * np.pi)
            psi_list.append(float(psi))
        return {
            "circular_variance": 0.5,
            "mean_resultant_length": 0.5,
            "residual_phases": psi_list,
            "is_harmonic": False,
            "consistency_score": 0,
        }

    # --- Compute residual phases for each replicate ---
    psi_list = []
    for rep in replicates:
        y = np.asarray(rep, dtype=float)
        phi_24 = _fit_phase(t, y, T_base)
        phi_12 = _fit_phase(t, y, T_half)
        psi = (phi_12 - 2.0 * phi_24) % (2.0 * np.pi)
        psi_list.append(float(psi))

    psi_arr = np.array(psi_list)

    # --- Circular statistics ---
    C = np.mean(np.cos(psi_arr))
    S = np.mean(np.sin(psi_arr))
    R = np.sqrt(C ** 2 + S ** 2)
    V = 1.0 - R  # circular variance

    # --- Consistency score ---
    if V < 0.2:
        score = -2
    elif V < variance_threshold:
        score = -1
    elif V > 0.7:
        score = 2
    elif V > 0.5:
        score = 1
    else:
        score = 0

    is_harmonic = bool(V < variance_threshold)

    return {
        "circular_variance": float(V),
        "mean_resultant_length": float(R),
        "residual_phases": psi_list,
        "is_harmonic": is_harmonic,
        "consistency_score": score,
    }
