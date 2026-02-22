"""
Hilbert Instantaneous Frequency — VMD-Hilbert disentanglement pipeline.

Uses the Hilbert transform to compute instantaneous frequency (IF) of VMD modes,
then measures IF coupling to distinguish true independent 12h oscillators from
circadian harmonics.

Key insight: if IF_12(t) ≈ 2·IF_24(t) at all times → harmonic.
If IF_12(t) drifts independently → independent oscillator.

References
----------
Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert
spectrum for nonlinear and non-stationary time series analysis.
Proc. R. Soc. Lond. A, 454, 903-995.
"""

import numpy as np
from scipy.signal import hilbert

from chord.bhdt.vmd import vmd_decompose


def hilbert_instantaneous_frequency(t, y):
    """Compute instantaneous frequency via Hilbert transform.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    y : array-like
        Signal values.

    Returns
    -------
    if_vals : ndarray
        Instantaneous frequency at each time point (cycles/hour).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Analytic signal: z(t) = y(t) + j·H[y(t)]
    z = hilbert(y)

    # Instantaneous phase: φ(t) = unwrap(arg(z(t)))
    inst_phase = np.unwrap(np.angle(z))

    # Instantaneous frequency: IF(t) = (1/2π) · dφ/dt
    dt = np.gradient(t)
    dphi = np.gradient(inst_phase)
    if_vals = dphi / (2.0 * np.pi * dt)

    # Clip to [0, Nyquist]
    dt_median = np.median(np.diff(t))
    nyquist = 0.5 / dt_median
    if_vals = np.clip(if_vals, 0.0, nyquist)

    return if_vals


def instantaneous_frequency_coupling(t, mode_24, mode_12, correlation_threshold=0.6):
    """Measure IF coupling between 24h and 12h modes.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    mode_24 : array-like
        24h mode time series.
    mode_12 : array-like
        12h mode time series.
    correlation_threshold : float
        Threshold for classifying as harmonic (default 0.6).

    Returns
    -------
    dict with keys:
        if_correlation : float — Pearson correlation between IF_12 and 2*IF_24
        if_ratio_mean : float — mean of IF_12 / IF_24
        if_ratio_std : float — std of IF_12 / IF_24
        is_harmonic : bool — True if coupling indicates harmonic relationship
        coupling_score : int — negative = harmonic, positive = independent
    """
    t = np.asarray(t, dtype=np.float64)
    mode_24 = np.asarray(mode_24, dtype=np.float64)
    mode_12 = np.asarray(mode_12, dtype=np.float64)

    # Compute IF for both modes
    if_24 = hilbert_instantaneous_frequency(t, mode_24)
    if_12 = hilbert_instantaneous_frequency(t, mode_12)

    # Trim edges to avoid Hilbert edge effects
    N = len(t)
    trim = max(2, int(0.1 * N))
    if_24_trimmed = if_24[trim:-trim]
    if_12_trimmed = if_12[trim:-trim]

    # Compute Pearson correlation between IF_12 and 2*IF_24
    if_24_doubled = 2.0 * if_24_trimmed
    # Handle degenerate case: nearly-constant IF (pure cosines)
    # Use coefficient of variation to detect near-constant signals
    mean_12 = np.mean(if_12_trimmed)
    mean_24d = np.mean(if_24_doubled)
    std_12 = np.std(if_12_trimmed)
    std_24d = np.std(if_24_doubled)
    cv_12 = std_12 / max(abs(mean_12), 1e-12)
    cv_24d = std_24d / max(abs(mean_24d), 1e-12)

    if cv_12 < 0.01 and cv_24d < 0.01:
        # Both IFs are near-constant → pure sinusoids.
        # Correlation is undefined (zero variance). The IF ratio alone
        # cannot distinguish harmonic from independent because BOTH
        # produce near-constant IF at their respective frequencies.
        # Return 0 (inconclusive) to let other evidence lines decide.
        corr = 0.0
    else:
        corr = float(np.corrcoef(if_12_trimmed, if_24_doubled)[0, 1])
        if np.isnan(corr):
            corr = 0.0

    # IF ratio statistics
    valid_mask = if_24_trimmed > 1e-12
    if np.sum(valid_mask) > 2:
        ratios = if_12_trimmed[valid_mask] / if_24_trimmed[valid_mask]
        ratio_mean = float(np.mean(ratios))
        ratio_std = float(np.std(ratios))
    else:
        ratio_mean = 0.0
        ratio_std = 999.0

    # Coefficient of variation of the ratio — normalized measure of IF drift
    ratio_cv = ratio_std / max(abs(ratio_mean), 1e-12)

    # --- Scoring ---
    # Negative = harmonic, positive = independent
    # For short signals (small N), correlation is unreliable due to Hilbert
    # edge effects, so we weight ratio statistics more heavily.
    is_short = len(if_24_trimmed) < 30

    if corr > max(0.8, correlation_threshold + 0.2) and abs(ratio_mean - 2.0) < 0.15:
        score = -3  # strong harmonic
    elif corr > correlation_threshold and abs(ratio_mean - 2.0) < 0.3:
        score = -2  # moderate harmonic
    elif is_short and abs(ratio_mean - 2.0) < 0.2 and ratio_cv < 0.35:
        # Short signal: correlation unreliable, but ratio is close to 2
        # with moderate variability → likely harmonic
        score = -2  # moderate harmonic
    elif is_short and abs(ratio_mean - 2.0) < 0.3 and ratio_cv < 0.5:
        score = -1  # weak harmonic
    elif not is_short and (corr < 0.2 or ratio_std > 0.5):
        score = 2   # strong independent
    elif abs(ratio_mean - 2.0) > 0.5:
        score = 2   # strong independent (ratio far from 2)
    elif ratio_cv > 0.5 and abs(ratio_mean - 2.0) > 0.3:
        score = 2   # strong independent
    elif corr < 0.4 and abs(ratio_mean - 2.0) > 0.2:
        score = 1   # moderate independent
    else:
        score = 0   # inconclusive

    is_harmonic = score < 0

    return {
        "if_correlation": corr,
        "if_ratio_mean": ratio_mean,
        "if_ratio_std": ratio_std,
        "is_harmonic": is_harmonic,
        "coupling_score": score,
    }


def vmd_hilbert_disentangle(t, y, T_base=24.0):
    """Full VMD + Hilbert disentanglement pipeline.

    Decomposes signal into 24h and 12h modes via VMD, then uses Hilbert
    instantaneous frequency coupling to classify the 12h component as
    harmonic or independent.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    y : array-like
        Signal values.
    T_base : float
        Base circadian period (default 24.0 hours).

    Returns
    -------
    dict with keys:
        classification_evidence : str — human-readable summary
        if_correlation : float
        if_ratio_mean : float
        if_ratio_std : float
        is_harmonic : bool
        mode_24_amplitude : float
        mode_12_amplitude : float
        vmd_converged : bool
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Run VMD decomposition
    vmd_result = vmd_decompose(
        t, y, K=2, alpha=2000,
        init_periods=[T_base, T_base / 2.0],
    )

    modes = vmd_result["modes"]
    periods = vmd_result["center_periods"]
    amplitudes = vmd_result["mode_amplitudes"]
    n_iter = vmd_result["n_iterations"]

    # Identify 24h and 12h modes by closest center_period
    idx_24 = int(np.argmin([abs(p - T_base) for p in periods]))
    idx_12 = int(np.argmin([abs(p - T_base / 2.0) for p in periods]))

    # Handle edge case: both modes converge to same frequency
    if idx_24 == idx_12:
        # Pick the one closer to T_base for 24h, other for 12h
        dists_24 = [abs(p - T_base) for p in periods]
        dists_12 = [abs(p - T_base / 2.0) for p in periods]
        if dists_24[idx_24] <= dists_12[idx_12]:
            dists_12[idx_24] = float("inf")
            idx_12 = int(np.argmin(dists_12))
        else:
            dists_24[idx_12] = float("inf")
            idx_24 = int(np.argmin(dists_24))

    # With K=2, if they still collide, return inconclusive
    if idx_24 == idx_12:
        return {
            "classification_evidence": "inconclusive: VMD modes converged to same frequency",
            "if_correlation": 0.0,
            "if_ratio_mean": 0.0,
            "if_ratio_std": 0.0,
            "is_harmonic": False,
            "mode_24_amplitude": float(amplitudes[0]),
            "mode_12_amplitude": float(amplitudes[1] if len(amplitudes) > 1 else 0.0),
            "vmd_converged": False,
        }

    mode_24 = modes[idx_24]
    mode_12 = modes[idx_12]
    amp_24 = float(amplitudes[idx_24])
    amp_12 = float(amplitudes[idx_12])
    T_12_vmd = float(periods[idx_12])

    # Check VMD convergence
    vmd_converged = n_iter < 500

    # VMD frequency deviation: how far is the 12h mode from exact T_base/2
    freq_deviation = abs(T_12_vmd - T_base / 2.0) / (T_base / 2.0)

    # Run IF coupling analysis
    coupling = instantaneous_frequency_coupling(t, mode_24, mode_12)

    # Combine IF coupling score with VMD frequency deviation
    # For short signals, VMD freq deviation is more reliable than IF correlation
    score = coupling["coupling_score"]
    corr = coupling["if_correlation"]
    ratio_mean = coupling["if_ratio_mean"]

    # VMD frequency deviation: only use as positive evidence for independence.
    # Do NOT penalize freq_deviation < 0.02 because both harmonic AND
    # independent 12h signals have VMD center frequency near 12.0h.
    # VMD center frequency reflects the dominant spectral peak, not causality.
    if freq_deviation > 0.05:
        # 12h mode period deviates >5% from T_base/2 → independent evidence
        if score < 0:
            score = max(score + 2, 1)  # override harmonic classification


    is_harmonic = score < 0

    if score <= -3:
        evidence = "strong harmonic"
    elif score <= -2:
        evidence = "moderate harmonic"
    elif score == -1:
        evidence = "weak harmonic"
    elif score >= 2:
        evidence = "strong independent"
    elif score >= 1:
        evidence = "moderate independent"
    else:
        evidence = "inconclusive"

    evidence += (f" (IF_corr={corr:.3f}, IF_ratio={ratio_mean:.3f}, "
                 f"freq_dev={freq_deviation:.3f}, "
                 f"amp_24={amp_24:.3f}, amp_12={amp_12:.3f})")

    return {
        "classification_evidence": evidence,
        "if_correlation": corr,
        "if_ratio_mean": ratio_mean,
        "if_ratio_std": coupling["if_ratio_std"],
        "is_harmonic": is_harmonic,
        "mode_24_amplitude": amp_24,
        "mode_12_amplitude": amp_12,
        "vmd_converged": vmd_converged,
    }
