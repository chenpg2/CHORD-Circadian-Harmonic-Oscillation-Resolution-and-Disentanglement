"""
VMD — Variational Mode Decomposition for circadian/ultradian rhythm analysis.

Implements the ADMM-based VMD algorithm (Dragomiretskiy & Zosso, 2014)
adapted for extracting BHDT evidence from time-series gene expression data.

References
----------
Dragomiretskiy, K., & Zosso, D. (2014). Variational Mode Decomposition.
IEEE Transactions on Signal Processing, 62(3), 531-544.
"""

import numpy as np


def vmd_decompose(t, y, K=3, alpha=2000, tau=0.0, tol=1e-7,
                  max_iter=500, init_periods=None):
    """Variational Mode Decomposition via ADMM.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    y : array-like
        Signal values.
    K : int
        Number of modes to extract.
    alpha : float
        Bandwidth constraint (higher = narrower bands).
    tau : float
        Noise tolerance for dual variable update (0 = exact reconstruction).
    tol : float
        Convergence tolerance on center frequency change.
    max_iter : int
        Maximum ADMM iterations.
    init_periods : list of float, optional
        Initial center periods in hours. Default [24.0, 12.0, 8.0].

    Returns
    -------
    dict with keys:
        modes : ndarray (K, N) — extracted modes in time domain
        center_frequencies : ndarray (K,) — in cycles/hour
        center_periods : ndarray (K,) — in hours
        mode_energies : ndarray (K,) — energy of each mode
        mode_amplitudes : ndarray (K,) — RMS amplitude of each mode
        n_iterations : int — iterations until convergence
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = len(y)

    if init_periods is None:
        init_periods = [24.0, 12.0, 8.0]

    # --- Mirror extension to reduce boundary effects ---
    half = N // 2
    y_mirror = np.concatenate([y[:half][::-1], y, y[half:][::-1]])
    T = len(y_mirror)

    # --- Frequency axis for the mirrored signal ---
    # Normalized frequencies: 0 to 0.5 (in cycles per sample)
    freqs = np.arange(T) / float(T)  # [0, 1/T, 2/T, ..., (T-1)/T]

    # Convert to cycles/hour using sampling interval
    dt = np.median(np.diff(t))
    fs = 1.0 / dt  # samples per hour

    # FFT of the mirrored signal
    f_hat = np.fft.fft(y_mirror)

    # --- Initialize center frequencies (in normalized freq: cycles/sample) ---
    omega = np.zeros(K)
    for k in range(K):
        if k < len(init_periods):
            # Convert period (hours) -> freq (cycles/hour) -> normalized (cycles/sample)
            omega[k] = (1.0 / init_periods[k]) / fs
        else:
            omega[k] = (0.5 / K) * (k + 1)

    # --- ADMM variables ---
    u_hat = np.zeros((K, T), dtype=complex)
    lambda_hat = np.zeros(T, dtype=complex)

    n_iter = 0
    for iteration in range(max_iter):
        omega_prev = omega.copy()

        # --- Update each mode ---
        for k in range(K):
            # Sum of all other modes
            sum_other = np.sum(u_hat, axis=0) - u_hat[k]

            # Wiener filter update
            numerator = f_hat - sum_other + lambda_hat / 2.0
            denominator = 1.0 + alpha * (freqs - omega[k]) ** 2
            u_hat[k] = numerator / denominator

        # --- Update center frequencies (center of gravity on positive freqs) ---
        for k in range(K):
            # Use only positive frequencies (0 to 0.5 in normalized)
            pos_mask = (freqs > 0) & (freqs <= 0.5)
            power = np.abs(u_hat[k, pos_mask]) ** 2
            total_power = np.sum(power)
            if total_power > 1e-20:
                omega[k] = np.sum(freqs[pos_mask] * power) / total_power

        # --- Update dual variable ---
        residual = f_hat - np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * residual

        # --- Check convergence ---
        n_iter = iteration + 1
        max_change = np.max(np.abs(omega - omega_prev))
        if max_change < tol:
            break

    # --- Extract modes in time domain (middle N samples) ---
    modes = np.zeros((K, N))
    for k in range(K):
        mode_full = np.real(np.fft.ifft(u_hat[k]))
        modes[k] = mode_full[half:half + N]

    # --- Convert center frequencies to physical units ---
    center_freqs_hz = omega * fs  # cycles per hour
    center_periods = np.where(center_freqs_hz > 1e-12,
                              1.0 / center_freqs_hz,
                              np.inf)

    # --- Mode energies and amplitudes ---
    mode_energies = np.sum(modes ** 2, axis=1)
    mode_amplitudes = np.sqrt(np.mean(modes ** 2, axis=1))

    return {
        "modes": modes,
        "center_frequencies": center_freqs_hz,
        "center_periods": center_periods,
        "mode_energies": mode_energies,
        "mode_amplitudes": mode_amplitudes,
        "n_iterations": n_iter,
    }


def vmd_evidence(t, y, T_base=24.0, K=3, alpha=2000):
    """Extract BHDT evidence from VMD decomposition.

    Runs VMD and scores how likely the 12h component is an independent
    oscillator vs. a harmonic of the circadian rhythm.

    Parameters
    ----------
    t : array-like
        Time points (hours).
    y : array-like
        Signal values.
    T_base : float
        Base circadian period (default 24.0 hours).
    K : int
        Number of VMD modes.
    alpha : float
        VMD bandwidth parameter.

    Returns
    -------
    dict with keys:
        vmd_score : int — positive favors independent, negative favors harmonic
        energy_ratio_12_24 : float — E_12 / E_24
        T_12_vmd : float — estimated 12h period from VMD
        T_24_vmd : float — estimated 24h period from VMD
        freq_deviation : float — |T_12 - T_base/2| / (T_base/2)
        harmonic_lock : float — |T_12 - T_24/2| / (T_24/2)
        vmd_result : dict — full VMD decomposition result
    """
    init_periods = [T_base, T_base / 2.0, T_base / 3.0]
    vmd_result = vmd_decompose(t, y, K=K, alpha=alpha, init_periods=init_periods)

    periods = vmd_result["center_periods"]
    energies = vmd_result["mode_energies"]

    # Find modes closest to 24h and 12h
    idx_24 = int(np.argmin([abs(p - T_base) for p in periods]))
    idx_12 = int(np.argmin([abs(p - T_base / 2.0) for p in periods]))

    # Avoid picking the same mode for both
    if idx_24 == idx_12:
        dists_24 = [abs(p - T_base) for p in periods]
        dists_12 = [abs(p - T_base / 2.0) for p in periods]
        # Assign to whichever is closer, then pick second-best for the other
        if dists_24[idx_24] <= dists_12[idx_12]:
            dists_12[idx_24] = np.inf
            idx_12 = int(np.argmin(dists_12))
        else:
            dists_24[idx_12] = np.inf
            idx_24 = int(np.argmin(dists_24))

    T_24_vmd = periods[idx_24]
    T_12_vmd = periods[idx_12]
    E_24 = energies[idx_24]
    E_12 = energies[idx_12]

    # --- Compute metrics ---
    energy_ratio = E_12 / max(E_24, 1e-12)

    # How much T_12 deviates from the expected T_base/2
    freq_deviation = abs(T_12_vmd - T_base / 2.0) / (T_base / 2.0)

    # How much T_12 deviates from T_24/2 (harmonic lock)
    harmonic_lock = abs(T_12_vmd - T_24_vmd / 2.0) / max(T_24_vmd / 2.0, 1e-12)

    # --- Scoring ---
    score = 0

    # Energy ratio evidence
    if energy_ratio > 0.3:
        score += 1
    if energy_ratio > 0.6:
        score += 1

    # Frequency deviation from expected harmonic
    if freq_deviation > 0.02:
        score += 1

    # Harmonic lock deviation
    if harmonic_lock > 0.03:
        score += 1

    # Counter-evidence: very weak 12h mode
    if energy_ratio < 0.1:
        score -= 1

    # Counter-evidence: locked to harmonic AND weak
    if harmonic_lock < 0.02 and energy_ratio < 0.3:
        score -= 1

    return {
        "vmd_score": score,
        "energy_ratio_12_24": energy_ratio,
        "T_12_vmd": T_12_vmd,
        "T_24_vmd": T_24_vmd,
        "freq_deviation": freq_deviation,
        "harmonic_lock": harmonic_lock,
        "vmd_result": vmd_result,
    }
