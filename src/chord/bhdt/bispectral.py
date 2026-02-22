"""
Bispectral Harmonic Coupling Test (BHCT) for BHDT.

Detects quadratic phase coupling (QPC) using the bispectrum to distinguish
harmonic artifacts from independent oscillations at the same frequency.

The power spectrum (2nd-order statistic) cannot distinguish harmonics from
independent oscillations. The bispectrum (3rd-order statistic) CAN:

- Harmonics (nonlinear waveform distortion): The 12h component is generated
  by quadratic nonlinearity of the 24h signal. This creates QPC — the
  bispectrum B(ω_base, ω_base) is significantly non-zero, and bicoherence
  b²(ω_base, ω_base) is high.

- Independent oscillators: The 12h component comes from a separate physical
  process. No QPC — bicoherence at (ω_base, ω_base) is near zero.

Key formulas:
    Bispectrum:  B(ω₁, ω₂) = E[X(ω₁) · X(ω₂) · X*(ω₁ + ω₂)]
    Bicoherence: b²(ω₁, ω₂) = |B(ω₁,ω₂)|² / (E[|X(ω₁)X(ω₂)|²] · E[|X(ω₁+ω₂)|²])

Implementation notes:
    For short time series (N=12-48), standard segment-based bispectrum
    estimation lacks power. We use DPSS (Slepian) multi-taper estimation
    which provides multiple independent spectral estimates from a single
    realization, enabling bicoherence estimation even with very short data.

    The surrogate test uses phase randomization to destroy phase coupling
    while preserving the power spectrum, then compares the observed
    bicoherence against the surrogate null distribution.

    Power limitations: The test has good power for signals with strong
    multi-harmonic coupling (e.g., sawtooth waves) but limited power for
    signals with only weak 2nd-harmonic coupling (e.g., cos + 0.2*cos(2wt))
    when N < 50. This is a fundamental limitation of bispectral analysis
    on short series, not a bug.

References:
    Kim & Powers (1979). Digital bispectral analysis and its applications
    to nonlinear wave interactions. IEEE Trans. Plasma Sci.

    Thomson (1982). Spectrum estimation and harmonic analysis.
    Proc. IEEE, 70(9), 1055-1096.
"""

import numpy as np
from typing import Dict, Optional

try:
    from scipy.signal.windows import dpss as _scipy_dpss
    _HAS_SCIPY_DPSS = True
except ImportError:
    _HAS_SCIPY_DPSS = False


# ============================================================================
# DFT at exact frequency (avoids FFT bin mismatch for short series)
# ============================================================================

def _dft_at_freq(y, t, f):
    """Compute the DFT of y at exact frequency f using direct summation.

    This avoids the frequency resolution limitations of FFT for short
    series where the target frequency may not align with any FFT bin.

    Parameters
    ----------
    y : array
        Signal values.
    t : array
        Time points (same units as 1/f).
    f : float
        Target frequency.

    Returns
    -------
    complex
        DFT coefficient X(f).
    """
    return np.sum(y * np.exp(-2j * np.pi * f * t))


# ============================================================================
# Multi-taper generation
# ============================================================================

def _get_tapers(N, NW, K):
    """Get DPSS (Slepian) tapers, with sine-taper fallback.

    Parameters
    ----------
    N : int
        Data length.
    NW : float
        Time-bandwidth product.
    K : int
        Number of tapers.

    Returns
    -------
    array of shape (K, N)
        Orthogonal tapers.
    """
    if _HAS_SCIPY_DPSS:
        return _scipy_dpss(N, NW, K)
    # Fallback: sine tapers (reasonable approximation)
    tapers = np.zeros((K, N))
    for k in range(K):
        tapers[k] = np.sqrt(2.0 / (N + 1)) * np.sin(
            np.pi * (k + 1) * np.arange(1, N + 1) / (N + 1)
        )
    return tapers


# ============================================================================
# Core bispectrum computation
# ============================================================================

def compute_bispectrum(t, y, T_base=24.0, NW=2.5):
    """Compute bispectrum and bicoherence at the self-coupling point (ω_base, ω_base).

    Uses multi-taper (DPSS) estimation with exact-frequency DFT to handle
    short time series where FFT bins may not align with target frequencies.

    Each taper provides an independent spectral estimate; averaging across
    tapers yields a bicoherence that is < 1 for uncoupled signals (random
    biphase across tapers) and near 1 for coupled signals (consistent biphase).

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values (or any signal).
    T_base : float
        Base period in hours (default 24.0 for circadian).
    NW : float
        Time-bandwidth product for DPSS tapers (default 2.5).
        Higher NW = more tapers but broader spectral leakage.

    Returns
    -------
    dict
        bispectrum_mag : float
            |B(ω_base, ω_base)|, magnitude of the averaged bispectrum.
        bicoherence : float
            b²(ω_base, ω_base), squared bicoherence in [0, 1].
        bispectrum_real : float
            Real part of the averaged bispectrum.
        bispectrum_imag : float
            Imaginary part of the averaged bispectrum.
        n_tapers : int
            Number of tapers used.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    N = len(t_arr)

    y_centered = y_arr - np.mean(y_arr)

    f_base = 1.0 / T_base
    f_harm = 2.0 / T_base

    # Determine number of tapers
    K = max(2, int(2 * NW) - 1)
    # For very short series, limit K to avoid over-smoothing
    K = min(K, max(2, N // 4))
    tapers = _get_tapers(N, NW, K)

    # Accumulate bispectrum across tapers
    B_list = []
    P12_list = []
    P3_list = []

    for k in range(K):
        y_tapered = y_centered * tapers[k]

        X1 = _dft_at_freq(y_tapered, t_arr, f_base)
        X3 = _dft_at_freq(y_tapered, t_arr, f_harm)

        B = X1 * X1 * np.conj(X3)
        B_list.append(B)
        P12_list.append(np.abs(X1) ** 4)   # |X1·X1|² = |X1|⁴
        P3_list.append(np.abs(X3) ** 2)

    B_avg = np.mean(B_list)
    P12_avg = np.mean(P12_list)
    P3_avg = np.mean(P3_list)

    denom = P12_avg * P3_avg
    if denom < 1e-30:
        bicoherence = 0.0
    else:
        bicoherence = float(np.abs(B_avg) ** 2 / denom)

    bicoherence = min(max(bicoherence, 0.0), 1.0)

    return {
        "bispectrum_mag": float(np.abs(B_avg)),
        "bicoherence": bicoherence,
        "bispectrum_real": float(B_avg.real),
        "bispectrum_imag": float(B_avg.imag),
        "n_tapers": K,
    }


# ============================================================================
# Phase-randomization surrogates
# ============================================================================

def _phase_randomize(y, rng):
    """Generate a phase-randomized surrogate preserving the power spectrum.

    Shuffles the phases of the DFT while keeping magnitudes intact,
    then inverse-transforms. This destroys any phase coupling while
    preserving the amplitude spectrum.

    Parameters
    ----------
    y : array
        Input signal (real-valued).
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    array
        Phase-randomized surrogate signal (real-valued).
    """
    N = len(y)
    Y = np.fft.fft(y - np.mean(y))
    magnitudes = np.abs(Y)

    # Generate random phases, respecting conjugate symmetry for real output
    random_phases = np.zeros(N)
    if N % 2 == 0:
        n_free = N // 2 - 1
        random_phases[1:N // 2] = rng.uniform(0, 2 * np.pi, n_free)
        random_phases[N // 2 + 1:] = -random_phases[1:N // 2][::-1]
    else:
        n_free = (N - 1) // 2
        random_phases[1:n_free + 1] = rng.uniform(0, 2 * np.pi, n_free)
        random_phases[n_free + 1:] = -random_phases[1:n_free + 1][::-1]

    Y_surr = magnitudes * np.exp(1j * random_phases)
    y_surr = np.fft.ifft(Y_surr).real
    y_surr += np.mean(y)
    return y_surr


# ============================================================================
# Main surrogate-based coupling test
# ============================================================================

def bispectral_coupling_test(t, y, T_base=24.0, n_surrogates=199, seed=None):
    """Bispectral Harmonic Coupling Test (BHCT).

    Tests for quadratic phase coupling (QPC) at the self-coupling point
    (ω_base, ω_base) using surrogate data. If QPC is detected, the 12h
    component is likely a harmonic artifact of the 24h rhythm rather than
    an independent oscillation.

    Uses multi-taper bicoherence estimation with phase-randomization
    surrogates. The surrogates preserve the power spectrum but destroy
    phase coupling, providing a null distribution for the bicoherence.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base period in hours (default 24.0).
    n_surrogates : int
        Number of phase-randomized surrogates (default 199).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        bicoherence : float
            Observed squared bicoherence b²(ω_base, ω_base).
        p_value : float
            Fraction of surrogates with bicoherence >= observed.
        bispectrum_mag : float
            Observed |B(ω_base, ω_base)|.
        surrogate_bicoherences : array
            Bicoherence values from all surrogates.
        n_surrogates : int
            Number of surrogates used.
        bhct_score : int
            Evidence score for BHDT integration (see bhct_evidence).
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Compute observed bicoherence
    obs = compute_bispectrum(t_arr, y_arr, T_base=T_base)
    obs_bic = obs["bicoherence"]

    # Generate surrogates and compute their bicoherences
    rng = np.random.RandomState(seed)
    surr_bics = np.zeros(n_surrogates)

    for i in range(n_surrogates):
        y_surr = _phase_randomize(y_arr, rng)
        surr_result = compute_bispectrum(t_arr, y_surr, T_base=T_base)
        surr_bics[i] = surr_result["bicoherence"]

    # p-value: fraction of surrogates with bicoherence >= observed
    # Using the conservative (n_exceed + 1) / (n_surrogates + 1) formula
    n_exceed = np.sum(surr_bics >= obs_bic)
    p_value = float(n_exceed + 1) / (n_surrogates + 1)

    score = _p_to_bhct_score(p_value)

    return {
        "bicoherence": obs_bic,
        "p_value": p_value,
        "bispectrum_mag": obs["bispectrum_mag"],
        "surrogate_bicoherences": surr_bics,
        "n_surrogates": n_surrogates,
        "bhct_score": score,
    }


# ============================================================================
# Evidence scoring for BHDT integration
# ============================================================================

def _p_to_bhct_score(p_value):
    """Convert p-value to BHCT evidence score.

    Score interpretation:
        Negative = evidence for harmonic (QPC detected)
        Positive = evidence for independent (no QPC)
        Zero     = inconclusive

    Parameters
    ----------
    p_value : float
        p-value from the surrogate test.

    Returns
    -------
    int
        Evidence score in {-3, -2, 0, +1, +2}.
    """
    if p_value < 0.01:
        return -3   # Strong harmonic evidence (QPC detected)
    elif p_value < 0.05:
        return -2   # Moderate harmonic evidence
    elif p_value > 0.5:
        return +2   # Strong independent evidence (no QPC)
    elif p_value > 0.2:
        return +1   # Weak independent evidence
    else:
        return 0    # Inconclusive


def bhct_evidence(t, y, T_base=24.0, n_surrogates=199, seed=None):
    """Wrapper returning an evidence score compatible with BHDT's scoring system.

    This is the main entry point for integration with the BHDT inference
    pipeline (Evidence 11).

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base period in hours (default 24.0).
    n_surrogates : int
        Number of surrogates (default 199).
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        score : int
            Evidence score: negative = harmonic, positive = independent.
        p_value : float
            Raw p-value from the surrogate test.
        bicoherence : float
            Observed squared bicoherence.
        label : str
            Human-readable interpretation.
    """
    result = bispectral_coupling_test(
        t, y, T_base=T_base, n_surrogates=n_surrogates, seed=seed
    )

    score = result["bhct_score"]

    if score <= -2:
        label = "strong_qpc_harmonic"
    elif score == -1:
        label = "weak_qpc_harmonic"
    elif score >= 2:
        label = "no_qpc_independent"
    elif score == 1:
        label = "weak_no_qpc"
    else:
        label = "inconclusive"

    return {
        "score": score,
        "p_value": result["p_value"],
        "bicoherence": result["bicoherence"],
        "label": label,
    }
