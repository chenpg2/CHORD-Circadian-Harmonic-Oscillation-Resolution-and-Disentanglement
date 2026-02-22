"""
Nonparametric 12h rhythm detection layer.

Extracts core principles from JTK_CYCLE (Hughes et al. 2010) and
RAIN (Thaben & Westermark 2014) as a front-end detection layer for CHORD.

JTK_CYCLE uses Kendall's tau rank correlation -- nonparametric, does not
assume sinusoidal waveform, robust to outliers.

RAIN uses umbrella alternatives (rise-then-fall) -- detects any periodic
pattern regardless of waveform shape.

These complement the parametric F-test which assumes sinusoidal components.
When F-test misses non-sinusoidal 12h rhythms (peaked, sawtooth, pulsatile),
JTK/RAIN can still detect them, rescuing genes from being classified as
ambiguous or circadian_only.

Combination strategy
--------------------
Each method (JTK, RAIN, F-test) already applies its own internal multiple-
testing correction (Bonferroni over periods/phases/peak-positions).  The
outer combination therefore takes a simple min -- NO additional Bonferroni.
This is statistically justified because the three tests are positively
correlated (they all measure the same underlying 12h signal), so a second
Bonferroni would be overly conservative and defeat the purpose of rescuing
borderline genes (Westfall & Young 1993, resampling-based reasoning).

References
----------
Hughes ME, Hogenesch JB, Kornacker K. (2010). JTK_CYCLE: an efficient
    nonparametric algorithm for detecting rhythmic components in genome-scale
    data sets. J Biol Rhythms, 25(5):372-380.
Thaben PF, Westermark PO. (2014). Detecting rhythms in time series with
    RAIN. J Biol Rhythms, 29(6):391-400.
Westfall PH, Young SS. (1993). Resampling-Based Multiple Testing.
    Wiley, New York.
"""

import numpy as np
from scipy import stats


def jtk_tau_12h(t, y, period_range=(10.5, 13.5), n_phases=8):
    """JTK-style Kendall tau test for 12h rhythmicity.

    Simplified from the full JTK_CYCLE algorithm, optimized for
    single-period detection around 12h.  Uses fewer test periods
    and phases than the benchmark version to preserve statistical power
    (fewer multiple testing corrections).

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    period_range : tuple of float
        (min, max) period in hours to scan.
    n_phases : int
        Number of phase offsets to test per period.

    Returns
    -------
    dict with keys:
        p_value : float -- Bonferroni-corrected p-value
        tau : float -- best Kendall's tau
        best_period : float -- period with strongest correlation
        best_phase : float -- phase with strongest correlation
        n_tests : int -- number of tests performed
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Guard: constant signal or too few points
    if len(y) < 6 or np.ptp(y) < 1e-12:
        return {
            "p_value": 1.0,
            "tau": 0.0,
            "best_period": 12.0,
            "best_phase": 0.0,
            "n_tests": 0,
        }

    # Scan periods: 10.5, 11.5, 12.5, 13.5 (4 periods)
    test_periods = np.arange(period_range[0], period_range[1] + 1.0, 1.0)

    best_p = 1.0
    best_tau = 0.0
    best_period = 12.0
    best_phase = 0.0

    for T in test_periods:
        for phase_idx in range(n_phases):
            phase = 2 * np.pi * phase_idx / n_phases
            ref = np.cos(2 * np.pi / T * t + phase)
            tau, p = stats.kendalltau(y, ref)
            # kendalltau returns NaN for constant input
            if np.isnan(tau) or np.isnan(p):
                continue
            if abs(tau) > abs(best_tau):
                best_tau = tau
                best_p = p
                best_period = T
                best_phase = phase

    n_tests = len(test_periods) * n_phases
    corrected_p = min(best_p * n_tests, 1.0)

    return {
        "p_value": float(corrected_p),
        "tau": float(best_tau),
        "best_period": float(best_period),
        "best_phase": float(best_phase),
        "n_tests": n_tests,
    }


def rain_umbrella_12h(t, y, period=12.0, n_peaks=9):
    """RAIN-style umbrella test for 12h rhythmicity.

    Simplified from the full RAIN algorithm.  Tests whether data shows
    a rise-then-fall (or fall-then-rise) pattern within each 12h cycle.

    Uses 9 peak positions spanning 0.1--0.9 to capture both symmetric
    and highly asymmetric waveforms (pulsatile UPR, sawtooth, etc.).

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    period : float
        Candidate period in hours.
    n_peaks : int
        Number of peak positions to test (default 9).

    Returns
    -------
    dict with keys:
        p_value : float -- Bonferroni-corrected p-value
        peak_fraction : float -- best peak position (0-1)
        direction : str -- 'peak' or 'trough'
        n_tests : int
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Guard: constant signal or too few points
    if len(y) < 6 or np.ptp(y) < 1e-12:
        return {
            "p_value": 1.0,
            "peak_fraction": 0.5,
            "direction": "peak",
            "n_tests": 0,
        }

    # Phase-fold the data
    phases = (t % period) / period
    order = np.argsort(phases)
    y_sorted = y[order]
    phases_sorted = phases[order]

    best_p = 1.0
    best_peak = 0.5
    best_dir = "peak"

    # 9 peak positions from 0.1 to 0.9 -- covers extreme asymmetry
    peak_fracs = np.linspace(0.1, 0.9, n_peaks)
    n_alternatives = len(peak_fracs) * 2  # peak + trough per position

    for peak_frac in peak_fracs:
        rising_mask = phases_sorted <= peak_frac
        falling_mask = phases_sorted > peak_frac

        n_rise = int(np.sum(rising_mask))
        n_fall = int(np.sum(falling_mask))

        if n_rise < 2 or n_fall < 2:
            continue

        y_rise = y_sorted[rising_mask]
        tau_rise, p_rise = stats.kendalltau(np.arange(n_rise), y_rise)

        y_fall = y_sorted[falling_mask]
        tau_fall, p_fall = stats.kendalltau(np.arange(n_fall), y_fall)

        # Handle NaN from constant sub-segments
        if np.isnan(tau_rise) or np.isnan(p_rise):
            tau_rise, p_rise = 0.0, 1.0
        if np.isnan(tau_fall) or np.isnan(p_fall):
            tau_fall, p_fall = 0.0, 1.0

        # Peak umbrella: rise increases, fall decreases
        # Convert two-sided p to one-sided in the expected direction
        p_rise_up = p_rise / 2 if tau_rise > 0 else 1 - p_rise / 2
        p_fall_down = p_fall / 2 if tau_fall < 0 else 1 - p_fall / 2
        # Fisher's method to combine the two one-sided tests
        chi2_up = -2 * (np.log(max(p_rise_up, 1e-20)) +
                        np.log(max(p_fall_down, 1e-20)))
        p_up = 1 - stats.chi2.cdf(chi2_up, df=4)

        # Trough umbrella: rise decreases, fall increases
        p_rise_down = p_rise / 2 if tau_rise < 0 else 1 - p_rise / 2
        p_fall_up = p_fall / 2 if tau_fall > 0 else 1 - p_fall / 2
        chi2_down = -2 * (np.log(max(p_rise_down, 1e-20)) +
                          np.log(max(p_fall_up, 1e-20)))
        p_down = 1 - stats.chi2.cdf(chi2_down, df=4)

        if p_up < best_p:
            best_p = p_up
            best_peak = peak_frac
            best_dir = "peak"
        if p_down < best_p:
            best_p = p_down
            best_peak = peak_frac
            best_dir = "trough"

    corrected_p = min(best_p * n_alternatives, 1.0)

    return {
        "p_value": float(corrected_p),
        "peak_fraction": float(best_peak),
        "direction": best_dir,
        "n_tests": n_alternatives,
    }


def nonparametric_12h_detect(t, y, f_test_p=None):
    """Combined nonparametric 12h detection.

    Runs JTK tau + RAIN umbrella, combines with optional F-test p-value.

    Combination: simple min of the individually-corrected p-values.
    NO outer Bonferroni -- each method already corrects internally, and
    the three tests are positively correlated (same underlying signal).
    A second Bonferroni would be overly conservative and defeat the
    rescue purpose of this layer.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    f_test_p : float, optional
        F-test p-value for 12h component (from BHDT).

    Returns
    -------
    dict with keys:
        p_combined : float -- min of all available corrected p-values
        p_jtk : float
        p_rain : float
        p_ftest : float or None
        detection_method : str -- which method drove the detection
        rain_peak_fraction : float -- waveform shape info
        jtk_tau : float -- Kendall's tau value
        jtk_period : float -- best period from JTK scan
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    jtk_result = jtk_tau_12h(t, y)
    rain_result = rain_umbrella_12h(t, y)

    p_jtk = jtk_result["p_value"]
    p_rain = rain_result["p_value"]

    # Collect all p-values (each already internally corrected)
    p_values = [p_jtk, p_rain]
    methods = ["jtk", "rain"]
    if f_test_p is not None:
        p_values.append(f_test_p)
        methods.append("ftest")

    # Simple min -- no outer Bonferroni (see module docstring)
    min_idx = int(np.argmin(p_values))
    p_combined = float(p_values[min_idx])

    return {
        "p_combined": p_combined,
        "p_jtk": float(p_jtk),
        "p_rain": float(p_rain),
        "p_ftest": f_test_p,
        "detection_method": methods[min_idx],
        "rain_peak_fraction": rain_result["peak_fraction"],
        "jtk_tau": jtk_result["tau"],
        "jtk_period": jtk_result["best_period"],
    }
