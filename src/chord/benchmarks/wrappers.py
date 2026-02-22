"""
Unified wrappers for classical rhythm detection methods.

Each wrapper takes (t, y) arrays and returns a dict with at minimum:
    {p_value, period_estimate, amplitude_estimate, method_name}

Dependencies: numpy, scipy (no R, no sklearn).
"""

import numpy as np
from scipy import signal, stats


def lomb_scargle(t, y, periods=None, n_bootstrap=200, seed=42):
    """Lomb-Scargle periodogram with bootstrap p-value.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    periods : list of float, optional
        Candidate periods to evaluate (hours). Default [24, 12, 8, 6, 4].
    n_bootstrap : int
        Number of shuffles for null distribution of max power.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        p_value, period_estimate, amplitude_estimate, method_name,
        powers (array of power at each candidate period).
    """
    if periods is None:
        periods = [24, 12, 8, 6, 4]

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    periods = np.asarray(periods, dtype=float)

    angular_freqs = 2.0 * np.pi / periods

    # Normalize y to zero mean for lombscargle
    y_centered = y - y.mean()

    powers = signal.lombscargle(t, y_centered, angular_freqs, normalize=True)

    best_idx = np.argmax(powers)
    best_period = periods[best_idx]
    best_power = powers[best_idx]

    # Amplitude estimate from Lomb-Scargle power (approximate)
    amplitude_estimate = np.sqrt(2.0 * best_power) * np.std(y_centered)

    # Bootstrap null: shuffle y, recompute max power each time
    rng = np.random.default_rng(seed)
    null_max_powers = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        y_shuf = rng.permutation(y_centered)
        null_powers = signal.lombscargle(t, y_shuf, angular_freqs, normalize=True)
        null_max_powers[i] = null_powers.max()

    p_value = float(np.mean(null_max_powers >= best_power))
    # Ensure p_value is never exactly 0 (finite-sample correction)
    if p_value == 0.0:
        p_value = 1.0 / (n_bootstrap + 1)

    return {
        "p_value": p_value,
        "period_estimate": float(best_period),
        "amplitude_estimate": float(amplitude_estimate),
        "method_name": "lomb_scargle",
        "powers": powers,
        "periods_tested": periods,
    }


def cosinor(t, y, period=12.0):
    """Single-component cosinor regression with F-test.

    Model: y = M + A*cos(2*pi*t/T - phi) + epsilon
    Reparameterised as: y = M + beta*cos(w*t) + gamma*sin(w*t)
    where A = sqrt(beta^2 + gamma^2), phi = atan2(gamma, beta).

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    period : float
        Fixed period T (hours).

    Returns
    -------
    dict
        p_value, period_estimate, amplitude_estimate, phase_estimate,
        mesor, method_name.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)

    omega = 2.0 * np.pi / period
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)

    # Design matrix: [1, cos(wt), sin(wt)]
    X = np.column_stack([np.ones(n), cos_t, sin_t])

    # OLS fit
    beta_hat, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    mesor, beta, gamma = beta_hat

    y_hat = X @ beta_hat
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    # F-test: full model (3 params) vs reduced model (intercept only, 1 param)
    df_model = 2  # beta and gamma
    df_resid = n - 3
    if df_resid <= 0 or ss_tot == 0:
        p_value = 1.0
        f_stat = 0.0
    else:
        ss_reg = ss_tot - ss_res
        ms_reg = ss_reg / df_model
        ms_res = ss_res / df_resid
        f_stat = ms_reg / ms_res if ms_res > 0 else 0.0
        p_value = float(1.0 - stats.f.cdf(f_stat, df_model, df_resid))

    amplitude = float(np.sqrt(beta ** 2 + gamma ** 2))
    phase = float(np.arctan2(gamma, beta))  # radians

    return {
        "p_value": p_value,
        "period_estimate": float(period),
        "amplitude_estimate": amplitude,
        "phase_estimate": phase,
        "mesor": float(mesor),
        "f_statistic": float(f_stat),
        "method_name": "cosinor",
    }


def harmonic_regression(t, y, periods=None):
    """Multi-harmonic OLS regression with per-component F-tests.

    Model: y = M + sum_k [A_k*cos(2*pi*t/T_k) + B_k*sin(2*pi*t/T_k)] + eps

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    periods : list of float, optional
        Periods for each harmonic component. Default [24, 12, 8].

    Returns
    -------
    dict
        p_value (overall), period_estimate (dominant), amplitude_estimate,
        method_name, components (list of per-component dicts).
    """
    if periods is None:
        periods = [24, 12, 8]

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    k = len(periods)

    # Build full design matrix
    X_full = [np.ones(n)]
    for T in periods:
        omega = 2.0 * np.pi / T
        X_full.append(np.cos(omega * t))
        X_full.append(np.sin(omega * t))
    X_full = np.column_stack(X_full)

    # Full model OLS
    beta_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    y_hat_full = X_full @ beta_full
    ss_res_full = np.sum((y - y_hat_full) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    p_full = 2 * k  # number of harmonic parameters (cos + sin per period)
    df_resid_full = n - (1 + p_full)

    # Overall F-test (full model vs intercept-only)
    if df_resid_full <= 0 or ss_tot == 0:
        overall_p = 1.0
        overall_f = 0.0
    else:
        ss_reg = ss_tot - ss_res_full
        ms_reg = ss_reg / p_full
        ms_res = ss_res_full / df_resid_full
        overall_f = ms_reg / ms_res if ms_res > 0 else 0.0
        overall_p = float(1.0 - stats.f.cdf(overall_f, p_full, df_resid_full))

    # Per-component partial F-tests and parameter extraction
    components = []
    best_amp = 0.0
    best_period = periods[0]

    for i, T in enumerate(periods):
        # Extract cos/sin coefficients for this component
        beta_cos = beta_full[1 + 2 * i]
        beta_sin = beta_full[1 + 2 * i + 1]
        amplitude = float(np.sqrt(beta_cos ** 2 + beta_sin ** 2))
        phase = float(np.arctan2(beta_sin, beta_cos))

        # Reduced model: drop this component's two columns
        cols_to_keep = list(range(X_full.shape[1]))
        cols_to_keep.remove(1 + 2 * i)
        cols_to_keep.remove(1 + 2 * i + 1)
        X_reduced = X_full[:, cols_to_keep]

        beta_red, _, _, _ = np.linalg.lstsq(X_reduced, y, rcond=None)
        y_hat_red = X_reduced @ beta_red
        ss_res_red = np.sum((y - y_hat_red) ** 2)

        df_num = 2
        if df_resid_full > 0 and ss_res_full > 0:
            f_partial = ((ss_res_red - ss_res_full) / df_num) / (
                ss_res_full / df_resid_full
            )
            comp_p = float(1.0 - stats.f.cdf(f_partial, df_num, df_resid_full))
        else:
            f_partial = 0.0
            comp_p = 1.0

        components.append(
            {
                "period": float(T),
                "amplitude": amplitude,
                "phase": phase,
                "p_value": comp_p,
                "f_statistic": float(f_partial),
            }
        )

        if amplitude > best_amp:
            best_amp = amplitude
            best_period = T

    return {
        "p_value": overall_p,
        "period_estimate": float(best_period),
        "amplitude_estimate": float(best_amp),
        "method_name": "harmonic_regression",
        "f_statistic": float(overall_f),
        "mesor": float(beta_full[0]),
        "components": components,
    }


def jtk_cycle(t, y, period_range=(10, 14), n_phases=24):
    """JTK_CYCLE nonparametric rhythm detection.

    Simplified implementation of Hughes et al. 2010.
    Tests for rhythmicity using Kendall's tau between data and
    reference cosine waveforms across candidate periods and phases.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    period_range : tuple of float
        (min_period, max_period) in hours to scan.
    n_phases : int
        Number of phase offsets to test per period.

    Returns
    -------
    dict
        p_value, period_estimate, amplitude_estimate, method_name,
        tau, phase, n_tests.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(t)

    best_p = 1.0
    best_period = (period_range[0] + period_range[1]) / 2.0
    best_tau = 0.0
    best_phase = 0.0

    # Test periods from period_range[0] to period_range[1] in 0.5h steps
    test_periods = np.arange(period_range[0], period_range[1] + 0.5, 0.5)

    for T in test_periods:
        for phase_idx in range(n_phases):
            phase = 2 * np.pi * phase_idx / n_phases
            # Reference waveform: cosine at this period and phase
            ref = np.cos(2 * np.pi / T * t + phase)
            # Kendall's tau correlation
            tau, p = stats.kendalltau(y, ref)
            if abs(tau) > abs(best_tau):
                best_tau = tau
                best_p = p
                best_period = T
                best_phase = phase

    # Bonferroni correction for multiple testing
    n_tests = len(test_periods) * n_phases
    corrected_p = min(best_p * n_tests, 1.0)

    # Amplitude estimate from cosinor fit at best period
    omega = 2 * np.pi / best_period
    X = np.column_stack([np.ones(N), np.cos(omega * t), np.sin(omega * t)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    amplitude = np.sqrt(beta[1] ** 2 + beta[2] ** 2)

    return {
        "p_value": float(corrected_p),
        "period_estimate": float(best_period),
        "amplitude_estimate": float(amplitude),
        "method_name": "jtk_cycle",
        "tau": float(best_tau),
        "phase": float(best_phase),
        "n_tests": n_tests,
    }


def rain(t, y, period=12.0):
    """RAIN: Rhythmicity Analysis Incorporating Non-parametric methods.

    Simplified implementation of Thaben & Westermark 2014.
    Tests for rhythmicity using umbrella alternatives — the data should
    rise then fall (or vice versa) within each cycle.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    period : float
        Candidate period in hours.

    Returns
    -------
    dict
        p_value, period_estimate, amplitude_estimate, method_name,
        peak_fraction.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(t)

    # Phase-fold the data
    phases = (t % period) / period  # 0 to 1
    order = np.argsort(phases)
    y_sorted = y[order]
    phases_sorted = phases[order]

    best_p = 1.0
    best_peak = 0.0

    # Test different peak positions (umbrella alternatives).
    # RAIN tests both "up-then-down" and "down-then-up" umbrella shapes
    # to handle arbitrary phase offsets.
    peak_fracs = np.linspace(0.1, 0.9, 17)
    n_alternatives = len(peak_fracs) * 2  # two directions per peak position

    for peak_frac in peak_fracs:
        rising_mask = phases_sorted <= peak_frac
        falling_mask = phases_sorted > peak_frac

        n_rise = np.sum(rising_mask)
        n_fall = np.sum(falling_mask)

        if n_rise < 2 or n_fall < 2:
            continue

        # Mann-Kendall trend test as proxy for Jonckheere-Terpstra
        y_rise = y_sorted[rising_mask]
        tau_rise, p_rise = stats.kendalltau(np.arange(n_rise), y_rise)

        y_fall = y_sorted[falling_mask]
        tau_fall, p_fall = stats.kendalltau(np.arange(n_fall), y_fall)

        # Direction 1: peak umbrella (rise increases, fall decreases)
        p_rise_up = p_rise / 2 if tau_rise > 0 else 1 - p_rise / 2
        p_fall_down = p_fall / 2 if tau_fall < 0 else 1 - p_fall / 2
        chi2_up = -2 * (
            np.log(max(p_rise_up, 1e-20)) + np.log(max(p_fall_down, 1e-20))
        )
        p_up = 1 - stats.chi2.cdf(chi2_up, df=4)

        # Direction 2: trough umbrella (rise decreases, fall increases)
        p_rise_down = p_rise / 2 if tau_rise < 0 else 1 - p_rise / 2
        p_fall_up = p_fall / 2 if tau_fall > 0 else 1 - p_fall / 2
        chi2_down = -2 * (
            np.log(max(p_rise_down, 1e-20)) + np.log(max(p_fall_up, 1e-20))
        )
        p_down = 1 - stats.chi2.cdf(chi2_down, df=4)

        for p_cand in (p_up, p_down):
            if p_cand < best_p:
                best_p = p_cand
                best_peak = peak_frac

    # Bonferroni correction for all alternatives tested
    corrected_p = min(best_p * n_alternatives, 1.0)

    # Amplitude estimate from cosinor fit
    omega = 2 * np.pi / period
    X = np.column_stack([np.ones(N), np.cos(omega * t), np.sin(omega * t)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    amplitude = np.sqrt(beta[1] ** 2 + beta[2] ** 2)

    return {
        "p_value": float(corrected_p),
        "period_estimate": float(period),
        "amplitude_estimate": float(amplitude),
        "method_name": "rain",
        "peak_fraction": float(best_peak),
    }


def pencil_method(t, y, T_base=24.0, n_components=None):
    """Matrix pencil method for oscillation detection.

    Decomposes signal into superimposed damped sinusoids via
    generalized eigenvalue decomposition of Hankel matrices.
    Used by Zhu 2017 (Cell Metabolism) for 12h rhythm detection.

    Note: This detects 12h oscillations but cannot distinguish
    harmonic artifacts from independent oscillators.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    T_base : float
        Base period (hours), used for context only.
    n_components : int or None
        Model order. Auto-detected from singular value gap if None.

    Returns
    -------
    dict
        p_value, period_estimate, amplitude_estimate, method_name,
        detected_periods, has_12h, n_components.
    """
    from chord.bhdt.inference import component_f_test

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(y)
    dt = np.median(np.diff(t))

    # Pencil parameter L (typically N/3 to N/2)
    L = N // 3

    if L < 2 or N - L < 2:
        return {
            "p_value": 1.0,
            "period_estimate": float("nan"),
            "amplitude_estimate": 0.0,
            "method_name": "pencil",
            "detected_periods": [],
            "has_12h": False,
            "n_components": 0,
        }

    # Build Hankel matrices
    H0 = np.zeros((N - L, L))
    H1 = np.zeros((N - L, L))
    for i in range(N - L):
        H0[i, :] = y[i : i + L]
        H1[i, :] = y[i + 1 : i + 1 + L]

    # SVD of H0 to determine model order
    U, s, Vt = np.linalg.svd(H0, full_matrices=False)

    # Auto-detect model order from singular value ratio
    if n_components is None:
        if len(s) > 1:
            ratios = s[:-1] / np.maximum(s[1:], 1e-15)
            n_components = min(np.argmax(ratios) + 1, L // 2, 6)
            n_components = max(n_components, 2)
        else:
            n_components = 1

    n_components = min(n_components, len(s))

    # Truncate to model order
    U_r = U[:, :n_components]
    S_r = np.diag(s[:n_components])
    Vt_r = Vt[:n_components, :]

    # Generalized eigenvalue problem via projection
    H0_r = U_r @ S_r @ Vt_r
    H1_r = np.zeros_like(H0_r)
    for i in range(N - L):
        H1_r[i, :] = y[i + 1 : i + 1 + L]

    H1_proj = U_r.T @ H1_r @ Vt_r.T
    H0_proj = U_r.T @ H0_r @ Vt_r.T

    # Eigenvalues of the pencil
    try:
        eigenvalues = np.linalg.eigvals(np.linalg.pinv(H0_proj) @ H1_proj)
    except np.linalg.LinAlgError:
        return {
            "p_value": 1.0,
            "period_estimate": float("nan"),
            "amplitude_estimate": 0.0,
            "method_name": "pencil",
            "detected_periods": [],
            "has_12h": False,
            "n_components": n_components,
        }

    # Extract frequencies and damping from eigenvalues
    # z_k = exp((alpha_k + j*omega_k) * dt)
    detected = []
    for z in eigenvalues:
        if abs(z) < 1e-10 or abs(z) > 10:
            continue
        omega = np.angle(z) / dt  # angular frequency
        alpha = np.log(abs(z)) / dt  # damping
        if omega < 0:
            omega = -omega
        period = 2 * np.pi / omega if omega > 1e-10 else float("inf")

        if 2 < period < 100:  # reasonable biological range
            detected.append(
                {
                    "period": float(period),
                    "damping": float(alpha),
                    "magnitude": float(abs(z)),
                }
            )

    # Sort by magnitude (strongest first)
    detected.sort(key=lambda x: -x["magnitude"])

    # Check for 12h component
    has_12h = any(10 < d["period"] < 14 for d in detected)
    period_12h = next(
        (d["period"] for d in detected if 10 < d["period"] < 14), float("nan")
    )

    # Amplitude estimate for 12h component
    if has_12h:
        omega_12 = 2 * np.pi / period_12h
        X = np.column_stack([np.ones(N), np.cos(omega_12 * t), np.sin(omega_12 * t)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        amp_12 = np.sqrt(beta[1] ** 2 + beta[2] ** 2)
    else:
        amp_12 = 0.0

    # P-value: use F-test for the 12h component significance
    if has_12h:
        f_result = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)
        p_val = f_result["p_value"]
    else:
        p_val = 1.0

    return {
        "p_value": float(p_val),
        "period_estimate": float(period_12h) if has_12h else float("nan"),
        "amplitude_estimate": float(amp_12),
        "method_name": "pencil",
        "detected_periods": [d["period"] for d in detected],
        "has_12h": has_12h,
        "n_components": n_components,
    }


def run_all_methods(t, y, test_periods=None):
    """Run all rhythm detection methods and collect results.

    Parameters
    ----------
    t : array-like
        Time points.
    y : array-like
        Observed values.
    test_periods : list of float, optional
        Periods to test. Default [24, 12, 8].

    Returns
    -------
    dict
        Results keyed by method name.
    """
    if test_periods is None:
        test_periods = [24, 12, 8]

    results = {}

    # Lomb-Scargle
    results["lomb_scargle"] = lomb_scargle(t, y, periods=test_periods)

    # Cosinor — run for each candidate period, keep best
    best_cosinor = None
    for T in test_periods:
        res = cosinor(t, y, period=T)
        if best_cosinor is None or res["p_value"] < best_cosinor["p_value"]:
            best_cosinor = res
    results["cosinor"] = best_cosinor

    # Harmonic regression
    results["harmonic_regression"] = harmonic_regression(
        t, y, periods=test_periods
    )

    # JTK_CYCLE — scan around 12h by default
    results["jtk_cycle"] = jtk_cycle(t, y, period_range=(10, 14))

    # RAIN — test at 12h period
    results["rain"] = rain(t, y, period=12.0)

    # Matrix pencil method
    results["pencil"] = pencil_method(t, y)

    return results
