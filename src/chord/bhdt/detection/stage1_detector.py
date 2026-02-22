"""
Stage 1: Liberal Multi-Method 12h Detection for CHORD.

Fuses four independent detection methods via Cauchy Combination Test (CCT)
to maximize recall for 12h rhythmic components. This stage does NOT perform
disentanglement — it only determines whether a gene has a detectable 12h
component worth analyzing further.

Detection methods:
  1. Parametric F-test (sinusoidal assumption)
  2. JTK_CYCLE Kendall tau (nonparametric, rank-based)
  3. RAIN umbrella (nonparametric, asymmetric waveforms)
  4. Harmonic regression partial F-test (multi-component)

Gate B (Residual 12h Test):
  After CCT detection, fits a K-harmonic 24h model (K selected by AICc)
  and tests the residuals for 12h periodicity. This eliminates false
  positives from circadian genes whose 12h component is entirely explained
  by non-sinusoidal 24h waveforms (the dominant FP source).

  Mathematical basis: Any non-sinusoidal periodic signal with period T
  contains Fourier harmonics at T/2, T/3, etc. If the 12h component is
  purely harmonic, removing the K-harmonic 24h model leaves no 12h signal
  in the residuals. If the 12h component is independent, residuals retain
  detectable 12h periodicity.

  Reference: Zhu et al. 2017 (Cell Metab, PMC5526350) — eigenvalue/pencil
  orthogonality test for separating harmonics from independent oscillators.

The CCT fusion is valid under arbitrary dependence (Liu & Xie, 2020),
which is critical since all four methods test the same underlying signal.

Output classification:
  - "has_12h" -> passed gate, enters Stage 2 disentanglement
  - "circadian_only" -> 24h detected, no 12h -> final classification
  - "non_rhythmic" -> nothing detected -> final classification
"""

import numpy as np

from chord.bhdt.detection.cauchy_combination import cauchy_combine
from chord.bhdt.inference import component_f_test
from chord.bhdt.nonparametric_detect import jtk_tau_12h, rain_umbrella_12h
from chord.bhdt.models import (
    _cos_sin_design, _fit_ols, fit_harmonic_model, _compute_aicc,
)


def _harmonic_regression_f_test(t, y, T_base=24.0):
    """Partial F-test for 12h component in a dual-harmonic regression.

    Fits y = M + A_24*cos(wt) + B_24*sin(wt) + A_12*cos(2wt) + B_12*sin(2wt) + e
    Tests H0: A_12 = B_12 = 0 via partial F-test.

    This differs from component_f_test in that it uses a 2-component model
    (24h + 12h only) rather than the full 3-component model, giving a
    cleaner test of the 12h component.
    """
    from scipy.stats import f as f_dist

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = len(t)

    # Full model: 24h + 12h
    periods_full = [T_base, T_base / 2.0]
    X_full = _cos_sin_design(t, periods_full)
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    rss_full = float(np.sum((y - X_full.dot(beta_full)) ** 2))
    p_full = X_full.shape[1]  # 5: intercept + 2*cos + 2*sin

    # Reduced model: 24h only
    periods_red = [T_base]
    X_red = _cos_sin_design(t, periods_red)
    beta_red = np.linalg.lstsq(X_red, y, rcond=None)[0]
    rss_red = float(np.sum((y - X_red.dot(beta_red)) ** 2))
    p_red = X_red.shape[1]  # 3: intercept + cos + sin

    df1 = p_full - p_red  # 2
    df2 = N - p_full
    if df2 <= 0 or rss_full <= 0:
        return 1.0

    F_stat = ((rss_red - rss_full) / df1) / (rss_full / df2)
    if F_stat < 0:
        return 1.0

    return float(1.0 - f_dist.cdf(F_stat, df1, df2))


def _select_K_by_aicc(t, y, T_base=24.0, K_max=4):
    """Select optimal K for K-harmonic 24h model via AICc (K=1..K_max)."""
    N = len(t)
    best_K = 1
    best_aicc = np.inf
    best_fit = None

    for K in range(1, K_max + 1):
        n_params = 1 + 2 * K + 1
        if N <= n_params + 1:
            break
        m = fit_harmonic_model(t, y, T_base=T_base, K=K)
        if m["aicc"] < best_aicc:
            best_aicc = m["aicc"]
            best_K = K
            best_fit = m

    if best_fit is None:
        best_fit = fit_harmonic_model(t, y, T_base=T_base, K=1)

    return best_K, best_fit


def _residual_12h_test(t, y, T_base=24.0, K_max=4, alpha_residual=0.05):
    """Gate B: Test whether 12h component's phase is independent of 24h.

    Compares two nested models via F-test:
      M_constrained (4 params): y = a0 + a1*cos(w24*t) + b1*sin(w24*t) + A*cos(w12*t - 2*phi_24)
      M_free (5 params): y = a0 + a1*cos(w24*t) + b1*sin(w24*t) + c1*cos(w12*t) + d1*sin(w12*t)

    The constrained model fixes 12h phase to the harmonic prediction (2*phi_24),
    allowing only amplitude to vary. The free model allows arbitrary 12h phase.
    F-test with df1=1, df2=N-5.
    """
    from scipy.stats import f as f_dist

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = len(t)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    T_12 = T_base / 2.0
    omega_12 = 2 * np.pi / T_12
    omega_24 = 2 * np.pi / T_base

    # Free model (5 params): intercept + cos/sin(24h) + cos/sin(12h)
    X_free = np.column_stack([
        np.ones(N),
        np.cos(omega_24 * t), np.sin(omega_24 * t),
        np.cos(omega_12 * t), np.sin(omega_12 * t),
    ])
    beta_free = np.linalg.lstsq(X_free, y, rcond=None)[0]
    rss_free = float(np.sum((y - X_free.dot(beta_free)) ** 2))

    # Extract phi_24 from free model
    a1, b1 = beta_free[1], beta_free[2]
    phi_24 = np.arctan2(b1, a1)
    phi_12_harmonic = 2 * phi_24

    # Constrained model (4 params): intercept + cos/sin(24h) + A*cos(w12*t - 2*phi_24)
    X_constrained = np.column_stack([
        np.ones(N),
        np.cos(omega_24 * t), np.sin(omega_24 * t),
        np.cos(omega_12 * t - phi_12_harmonic),
    ])
    beta_constrained = np.linalg.lstsq(X_constrained, y, rcond=None)[0]
    rss_constrained = float(np.sum((y - X_constrained.dot(beta_constrained)) ** 2))

    m0_r2 = 1.0 - rss_free / max(ss_tot, 1e-10)

    # F-test: constrained (4 params) vs free (5 params)
    df1_phase = 1
    df2_phase = N - 5
    if df2_phase > 0 and rss_free > 0 and rss_constrained >= rss_free:
        F_phase = ((rss_constrained - rss_free) / df1_phase) / (rss_free / df2_phase)
        p_phase = float(1.0 - f_dist.cdf(max(F_phase, 0), df1_phase, df2_phase))
    else:
        p_phase = 1.0

    passed = p_phase < alpha_residual

    return {
        "passed": passed,
        "p_residual_12h": p_phase,
        "p_phase_freedom": p_phase,
        "best_K": 2,
        "m0_r2": m0_r2,
        "residual_amp_ratio": 0.0,
    }


def stage1_detect(t, y, T_base=24.0, alpha_detect=0.10,
                  gate_b_enabled=True, alpha_residual=0.05, K_max=4):
    """Stage 1: Liberal multi-method 12h detection with Gate B residual test.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base circadian period (default 24.0).
    alpha_detect : float
        Gate threshold for CCT p-value (default 0.10, liberal).
    gate_b_enabled : bool
        Whether to apply Gate B residual 12h test (default True).
        When enabled, genes that pass CCT must also show significant
        12h signal in residuals after removing K-harmonic 24h model.
    alpha_residual : float
        Significance threshold for Gate B residual test (default 0.05).
    K_max : int
        Maximum harmonics for Gate B 24h model (default 4).

    Returns
    -------
    dict with keys:
        p_detect, p_ftest, p_jtk, p_rain, p_harmreg, p_24,
        passed, stage1_class, best_detector, detection_strength,
        waveform_hint, gate_b_result (dict or None)
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    periods_3comp = [T_base, T_base / 2.0, T_base / 3.0]

    f_test_12 = component_f_test(t, y, periods_3comp, test_period_idx=1)
    p_ftest = f_test_12["p_value"]

    jtk_result = jtk_tau_12h(t, y, period_range=(11.5, 12.5), n_phases=8)
    p_jtk = jtk_result["p_value"]

    rain_result = rain_umbrella_12h(t, y, period=T_base / 2.0)
    p_rain = rain_result["p_value"]

    p_harmreg = _harmonic_regression_f_test(t, y, T_base=T_base)

    f_test_24 = component_f_test(t, y, periods_3comp, test_period_idx=0)
    p_24 = f_test_24["p_value"]

    p_values_12h = [p_ftest, p_jtk, p_rain, p_harmreg]
    p_values_clean = [p if np.isfinite(p) else 1.0 for p in p_values_12h]
    p_detect = cauchy_combine(p_values_clean)

    detector_names = ["ftest", "jtk", "rain", "harmreg"]
    best_idx = int(np.argmin(p_values_clean))
    best_detector = detector_names[best_idx]

    np_best = min(p_jtk, p_rain)
    if np_best < 0.05 and p_ftest > 0.10:
        waveform_hint = "non_sinusoidal"
    elif p_ftest < 0.05:
        waveform_hint = "sinusoidal"
    else:
        waveform_hint = "unknown"

    passed = p_detect < alpha_detect
    detection_strength = -np.log10(max(p_detect, 1e-20))

    # Gate B: Phase-freedom test — computes p_phase_freedom for Stage 2 evidence.
    # Previously used as a hard gate (blocking genes with p >= 0.05), but this
    # destroyed sensitivity because: (1) the 1-df F-test has low power when
    # A_12/A_24 is small or phi_12 ≈ 2*phi_24 by biological coincidence, and
    # (2) "phase locked" ≠ "harmonic" (independent 12h can have phi_12 ≈ 2*phi_24).
    # Now: compute and store for Stage 2 soft evidence, but do NOT block passage.
    gate_b_result = None
    if gate_b_enabled and passed and p_24 < 0.05:
        gate_b_result = _residual_12h_test(
            t, y, T_base=T_base, K_max=K_max, alpha_residual=alpha_residual
        )
        # Gate B result stored for Stage 2 evidence scoring.
        # No hard blocking — Stage 2 handles disentanglement.

    if p_24 > 0.5 and p_detect > 0.5:
        stage1_class = "non_rhythmic"
        passed = False
    elif p_24 < 0.05 and not passed:
        stage1_class = "circadian_only"
    elif passed:
        stage1_class = "has_12h"
    else:
        stage1_class = "non_rhythmic"
        passed = False

    return {
        "p_detect": float(p_detect),
        "p_ftest": float(p_ftest) if np.isfinite(p_ftest) else 1.0,
        "p_jtk": float(p_jtk),
        "p_rain": float(p_rain),
        "p_harmreg": float(p_harmreg),
        "p_24": float(p_24),
        "passed": passed,
        "stage1_class": stage1_class,
        "best_detector": best_detector,
        "detection_strength": float(detection_strength),
        "waveform_hint": waveform_hint,
        "gate_b_result": gate_b_result,
    }
