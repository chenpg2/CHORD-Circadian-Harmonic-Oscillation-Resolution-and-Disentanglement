"""Numerical verification of the three analytic lemmas underpinning CHORD's two-axis reframe.

These lemmas are the "concede-with-math" backbone of the JBR revision (see
docs/plans/2026-07-02-001-refactor-chord-jbr-revision-plan.md, unit U4). Each is
checked numerically here so the manuscript supplement states only confirmed results.

Lemma 1 (peak-symmetry redundancy). For g(theta) = A1 cos(theta) + A2 cos(2 theta + delta),
    at the locked phase delta = 0 the ratio of the two daily peak heights (measured from the
    global trough) is exactly
        rho(r) = ((4r - 1)/(4r + 1))**2,   r = A2/A1,
    and a second maximum exists (twin peaks) only for r > 1/4. Hence rho is a deterministic
    function of (r, delta): the "peak symmetry" feature carries no information beyond the
    amplitude ratio r and the relative phase delta.

Lemma 2 (half-period-symmetry / fundamental-suppression invariant). For the intersection
    archetype s(t) = h(t) + h(t - (1/2 + beta) T) built from a smooth unimodal bump h with
    period T, the Fourier magnitudes obey
        |s_1| = |h_1| * 2|sin(pi beta)|,   |s_2| = |h_2| * 2|cos(2 pi beta)|,
    so R = A(2f0)/A(f0) = (|h_2|/|h_1|) * |cos(2 pi beta)| / |sin(pi beta)| -> infinity as
    beta -> 0. Fundamental suppression (R >> 1, even-harmonic dominance) is the invariant that
    separates a near-symmetric intersection from a fundamental-dominated harmonic.

Lemma 3 (phase-freedom power). The nested F-test locking the 2f0 phase to twice the f0 phase
    has noncentrality
        lambda = (N/2) (A2/sigma)**2 sin^2(Delta),   Delta = delta_true - c0,
    so its power is governed by N x SNR^2 x sin^2(Delta) and collapses to the size alpha as
    Delta -> 0 (entrainment), A2 -> 0 (low SNR), or N small.

Run:  python scripts/verify_analytic_lemmas.py
Deterministic (fixed seeds). Prints PASS/FAIL for each lemma.
"""
from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy import stats

SEED = 42


# --------------------------------------------------------------------------------------
# Lemma 1: rho(r) = ((4r-1)/(4r+1))^2 at delta = 0, twin peaks only for r > 1/4
# --------------------------------------------------------------------------------------
def _peak_ratio_numeric(r: float, n_theta: int = 200_000) -> tuple[int, float]:
    """Return (n_maxima, peak_height_ratio) for g = cos(theta) + r cos(2 theta) on [0, 2pi)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    g = np.cos(theta) + r * np.cos(2.0 * theta)
    gmin = g.min()
    # local maxima on the circular grid
    left = np.roll(g, 1)
    right = np.roll(g, -1)
    is_max = (g > left) & (g > right)
    peaks = np.sort(g[is_max])[::-1]
    if peaks.size < 2:
        return int(peaks.size), float("nan")
    heights = peaks - gmin
    top2 = np.sort(heights)[::-1][:2]
    return int(is_max.sum()), float(top2[1] / top2[0])


def verify_lemma1(tol: float = 2e-3) -> bool:
    ok = True
    print("Lemma 1 — rho(r) = ((4r-1)/(4r+1))^2 at delta=0")
    for r in [0.30, 0.50, 0.75, 1.0, 1.5, 3.0]:
        n_max, ratio = _peak_ratio_numeric(r)
        predicted = ((4 * r - 1) / (4 * r + 1)) ** 2
        err = abs(ratio - predicted)
        status = "ok" if err < tol else "FAIL"
        ok = ok and err < tol
        print(f"  r={r:4.2f}  n_maxima={n_max}  rho_numeric={ratio:.5f}  "
              f"rho_formula={predicted:.5f}  |err|={err:.2e}  {status}")
    # twin-peak threshold: r < 1/4 -> single maximum; r > 1/4 -> two maxima
    n_below, _ = _peak_ratio_numeric(0.20)
    n_above, _ = _peak_ratio_numeric(0.30)
    thr_ok = (n_below == 1) and (n_above == 2)
    ok = ok and thr_ok
    print(f"  twin-peak threshold: r=0.20 -> {n_below} max, r=0.30 -> {n_above} max  "
          f"{'ok' if thr_ok else 'FAIL'}")
    print(f"  => Lemma 1 {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------------------
# Lemma 2: |s_1| = |h_1| 2|sin(pi beta)|,  |s_2| = |h_2| 2|cos(2 pi beta)|
# --------------------------------------------------------------------------------------
def _fourier_mag(x: np.ndarray, k: int) -> float:
    """Magnitude of the k-th Fourier component of a length-N periodic sample."""
    n = x.size
    coeff = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n) / n)) / n
    return float(abs(coeff))


def verify_lemma2(tol: float = 1e-3) -> bool:
    ok = True
    print("Lemma 2 — fundamental suppression of the intersection archetype")
    n = 4096
    t = np.arange(n) / n  # one period, T = 1
    # smooth unimodal bump (von Mises-like), strictly positive, non-sinusoidal
    kappa = 2.0
    h = np.exp(kappa * np.cos(2.0 * np.pi * t))
    h1, h2 = _fourier_mag(h, 1), _fourier_mag(h, 2)
    print(f"  bump harmonics: |h_1|={h1:.5f}  |h_2|={h2:.5f}")
    for beta in [0.0, 0.02, 0.05, 0.10, 0.20]:
        shift = int(round((0.5 + beta) * n)) % n
        s = h + np.roll(h, shift)  # lambda = 1 anti-phase superposition
        s1, s2 = _fourier_mag(s, 1), _fourier_mag(s, 2)
        pred_s1 = h1 * 2.0 * abs(np.sin(np.pi * beta))
        pred_s2 = h2 * 2.0 * abs(np.cos(2.0 * np.pi * beta))
        e1, e2 = abs(s1 - pred_s1), abs(s2 - pred_s2)
        R = s2 / s1 if s1 > 1e-12 else float("inf")
        status = "ok" if (e1 < tol and e2 < tol) else "FAIL"
        ok = ok and (e1 < tol and e2 < tol)
        print(f"  beta={beta:4.2f}  |s1|={s1:.5f}(pred {pred_s1:.5f})  "
              f"|s2|={s2:.5f}(pred {pred_s2:.5f})  R=A2/A1={R:8.3f}  {status}")
    print("  (R -> inf as beta -> 0: fundamental suppressed, even-harmonic dominant)")
    print(f"  => Lemma 2 {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------------------
# Lemma 3: nested-F noncentrality lambda = (N/2)(A2/sigma)^2 sin^2(Delta)
# --------------------------------------------------------------------------------------
def _rss_full(t: np.ndarray, y: np.ndarray, w: float) -> float:
    """Full model: free f0 and 2f0 phases (linear OLS on 4 sinusoids)."""
    X = np.column_stack([np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t), np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sum((y - X @ beta) ** 2))


def _rss_restricted(t: np.ndarray, y: np.ndarray, w: float, c0: float,
                    n_grid: int = 360) -> float:
    """Restricted model: 2f0 phase locked to 2*(f0 phase) + c0. Grid over the single
    free phase phi1; amplitudes are linear given phi1."""
    best = np.inf
    ones = np.ones_like(t)
    for phi1 in np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False):
        X = np.column_stack([np.cos(w * t + phi1),
                             np.cos(2 * w * t + 2 * phi1 + c0), ones])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ beta) ** 2))
        if rss < best:
            best = rss
    return best


def _theory_power(fcrit: float, df2: int, lam: float) -> float:
    """Noncentral-F survival at fcrit; guard the lam=0 scipy instability (central F -> alpha)."""
    if lam < 1e-6:
        return float(stats.f.sf(fcrit, 1, df2))
    return float(stats.ncf.sf(fcrit, 1, df2, lam))


def _emp_power(rng, t, w, c0, A1, A2, Delta, sigma, N, p_full, fcrit, n_trials) -> float:
    rej = 0
    for _ in range(n_trials):
        phi1 = rng.uniform(0, 2 * np.pi)
        y = (A1 * np.cos(w * t + phi1)
             + A2 * np.cos(2 * w * t + 2 * phi1 + c0 + Delta)
             + rng.normal(0, sigma, N))
        rss_f = _rss_full(t, y, w)
        rss_r = _rss_restricted(t, y, w, c0)
        F = ((rss_r - rss_f) / 1.0) / (rss_f / (N - p_full))
        rej += int(F > fcrit)
    return rej / n_trials


def verify_lemma3(tol_highSNR: float = 0.05, tol_size: float = 0.04) -> bool:
    """The noncentrality lambda = (N/2)(A2/sigma)^2 sin^2(Delta) is the high-A1-SNR /
    subordinate-2f0 (A2 << A1) leading-order result. We verify the three load-bearing
    claims: (i) Delta=0 -> power ~ alpha (entrainment collapse); (ii) power is monotone
    increasing in lambda; (iii) at high lambda it agrees with noncentral-F. Moderate-lambda
    gaps are expected (nonlinear restricted model) and reported, not gated."""
    print("Lemma 3 — phase-freedom nested-F power vs leading-order noncentral-F")
    T = 24.0
    w = 2 * np.pi / T
    N = 48
    t = np.linspace(0.0, 2 * T, N, endpoint=False)  # 2 full periods
    c0 = 0.0
    A1, sigma = 8.0, 1.0  # high fundamental SNR so the leading-order limit applies
    n_trials = 800
    alpha = 0.05
    p_full = 5
    fcrit = float(stats.f.ppf(1 - alpha, 1, N - p_full))
    rng = default_rng(SEED)

    print(f"  regime: A1={A1} (high f0-SNR), A2 subordinate, N={N}, sigma={sigma}")
    emps, lams = [], []
    for Delta in [0.0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
        A2 = 1.0
        lam = (N / 2.0) * (A2 / sigma) ** 2 * np.sin(Delta) ** 2
        emp = _emp_power(rng, t, w, c0, A1, A2, Delta, sigma, N, p_full, fcrit, n_trials)
        theo = _theory_power(fcrit, N - p_full, lam)
        emps.append(emp); lams.append(lam)
        print(f"  Delta={Delta:5.3f}  lambda={lam:6.2f}  power_emp={emp:.3f}  "
              f"power_theory={theo:.3f}")

    size_ok = abs(emps[0] - alpha) < tol_size          # (i) Delta=0 collapse to alpha
    mono_ok = all(emps[i] <= emps[i + 1] + 0.03 for i in range(len(emps) - 1))  # (ii) monotone
    theo_hi = _theory_power(fcrit, N - p_full, lams[-1])
    highsnr_ok = abs(emps[-1] - theo_hi) < tol_highSNR  # (iii) high-lambda agreement
    print(f"  (i) size at Delta=0: emp={emps[0]:.3f} ~ alpha={alpha}  "
          f"{'ok' if size_ok else 'FAIL'}")
    print(f"  (ii) power monotone increasing in lambda: {'ok' if mono_ok else 'FAIL'}")
    print(f"  (iii) high-lambda agreement (lambda={lams[-1]:.0f}): "
          f"emp={emps[-1]:.3f} vs theory={theo_hi:.3f}  {'ok' if highsnr_ok else 'FAIL'}")

    # (iv) SNR scaling at fixed Delta=pi/2: lambda ~ A2^2, power must rise monotonically
    scale = [_emp_power(rng, t, w, c0, A1, a2, np.pi / 2, sigma, N, p_full, fcrit, n_trials)
             for a2 in [0.4, 0.8, 1.4]]
    scale_ok = scale[0] < scale[1] < scale[2]
    print(f"  (iv) SNR scaling (A2=0.4/0.8/1.4 at Delta=pi/2): "
          f"power={scale[0]:.3f}/{scale[1]:.3f}/{scale[2]:.3f}  "
          f"{'ok' if scale_ok else 'FAIL'}")

    ok = size_ok and mono_ok and highsnr_ok and scale_ok
    print(f"  => Lemma 3 {'PASS' if ok else 'FAIL'} "
          f"(formula is the high-SNR leading-order noncentrality)\n")
    return ok


def main() -> int:
    print("=" * 74)
    print("CHORD analytic-lemma verification (U4)")
    print("=" * 74)
    r1 = verify_lemma1()
    r2 = verify_lemma2()
    r3 = verify_lemma3()
    allok = r1 and r2 and r3
    print("=" * 74)
    print(f"OVERALL: {'ALL LEMMAS PASS' if allok else 'SOME LEMMAS FAILED'}")
    print("=" * 74)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
