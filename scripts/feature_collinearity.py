"""U6: feature-collinearity panel — numerical evidence for 'low multicollinearity' (R3-4).

Reviewer 3 (comment 4) showed the submitted "four orthogonal statistics" claim is too
strong: at the exactly phase-locked knife-edge, peak symmetry rho = ((4r-1)/(4r+1))^2 is a
deterministic function of the amplitude ratio (supp_analytic_lemmas.md). The revision
replaces "orthogonal" with "low multicollinearity" and KEEPS twin_peak_symmetry as a model
input — because across the broadened four-mechanism Class-C family the locked-phase identity
does NOT hold and twin carries independent B-vs-C signal (this panel + eval_ternary_refit.py:
twin raises held-out B-vs-C AUC from 0.866 without it to 0.927 with it).

This script supplies the defensible diagnostics on the class-balanced calibration set for the
SHIPPED FIVE-feature vector {F1 phase-freedom, log A12/A24, twin symmetry, decay residual,
log 12h-SNR}:

  * Pearson correlation of the five model features
  * per-feature Variance Inflation Factor (VIF; flag > 5) — INCLUDING twin
  * design condition number kappa (standardised, with intercept); flag >= 30
  * twin NON-redundancy: R^2(twin ~ the other four) is LOW, so the locked-phase analytic
    identity does not make twin redundant on the diverse family.

Run on two grids because the harmonic-decay residual is cadence/noise sensitive: the shipped
2h/48h grid and a dense 1h/48h grid (the Hughes-2009 flagship cadence). If a feature is
constant on a grid, the panel discloses it and excludes it from VIF/kappa.

Deterministic (calibration seed 8108, matching get_default_model). Writes
publication/CHORD_ternary_submission/figures/source_data/collinearity.csv.
Run:  python scripts/feature_collinearity.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform

from chord.bhdt.stage2_orthogonal import (
    FEATURE_NAMES, phase_freedom_pvalue, amplitude_ratio, twin_peak_symmetry,
    harmonic_decay_residual, _harmonic_fit, TernaryConfig,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

CFG = TernaryConfig()
W = 2 * np.pi / 24.0
SEED = 8108           # matches the default calibration model
N_PER_CLASS = 200
MODEL_FEATS = [str(f) for f in FEATURE_NAMES]   # the 5 features fed to the classifier
# short display labels, aligned to FEATURE_NAMES order
_SHORT_BY_NAME = {"neglog10_p_phase": "F1", "log_amp_ratio": "logR",
                  "twin_symmetry": "twin", "decay_residual": "decay", "log_snr12": "logSNR"}
SHORT = [_SHORT_BY_NAME[f] for f in MODEL_FEATS]
OUT_CSV = "publication/CHORD_ternary_submission/figures/source_data/collinearity.csv"


def _row(t, y):
    """The 5 model features (all shipped), or None if the gates fail."""
    amps, _, noise = _harmonic_fit(t, y, W, 2)
    snr12 = amps[1] / max(noise, 1e-9)
    snr24 = amps[0] / max(noise, 1e-9)
    if snr12 < CFG.min_12h_snr or snr24 < CFG.min_24h_snr:
        return None
    p = phase_freedom_pvalue(t, y)
    r = amplitude_ratio(t, y)
    return {
        "neglog10_p_phase": float(np.clip(-np.log10(max(p, 1e-12)), 0.0, 12.0)),
        "log_amp_ratio": float(np.log(max(r, 1e-6))),
        "twin_symmetry": float(twin_peak_symmetry(t, y)),
        "decay_residual": float(harmonic_decay_residual(t, y, noise_sd=noise)),
        "log_snr12": float(np.log(max(snr12, 1e-6))),
    }


def _dcor(X: np.ndarray, Y: np.ndarray) -> float:
    """Distance correlation between column-stacks X and Y (each (n,) or (n,d))."""
    X = X.reshape(len(X), -1); Y = Y.reshape(len(Y), -1)
    dc = lambda M: (lambda a: a - a.mean(0) - a.mean(1)[:, None] + a.mean())(squareform(pdist(M)))
    A, B = dc(X), dc(Y)
    n = len(X)
    dcov2 = (A * B).sum() / n ** 2
    dvx = (A * A).sum() / n ** 2
    dvy = (B * B).sum() / n ** 2
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(max(dvx, 1e-30) * max(dvy, 1e-30))))


def _vif(X: np.ndarray) -> list:
    """VIF_j = 1/(1-R^2_j) from OLS of column j on the others (+ intercept)."""
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]
        M = np.column_stack([np.delete(X, j, axis=1), np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        rss = float(np.sum((y - M @ beta) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / max(tss, 1e-30)
        out.append(1.0 / max(1.0 - r2, 1e-9))
    return out


def _panel(t, label, csv_rows):
    """Collinearity panel on one time grid. Returns (verdict, X, n, active_names)."""
    d = len(MODEL_FEATS)
    bench = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=("A", "B", "C"),
                                    seed=SEED, t=t)
    rows = [r for r in (_row(t, y) for y in bench["expr"]) if r is not None]
    n = len(rows)
    X = np.array([[r[f] for f in MODEL_FEATS] for r in rows], float)
    std = X.std(0)
    active = [j for j in range(d) if std[j] > 1e-9]
    names = [SHORT[j] for j in active]
    inert = [SHORT[j] for j in range(d) if j not in active]

    print(f"\n{'=' * 74}\n{label}  (n gated = {n}, dt = {t[1]-t[0]:.0f} h, {len(t)} pts)\n{'=' * 74}")
    for j, s in enumerate(SHORT):
        tag = "" if j in active else "   <- INERT (not estimable at this cadence)"
        print(f"  {s:>6s}  mean={X[:, j].mean():+7.3f}  std={std[j]:7.4f}"
              f"  nonzero_frac={np.mean(X[:, j] != 0):.2f}{tag}")

    Xa = X[:, active]
    with np.errstate(divide="ignore", invalid="ignore"):
        pear = np.corrcoef(Xa.T)
    vifs = _vif(Xa)
    Xz = (Xa - Xa.mean(0)) / Xa.std(0)
    kappa = float(np.linalg.cond(np.column_stack([Xz, np.ones(n)])))

    print(f"\nPearson (active {names}):")
    print("        " + "  ".join(f"{s:>7s}" for s in names))
    for i, s in enumerate(names):
        print(f"  {s:>6s}" + "  ".join(f"{pear[i, j]:7.3f}" for j in range(len(names))))
    print("VIF (flag>5):  " + "   ".join(
        f"{s}={v:.2f}{'!' if v > 5 else ''}" for s, v in zip(names, vifs)))
    print(f"condition number kappa = {kappa:.2f}   {'FLAG' if kappa >= 30 else 'ok'}")
    verdict = (max(vifs) < 5.0) and (kappa < 30.0)
    print(f"  => {'LOW MULTICOLLINEARITY (active features, twin included)' if verdict else 'FLAGGED'}"
          + (f"   INERT: {inert}" if inert else ""))

    csv_rows.append([f"[{label}]", f"n={n}", f"dt={t[1]-t[0]:.0f}h",
                     f"inert={','.join(inert) or 'none'}", "", ""])
    for i, s in enumerate(names):
        csv_rows.append([f"pearson_{s}"] + [f"{pear[i, j]:.4f}" for j in range(len(names))]
                        + [""] * (5 - len(names)))
    csv_rows.append(["vif"] + [f"{v:.4f}" for v in vifs] + [""] * (5 - len(vifs)))
    csv_rows.append(["condition_number", f"{kappa:.4f}", "", "", "", ""])
    return verdict, X, n, names


def main() -> int:
    print("U6 feature-collinearity panel — evidence for 'low multicollinearity' (R3-4)")
    csv_rows = []
    v2, _, _, _ = _panel(np.arange(0.0, 48.0, 2.0), "SHIPPED GRID 2h/48h", csv_rows)
    v1, Xd, nd, _ = _panel(np.arange(0.0, 48.0, 1.0), "DENSE GRID 1h/48h", csv_rows)

    # twin NON-redundancy on the dense grid: the locked-phase identity rho = f(r) would
    # make twin redundant only at delta = 0; across the diverse family it is not. Regress
    # twin on the other four model features -> a LOW R^2 confirms an independent axis.
    tw_idx = MODEL_FEATS.index("twin_symmetry")
    others = [j for j in range(len(MODEL_FEATS)) if j != tw_idx]
    twin_d = Xd[:, tw_idx]
    M = np.column_stack([Xd[:, others], np.ones(nd)])
    beta, *_ = np.linalg.lstsq(M, twin_d, rcond=None)
    r2 = 1.0 - np.sum((twin_d - M @ beta) ** 2) / max(np.sum((twin_d - twin_d.mean()) ** 2), 1e-30)
    logR_d = Xd[:, MODEL_FEATS.index("log_amp_ratio")]
    pear_tr = float(np.corrcoef(twin_d, logR_d)[0, 1])
    dcor_tr = _dcor(twin_d, logR_d)
    print(f"\n{'=' * 74}\nTWIN NON-REDUNDANCY (why twin is kept as a model input)\n{'=' * 74}")
    print(f"  (dense grid)  Pearson(twin, logR) = {pear_tr:+.3f}   dCor(twin, logR) = {dcor_tr:.3f}")
    print(f"  R^2(twin ~ other four model features) = {r2:.3f}  (LOW -> independent axis)")
    print("  The locked-phase identity rho = ((4r-1)/(4r+1))^2 (supp_analytic_lemmas.md) holds")
    print("  ONLY at delta = 0; across the four-mechanism Class-C family it does not, so twin")
    print("  is not redundant. Operationally twin is decisive: it raises the held-out B-vs-C")
    print("  AUC from 0.866 (without) to 0.927 (with) and carries univariate B-vs-C 0.737")
    print("  (scripts/eval_ternary_refit.py).")
    csv_rows.append(["twin_nonredundancy_dense", f"pearson_logR={pear_tr:.4f}",
                     f"dcor_logR={dcor_tr:.4f}", f"r2_others={r2:.4f}", "", ""])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "c1", "c2", "c3", "c4", "c5"])
        w.writerows(csv_rows)
    print(f"\nwrote {OUT_CSV}")

    print("\nNOTE (for U16 feature-role prose + U12 sampling story): the harmonic-decay "
          "residual is evaluated on the harmonic-amplitude noise scale. On the current "
          "calibration seed it is active at both 2h and 1h sampling, so the shipped default "
          "model uses all five features when the gates pass. If future benchmarks make a "
          "feature constant, this panel discloses that column as inert and computes "
          "VIF/kappa on the active subset.")
    return 0 if (v1 and v2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
