"""U5 B-vs-C gate (RESOLVED: KEEP twin_peak_symmetry). Does dropping twin preserve the
ternary contrast? On the broadened four-mechanism Class-C family it does NOT, so the
shipped method retains all FIVE features.

The submitted method used FIVE features {F1 phase-freedom, log A12/A24, twin_symmetry,
decay_residual, log SNR12}. Review PF3 (adversarial) warned that removing twin_peak_symmetry
(which the manuscript credits with the B-vs-C AUC 0.87-0.95) could collapse the
autonomous-vs-intersection axis. An early single-mechanism benchmark made twin look
redundant (DROP4 tied FULL5 near 1.0), but the broadened Class-C family (four structurally
distinct intersection mechanisms: harmonic ODE, rectified, saturating, product) reverses
that verdict. This script decides it EMPIRICALLY on a held-out benchmark, comparing the
pairwise AUCs of:

    FULL5  : {F1, logR, twin, decay, logSNR}   (submitted; SHIPPED)
    DROP4  : {F1, logR, decay, logSNR}          (twin removed)
    ALT5   : DROP4 + relative-phase delta_hat   (secondary comparison)

Gate: ship DROP4 iff its B_independent-vs-C_intersection AUC does NOT fall materially below
FULL5 (>= FULL5 lower 95% CI) AND stays above a pre-registered floor of 0.87.

Result on the diverse family: FULL5 B-vs-C 0.927 vs DROP4 0.866 (below the 0.87 floor);
twin's univariate B-vs-C is 0.737, the strongest single feature. The gate FAILS for DROP4
-> keep twin (the shipped five-feature vector). Saturating (fully fundamental-suppressed) is
gated out of this 24h-bearing AUC set and handled as 'ambiguous' by the identifiability
floor (Option A; scripts/measure_option_a_impact.py).

Deterministic. Run:  python scripts/eval_ternary_refit.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from chord.bhdt.stage2_orthogonal import (
    phase_freedom_pvalue, amplitude_ratio, twin_peak_symmetry,
    harmonic_decay_residual, relative_phase, _harmonic_fit,
    _fit_multinomial, _softmax, TernaryConfig,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

CLASSES = ("A", "B", "C")
CIDX = {c: i for i, c in enumerate(CLASSES)}
CFG = TernaryConfig()
GATE_FLOOR = 0.87           # pre-registered B-vs-C floor
TRAIN_SEED = 8108           # matches get_default_model()
TEST_SEED = 20260703        # held out from the model's calibration seed
N_PER_CLASS = 200
BOOT = 2000
W = 2 * np.pi / 24.0


def _auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """AUC of `scores` separating pos (bool) from ~pos, tie-corrected (Mann-Whitney)."""
    s = np.asarray(scores, float); p = np.asarray(pos, bool)
    npos, nneg = int(p.sum()), int((~p).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[p].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _boot_ci(scores, pos, rng, n=BOOT):
    idx = np.arange(len(scores))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        a = _auc(scores[b], pos[b])
        if np.isfinite(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _features(t, y):
    """All candidate features for one gene, matching extract_features' transforms.
    Returns (row_dict, passed_gates) — None row if the gene fails the fit/gates."""
    amps, _, noise = _harmonic_fit(t, y, W, 2)
    snr12 = amps[1] / max(noise, 1e-9)
    snr24 = amps[0] / max(noise, 1e-9)
    if snr12 < CFG.min_12h_snr or snr24 < CFG.min_24h_snr:   # same gates as fit_ternary_model
        return None
    p = phase_freedom_pvalue(t, y)
    r = amplitude_ratio(t, y)
    delta = relative_phase(t, y)
    return {
        "F1": float(np.clip(-np.log10(max(p, 1e-12)), 0.0, 12.0)),
        "logR": float(np.log(max(r, 1e-6))),
        "twin": float(twin_peak_symmetry(t, y)),
        "decay": float(harmonic_decay_residual(t, y, noise_sd=noise)),
        "logSNR": float(np.log(max(snr12, 1e-6))),
        # delta_hat encoded as distance to the nearest locked phase {0, +-pi}, in [0, pi/2]
        "dphase": float(min(abs(delta), abs(abs(delta) - np.pi))),
    }


def _build(seed):
    bench = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=CLASSES, seed=seed)
    t = bench["t"]
    rows, ys = [], []
    for y, lab in zip(bench["expr"], bench["labels"]):
        if lab not in CIDX:
            continue
        f = _features(t, y)
        if f is None:
            continue
        rows.append(f); ys.append(CIDX[lab])
    return rows, np.array(ys)


def _matrix(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], float)


def _fit_predict(Xtr, ytr, Xte):
    """Standardise on train, class-balanced multinomial, return test P(A,B,C)."""
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-9] = 1.0
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = np.where(counts > 0, len(ytr) / (3 * np.maximum(counts, 1)), 0.0)
    W_ = _fit_multinomial((Xtr - mean) / std, ytr, 3, l2=1.0, class_weight=cw)
    Zte = np.column_stack([(Xte - mean) / std, np.ones(len(Xte))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _softmax(Zte @ W_.T)


def _contrasts(proba, yte, rng):
    """Pairwise AUCs (+CI) from test posteriors. proba columns = A,B,C (idx 0,1,2)."""
    A, B, C = yte == 0, yte == 1, yte == 2
    out = {}
    # autonomous (B) vs driven {A,C}
    m = np.ones(len(yte), bool)
    out["B_vs_driven"] = (_auc(proba[:, 1][m], B[m]), _boot_ci(proba[:, 1][m], B[m], rng))
    # B vs C (the axis at risk)
    m = B | C
    sc = proba[m, 1] / np.clip(proba[m, 1] + proba[m, 2], 1e-12, None)
    out["B_vs_C"] = (_auc(sc, B[m]), _boot_ci(sc, B[m], rng))
    # A (harmonic) vs C (intersection)
    m = A | C
    sc = proba[m, 0] / np.clip(proba[m, 0] + proba[m, 2], 1e-12, None)
    out["A_vs_C"] = (_auc(sc, A[m]), _boot_ci(sc, A[m], rng))
    return out


def main() -> int:
    print("=" * 72)
    print("U5 feature-set evaluation on held-out ternary benchmark")
    print("=" * 72)
    tr_rows, ytr = _build(TRAIN_SEED)
    te_rows, yte = _build(TEST_SEED)
    print(f"train gated n={len(ytr)} (A/B/C={np.bincount(ytr)}), "
          f"test gated n={len(yte)} (A/B/C={np.bincount(yte)})")

    variants = {
        "FULL5 (submitted, +twin)": ["F1", "logR", "twin", "decay", "logSNR"],
        "DROP4 (proposed, no twin)": ["F1", "logR", "decay", "logSNR"],
        "ALT5  (DROP4 + delta_hat)": ["F1", "logR", "decay", "logSNR", "dphase"],
    }
    rng = np.random.default_rng(0)
    results = {}
    for name, cols in variants.items():
        proba = _fit_predict(_matrix(tr_rows, cols), ytr, _matrix(te_rows, cols))
        results[name] = _contrasts(proba, yte, rng)
        print(f"\n{name}  features={cols}")
        for k, (a, (lo, hi)) in results[name].items():
            print(f"    {k:14s} AUC = {a:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")

    # Univariate B-vs-C AUC of each raw feature (which feature carries the axis?)
    print("\nUnivariate B-vs-C AUC (which feature separates autonomous from intersection):")
    Bte = yte == 1; Cte = yte == 2; m = Bte | Cte
    for c in ["twin", "logR", "F1", "decay", "dphase"]:
        col = _matrix(te_rows, [c])[:, 0][m]
        a = _auc(col, Bte[m]); a = max(a, 1 - a)  # direction-agnostic
        print(f"    {c:8s} |AUC-0.5|-oriented = {a:.3f}")

    # ---- Gate decision -------------------------------------------------------
    # Compare against FULL5's point estimate within a 0.01-AUC tolerance: FULL5's
    # bootstrap CI can be degenerate (all replicates = 1.000) on a single-generator C
    # benchmark, so a strict ">= lower-CI" test is meaningless; use an absolute tolerance.
    # On the broadened four-mechanism family the CIs are non-degenerate and DROP4 fails.
    TOL = 0.01
    full_bc, _ = results["FULL5 (submitted, +twin)"]["B_vs_C"]
    drop_bc, _ = results["DROP4 (proposed, no twin)"]["B_vs_C"]
    alt_bc, _ = results["ALT5  (DROP4 + delta_hat)"]["B_vs_C"]
    print("\n" + "-" * 72)
    print(f"GATE: B-vs-C  FULL5={full_bc:.4f}  DROP4={drop_bc:.4f}  "
          f"ALT5={alt_bc:.4f}  (tol={TOL}, floor={GATE_FLOOR})")
    passed = (drop_bc >= full_bc - TOL) and (drop_bc >= GATE_FLOOR)
    if passed:
        print(f"  => PASS: DROP4 preserves B-vs-C (within {TOL} of FULL5, above floor). "
              "Ship the 4-feature vector (twin redundant on THIS benchmark).")
    elif alt_bc >= max(full_bc - TOL, GATE_FLOOR):
        print("  => DROP4 marginal, ALT5 recovers: ship DROP4 + delta_hat (5th feature).")
    else:
        print("  => FAIL: neither recovers B-vs-C — retain a shape feature (keep twin).")
    print("-" * 72)
    return 0 if (passed or alt_bc >= max(full_bc - TOL, GATE_FLOOR)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
