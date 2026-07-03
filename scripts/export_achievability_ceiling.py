"""R1-b: is the near-perfect harmonic-vs-intersection AUC a simulator<->feature artifact?

The exact Bayes-optimal discriminator is intractable for these nonlinear/ODE generators, so we
report an ACHIEVABILITY CEILING: a much richer interpretable representation of the full waveform
— the first six harmonic amplitudes, the relative phases of harmonics 2 and 3, and the 12h-SNR
(nine features) — fit with the same calibrated multinomial. If this rich representation does not
materially beat CHORD's five features, then CHORD is not leaving discriminative information on the
table, and the high AUCs reflect genuine class separability rather than feature suboptimality or a
lucky simulator<->feature match. The residual non-identifiability (AUC -> 0.5 as the generator
families overlap) is the analytic limit conceded in supp_analytic_lemmas.md. Deterministic.

Writes figures/source_data/achievability_ceiling.csv. Run: python scripts/export_achievability_ceiling.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from chord.bhdt.stage2_orthogonal import (
    extract_features, _harmonic_fit, TernaryConfig, _fit_multinomial, _softmax,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark
from eval_ternary_refit import _contrasts, CIDX, CLASSES, TRAIN_SEED, TEST_SEED

CFG = TernaryConfig()
W = 2 * np.pi / 24.0
N_PER_CLASS = 200
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data", "achievability_ceiling.csv")


def _rich(t: np.ndarray, y: np.ndarray, k: int = 6) -> list:
    """Rich full-waveform representation: 6 log harmonic amplitudes + relative phases of
    harmonics 2,3 vs the fundamental + log 12h-SNR (nine phase-aware spectral features)."""
    n = len(t)
    cols = [np.ones(n)]
    for j in range(1, k + 1):
        cols += [np.cos(j * W * t), np.sin(j * W * t)]
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    amp = [float(np.hypot(beta[1 + 2 * (j - 1)], beta[2 + 2 * (j - 1)])) for j in range(1, k + 1)]
    phi = [float(np.arctan2(beta[2 + 2 * (j - 1)], beta[1 + 2 * (j - 1)])) for j in range(1, k + 1)]
    noise = float(np.sqrt(max(np.sum((y - X @ beta) ** 2) / max(n - (1 + 2 * k), 1), 1e-12)))
    d2 = float(np.mod(phi[1] - 2 * phi[0] + np.pi, 2 * np.pi) - np.pi)
    d3 = float(np.mod(phi[2] - 3 * phi[0] + np.pi, 2 * np.pi) - np.pi)
    feats = [np.log(max(a, 1e-9)) for a in amp] + [d2, d3,
             float(np.log(max(amp[1] / max(noise, 1e-9), 1e-6)))]
    return feats


def _build(seed: int):
    b = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=CLASSES, seed=seed)
    t = b["t"]
    Xc, Xr, ys = [], [], []
    for y, lab in zip(b["expr"], b["labels"]):
        amps, _, noise = _harmonic_fit(t, y, W, 2)
        s12, s24 = amps[1] / max(noise, 1e-9), amps[0] / max(noise, 1e-9)
        if s12 < CFG.min_12h_snr or s24 < CFG.min_24h_snr:
            continue
        Xc.append(extract_features(t, y, noise_sd=noise))
        Xr.append(_rich(t, y))
        ys.append(CIDX[lab])
    return np.array(Xc), np.array(Xr), np.array(ys)


def _fit_predict(Xtr, ytr, Xte):
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-9] = 1.0
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = np.where(counts > 0, len(ytr) / (3 * np.maximum(counts, 1)), 0.0)
    W_ = _fit_multinomial((Xtr - mean) / std, ytr, 3, l2=1.0, class_weight=cw)
    Zte = np.column_stack([(Xte - mean) / std, np.ones(len(Xte))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _softmax(Zte @ W_.T)


def main() -> int:
    Xc_tr, Xr_tr, ytr = _build(TRAIN_SEED)
    Xc_te, Xr_te, yte = _build(TEST_SEED)
    print(f"gated train n={len(ytr)}, test n={len(yte)}")
    print(f"{'representation':>22s}  {'B-vs-driven':>12s} {'B-vs-C':>8s} {'A-vs-C':>8s}")
    out = {}
    for name, Xtr, Xte in [("CHORD (5 features)", Xc_tr, Xc_te),
                           ("rich spectrum (9, ceiling)", Xr_tr, Xr_te)]:
        proba = _fit_predict(Xtr, ytr, Xte)
        c = _contrasts(proba, yte, np.random.default_rng(TEST_SEED))
        out[name] = (c["B_vs_driven"][0], c["B_vs_C"][0], c["A_vs_C"][0])
        print(f"{name:>22s}  {out[name][0]:12.3f} {out[name][1]:8.3f} {out[name][2]:8.3f}")
    a = out["CHORD (5 features)"]; b = out["rich spectrum (9, ceiling)"]
    print(f"\ngap (ceiling - CHORD): B-vs-driven {b[0]-a[0]:+.3f}  "
          f"B-vs-C {b[1]-a[1]:+.3f}  A-vs-C {b[2]-a[2]:+.3f}")
    print("small/negative gaps => CHORD's features are near-optimal; the high AUCs are real "
          "separability, not a simulator<->feature artifact.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["representation", "auc_B_vs_driven", "auc_B_vs_C", "auc_A_vs_C"])
        for name in out:
            w.writerow([name] + [f"{v:.4f}" for v in out[name]])
    print(f"wrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
