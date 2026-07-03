"""R3-m3: robustness of disentanglement to sparser sampling. The generative benchmark is
built at 1h, 2h, 3h, and 4h cadence over 48h; the five-feature model is re-fit and evaluated
on a held-out set at each cadence, tracking the three pairwise AUCs and the gated retention.

Coarser sampling degrades classification only through the fewer-points -> lower-SNR route
(consistent with the identifiability budget, Figure 5): the harmonic-vs-intersection and
autonomous-vs-driven axes are near-flat down to 3h and soften at 4h (12 points/48h), where the
harmonic fits lose degrees of freedom. Reuses scripts/eval_ternary_refit.py. Deterministic.

Writes figures/source_data/downsampling.csv. Run:  python scripts/export_downsampling.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from eval_ternary_refit import (  # noqa: E402 (same dir on sys.path when run as a script)
    _features, _matrix, _fit_predict, _contrasts, CIDX, CLASSES, TRAIN_SEED, TEST_SEED,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

FULL5 = ["F1", "logR", "twin", "decay", "logSNR"]
CADENCES = [1.0, 2.0, 3.0, 4.0]
N_PER_CLASS = 200
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data", "downsampling.csv")


def _build_at(t: np.ndarray, seed: int):
    b = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=CLASSES, seed=seed, t=t)
    rows, ys = [], []
    for y, lab in zip(b["expr"], b["labels"]):
        f = _features(t, y)
        if f is None:
            continue
        rows.append(f); ys.append(CIDX[lab])
    return rows, np.array(ys)


def main() -> int:
    print(f"{'cadence':>8s} {'pts':>4s} {'gated':>6s}  {'B-vs-driven':>12s} {'B-vs-C':>8s} {'A-vs-C':>8s}")
    results = []
    for dt in CADENCES:
        t = np.arange(0.0, 48.0, dt)
        tr, ytr = _build_at(t, TRAIN_SEED)
        te, yte = _build_at(t, TEST_SEED)
        proba = _fit_predict(_matrix(tr, FULL5), ytr, _matrix(te, FULL5))
        rng = np.random.default_rng(TEST_SEED)
        c = _contrasts(proba, yte, rng)
        bd, bc, ac = c["B_vs_driven"][0], c["B_vs_C"][0], c["A_vs_C"][0]
        results.append((dt, len(t), len(yte), bd, bc, ac))
        print(f"{dt:6.0f}h  {len(t):4d} {len(yte):6d}  {bd:12.3f} {bc:8.3f} {ac:8.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cadence_h", "n_points", "n_gated_test", "auc_B_vs_driven", "auc_B_vs_C", "auc_A_vs_C"])
        for dt, npt, ng, bd, bc, ac in results:
            w.writerow([f"{dt:.0f}", npt, ng, f"{bd:.4f}", f"{bc:.4f}", f"{ac:.4f}"])
    print(f"\nwrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
