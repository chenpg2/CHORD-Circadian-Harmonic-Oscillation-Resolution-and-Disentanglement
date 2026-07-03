"""R3-m5: classifier hyperparameter sensitivity. The three disentanglement AUCs are
recomputed on the held-out set while sweeping the L2 penalty and the class-weighting of the
multinomial (the abstention threshold is rank-free, so it does not affect AUC — it only trades
coverage for confidence, quantified separately by the abstention rate). Reuses
scripts/eval_ternary_refit.py. Deterministic.

Writes figures/source_data/hyperparameter.csv. Run:  python scripts/export_hyperparameter.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from eval_ternary_refit import (  # noqa: E402
    _build, _matrix, _contrasts, TRAIN_SEED, TEST_SEED,
)
from chord.bhdt.stage2_orthogonal import _fit_multinomial, _softmax

FULL5 = ["F1", "logR", "twin", "decay", "logSNR"]
L2_GRID = [0.3, 1.0, 3.0]
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data", "hyperparameter.csv")


def _fit_predict(Xtr, ytr, Xte, l2, balanced):
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-9] = 1.0
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = (np.where(counts > 0, len(ytr) / (3 * np.maximum(counts, 1)), 0.0)
          if balanced else np.ones(3))
    W = _fit_multinomial((Xtr - mean) / std, ytr, 3, l2=l2, class_weight=cw)
    Zte = np.column_stack([(Xte - mean) / std, np.ones(len(Xte))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _softmax(Zte @ W.T)


def main() -> int:
    tr, ytr = _build(TRAIN_SEED)
    te, yte = _build(TEST_SEED)
    Xtr, Xte = _matrix(tr, FULL5), _matrix(te, FULL5)

    print(f"{'l2':>5s} {'weighting':>10s}  {'B-vs-driven':>12s} {'B-vs-C':>8s} {'A-vs-C':>8s}")
    rows = []
    for l2 in L2_GRID:
        for bal in (True, False):
            proba = _fit_predict(Xtr, ytr, Xte, l2, bal)
            c = _contrasts(proba, yte, np.random.default_rng(TEST_SEED))
            bd, bc, ac = c["B_vs_driven"][0], c["B_vs_C"][0], c["A_vs_C"][0]
            tag = "balanced" if bal else "uniform"
            rows.append((l2, tag, bd, bc, ac))
            print(f"{l2:5.1f} {tag:>10s}  {bd:12.3f} {bc:8.3f} {ac:8.3f}")

    arr = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"\nAUC range across all {len(rows)} settings: "
          f"B-vs-driven [{arr[:,0].min():.3f},{arr[:,0].max():.3f}]  "
          f"B-vs-C [{arr[:,1].min():.3f},{arr[:,1].max():.3f}]  "
          f"A-vs-C [{arr[:,2].min():.3f},{arr[:,2].max():.3f}]")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["l2", "class_weighting", "auc_B_vs_driven", "auc_B_vs_C", "auc_A_vs_C"])
        for l2, tag, bd, bc, ac in rows:
            w.writerow([f"{l2:.1f}", tag, f"{bd:.4f}", f"{bc:.4f}", f"{ac:.4f}"])
    print(f"wrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
