"""Export Figure 3b disentanglement AUCs (the three pairwise contrasts) with the shipped
five-feature model on a held-out benchmark. Reuses scripts/eval_ternary_refit.py so the
numbers match the B-vs-C gate exactly (train seed 8108, test seed 20260703).

Panel a (recovery of 38 known 12h genes; fig3_detection.csv) is dataset-derived and
model-independent, so it is NOT regenerated here. Deterministic.
Run:  python scripts/export_fig3_data.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from eval_ternary_refit import (  # noqa: E402  (same dir on sys.path when run as a script)
    _build, _matrix, _fit_predict, _contrasts, TRAIN_SEED, TEST_SEED,
)

FULL5 = ["F1", "logR", "twin", "decay", "logSNR"]      # the shipped five-feature vector
LEGACY_AUTONOMOUS_REF = 0.773                          # additive 12-evidence discriminator
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data", "fig3_disentangle.csv")


def _row(name: str, tup, ref: str = "") -> tuple:
    auc, (lo, hi) = tup
    return (name, f"{auc:.3f}", f"{lo:.3f}", f"{hi:.3f}", ref)


def main() -> int:
    tr_rows, ytr = _build(TRAIN_SEED)
    te_rows, yte = _build(TEST_SEED)
    Xtr, Xte = _matrix(tr_rows, FULL5), _matrix(te_rows, FULL5)
    proba = _fit_predict(Xtr, ytr, Xte)
    rng = np.random.default_rng(TEST_SEED)
    c = _contrasts(proba, yte, rng)
    rows = [
        _row("autonomous vs driven", c["B_vs_driven"], f"{LEGACY_AUTONOMOUS_REF:.3f}"),
        _row("oscillator vs intersection", c["B_vs_C"]),
        _row("harmonic vs intersection", c["A_vs_C"]),
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "auc", "lo", "hi", "ref"])
        w.writerows(rows)
    for r in rows:
        print(f"  {r[0]:28s} AUC={r[1]}  [{r[2]}, {r[3]}]" + (f"  (legacy {r[4]})" if r[4] else ""))
    print(f"wrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
