"""R3-m2: three-class confusion matrix + posterior calibration for the ternary classifier.

On a held-out benchmark (test seed 20260703, disjoint from the model's calibration seed 8108),
the shipped five-feature model classifies every gated (24h-bearing) A/B/C gene. We report:

  * the 3x3 confusion matrix (true vs argmax-predicted) and per-class precision/recall,
    plus the abstention rate (max posterior < abstain_prob);
  * posterior calibration: genes binned by predicted confidence (max posterior), with the
    empirical accuracy per bin and the Expected Calibration Error (ECE).

Writes figures/source_data/{confusion.csv, calibration.csv}. Deterministic.
Run:  python scripts/export_confusion_calibration.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from chord.bhdt.stage2_orthogonal import (
    get_default_model, extract_features, _harmonic_fit, TernaryConfig,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

CFG = TernaryConfig()
W = 2 * np.pi / 24.0
TEST_SEED = 20260703
N_PER_CLASS = 200
CLASSES = ["A", "B", "C"]
N_BINS = 8
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data")


def main() -> int:
    model = get_default_model()
    bench = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=("A", "B", "C"),
                                    seed=TEST_SEED)
    t = bench["t"]
    true_i, pred_i, conf, correct = [], [], [], []
    n_abstain = 0
    for y, lab in zip(bench["expr"], bench["labels"]):
        amps, _, noise = _harmonic_fit(t, y, W, 2)
        s12 = amps[1] / max(noise, 1e-9)
        s24 = amps[0] / max(noise, 1e-9)
        if s12 < CFG.min_12h_snr or s24 < CFG.min_24h_snr:   # gated set (the multinomial's domain)
            continue
        proba = model.predict_proba(extract_features(t, y, noise_sd=noise))
        p = np.array([proba[c] for c in CLASSES])
        ti, pi = CLASSES.index(lab), int(np.argmax(p))
        true_i.append(ti); pred_i.append(pi)
        conf.append(float(p.max())); correct.append(pi == ti)
        n_abstain += int(p.max() < CFG.abstain_prob)

    true_i = np.array(true_i); pred_i = np.array(pred_i)
    conf = np.array(conf); correct = np.array(correct, bool)
    n = len(true_i)

    # ---- confusion matrix (rows=true, cols=argmax-predicted) --------------------------
    cm = np.zeros((3, 3), int)
    for ti, pi in zip(true_i, pred_i):
        cm[ti, pi] += 1
    acc = correct.mean()
    print("=" * 70)
    print(f"Confusion matrix (held-out test, n={n} gated; overall accuracy {acc:.3f}, "
          f"abstain {n_abstain}/{n}={n_abstain/n:.1%})")
    print("            pred_A  pred_B  pred_C   recall")
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        print(f"  true_{c}    {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}   {rec:.3f}")
    prec = [cm[i, i] / max(cm[:, i].sum(), 1) for i in range(3)]
    print("  precision " + "  ".join(f"{prec[i]:6.3f}" for i in range(3)))

    # ---- calibration: bin by confidence, accuracy per bin, ECE ------------------------
    edges = np.linspace(1.0 / 3.0, 1.0, N_BINS + 1)   # 3-class max-posterior in [1/3, 1]
    rows_cal, ece = [], 0.0
    for b in range(N_BINS):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < N_BINS - 1 else conf <= edges[b + 1])
        if m.sum() == 0:
            continue
        mc, ma, cnt = conf[m].mean(), correct[m].mean(), int(m.sum())
        ece += (cnt / n) * abs(mc - ma)
        rows_cal.append((0.5 * (edges[b] + edges[b + 1]), mc, ma, cnt))
    print(f"\nCalibration: ECE = {ece:.3f} over {N_BINS} confidence bins")
    for mid, mc, ma, cnt in rows_cal:
        print(f"  bin~{mid:.2f}  mean_conf={mc:.3f}  accuracy={ma:.3f}  n={cnt}")

    # ---- write CSVs -------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "confusion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "count", "row_frac"])
        for i, ct in enumerate(CLASSES):
            for j, cp in enumerate(CLASSES):
                w.writerow([ct, cp, cm[i, j], f"{cm[i, j] / max(cm[i].sum(), 1):.4f}"])
    with open(os.path.join(OUT, "calibration.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_mid", "mean_conf", "accuracy", "count", "ece"])
        for mid, mc, ma, cnt in rows_cal:
            w.writerow([f"{mid:.4f}", f"{mc:.4f}", f"{ma:.4f}", cnt, f"{ece:.4f}"])
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'confusion.csv'))} + calibration.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
