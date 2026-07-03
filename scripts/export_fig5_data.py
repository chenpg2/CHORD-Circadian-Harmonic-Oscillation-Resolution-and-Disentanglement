"""Export Figure 5 data (identifiability budget) by REUSING the validated method of
plan/experiments/exp_identifiability_budget.py: per-cell train/test refit of the
multinomial, realised 12h-SNR = fitted A12 / (noise/sqrt(M)), three cadences.

Panel a: autonomy AUC vs realised 12h-SNR for N=12/24/48 (finer noise grid).
Panel b: sqrt(M) replicate gain at a fixed low single-series SNR.
Non-visual export (figure built in R)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chord.bhdt.stage2_orthogonal import (  # noqa: E402
    extract_features, _harmonic_fit, _fit_multinomial, _softmax,
)
from chord.simulation.generator import intersection_harmonic  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "publication", "CHORD_ternary_submission", "figures", "source_data")
W = 2 * np.pi / 24.0


def gen_B(t, A12, noise, rng):
    y = 2.0 * np.cos(W * t - rng.uniform(0, 2 * np.pi)) + \
        A12 * np.cos(2 * np.pi * t / rng.uniform(11.5, 12.5) - rng.uniform(0, 2 * np.pi))
    return y + rng.normal(0, noise, len(t))


def gen_A(t, A12, noise, rng):
    phi = rng.uniform(0, 2 * np.pi); A1 = A12 / np.exp(-0.5); y = np.zeros_like(t)
    for k in range(1, 5):
        y = y + A1 * np.exp(-0.5 * (k - 1)) * np.cos(k * (W * t - phi))
    return y + rng.normal(0, noise, len(t))


def gen_C(t, noise, rng):
    return intersection_harmonic(t=t, A=2.0, A_s=rng.uniform(0.8, 0.95),
                                 A_d=rng.uniform(0.8, 0.95), d0=rng.uniform(0.12, 0.2),
                                 phi_s=rng.uniform(0, 2 * np.pi), noise_sd=noise,
                                 seed=int(rng.randint(1, 2**31)))["y"]


def amp12(y, t):
    return _harmonic_fit(np.asarray(t, float), np.asarray(y, float), W, 2)[0][1]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)))


def build(t, noise, ng, seed):
    rng = np.random.RandomState(seed); rows = []; labs = []
    for _ in range(ng):
        rows.append(gen_A(t, 0.6, noise, rng)); labs.append(0)
        rows.append(gen_B(t, 0.6, noise, rng)); labs.append(1)
        rows.append(gen_C(t, noise, rng)); labs.append(2)
    return np.array(rows), np.array(labs)


def cell(t, noise, M):
    eff = noise / np.sqrt(M)
    Xtr_y, ytr = build(t, eff, 90, 1); Xte_y, yte = build(t, eff, 60, 2)

    def feats(Y):
        F = []; keep = []
        for y in Y:
            a, _, nz = _harmonic_fit(t, y, W, 2)
            if a[1] / max(nz, 1e-9) < 1.0 or a[0] / max(nz, 1e-9) < 2.0:
                keep.append(False); F.append(None)
            else:
                keep.append(True); F.append(extract_features(t, y, noise_sd=nz))
        return F, np.array(keep)

    Ftr, ktr = feats(Xtr_y); Fte, kte = feats(Xte_y)
    Xtr = np.array([f for f, k in zip(Ftr, ktr) if k]); ytr2 = ytr[ktr]
    Xte = np.array([f for f, k in zip(Fte, kte) if k]); yte2 = yte[kte]
    snr = np.median([amp12(y, t) for y in Xte_y]) / eff
    if len(set(ytr2)) < 3 or len(Xte) < 10:
        return snr, float("nan")
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-9] = 1
    cnt = np.bincount(ytr2, minlength=3).astype(float)
    cw = np.where(cnt > 0, len(ytr2) / (3 * np.maximum(cnt, 1)), 0.0)
    Wt = _fit_multinomial((Xtr - mean) / std, ytr2, 3, l2=1.0, class_weight=cw)
    P = _softmax(np.column_stack([(Xte - mean) / std, np.ones(len(Xte))]) @ Wt.T)
    return snr, auc(P[yte2 == 1, 1], P[(yte2 == 0) | (yte2 == 2), 1])


def main():
    cadences = [(12, np.arange(0, 48, 4)), (24, np.arange(0, 48, 2)), (48, np.arange(0, 48, 1))]
    noises = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4]
    with open(os.path.join(OUT, "fig5_snr.csv"), "w") as f:
        f.write("N,snr,auc\n")
        for N, t in cadences:
            for nz in noises:
                snr, a = cell(t, nz, 1)
                if a == a:
                    f.write(f"{N},{snr:.4f},{a:.4f}\n")
            print(f"N={N} done")

    t = np.arange(0, 48, 2)   # N=24 series
    with open(os.path.join(OUT, "fig5_replicates.csv"), "w") as f:
        f.write("M,eff_snr,auc\n")
        for M in (1, 2, 4, 8, 16):
            snr, a = cell(t, 1.0, M)   # single-series SNR ~0.6 at noise=1.0, lifted by sqrt(M)
            f.write(f"{M},{snr:.4f},{a:.4f}\n")
        print("replicates done")


if __name__ == "__main__":
    main()
