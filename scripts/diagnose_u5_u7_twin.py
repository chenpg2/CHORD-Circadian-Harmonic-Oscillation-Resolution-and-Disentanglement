"""Diagnostics for U5/U7 twin-symmetry investigation.

This is intentionally read-only with respect to CHORD model code. It checks:

* the broadened Class-C generators have only 24h input drives and non-constant
  12h output after their interaction/readout;
* the U5 ablation feature rows match ``extract_features`` for the retained four;
* the diverse benchmark's gated scenario mix, five-feature VIF, and per-mechanism
  twin/logR behavior;
* a small held-out seed sweep for FULL5 vs DROP4 B-vs-C AUC.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations_with_replacement

import numpy as np

from chord.bhdt.stage2_orthogonal import (
    TernaryConfig,
    _fit_multinomial,
    _harmonic_fit,
    _softmax,
    amplitude_ratio,
    extract_features,
    harmonic_decay_residual,
    phase_freedom_pvalue,
    relative_phase,
    twin_peak_symmetry,
)
from chord.simulation.generator import (
    intersection_harmonic,
    intersection_product,
    intersection_rectified,
    intersection_saturating,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark


CFG = TernaryConfig()
W = 2.0 * np.pi / 24.0
CLASSES = ("A", "B", "C")
CIDX = {c: i for i, c in enumerate(CLASSES)}


def amp(y: np.ndarray, t: np.ndarray, period: float) -> float:
    w = 2.0 * np.pi / period
    x = np.column_stack([np.cos(w * t), np.sin(w * t)])
    beta = np.linalg.lstsq(x, np.asarray(y) - np.mean(y), rcond=None)[0]
    return float(np.hypot(beta[0], beta[1]))


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    scores = np.asarray(scores, float)
    pos = np.asarray(pos, bool)
    npos = int(pos.sum())
    nneg = int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def harmonic_table(name: str, t: np.ndarray, series: dict[str, np.ndarray]) -> None:
    print(f"\n{name}")
    for label, y in series.items():
        a24 = amp(y, t, 24.0)
        a12 = amp(y, t, 12.0)
        std = float(np.std(y))
        half = float(np.sqrt(np.mean((y - np.roll(y, len(y) // 2)) ** 2)) / max(std, 1e-12))
        ratio = a12 / max(a24, 1e-12)
        print(
            f"  {label:16s} std={std:.6f} A24={a24:.6f} "
            f"A12={a12:.6f} R12/24={ratio:.3g} halfdiff/std={half:.3e}"
        )


def check_generators() -> None:
    print("=" * 78)
    print("Generator Fourier checks (dense, noise-free, symmetric parameters)")
    print("=" * 78)
    t = np.linspace(0.0, 24.0, 4096, endpoint=False)
    p = 0.9 * np.cos(W * t)
    q = 0.9 * np.cos(W * t - np.pi)

    # ODE output is sampled over two days because the generator integrates to a
    # periodic regime and returns the requested grid.
    td = np.linspace(0.0, 48.0, 8192, endpoint=False)
    ode = intersection_harmonic(t=td, A=3.0, A_s=0.9, A_d=0.9, d0=0.15, noise_sd=1e-12, seed=1)
    synth = 1.0 + 0.9 * np.cos(W * td)
    degr = 0.15 * (1.0 + 0.9 * np.cos(W * td - np.pi))
    harmonic_table(
        "intersection_harmonic inputs/output",
        td,
        {"synth_24h": synth, "degr_24h": degr, "ode_output": ode["y_clean"]},
    )

    gain = 2.0
    softplus = lambda z: np.log1p(np.exp(-np.abs(gain * z))) + np.maximum(gain * z, 0.0)
    raw_rect = softplus(p) + softplus(q)
    harmonic_table(
        "intersection_rectified replicated equation",
        t,
        {
            "drive_p": p,
            "drive_q": q,
            "branch_p": softplus(p),
            "branch_q": softplus(q),
            "raw_sum": raw_rect,
        },
    )

    gain = 3.0
    branch_p = np.tanh(gain * np.maximum(p, 0.0))
    branch_q = np.tanh(gain * np.maximum(q, 0.0))
    raw_sat = branch_p + branch_q
    harmonic_table(
        "intersection_saturating replicated equation",
        t,
        {
            "drive_p": p,
            "drive_q": q,
            "branch_p": branch_p,
            "branch_q": branch_q,
            "raw_sum": raw_sat,
        },
    )

    prod_p = 1.0 + 0.9 * np.cos(W * t)
    prod_q = 1.0 + 0.9 * np.cos(W * t - np.pi)
    raw_prod = prod_p * prod_q
    harmonic_table(
        "intersection_product replicated equation",
        t,
        {"drive_p": prod_p, "drive_q": prod_q, "raw_product": raw_prod},
    )

    for label, fn in [
        ("rectified", intersection_rectified),
        ("saturating", intersection_saturating),
        ("product", intersection_product),
    ]:
        res = fn(t=np.arange(0.0, 48.0, 2.0), A=3.0, A_s=0.9, A_d=0.9, noise_sd=1e-12, seed=1)
        yc = res["y_clean"]
        print(
            f"  generator_call {label:10s}: scenario={res['truth']['scenario']} "
            f"class={res['truth']['class_12h']} independent={res['truth']['has_independent_12h']} "
            f"harmonic={res['truth']['has_harmonic_12h']} std={np.std(yc):.6f}"
        )


def feature_row(t: np.ndarray, y: np.ndarray):
    amps, _, noise = _harmonic_fit(t, y, W, 2)
    snr12 = amps[1] / max(noise, 1e-9)
    snr24 = amps[0] / max(noise, 1e-9)
    if snr12 < CFG.min_12h_snr or snr24 < CFG.min_24h_snr:
        return None
    p = phase_freedom_pvalue(t, y)
    r = amplitude_ratio(t, y)
    delta = relative_phase(t, y)
    retained = np.array(
        [
            float(np.clip(-np.log10(max(p, 1e-12)), 0.0, 12.0)),
            float(np.log(max(r, 1e-6))),
            float(harmonic_decay_residual(t, y, noise_sd=noise)),
            float(np.log(max(snr12, 1e-6))),
        ]
    )
    return {
        "F1": retained[0],
        "logR": retained[1],
        "twin": float(twin_peak_symmetry(t, y)),
        "decay": retained[2],
        "logSNR": retained[3],
        "dphase": float(min(abs(delta), abs(abs(delta) - np.pi))),
        "delta": float(delta),
        "retained": retained,
        "noise": float(noise),
    }


def build_rows(seed: int, n_per_class: int = 200):
    bench = build_ternary_benchmark(n_per_class=n_per_class, classes=CLASSES, seed=seed)
    t = np.asarray(bench["t"], float)
    rows = []
    max_feature_diff = 0.0
    raw_counts = Counter(zip(bench["labels"], [tr["scenario"] for tr in bench["truth"]]))
    gate_stats = defaultdict(lambda: {"n": 0, "snr12": [], "snr24": [], "pass12": 0, "pass24": 0, "pass_both": 0})
    for y, lab, truth in zip(bench["expr"], bench["labels"], bench["truth"]):
        if lab not in CIDX:
            continue
        amps, _, noise = _harmonic_fit(t, y, W, 2)
        snr12 = float(amps[1] / max(noise, 1e-9))
        snr24 = float(amps[0] / max(noise, 1e-9))
        gs = gate_stats[(lab, truth["scenario"])]
        gs["n"] += 1
        gs["snr12"].append(snr12)
        gs["snr24"].append(snr24)
        gs["pass12"] += int(snr12 >= CFG.min_12h_snr)
        gs["pass24"] += int(snr24 >= CFG.min_24h_snr)
        gs["pass_both"] += int(snr12 >= CFG.min_12h_snr and snr24 >= CFG.min_24h_snr)
        row = feature_row(t, y)
        if row is None:
            continue
        ref = extract_features(t, y, noise_sd=row["noise"])
        max_feature_diff = max(max_feature_diff, float(np.max(np.abs(ref - row["retained"]))))
        row["label"] = lab
        row["yidx"] = CIDX[lab]
        row["scenario"] = truth["scenario"]
        rows.append(row)
    return bench, rows, max_feature_diff, raw_counts, gate_stats


def matrix(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], float)


def vifs(x: np.ndarray) -> list[float]:
    out = []
    for j in range(x.shape[1]):
        y = x[:, j]
        design = np.column_stack([np.delete(x, j, axis=1), np.ones(len(y))])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        rss = float(np.sum((y - design @ beta) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / max(tss, 1e-30)
        out.append(float(1.0 / max(1.0 - r2, 1e-9)))
    return out


def r2(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([x, np.ones(len(y))])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    rss = float(np.sum((y - design @ beta) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - rss / max(tss, 1e-30)


def poly2(x: np.ndarray) -> np.ndarray:
    cols = [x[:, j] for j in range(x.shape[1])]
    for a, b in combinations_with_replacement(range(x.shape[1]), 2):
        cols.append(x[:, a] * x[:, b])
    return np.column_stack(cols)


def fit_predict(xtr, ytr, xte):
    mean = xtr.mean(0)
    std = xtr.std(0)
    std[std < 1e-9] = 1.0
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = np.where(counts > 0, len(ytr) / (3 * np.maximum(counts, 1)), 0.0)
    weights = _fit_multinomial((xtr - mean) / std, ytr, 3, l2=1.0, class_weight=cw)
    zte = np.column_stack([(xte - mean) / std, np.ones(len(xte))])
    return _softmax(zte @ weights.T)


def bc_auc(proba: np.ndarray, y: np.ndarray) -> float:
    m = (y == 1) | (y == 2)
    score = proba[m, 1] / np.clip(proba[m, 1] + proba[m, 2], 1e-12, None)
    return auc(score, y[m] == 1)


def check_features_and_reversal() -> None:
    print("\n" + "=" * 78)
    print("Feature, benchmark, VIF, and reversal checks")
    print("=" * 78)
    train_bench, train_rows, train_diff, train_raw, train_gate = build_rows(8108)
    print(f"train seed=8108 raw class counts={dict(Counter(train_bench['labels']))}")
    print(f"train gated counts={dict(Counter(r['label'] for r in train_rows))}")
    print(f"train max |diagnostic retained features - extract_features|={train_diff:.3e}")
    print(
        "train raw C mix="
        + str(dict(Counter(k[1] for k, v in train_raw.items() if k[0] == "C" for _ in range(v))))
    )
    print(
        "train gated C mix="
        + str(dict(Counter(r["scenario"] for r in train_rows if r["label"] == "C")))
    )
    print("train C gate pass by mechanism:")
    print("  scenario                    n pass12 pass24 pass_both  median_snr12 median_snr24")
    for (lab, scenario), gs in sorted(train_gate.items()):
        if lab != "C":
            continue
        print(
            f"  {scenario:27s} {gs['n']:3d} {gs['pass12']:6d} {gs['pass24']:6d} "
            f"{gs['pass_both']:9d} {np.median(gs['snr12']):12.3f} {np.median(gs['snr24']):12.3f}"
        )

    cols5 = ["F1", "logR", "twin", "decay", "logSNR"]
    x5 = matrix(train_rows, cols5)
    vif5 = vifs(x5)
    pear = float(np.corrcoef(x5[:, 2], x5[:, 1])[0, 1])
    lin_r2 = r2(x5[:, 2], np.delete(x5, 2, axis=1))
    geom = np.column_stack([x5[:, 1], [r["delta"] for r in train_rows]])
    geom_r2 = r2(x5[:, 2], poly2(geom))
    print("five-feature VIF: " + " ".join(f"{c}={v:.2f}" for c, v in zip(cols5, vif5)))
    print(f"Pearson(twin, logR)={pear:+.3f}")
    print(f"R2(twin ~ other 4 features, linear)={lin_r2:.3f}")
    print(f"R2(twin ~ poly2(logR, raw_delta))={geom_r2:.3f}")

    print("\nC-mechanism feature geometry on train gated rows:")
    print("  scenario                    n   mean_logR  mean_twin  corr(twin,logR)  slope")
    for scenario in sorted({r["scenario"] for r in train_rows if r["label"] == "C"}):
        grp = [r for r in train_rows if r["scenario"] == scenario]
        x = np.array([r["logR"] for r in grp])
        y = np.array([r["twin"] for r in grp])
        corr = float(np.corrcoef(x, y)[0, 1]) if len(grp) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        slope = float(np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0][0])
        print(f"  {scenario:27s} {len(grp):3d} {x.mean():+10.3f} {y.mean():10.3f} {corr:+16.3f} {slope:+7.3f}")

    ytr = np.array([r["yidx"] for r in train_rows])
    seed_rows = []
    for seed in [20260703, 20260704, 11, 42, 99]:
        _, test_rows, test_diff, _, _ = build_rows(seed)
        yte = np.array([r["yidx"] for r in test_rows])
        full = bc_auc(fit_predict(matrix(train_rows, cols5), ytr, matrix(test_rows, cols5)), yte)
        drop_cols = ["F1", "logR", "decay", "logSNR"]
        drop = bc_auc(fit_predict(matrix(train_rows, drop_cols), ytr, matrix(test_rows, drop_cols)), yte)
        seed_rows.append((seed, len(test_rows), test_diff, full, drop, full - drop))
    print("\nHeld-out seed sweep, B-vs-C AUC:")
    print("  seed        gated_n  max_feature_diff     FULL5   DROP4   delta")
    for seed, n, diff, full, drop, delta in seed_rows:
        print(f"  {seed:<10d} {n:7d} {diff:17.3e} {full:7.3f} {drop:7.3f} {delta:7.3f}")

    _, test_rows, _, _, test_gate = build_rows(20260703)
    print("test seed=20260703 gated C mix=" + str(dict(Counter(r["scenario"] for r in test_rows if r["label"] == "C"))))
    print("test seed=20260703 C gate pass by mechanism:")
    print("  scenario                    n pass12 pass24 pass_both  median_snr12 median_snr24")
    for (lab, scenario), gs in sorted(test_gate.items()):
        if lab != "C":
            continue
        print(
            f"  {scenario:27s} {gs['n']:3d} {gs['pass12']:6d} {gs['pass24']:6d} "
            f"{gs['pass_both']:9d} {np.median(gs['snr12']):12.3f} {np.median(gs['snr24']):12.3f}"
        )


def main() -> int:
    check_generators()
    check_features_and_reversal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
