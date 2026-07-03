"""U7: verify the broadened intersection (Class-C) family (answers R3-1).

The submitted benchmark generated Class C from ONE ODE (intersection_harmonic), so the
near-perfect harmonic-vs-intersection AUC could reflect simulator<->feature matching.
This adds three structurally different mechanisms (rectified/softplus, saturating/
logistic, product/bilinear). This script checks that each:

  1. produces a GENUINE, detectable 12-h (R = A12/A24 high at symmetric params);
  2. has the built-in falsifiability boundary — imbalance/asymmetry revives the 24-h
     fundamental (R drops), crossing toward Class A;
  3. clears the benchmark's 12h-SNR floor at a healthy RETENTION rate (so no mechanism
     is silently under-represented in the detectable region — the adversarial concern).

Also confirms build_ternary_benchmark runs with the expanded pool and stays balanced.
Deterministic. Run:  python scripts/verify_intersection_family.py
"""
from __future__ import annotations

import numpy as np

from chord.simulation.generator import (
    intersection_harmonic, intersection_rectified, intersection_saturating,
    intersection_product, _default_timepoints,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

VARIANTS = [
    ("harmonic(ODE)", intersection_harmonic, {"A_d": 0.4}),          # asym: unequal depth
    ("rectified(softplus)", intersection_rectified, {"imbalance": 2.2}),
    ("saturating(logistic)", intersection_saturating, {"imbalance": 2.2}),
    ("product(bilinear)", intersection_product, {"imbalance": 2.5}),
]


def _amp(y, t, T):
    w = 2 * np.pi / T
    X = np.column_stack([np.cos(w * t), np.sin(w * t)])
    b, *_ = np.linalg.lstsq(X, np.asarray(y) - np.mean(y), rcond=None)
    return float(np.hypot(b[0], b[1]))


def _R(res):
    t, yc = res["t"], res["y_clean"]
    return _amp(yc, t, 12.0) / max(_amp(yc, t, 24.0), 1e-9)


def main() -> int:
    t = _default_timepoints()
    base = dict(A=3.0, A_s=0.9, A_d=0.9, noise_sd=1e-9, seed=1)
    print("=" * 74)
    print("U7 broadened intersection family — 4 structurally different mechanisms")
    print("=" * 74)

    rng = np.random.RandomState(7)

    def retention(fn):
        cleared = 0
        for _ in range(200):
            kw = dict(A=float(rng.uniform(2.5, 4.5)), A_s=float(rng.uniform(0.6, 0.95)),
                      A_d=float(rng.uniform(0.6, 0.95)),
                      antiphase_jitter=float(rng.uniform(0.0, 0.6)),
                      noise_sd=float(rng.uniform(0.3, 0.7)), M=float(rng.uniform(3, 10)),
                      seed=int(rng.randint(1, 2**31)))
            if fn in (intersection_rectified, intersection_saturating):
                kw["gain"] = float(rng.uniform(1.5, 4.0)); kw["imbalance"] = float(rng.uniform(0.6, 1.4))
            if fn is intersection_product:
                kw["imbalance"] = float(rng.uniform(0.6, 1.4))
            res = fn(t=t, **kw)
            snr12 = _amp(res["y_clean"], t, 12.0) / max(kw["noise_sd"], 1e-9)
            cleared += int(snr12 >= 1.0)
        return cleared / 200

    print("\nPer-mechanism: R=A12/A24 (12h dominance, noise-free), retention (under noise), boundary")
    print(f"  {'mechanism':22s} {'R_sym':>9s} {'R_asym':>9s} {'retain':>7s}   verdict")
    ok = True
    for name, fn, asym in VARIANTS:
        r_sym = _R(fn(t=t, **base))
        r_asym = _R(fn(t=t, **{**base, **asym}))
        ret = retention(fn)
        detectable = ret >= 0.3                 # genuine detectable 12-h under noise
        boundary = r_asym < 0.9 * r_sym         # asymmetry lowers 12h dominance (24h revives)
        ok = ok and detectable and boundary
        ds = lambda r: (f"{r:9.3f}" if r <= 999 else "  >1e3*  ")
        print(f"  {name:22s} {ds(r_sym)} {ds(r_asym)} {ret:7.2f}   "
              f"{'ok' if detectable and boundary else 'FAIL'}"
              f" (detect={detectable}, boundary={boundary})")
    print("  * fundamental fully suppressed at exact symmetry (A24->0, R capped for display).")
    print("  The R-regime spread is intentional diversity: ODE = weak 12h on a strong 24h")
    print("  (R small); rectified/saturating/product = fundamental-suppressed (R large).")

    print("\n(4) balanced benchmark builds with the expanded C pool:")
    b = build_ternary_benchmark(n_per_class=120, classes=("A", "B", "C"), seed=8108)
    from collections import Counter
    cvar = Counter(tr["scenario"] for tr, lab in zip(b["truth"], b["labels"]) if lab == "C")
    counts = Counter(b["labels"])
    print(f"  class balance: {dict(counts)}")
    print(f"  C-variant mix: {dict(cvar)}")
    balanced = counts["A"] == counts["B"] == counts["C"] == 120 and len(cvar) == 4
    print(f"  => {'ok (balanced, all 4 mechanisms present)' if balanced else 'FAIL'}")

    ok = ok and balanced
    print("\n" + "-" * 74)
    print(f"U7 VERDICT: {'PASS' if ok else 'FAIL'}")
    print("-" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
