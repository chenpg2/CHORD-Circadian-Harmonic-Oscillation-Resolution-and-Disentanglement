"""Quantify the paper-level impact of Option A (label no-24h 12h rhythms 'ambiguous').

Option A changes ONLY the "no-24h corner": genes with a detectable 12h but no 24h
fundamental (snr24 < min_24h_snr). The current gate forces these to autonomous B; Option A
would return 'ambiguous' instead. This script measures, on the full 4-mechanism benchmark:

  * the partition of every gene into dropped / no-24h-corner / gated(24h-bearing);
  * per class and per C-mechanism;
  * the ground-truth composition of the no-24h corner (what A relabels);
  * confirmation that the headline disentanglement AUCs live on the gated set, which
    Option A does NOT touch -> those AUCs are unchanged.

Deterministic (seed 8108). Run:  python scripts/measure_option_a_impact.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from chord.bhdt.stage2_orthogonal import _harmonic_fit, TernaryConfig
from chord.simulation.ternary_benchmark import build_ternary_benchmark

CFG = TernaryConfig()
W = 2 * np.pi / 24.0
N_PER_CLASS = 200


def _snr(t, y):
    amps, _, noise = _harmonic_fit(t, y, W, 2)
    return amps[1] / max(noise, 1e-9), amps[0] / max(noise, 1e-9)  # snr12, snr24


def main() -> int:
    b = build_ternary_benchmark(n_per_class=N_PER_CLASS, classes=("A", "B", "C"), seed=8108)
    t = b["t"]
    part = defaultdict(Counter)
    cmech = defaultdict(Counter)
    no24_truth = Counter()
    for y, lab, tr in zip(b["expr"], b["labels"], b["truth"]):
        s12, s24 = _snr(t, y)
        if s12 < CFG.min_12h_snr:
            bucket = "dropped(no 12h)"
        elif s24 < CFG.min_24h_snr:
            bucket = "no-24h corner"
            no24_truth[lab] += 1
        else:
            bucket = "gated(24h-bearing)"
        part[lab][bucket] += 1
        if lab == "C":
            cmech[tr["scenario"]][bucket] += 1

    print("=" * 76)
    print("Option-A impact measurement (4-mechanism benchmark, seed 8108, n=200/class)")
    print("=" * 76)
    print("\nPer-class partition:")
    for cls in ("A", "B", "C"):
        print(f"  {cls}: {dict(part[cls])}")
    print("\nPer C-mechanism partition (which mechanisms fall into the no-24h corner):")
    for m, c in cmech.items():
        print(f"  {m:26s}: {dict(c)}")

    n_gated = sum(part[c]["gated(24h-bearing)"] for c in ("A", "B", "C"))
    n_no24 = sum(no24_truth.values())
    print("\n" + "-" * 76)
    print(f"GATED (24h-bearing) set: n={n_gated}")
    print("  >>> The headline disentanglement AUCs (B-vs-driven, B-vs-C, A-vs-C) are computed")
    print("      ONLY on this set. Option A does not change gate membership or the fitted")
    print("      multinomial here, so these AUCs are UNCHANGED by Option A.")
    print("-" * 76)
    print(f"NO-24h CORNER: n={n_no24}  <<< the ONLY genes Option A relabels (B -> ambiguous)")
    print(f"  ground-truth composition: {dict(no24_truth)}")
    trueB = no24_truth.get("B", 0)
    trueCA = no24_truth.get("C", 0) + no24_truth.get("A", 0)
    print(f"  correct-B calls that become 'ambiguous' (the COST of A):      {trueB}"
          f"  = {trueB / (N_PER_CLASS):.0%} of the B class")
    print(f"  WRONG-B calls (really C/A) corrected to 'ambiguous' (the GAIN): {trueCA}")
    print("-" * 76)

    print("\nBOTTOM LINE for the paper:")
    print(f"  * Headline pairwise AUCs: UNCHANGED (they live on the {n_gated} gated genes).")
    print(f"  * Option A only relabels {n_no24} no-24h genes from a (partly over-claimed) 'B'")
    print(f"    to 'ambiguous' — trading {trueB} honest-but-unidentifiable B calls for")
    print(f"    {trueCA} corrected mis-calls. This is the identifiability floor, not a")
    print("    performance regression on the identifiable (24h-bearing) genes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
