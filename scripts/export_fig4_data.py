"""Export data for Figure 4 (XBP1-LKO interventional validation) to CSV for R plotting.

Non-visual data conversion only: reproduces exp_xbp1ko_validation.py's classification
and amplitude ratios, writes CSVs. No plotting here (figures are built in R).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chord.bhdt.stage2_orthogonal import (  # noqa: E402
    get_default_model, disentangle_ternary, _harmonic_fit,
)

NPZ = os.path.expanduser("~/.chord_cache/gse130890_xbp1ko.npz")
OUT = os.path.join(os.path.dirname(__file__), "..", "publication", "CHORD_ternary_submission", "figures", "source_data")
W = 2 * np.pi / 24.0

CIRCADIAN = ["Arntl", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Nr1d1", "Nr1d2",
             "Dbp", "Tef", "Nfil3"]
# canonical XBP1-axis autonomous 12h genes; the panel-b trace picks the strongest collapser
AUTONOMOUS_TRACE = ["Dnajb9", "Edem1", "Hspa5", "Manf", "Herpud1", "Hyou1", "Creld2", "Sel1l"]
TRACE_CIRCADIAN = "Arntl"      # core clock 24h control


def lognorm(x):
    L = np.log2(np.asarray(x, float) + 1.0)
    return L - np.median(L, axis=0, keepdims=True)


def amp(y, t, period):
    a, _, nz = _harmonic_fit(t, y, W, 2)
    return (a[1] if period == 12 else a[0]), nz


def main():
    d = np.load(NPZ, allow_pickle=True)
    t = d["timepoints"].astype(float)
    wt, ko = lognorm(d["wt_expr"]), lognorm(d["ko_expr"])
    names = np.array([str(x) for x in d["gene_names"]]); low = np.char.lower(names)
    keep = np.all(np.isfinite(wt), axis=1) & np.all(np.isfinite(ko), axis=1)
    model = get_default_model()

    # per-gene: two-tier partition + KO/WT 12h ratio (panel a).
    #   autonomous   = classifier-confident B (24h-bearing)
    #   driven       = classifier-confident A/C (24h-bearing)
    #   unidentified = no-24h program pool CHORD abstains on (resolved by the intervention)
    rows = []
    for i in np.where(keep)[0]:
        a12w, nzw = amp(wt[i], t, 12)
        if a12w / max(nzw, 1e-9) < 1.0:
            continue
        a24w, _ = amp(wt[i], t, 24)
        snr24 = a24w / max(nzw, 1e-9)
        cls = disentangle_ternary(t, wt[i], model=model)["class"]
        a12k, _ = amp(ko[i], t, 12)
        ratio = a12k / max(a12w, 1e-9)
        if snr24 < 2.0:
            grp = "unidentified"
        elif cls == "B_independent":
            grp = "autonomous"
        elif cls in ("A_harmonic", "C_intersection"):
            grp = "driven"
        else:                          # 24h-bearing ambiguous (abstain band): skip
            continue
        rows.append((names[i], grp, ratio))
    with open(os.path.join(OUT, "fig4_ratios.csv"), "w") as f:
        f.write("gene,group,ko_wt_12h\n")
        for g, grp, r in rows:
            f.write(f"{g},{grp},{r:.4f}\n")
    for g in ("autonomous", "driven", "unidentified"):
        v = [r for _, gg, r in rows if gg == g]
        print(f"{g:12s} n={len(v):>4} median KO/WT 12h={np.median(v):.2f}" if v
              else f"{g:12s} n=0")

    # circadian 24h control (panel b annotation): KO/WT 24h
    circ = []
    for g in CIRCADIAN:
        ix = np.where(low == g.lower())[0]
        if not len(ix) or not keep[ix[0]]:
            continue
        i = ix[0]; a24w, nzw = amp(wt[i], t, 24)
        if a24w / max(nzw, 1e-9) < 2.0:
            continue
        a24k, _ = amp(ko[i], t, 24)
        circ.append((g, a24k / max(a24w, 1e-9)))
    with open(os.path.join(OUT, "fig4_circadian24.csv"), "w") as f:
        f.write("gene,ko_wt_24h\n")
        for g, r in circ:
            f.write(f"{g},{r:.4f}\n")
    print(f"circadian 24h KO/WT median={np.median([r for _, r in circ]):.2f} (n={len(circ)})")

    # example traces (panel c) chosen data-driven so the collapse is visible:
    #  - autonomous: strong WT 12h AND clear collapse in KO (high amp, low KO/WT)
    #  - circadian:  strong WT 24h that is preserved in KO (KO/WT 24h near 1)
    def pick_autonomous():
        # a recognizable XBP1-axis autonomous gene with the clearest collapse in the KO
        best, best_ratio = None, 1e9
        for g in AUTONOMOUS_TRACE:
            ix = np.where(low == g.lower())[0]
            if not len(ix) or not keep[ix[0]]:
                continue
            i = ix[0]; a12w, nzw = amp(wt[i], t, 12)
            if a12w / max(nzw, 1e-9) < 1.5:      # a clean, well-resolved WT 12h
                continue
            ratio = amp(ko[i], t, 12)[0] / max(a12w, 1e-9)
            if ratio < best_ratio:
                best_ratio, best = ratio, i
        return best

    def pick_circadian():
        best, best_amp = None, -1
        for g in ["Arntl", "Dbp", "Nr1d1", "Per2"]:
            ix = np.where(low == g.lower())[0]
            if not len(ix) or not keep[ix[0]]:
                continue
            i = ix[0]; a24w, nzw = amp(wt[i], t, 24)
            if a24w / max(nzw, 1e-9) > best_amp:
                best_amp, best = a24w / max(nzw, 1e-9), i
        return best

    ia, ic = pick_autonomous(), pick_circadian()
    with open(os.path.join(OUT, "fig4_traces.csv"), "w") as f:
        f.write("gene,role,genotype,time,expr\n")
        for i, role in [(ia, "autonomous 12h"), (ic, "circadian 24h")]:
            if i is None:
                continue
            for geno, mat in [("WT", wt), ("XBP1-LKO", ko)]:
                for tt, yy in zip(t, mat[i]):
                    f.write(f"{names[i]},{role},{geno},{tt:.0f},{yy:.4f}\n")
    if ia is not None:
        kr = amp(ko[ia], t, 12)[0] / max(amp(wt[ia], t, 12)[0], 1e-9)
        print(f"traces: autonomous={names[ia]} (KO/WT 12h={kr:.2f}), "
              f"circadian={names[ic] if ic is not None else 'NA'} -> fig4_traces.csv")
    else:
        print("traces: no autonomous example matched the strict criteria")


if __name__ == "__main__":
    main()
