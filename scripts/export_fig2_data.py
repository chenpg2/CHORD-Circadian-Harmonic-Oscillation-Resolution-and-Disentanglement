"""Export Figure 2 data: the three orthogonal statistics by true class, from the
ternary benchmark. Non-visual data export (figure built in R)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chord.simulation.ternary_benchmark import build_ternary_benchmark  # noqa: E402
from chord.bhdt.stage2_orthogonal import extract_features, _harmonic_fit  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "publication", "CHORD_ternary_submission", "figures", "source_data")
W = 2 * np.pi / 24.0
LAB = {"A": "harmonic", "B": "autonomous", "C": "intersection"}


def main():
    bench = build_ternary_benchmark(n_per_class=200, classes=("A", "B", "C"), seed=2024)
    t = bench["t"]
    rows = []
    for y, lab in zip(bench["expr"], bench["labels"]):
        if lab not in LAB:
            continue
        _, _, nz = _harmonic_fit(t, y, W, 2)
        f = extract_features(t, y, 24.0, noise_sd=nz)
        # f = [neglog10_p_phase, log_amp_ratio, twin_symmetry, decay_residual, log_snr12]
        rows.append((LAB[lab], f[0], f[2], f[1]))   # phase_freedom, peak_symmetry, log_amp_ratio
    with open(os.path.join(OUT, "fig2_axes.csv"), "w") as fh:
        fh.write("class,phase_freedom,peak_symmetry,log_amp_ratio\n")
        for c, pf, ps, ar in rows:
            fh.write(f"{c},{pf:.4f},{ps:.4f},{ar:.4f}\n")
    print(f"fig2_axes.csv: {len(rows)} genes "
          f"({sum(c=='autonomous' for c,_,_,_ in rows)} autonomous, "
          f"{sum(c=='harmonic' for c,_,_,_ in rows)} harmonic, "
          f"{sum(c=='intersection' for c,_,_,_ in rows)} intersection)")


if __name__ == "__main__":
    main()
