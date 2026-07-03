"""R3-2: the entrainment / identifiability boundary of the phase-freedom axis.

An entrained autonomous oscillator whose 12h phase is pulled toward the locked value looks like
a harmonic. We build a 24h+12h signal with relative phase delta = phi_12 - 2*phi_24 and sweep
delta from 0 (fully entrained -> phase-locked) to pi/2 (free), reporting the fraction the
phase-freedom F-test calls free-phase (p < 0.05, i.e. correctly autonomous). The empirical curve
follows the analytic non-centrality lambda proportional to (A_12/sigma)^2 sin^2(delta)
(supp_analytic_lemmas.md): near delta = 0 the oscillator is unidentifiable from a harmonic; the
call becomes reliable once |delta| clears the SNR-set boundary. Deterministic.

Writes figures/source_data/entrainment.csv. Run:  python scripts/export_entrainment.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

from chord.bhdt.stage2_orthogonal import phase_freedom_pvalue

W = 2 * np.pi / 24.0
OUT = os.path.join(os.path.dirname(__file__), "..", "publication",
                   "CHORD_ternary_submission", "figures", "source_data", "entrainment.csv")


def main() -> int:
    t = np.arange(0.0, 48.0, 2.0)           # 24 points / 48 h
    A24, A12, sigma = 2.0, 1.2, 0.5         # SNR12 = A12/sigma = 2.4
    M = 400
    deltas = np.linspace(0.0, np.pi / 2, 11)
    rng = np.random.default_rng(20260703)

    print(f"SNR12 = {A12/sigma:.1f};  M = {M} per delta")
    print(f"{'delta(rad)':>10s} {'delta(deg)':>10s} {'free-phase rate':>16s} {'lambda~sin^2':>13s}")
    rows = []
    for d in deltas:
        free = 0
        for _ in range(M):
            ph24 = rng.uniform(0, 2 * np.pi)          # random absolute phase; only delta matters
            ph12 = 2 * ph24 + d
            y = (A24 * np.cos(W * t - ph24) + A12 * np.cos(2 * W * t - ph12)
                 + rng.normal(0, sigma, len(t)))
            free += int(phase_freedom_pvalue(t, y) < 0.05)
        rate = free / M
        lam = (A12 / sigma) ** 2 * np.sin(d) ** 2      # relative non-centrality (analytic)
        rows.append((d, np.degrees(d), rate, lam))
        print(f"{d:10.3f} {np.degrees(d):10.1f} {rate:16.3f} {lam:13.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_rad", "delta_deg", "free_phase_rate", "lambda_rel"])
        for d, dd, r, lam in rows:
            w.writerow([f"{d:.4f}", f"{dd:.2f}", f"{r:.4f}", f"{lam:.4f}"])
    print(f"\nwrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
