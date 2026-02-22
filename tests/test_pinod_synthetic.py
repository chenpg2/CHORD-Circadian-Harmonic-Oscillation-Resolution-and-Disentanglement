"""
Comprehensive PINOD validation on synthetic data.

Tests PINOD on the same 8 scenarios used in Phase 1 BHDT validation,
plus specific regression tests for the 3 problems identified in Phase 1.
"""

import sys
sys.path.insert(0, '/home/data2/fangcong2/ovary_aging/scripts/chord/src')

import numpy as np
import torch
from chord.simulation import generator as gen
from chord.pinod.networks import PINODSingleGene
from chord.pinod.trainer import train_single_gene
from chord.pinod.analysis import extract_oscillator_params, classify_gene_pinod


def run_pinod_on_scenario(data, label, n_epochs=500, lambda_sparse=0.005):
    """Train PINOD on a single scenario and return classification."""
    t = data["t"]
    y = data["y"]
    truth = data["truth"]

    model = PINODSingleGene(
        n_timepoints=len(t),
        n_oscillators=3,
        period_inits=[24.0, 12.0, 8.0],
        decoder_type="linear",
    )

    result = train_single_gene(
        model=model,
        expr=y,
        timepoints=t,
        n_epochs=n_epochs,
        lr=1e-3,
        lambda_phys=0.01,
        lambda_sparse=lambda_sparse,
        lambda_period=0.1,
        patience=80,
    )

    analysis = extract_oscillator_params(result["model"], y, t)
    classification = classify_gene_pinod(analysis)

    return {
        "label": label,
        "truth": truth,
        "classification": classification["classification"],
        "confidence": classification["confidence"],
        "evidence": classification["evidence"],
        "periods": [o["period"] for o in analysis["oscillators"]],
        "gammas": [o["gamma"] for o in analysis["oscillators"]],
        "amplitudes": [o["amplitude_rms"] for o in analysis["oscillators"]],
        "active": [o["active"] for o in analysis["oscillators"]],
        "r2": analysis["reconstruction_r2"],
        "best_loss": result["parameters"]["best_loss"],
    }


# ============================================================================
# Run all 8 Phase 1 scenarios + 3 regression tests
# ============================================================================

print("=" * 70)
print("PINOD Synthetic Data Validation")
print("=" * 70)

scenarios = [
    # Phase 1 original 8 scenarios
    ("Independent 12h (T=11.8)", gen.independent_superposition(seed=42)),
    ("Sawtooth harmonic", gen.sawtooth_harmonic(seed=42)),
    ("Pure noise", gen.pure_noise(seed=42)),
    ("Pure 24h only", gen.pure_circadian(seed=42)),
    ("Pure 12h only", gen.pure_ultradian(seed=42)),
    ("24h + indep 12h (T=12.0)", gen.independent_superposition(seed=42, T_12=12.0)),
    ("24h + weak 12h harmonic", gen.peaked_harmonic(seed=42, A=1.0)),
    ("Peaked wave (multi-harmonic)", gen.peaked_harmonic(seed=42)),

    # Phase 1 regression tests
    ("REGRESSION: Damped 12h", gen.damped_ultradian(seed=42, gamma=0.02)),
    ("REGRESSION: Asymmetric 12h", gen.asymmetric_ultradian(seed=42)),
    ("REGRESSION: Low SNR 12h", gen.low_snr_ultradian(seed=42)),
]

# Expected classifications
expected = {
    "Independent 12h (T=11.8)": "independent_ultradian",
    "Sawtooth harmonic": "harmonic",
    "Pure noise": "non_rhythmic",
    "Pure 24h only": "circadian_only",
    "Pure 12h only": "independent_ultradian",
    "24h + indep 12h (T=12.0)": "independent_ultradian",
    "24h + weak 12h harmonic": "harmonic",
    "Peaked wave (multi-harmonic)": "harmonic",
    "REGRESSION: Damped 12h": "damped_ultradian",
    "REGRESSION: Asymmetric 12h": "independent_ultradian",
    "REGRESSION: Low SNR 12h": "independent_ultradian",
}

results = []
for label, data in scenarios:
    print(f"\n--- {label} ---")
    r = run_pinod_on_scenario(data, label)
    results.append(r)

    exp = expected[label]
    # Allow some flexibility in classification matching
    correct = r["classification"] == exp
    partial = (
        (exp == "independent_ultradian" and r["classification"] in ["independent_ultradian", "likely_independent_ultradian", "multi_ultradian"])
        or (exp == "harmonic" and r["classification"] in ["harmonic", "circadian_only"])
        or (exp == "non_rhythmic" and r["classification"] in ["non_rhythmic", "circadian_only"])
        or (exp == "damped_ultradian" and r["classification"] in ["damped_ultradian", "independent_ultradian"])
    )

    status = "CORRECT" if correct else ("PARTIAL" if partial else "WRONG")
    print(f"  Expected: {exp}")
    print(f"  Got:      {r['classification']} (confidence={r['confidence']:.2f})")
    print(f"  Periods:  {[f'{p:.2f}' for p in r['periods']]}")
    print(f"  Gammas:   {[f'{g:.4f}' for g in r['gammas']]}")
    print(f"  Amps:     {[f'{a:.4f}' for a in r['amplitudes']]}")
    print(f"  Active:   {r['active']}")
    print(f"  R2:       {r['r2']:.4f}")
    print(f"  Status:   {status}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
n_correct = 0
n_partial = 0
n_wrong = 0
for r in results:
    exp = expected[r["label"]]
    correct = r["classification"] == exp
    partial = (
        (exp == "independent_ultradian" and r["classification"] in ["independent_ultradian", "likely_independent_ultradian", "multi_ultradian"])
        or (exp == "harmonic" and r["classification"] in ["harmonic", "circadian_only"])
        or (exp == "non_rhythmic" and r["classification"] in ["non_rhythmic", "circadian_only"])
        or (exp == "damped_ultradian" and r["classification"] in ["damped_ultradian", "independent_ultradian"])
    )
    if correct:
        n_correct += 1
        status = "OK"
    elif partial:
        n_partial += 1
        status = "PARTIAL"
    else:
        n_wrong += 1
        status = "**WRONG**"
    print(f"  [{status:8s}] {r['label']:40s} -> {r['classification']}")

print(f"\nCorrect: {n_correct}/{len(results)}, Partial: {n_partial}/{len(results)}, Wrong: {n_wrong}/{len(results)}")
