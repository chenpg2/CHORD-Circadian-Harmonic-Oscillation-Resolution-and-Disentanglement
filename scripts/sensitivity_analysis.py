#!/usr/bin/env python3
"""
Sensitivity analysis for CHORD classification thresholds.

Perturbs each key threshold by +/-20% and measures F1 change on synthetic
data. If F1 changes < 5%, thresholds are robust to perturbation.

Output: CSV with columns: parameter, default_value, perturbed_value,
        perturbation_pct, tier1_f1, tier2_f1, delta_tier1_f1, delta_tier2_f1

Usage:
    python sensitivity_analysis.py --reps 20
    python sensitivity_analysis.py --reps 5 --quick
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from chord.simulation.generator import generate_all_scenarios
from chord.bhdt.classifier import classify_gene, CHORDConfig


def _compute_binary_metrics(y_true, y_pred):
    """Compute F1 score."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def evaluate_config(config, all_data):
    """Run CHORD with given config on all synthetic data, return metrics."""
    y_true_detect = []
    y_pred_detect = []
    y_true_disent = []
    y_pred_disent = []

    for d in all_data:
        try:
            result = classify_gene(d["t"], d["y"], config=config)
        except Exception:
            result = {"stage1_passed": False, "classification": "error"}

        y_true_detect.append(1 if d["gt_has_12h"] else 0)
        y_pred_detect.append(1 if result.get("stage1_passed", False) else 0)

        if d["gt_has_12h"]:
            y_true_disent.append(1 if d["gt_independent"] else 0)
            cls = result.get("classification", "")
            y_pred_disent.append(
                1 if cls in ("independent_ultradian", "likely_independent_ultradian") else 0
            )

    t1 = _compute_binary_metrics(y_true_detect, y_pred_detect)
    if y_true_disent:
        t2 = _compute_binary_metrics(y_true_disent, y_pred_disent)
    else:
        t2 = {"f1": np.nan}

    return t1["f1"], t2["f1"]


# Key thresholds to perturb
THRESHOLD_SPECS = [
    # (param_name, how_to_set, default_value)
    ("confidence_thresholds[0]", "ct0", 0.6),
    ("confidence_thresholds[1]", "ct1", 0.3),
    ("confidence_divisor", "confidence_divisor", 6.0),
    ("bf_small_n_weight", "bf_small_n_weight", 0.5),
    ("vmd_amp_gate_default", "vmd_amp_gate_default", 0.15),
    ("bispectral_nonsin_discount", "bispectral_nonsin_discount", 0.5),
    ("amp_ratio_asymmetry_gate", "amp_ratio_asymmetry_gate", 0.5),
    ("m0_r2_discount_threshold", "m0_r2_discount_threshold", 0.5),
    ("h24_dominance_amp_ratio_max", "h24_dominance_amp_ratio_max", 0.35),
    ("h24_dominance_penalty", "h24_dominance_penalty", -3.0),
]


def make_perturbed_config(param_key, perturbed_value):
    """Create a CHORDConfig with one parameter perturbed."""
    config = CHORDConfig()

    if param_key == "ct0":
        ct = list(config.confidence_thresholds)
        ct[0] = perturbed_value
        config.confidence_thresholds = tuple(ct)
    elif param_key == "ct1":
        ct = list(config.confidence_thresholds)
        ct[1] = perturbed_value
        config.confidence_thresholds = tuple(ct)
    else:
        setattr(config, param_key, perturbed_value)

    return config


def run_sensitivity(n_reps=20, seed=42, verbose=True):
    """Run sensitivity analysis on all key thresholds."""
    t = np.arange(0, 48, 1.0)

    # Generate synthetic data
    if verbose:
        print(f"Generating synthetic data: 15 scenarios x {n_reps} reps...")
    all_data = []
    for rep in range(n_reps):
        scenarios = generate_all_scenarios(t, seed=seed + rep, n_replicates=1)
        for s in scenarios:
            truth = s["truth"]
            has_ind = truth.get("has_independent_12h", False)
            has_harm = truth.get("has_harmonic_12h", False)
            all_data.append({
                "y": s["y"], "t": t,
                "gt_has_12h": has_ind or has_harm,
                "gt_independent": has_ind,
            })

    if verbose:
        print(f"Total samples: {len(all_data)}")

    # Baseline evaluation
    if verbose:
        print("\nEvaluating baseline (default thresholds)...")
    baseline_config = CHORDConfig()
    t0 = time.time()
    baseline_t1_f1, baseline_t2_f1 = evaluate_config(baseline_config, all_data)
    if verbose:
        print(f"  Baseline: Tier-1 F1={baseline_t1_f1:.3f}, Tier-2 F1={baseline_t2_f1:.3f} ({time.time()-t0:.1f}s)")

    # Perturb each threshold
    results = []
    results.append({
        "parameter": "BASELINE",
        "param_key": "",
        "default_value": np.nan,
        "perturbed_value": np.nan,
        "perturbation_pct": 0,
        "tier1_f1": baseline_t1_f1,
        "tier2_f1": baseline_t2_f1,
        "delta_tier1_f1": 0.0,
        "delta_tier2_f1": 0.0,
    })

    for param_name, param_key, default_val in THRESHOLD_SPECS:
        for pct in [-20, +20]:
            perturbed_val = default_val * (1.0 + pct / 100.0)

            if verbose:
                print(f"\n  {param_name}: {default_val} -> {perturbed_val:.4f} ({pct:+d}%)...")

            config = make_perturbed_config(param_key, perturbed_val)
            t0 = time.time()
            t1_f1, t2_f1 = evaluate_config(config, all_data)
            elapsed = time.time() - t0

            delta_t1 = t1_f1 - baseline_t1_f1
            delta_t2 = t2_f1 - baseline_t2_f1 if np.isfinite(t2_f1) and np.isfinite(baseline_t2_f1) else np.nan

            if verbose:
                print(f"    Tier-1 F1={t1_f1:.3f} (delta={delta_t1:+.3f}), "
                      f"Tier-2 F1={t2_f1:.3f} (delta={delta_t2:+.3f}) ({elapsed:.1f}s)")

            results.append({
                "parameter": param_name,
                "param_key": param_key,
                "default_value": default_val,
                "perturbed_value": perturbed_val,
                "perturbation_pct": pct,
                "tier1_f1": t1_f1,
                "tier2_f1": t2_f1,
                "delta_tier1_f1": delta_t1,
                "delta_tier2_f1": delta_t2,
            })

    return pd.DataFrame(results), baseline_t1_f1, baseline_t2_f1


def print_summary(df, baseline_t1, baseline_t2):
    """Print summary of sensitivity analysis."""
    print("\n" + "=" * 80)
    print("  CHORD Threshold Sensitivity Analysis")
    print("=" * 80)
    print(f"\nBaseline: Tier-1 F1={baseline_t1:.3f}, Tier-2 F1={baseline_t2:.3f}")

    print("\nPerturbation results:")
    perturbed = df[df["parameter"] != "BASELINE"]
    cols = ["parameter", "perturbation_pct", "default_value", "perturbed_value",
            "tier1_f1", "delta_tier1_f1", "tier2_f1", "delta_tier2_f1"]
    print(perturbed[cols].to_string(index=False, float_format="%.3f"))

    # Check stability
    max_delta_t1 = perturbed["delta_tier1_f1"].abs().max()
    max_delta_t2 = perturbed["delta_tier2_f1"].abs().max()
    print(f"\nMax |delta Tier-1 F1| = {max_delta_t1:.4f}")
    print(f"Max |delta Tier-2 F1| = {max_delta_t2:.4f}")

    if max_delta_t1 < 0.05 and (np.isnan(max_delta_t2) or max_delta_t2 < 0.05):
        print("\nConclusion: All thresholds are ROBUST to +/-20% perturbation (|delta F1| < 0.05).")
    else:
        print("\nWARNING: Some thresholds show sensitivity > 0.05.")
        sensitive = perturbed[perturbed["delta_tier1_f1"].abs() >= 0.05]
        if len(sensitive) > 0:
            print("Sensitive parameters (Tier-1):")
            print(sensitive[["parameter", "perturbation_pct", "delta_tier1_f1"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHORD Threshold Sensitivity Analysis")
    parser.add_argument("--reps", type=int, default=20, help="Synthetic replicates")
    parser.add_argument("--output", type=str, default="results/benchmark/sensitivity_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 5 reps")
    args = parser.parse_args()

    if args.quick:
        args.reps = 5

    print(f"CHORD Threshold Sensitivity Analysis (reps={args.reps})")
    df, baseline_t1, baseline_t2 = run_sensitivity(n_reps=args.reps, seed=args.seed)
    print_summary(df, baseline_t1, baseline_t2)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
