#!/usr/bin/env python3
"""
5-fold cross-validation for CHORD threshold stability.

Demonstrates that CHORD's classification thresholds are not overfit to
synthetic training data. Splits synthetic scenarios into 5 folds,
evaluates each fold with default thresholds.

Output: CV results CSV with per-fold and mean +/- std metrics.

Usage:
    python cross_validation.py --reps 50 --folds 5
    python cross_validation.py --reps 10 --folds 5 --quick
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
    """Compute sensitivity, specificity, precision, F1."""
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
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def run_cv(n_reps=50, n_folds=5, seed=42, verbose=True):
    """Run 5-fold CV on synthetic data with default CHORD thresholds."""
    t = np.arange(0, 48, 1.0)  # 48 timepoints @ 1h

    # Generate all data
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
                "scenario": truth.get("scenario", "unknown"),
                "rep": rep,
            })

    if verbose:
        print(f"Total samples: {len(all_data)}")

    # Create fold indices (stratified by scenario for balance)
    indices = np.arange(len(all_data))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    fold_size = len(indices) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(indices)
        folds.append(indices[start:end])

    fold_results = []
    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{n_folds} ({len(test_idx)} test samples)...")

        config = CHORDConfig()
        t_start = time.time()

        y_true_detect = []
        y_pred_detect = []
        y_true_disent = []
        y_pred_disent = []

        for idx in test_idx:
            d = all_data[idx]
            try:
                result = classify_gene(d["t"], d["y"], config=config)
            except Exception:
                result = {"stage1_passed": False, "classification": "error"}

            # Tier-1: detection
            y_true_detect.append(1 if d["gt_has_12h"] else 0)
            y_pred_detect.append(1 if result.get("stage1_passed", False) else 0)

            # Tier-2: disentanglement (only for genes with any 12h)
            if d["gt_has_12h"]:
                y_true_disent.append(1 if d["gt_independent"] else 0)
                cls = result.get("classification", "")
                y_pred_disent.append(
                    1 if cls in ("independent_ultradian", "likely_independent_ultradian") else 0
                )

        elapsed = time.time() - t_start

        # Tier-1 metrics
        t1 = _compute_binary_metrics(y_true_detect, y_pred_detect)

        # Tier-2 metrics
        if y_true_disent:
            t2 = _compute_binary_metrics(y_true_disent, y_pred_disent)
        else:
            t2 = {"sensitivity": np.nan, "specificity": np.nan,
                   "precision": np.nan, "f1": np.nan}

        row = {
            "fold": fold_idx + 1,
            "n_test": len(test_idx),
            "tier1_sensitivity": t1["sensitivity"],
            "tier1_specificity": t1["specificity"],
            "tier1_precision": t1["precision"],
            "tier1_f1": t1["f1"],
            "tier2_sensitivity": t2["sensitivity"],
            "tier2_specificity": t2["specificity"],
            "tier2_precision": t2["precision"],
            "tier2_f1": t2["f1"],
            "wall_time_s": elapsed,
        }
        fold_results.append(row)

        if verbose:
            print(f"    Tier-1 F1={t1['f1']:.3f}  Tier-2 F1={t2['f1']:.3f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(fold_results)
    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("  CHORD 5-Fold Cross-Validation Results")
    print("=" * 70)
    print("\nPer-fold results:")
    print(df.to_string(index=False, float_format="%.3f"))

    print("\nSummary (mean +/- std):")
    for col in ["tier1_f1", "tier1_sensitivity", "tier1_specificity",
                 "tier2_f1", "tier2_sensitivity", "tier2_specificity"]:
        vals = df[col].dropna()
        print(f"  {col:25s}: {vals.mean():.3f} +/- {vals.std():.3f}")

    tier1_std = df["tier1_f1"].std()
    tier2_std = df["tier2_f1"].dropna().std()
    print(f"\nConclusion: Tier-1 F1 std = {tier1_std:.4f}, Tier-2 F1 std = {tier2_std:.4f}")
    if tier1_std < 0.05 and tier2_std < 0.05:
        print("  -> Thresholds are STABLE (std < 0.05). Not overfit.")
    else:
        print("  -> WARNING: High variance detected. Thresholds may be overfit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHORD 5-Fold Cross-Validation")
    parser.add_argument("--reps", type=int, default=50, help="Synthetic replicates")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", type=str, default="results/benchmark/cv_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 reps")
    args = parser.parse_args()

    if args.quick:
        args.reps = 10

    print(f"CHORD 5-Fold Cross-Validation (reps={args.reps}, folds={args.folds})")
    df = run_cv(n_reps=args.reps, n_folds=args.folds, seed=args.seed)
    print_summary(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
