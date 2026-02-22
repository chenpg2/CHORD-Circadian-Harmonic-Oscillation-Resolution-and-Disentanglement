#!/usr/bin/env python3
"""
BMAL1 KO Biological Validation for CHORD.

Tests the biological prediction that:
- Genes classified as "harmonic" in WT should LOSE their 12h signal in
  BMAL1-KO (because the 12h component is driven by the 24h clock)
- Genes classified as "independent_ultradian" in WT should RETAIN their
  12h signal in BMAL1-KO (because the 12h oscillator is autonomous)

Uses Zhu 2023 WT and BMAL1-KO mouse liver RNA-seq data.

Output:
- Per-gene WT/KO classification comparison CSV
- Summary statistics with Fisher's exact test
- Concordance rates

Usage:
    python bmal1ko_validation.py
    python bmal1ko_validation.py --output results/benchmark/bmal1ko_validation.csv
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from chord.bhdt.classifier import classify_gene, CHORDConfig


def run_bmal1ko_validation(cache_dir="~/.chord_cache", verbose=True):
    """Run BMAL1 KO validation analysis."""
    from chord.data.geo_loader import load_zhu2023_bmal1ko

    # Load data
    if verbose:
        print("Loading Zhu 2023 BMAL1-KO data...")
    data = load_zhu2023_bmal1ko(cache_dir=cache_dir)
    wt = data["wt"]
    ko = data["ko"]

    expr_wt = wt["expr"]
    tp_wt = wt["timepoints"]
    genes_wt = wt["gene_names"]

    expr_ko = ko["expr"]
    tp_ko = ko["timepoints"]
    genes_ko = ko["gene_names"]

    if verbose:
        print(f"  WT: {expr_wt.shape[0]} genes x {expr_wt.shape[1]} timepoints")
        print(f"  KO: {expr_ko.shape[0]} genes x {expr_ko.shape[1]} timepoints")

    # Find common genes
    wt_gene_idx = {g: i for i, g in enumerate(genes_wt)}
    ko_gene_idx = {g: i for i, g in enumerate(genes_ko)}
    common_genes = sorted(set(genes_wt) & set(genes_ko))
    if verbose:
        print(f"  Common genes: {len(common_genes)}")

    # Run CHORD on both conditions
    config = CHORDConfig()
    results = []

    for gi, gene in enumerate(common_genes):
        if verbose and (gi + 1) % 500 == 0:
            print(f"  Processing {gi + 1}/{len(common_genes)}...")

        y_wt = expr_wt[wt_gene_idx[gene]]
        y_ko = expr_ko[ko_gene_idx[gene]]

        # Skip genes with zero variance
        if np.nanstd(y_wt) < 1e-10 or np.nanstd(y_ko) < 1e-10:
            continue

        try:
            r_wt = classify_gene(tp_wt, y_wt, config=config)
        except Exception:
            r_wt = {"classification": "error", "stage1_passed": False, "confidence": 0.0}

        try:
            r_ko = classify_gene(tp_ko, y_ko, config=config)
        except Exception:
            r_ko = {"classification": "error", "stage1_passed": False, "confidence": 0.0}

        cls_wt = r_wt.get("classification", "error")
        cls_ko = r_ko.get("classification", "error")
        has12h_wt = r_wt.get("stage1_passed", False)
        has12h_ko = r_ko.get("stage1_passed", False)

        results.append({
            "gene": gene,
            "wt_classification": cls_wt,
            "ko_classification": cls_ko,
            "wt_has_12h": has12h_wt,
            "ko_has_12h": has12h_ko,
            "wt_confidence": r_wt.get("confidence", 0.0),
            "ko_confidence": r_ko.get("confidence", 0.0),
            "wt_p_detect": r_wt.get("stage1_p_detect", 1.0),
            "ko_p_detect": r_ko.get("stage1_p_detect", 1.0),
        })

    df = pd.DataFrame(results)
    return df


def compute_validation_stats(df, verbose=True):
    """Compute concordance statistics and Fisher's exact test."""
    stats = {}

    # Count WT classifications
    wt_counts = df["wt_classification"].value_counts()
    stats["wt_classification_counts"] = wt_counts.to_dict()

    # Focus on genes classified as harmonic or independent in WT
    harmonic_wt = df[df["wt_classification"] == "harmonic"]
    independent_wt = df[df["wt_classification"].isin(
        ["independent_ultradian", "likely_independent_ultradian"]
    )]

    stats["n_harmonic_wt"] = len(harmonic_wt)
    stats["n_independent_wt"] = len(independent_wt)

    # Prediction: harmonic genes should LOSE 12h in KO
    if len(harmonic_wt) > 0:
        harmonic_lost_12h = (~harmonic_wt["ko_has_12h"]).sum()
        harmonic_kept_12h = harmonic_wt["ko_has_12h"].sum()
        stats["harmonic_lost_12h_in_ko"] = int(harmonic_lost_12h)
        stats["harmonic_kept_12h_in_ko"] = int(harmonic_kept_12h)
        stats["harmonic_loss_rate"] = harmonic_lost_12h / len(harmonic_wt)
    else:
        stats["harmonic_loss_rate"] = np.nan

    # Prediction: independent genes should RETAIN 12h in KO
    if len(independent_wt) > 0:
        indep_kept_12h = independent_wt["ko_has_12h"].sum()
        indep_lost_12h = (~independent_wt["ko_has_12h"]).sum()
        stats["independent_kept_12h_in_ko"] = int(indep_kept_12h)
        stats["independent_lost_12h_in_ko"] = int(indep_lost_12h)
        stats["independent_retention_rate"] = indep_kept_12h / len(independent_wt)
    else:
        stats["independent_retention_rate"] = np.nan

    # Fisher's exact test: is the retention rate different between groups?
    # Contingency table:
    #                    Lost 12h in KO | Kept 12h in KO
    # Harmonic in WT:        a          |      b
    # Independent in WT:     c          |      d
    if len(harmonic_wt) > 0 and len(independent_wt) > 0:
        a = int((~harmonic_wt["ko_has_12h"]).sum())
        b = int(harmonic_wt["ko_has_12h"].sum())
        c = int((~independent_wt["ko_has_12h"]).sum())
        d = int(independent_wt["ko_has_12h"].sum())

        table = np.array([[a, b], [c, d]])
        odds_ratio, p_value = fisher_exact(table, alternative="greater")
        stats["fisher_table"] = table.tolist()
        stats["fisher_odds_ratio"] = odds_ratio
        stats["fisher_p_value"] = p_value
    else:
        stats["fisher_p_value"] = np.nan

    # Also check: do ambiguous genes behave differently?
    ambiguous_wt = df[df["wt_classification"] == "ambiguous"]
    if len(ambiguous_wt) > 0:
        stats["n_ambiguous_wt"] = len(ambiguous_wt)
        stats["ambiguous_loss_rate"] = float((~ambiguous_wt["ko_has_12h"]).sum() / len(ambiguous_wt))

    if verbose:
        print("\n" + "=" * 70)
        print("  BMAL1 KO Biological Validation Results")
        print("=" * 70)
        print(f"\nWT classification distribution:")
        for cls, count in sorted(wt_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls:35s}: {count}")

        print(f"\nHarmonic genes in WT (n={stats['n_harmonic_wt']}):")
        if stats["n_harmonic_wt"] > 0:
            print(f"  Lost 12h in KO: {stats['harmonic_lost_12h_in_ko']} ({stats['harmonic_loss_rate']:.1%})")
            print(f"  Kept 12h in KO: {stats['harmonic_kept_12h_in_ko']}")

        print(f"\nIndependent genes in WT (n={stats['n_independent_wt']}):")
        if stats["n_independent_wt"] > 0:
            print(f"  Kept 12h in KO: {stats['independent_kept_12h_in_ko']} ({stats['independent_retention_rate']:.1%})")
            print(f"  Lost 12h in KO: {stats['independent_lost_12h_in_ko']}")

        if not np.isnan(stats.get("fisher_p_value", np.nan)):
            print(f"\nFisher's exact test (one-sided):")
            print(f"  Contingency table: {stats['fisher_table']}")
            print(f"  Odds ratio: {stats['fisher_odds_ratio']:.2f}")
            print(f"  p-value: {stats['fisher_p_value']:.4f}")
            if stats["fisher_p_value"] < 0.05:
                print("  -> SIGNIFICANT: Harmonic genes lose 12h in KO at higher rate than independent genes")
            else:
                print("  -> Not significant at alpha=0.05")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMAL1 KO Biological Validation")
    parser.add_argument("--output", type=str, default="results/benchmark/bmal1ko_validation.csv")
    parser.add_argument("--cache-dir", type=str, default="~/.chord_cache")
    args = parser.parse_args()

    print("CHORD BMAL1 KO Biological Validation")
    df = run_bmal1ko_validation(cache_dir=args.cache_dir)
    stats = compute_validation_stats(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nPer-gene results saved to {args.output}")

    # Save summary stats
    summary_path = args.output.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Summary saved to {summary_path}")
