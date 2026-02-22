"""
Multi-tissue 12h rhythm atlas using Mure2018 baboon data (60 tissues).

Runs BHDT V2 genome-wide on 10 representative baboon tissues to create
a cross-tissue 12h rhythm atlas. Uses F-test pre-screening for speed.

Usage:
    PYTHONPATH=src python analysis_multi_tissue.py
"""

import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from chord.data.geo_loader import load_mure2018
from chord.bhdt.inference import component_f_test, bhdt_analytic

# ============================================================================
# Configuration
# ============================================================================

TISSUES = {
    "LIV": "Liver",
    "HEA": "Heart",
    "KIC": "Kidney cortex",
    "LUN": "Lung",
    "CER": "Cerebellum",
    "AOR": "Aorta",
    "WAT": "White adipose",
    "SPL": "Spleen",
    "PAN": "Pancreas",
    "SMM": "Skeletal muscle",
}

# UPR/ER stress genes to track across tissues
UPR_GENES = {"HSPA5", "XBP1", "ATF4", "ATF6", "PDIA4", "PDIA6",
             "DDIT3", "ERN1", "EIF2AK3", "CALR", "CANX", "HSP90B1"}

# 12h classification categories
TWELVE_H_CLASSES = {"independent_ultradian", "likely_independent_ultradian"}

# Pre-filter threshold for F-test screen
FTEST_PREFILTER_P = 0.3


# ============================================================================
# Core analysis
# ============================================================================

def fast_screen_12h(t, y):
    """Quick F-test screen for 12h component. Returns p-value or 1.0 on error."""
    try:
        f12 = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)
        return f12["p_value"]
    except Exception:
        return 1.0


def analyze_tissue(tissue_code, tissue_name):
    """Run genome-wide BHDT V2 on a single tissue with F-test pre-screening."""
    print(f"\n{'='*70}")
    print(f"  Analyzing {tissue_code} ({tissue_name})")
    print(f"{'='*70}")

    # Load data
    t0 = time.time()
    try:
        data = load_mure2018(tissue=tissue_code)
    except Exception as e:
        print(f"  ERROR loading {tissue_code}: {e}")
        return None
    t = data["timepoints"]
    expr = data["expr"]
    gene_names = data["gene_names"]
    n_genes = len(gene_names)
    print(f"  Loaded {n_genes} genes, {len(t)} timepoints")

    # Phase 1: F-test pre-screen
    t1 = time.time()
    pass_screen = []
    for i in range(n_genes):
        p12 = fast_screen_12h(t, expr[i])
        if p12 < FTEST_PREFILTER_P:
            pass_screen.append(i)
    n_pass = len(pass_screen)
    print(f"  F-test pre-screen: {n_pass}/{n_genes} genes pass (p<{FTEST_PREFILTER_P})")
    print(f"  Pre-screen took {time.time()-t1:.1f}s")

    # Phase 2: Full BHDT V2 on genes that pass screen
    t2 = time.time()
    results = {}  # gene_name -> classification dict
    for idx, i in enumerate(pass_screen):
        gene = gene_names[i]
        y = expr[i]
        try:
            r = bhdt_analytic(t, y, classifier_version="v2")
            results[gene] = {
                "classification": r["classification"],
                "log_bf": r["log_bayes_factor"],
                "bayes_factor": r["bayes_factor"],
                "amp_ratio": _get_amp_ratio(r),
                "p12_ftest": component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)["p_value"],
            }
        except Exception as e:
            results[gene] = {"classification": "error", "log_bf": 0, "bayes_factor": 0,
                             "amp_ratio": 0, "p12_ftest": 1.0}

        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx+1}/{n_pass} genes...")

    # Genes that didn't pass screen are non_rhythmic (for 12h purposes)
    screened_genes = {gene_names[i] for i in pass_screen}
    for i in range(n_genes):
        gene = gene_names[i]
        if gene not in screened_genes:
            results[gene] = {"classification": "non_rhythmic", "log_bf": 0,
                             "bayes_factor": 0, "amp_ratio": 0, "p12_ftest": 1.0}

    elapsed = time.time() - t2
    print(f"  Full BHDT V2 on {n_pass} genes took {elapsed:.1f}s")

    # Tally classifications
    class_counts = defaultdict(int)
    for gene, r in results.items():
        class_counts[r["classification"]] += 1

    n_12h = sum(class_counts.get(c, 0) for c in TWELVE_H_CLASSES)
    pct_12h = 100.0 * n_12h / n_genes if n_genes > 0 else 0

    # Top 12h genes by evidence (log_bf + amp_ratio)
    twelve_h_genes = []
    for gene, r in results.items():
        if r["classification"] in TWELVE_H_CLASSES:
            # Composite score: log_bf + 2*amp_ratio (amp_ratio is strong indicator)
            score = r["log_bf"] + 2.0 * r["amp_ratio"]
            twelve_h_genes.append((gene, r["classification"], r["log_bf"],
                                   r["amp_ratio"], r["p12_ftest"], score))
    twelve_h_genes.sort(key=lambda x: -x[5])  # sort by score descending

    total_time = time.time() - t0
    print(f"  Total: {n_12h} 12h genes ({pct_12h:.1f}%), took {total_time:.1f}s")

    return {
        "tissue": tissue_code,
        "tissue_name": tissue_name,
        "n_genes": n_genes,
        "n_12h": n_12h,
        "pct_12h": pct_12h,
        "class_counts": dict(class_counts),
        "top_12h": twelve_h_genes[:20],
        "all_12h_genes": {g[0] for g in twelve_h_genes},
        "results": results,
        "upr_results": {g: results.get(g, {"classification": "not_found"})
                        for g in UPR_GENES},
    }


def _get_amp_ratio(r):
    """Extract A_12/A_24 amplitude ratio from BHDT result."""
    try:
        m1 = r.get("m1", {})
        comps = m1.get("components", [])
        amps = {c["T"]: c["A"] for c in comps}
        a24 = amps.get(24.0, 1e-10)
        a12 = amps.get(12.0, 0)
        return a12 / max(a24, 1e-10)
    except Exception:
        return 0.0


# ============================================================================
# Reporting
# ============================================================================

def print_summary_table(tissue_results):
    """Print the cross-tissue summary table."""
    print("\n" + "=" * 100)
    print("  MULTI-TISSUE 12h RHYTHM ATLAS — Summary")
    print("=" * 100)

    header = (f"{'Tissue':<6} | {'Name':<16} | {'Total':>6} | {'12h':>5} | "
              f"{'%12h':>5} | {'Circadian':>9} | {'Harmonic':>8} | "
              f"{'Ambiguous':>9} | {'NonRhyth':>8}")
    print(header)
    print("-" * len(header))

    for tr in tissue_results:
        cc = tr["class_counts"]
        circ = cc.get("circadian_only", 0)
        harm = cc.get("harmonic", 0)
        ambig = cc.get("ambiguous", 0)
        nr = cc.get("non_rhythmic", 0)
        print(f"{tr['tissue']:<6} | {tr['tissue_name']:<16} | {tr['n_genes']:>6} | "
              f"{tr['n_12h']:>5} | {tr['pct_12h']:>4.1f}% | {circ:>9} | "
              f"{harm:>8} | {ambig:>9} | {nr:>8}")


def print_top_genes(tissue_results):
    """Print top 20 12h genes per tissue."""
    for tr in tissue_results:
        print(f"\n--- Top 20 12h genes in {tr['tissue']} ({tr['tissue_name']}) ---")
        if not tr["top_12h"]:
            print("  (none)")
            continue
        print(f"  {'Gene':<12} {'Classification':<30} {'logBF':>6} {'AmpR':>6} {'p12':>8}")
        for gene, cls, lbf, ar, p12, score in tr["top_12h"]:
            print(f"  {gene:<12} {cls:<30} {lbf:>6.2f} {ar:>6.2f} {p12:>8.4f}")


def cross_tissue_analysis(tissue_results):
    """Identify conserved 12h genes and tissue-specific patterns."""
    print("\n" + "=" * 100)
    print("  CROSS-TISSUE ANALYSIS")
    print("=" * 100)

    # Count how many tissues each 12h gene appears in
    gene_tissue_count = defaultdict(list)
    for tr in tissue_results:
        for gene in tr["all_12h_genes"]:
            gene_tissue_count[gene].append(tr["tissue"])

    # Conserved 12h genes (>= 3 tissues)
    conserved = {g: ts for g, ts in gene_tissue_count.items() if len(ts) >= 3}
    print(f"\n--- Conserved 12h genes (present in >=3 tissues): {len(conserved)} ---")
    for gene, tissues in sorted(conserved.items(), key=lambda x: -len(x[1])):
        print(f"  {gene:<12} ({len(tissues)} tissues): {', '.join(sorted(tissues))}")

    # Genes in >= 2 tissues
    shared_2 = {g: ts for g, ts in gene_tissue_count.items() if len(ts) >= 2}
    print(f"\n--- 12h genes shared across >=2 tissues: {len(shared_2)} ---")

    # Tissue ranking by 12h gene count
    print(f"\n--- Tissue ranking by 12h gene count ---")
    ranked = sorted(tissue_results, key=lambda x: -x["n_12h"])
    for tr in ranked:
        bar = "#" * (tr["n_12h"] // 10)
        print(f"  {tr['tissue']:<6} {tr['tissue_name']:<16} {tr['n_12h']:>5} 12h genes  {bar}")

    # UPR/ER stress gene analysis
    print(f"\n--- UPR/ER stress genes across tissues ---")
    print(f"  {'Gene':<12}", end="")
    for tr in tissue_results:
        print(f" {tr['tissue']:>6}", end="")
    print()
    for gene in sorted(UPR_GENES):
        print(f"  {gene:<12}", end="")
        for tr in tissue_results:
            upr = tr["upr_results"].get(gene, {})
            cls = upr.get("classification", "N/A")
            if cls in TWELVE_H_CLASSES:
                marker = "  12h"
            elif cls == "harmonic":
                marker = " harm"
            elif cls == "circadian_only":
                marker = " circ"
            elif cls == "non_rhythmic":
                marker = "   NR"
            elif cls == "ambiguous":
                marker = "  amb"
            elif cls == "not_found":
                marker = "    -"
            else:
                marker = f" {cls[:4]}"
            print(f" {marker:>6}", end="")
        print()

    return conserved, shared_2


def save_results(tissue_results, conserved, output_path="multi_tissue_12h_atlas.csv"):
    """Save all results to CSV."""
    rows = []
    for tr in tissue_results:
        for gene, r in tr["results"].items():
            rows.append({
                "tissue": tr["tissue"],
                "tissue_name": tr["tissue_name"],
                "gene": gene,
                "classification": r["classification"],
                "log_bf": r["log_bf"],
                "amp_ratio": r["amp_ratio"],
                "p12_ftest": r["p12_ftest"],
                "is_12h": r["classification"] in TWELVE_H_CLASSES,
                "is_conserved_12h": gene in conserved,
                "is_upr_gene": gene in UPR_GENES,
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path} ({len(df)} rows)")
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 100)
    print("  CHORD Multi-Tissue 12h Rhythm Atlas")
    print("  Mure et al. 2018 — Baboon 60-tissue dataset")
    print("  Running BHDT V2 genome-wide on 10 representative tissues")
    print("=" * 100)

    overall_start = time.time()
    tissue_results = []

    for tissue_code, tissue_name in TISSUES.items():
        result = analyze_tissue(tissue_code, tissue_name)
        if result is not None:
            tissue_results.append(result)

    if not tissue_results:
        print("ERROR: No tissues were successfully analyzed.")
        sys.exit(1)

    # Summary table
    print_summary_table(tissue_results)

    # Top genes per tissue
    print_top_genes(tissue_results)

    # Cross-tissue analysis
    conserved, shared_2 = cross_tissue_analysis(tissue_results)

    # Save
    df = save_results(tissue_results, conserved)

    total_time = time.time() - overall_start
    print(f"\nTotal analysis time: {total_time/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
