"""
Zhang2014 multi-tissue 12h rhythm atlas using mouse circadian atlas (GSE54650).

Runs BHDT V2 genome-wide on 12 mouse tissues to create a cross-tissue
12h rhythm atlas. Compares results with Zhu 2017 eigenvalue/pencil findings.

Key validation: Does CHORD's tissue-level 12h gene distribution match
Zhu 2017's ranking (BAT > skeletal muscle > WAT > aorta > lung > liver
> heart > kidney > hypothalamus)?

Usage:
    PYTHONPATH=src python analysis_zhang2014_multi_tissue.py
"""

import sys
import time
import warnings
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from chord.data.geo_loader import load_zhang2014, ZHANG2014_TISSUES
from chord.data.known_genes import KNOWN_12H_GENES_ZHU2017, CORE_CIRCADIAN_GENES
from chord.bhdt.inference import component_f_test, bhdt_analytic

# ============================================================================
# Configuration
# ============================================================================

# 12h classification categories
TWELVE_H_CLASSES = {"independent_ultradian", "likely_independent_ultradian"}

# Pre-filter threshold for F-test screen
FTEST_PREFILTER_P = 0.1  # stricter than 0.3 to reduce computation

# Zhu 2017 tissue ranking for 12h genes (eigenvalue/pencil method)
# From Zhu et al. 2017 Table S1 — approximate 12h gene counts
ZHU2017_TISSUE_RANKING = {
    "BFat": "BAT (highest 12h)",
    "Mus": "Skeletal muscle",
    "WFat": "WAT",
    "Aor": "Aorta",
    "Lun": "Lung",
    "Liv": "Liver",
    "Hrt": "Heart",
    "Kid": "Kidney",
    "Hyp": "Hypothalamus (lowest 12h)",
}

# UPR/ER stress genes to track (mouse symbols)
UPR_GENES = {
    "Xbp1", "Hspa5", "Atf4", "Atf6", "Pdia4", "Pdia6",
    "Ddit3", "Ero1l", "Calr", "Canx", "Hsp90b1", "Dnajb9",
}


# ============================================================================
# Core analysis
# ============================================================================

def fast_screen_12h(t, y):
    """Quick F-test screen for 12h component. Returns p-value or 1.0."""
    try:
        f12 = component_f_test(t, y, [24.0, 12.0, 8.0], test_period_idx=1)
        return f12["p_value"]
    except Exception:
        return 1.0


def analyze_tissue(tissue_code, tissue_name, cache_dir=".zhang2014_cache"):
    """Run genome-wide BHDT V2 on a single Zhang2014 tissue with caching."""
    # Check for cached results
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "{}.pkl".format(tissue_code))
    if os.path.exists(cache_file):
        print("\n  Loading cached results for {} ({})".format(
            tissue_code, tissue_name))
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("\n" + "=" * 70)
    print("  Analyzing {} ({})".format(tissue_code, tissue_name))
    print("=" * 70)

    t0 = time.time()
    try:
        data = load_zhang2014(tissue=tissue_code)
    except Exception as e:
        print("  ERROR loading {}: {}".format(tissue_code, e))
        return None

    t = data["timepoints"]
    expr = data["expr"]
    gene_names = data["gene_names"]
    n_genes = len(gene_names)
    print("  Loaded {} genes, {} timepoints (CT{}-CT{})".format(
        n_genes, len(t), int(t[0]), int(t[-1])))

    # Phase 1: F-test pre-screen
    t1 = time.time()
    pass_screen = []
    for i in range(n_genes):
        p12 = fast_screen_12h(t, expr[i])
        if p12 < FTEST_PREFILTER_P:
            pass_screen.append(i)
    n_pass = len(pass_screen)
    print("  F-test pre-screen: {}/{} genes pass (p<{})".format(
        n_pass, n_genes, FTEST_PREFILTER_P))
    print("  Pre-screen took {:.1f}s".format(time.time() - t1))

    # Phase 2: Full BHDT V2 on genes that pass screen
    t2 = time.time()
    results = {}
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
                "p12_ftest": component_f_test(
                    t, y, [24.0, 12.0, 8.0], test_period_idx=1
                )["p_value"],
            }
        except Exception:
            results[gene] = {
                "classification": "error", "log_bf": 0,
                "bayes_factor": 0, "amp_ratio": 0, "p12_ftest": 1.0,
            }

        if (idx + 1) % 500 == 0:
            print("    Processed {}/{} genes...".format(idx + 1, n_pass))

    # Genes that didn't pass screen
    screened_genes = {gene_names[i] for i in pass_screen}
    for i in range(n_genes):
        gene = gene_names[i]
        if gene not in screened_genes:
            results[gene] = {
                "classification": "non_rhythmic", "log_bf": 0,
                "bayes_factor": 0, "amp_ratio": 0, "p12_ftest": 1.0,
            }

    elapsed = time.time() - t2
    print("  Full BHDT V2 on {} genes took {:.1f}s".format(n_pass, elapsed))

    # Tally
    class_counts = defaultdict(int)
    for gene, r in results.items():
        class_counts[r["classification"]] += 1

    n_12h = sum(class_counts.get(c, 0) for c in TWELVE_H_CLASSES)
    pct_12h = 100.0 * n_12h / n_genes if n_genes > 0 else 0

    # Top 12h genes
    twelve_h_genes = []
    for gene, r in results.items():
        if r["classification"] in TWELVE_H_CLASSES:
            score = r["log_bf"] + 2.0 * r["amp_ratio"]
            twelve_h_genes.append((
                gene, r["classification"], r["log_bf"],
                r["amp_ratio"], r["p12_ftest"], score
            ))
    twelve_h_genes.sort(key=lambda x: -x[5])

    total_time = time.time() - t0
    print("  Total: {} 12h genes ({:.1f}%), took {:.1f}s".format(
        n_12h, pct_12h, total_time))

    result = {
        "tissue": tissue_code,
        "tissue_name": tissue_name,
        "n_genes": n_genes,
        "n_12h": n_12h,
        "pct_12h": pct_12h,
        "class_counts": dict(class_counts),
        "top_12h": twelve_h_genes[:20],
        "all_12h_genes": {g[0] for g in twelve_h_genes},
        "results": results,
        "upr_results": {
            g: results.get(g, {"classification": "not_found"})
            for g in UPR_GENES
        },
    }

    # Cache result
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print("  Cached to {}".format(cache_file))

    return result


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
    """Print cross-tissue summary."""
    print("\n" + "=" * 100)
    print("  ZHANG2014 MULTI-TISSUE 12h RHYTHM ATLAS — Summary")
    print("=" * 100)

    header = (
        "{:<6} | {:<20} | {:>6} | {:>5} | {:>5} | {:>9} | {:>8} | "
        "{:>9} | {:>8}".format(
            "Tissue", "Name", "Total", "12h", "%12h",
            "Circadian", "Harmonic", "Ambiguous", "NonRhyth"
        )
    )
    print(header)
    print("-" * len(header))

    for tr in tissue_results:
        cc = tr["class_counts"]
        circ = cc.get("circadian_only", 0)
        harm = cc.get("harmonic", 0)
        ambig = cc.get("ambiguous", 0)
        nr = cc.get("non_rhythmic", 0)
        print(
            "{:<6} | {:<20} | {:>6} | {:>5} | {:>4.1f}% | {:>9} | "
            "{:>8} | {:>9} | {:>8}".format(
                tr["tissue"], tr["tissue_name"], tr["n_genes"],
                tr["n_12h"], tr["pct_12h"], circ, harm, ambig, nr
            )
        )


def compare_with_zhu2017(tissue_results):
    """Compare CHORD tissue ranking with Zhu 2017."""
    print("\n" + "=" * 100)
    print("  COMPARISON WITH ZHU 2017 EIGENVALUE/PENCIL RESULTS")
    print("=" * 100)

    # Zhu 2017 ranking (approximate order from their paper)
    zhu_order = ["BFat", "Mus", "WFat", "Aor", "Lun", "Liv",
                 "Hrt", "Kid", "Hyp"]

    # CHORD ranking
    chord_ranked = sorted(tissue_results, key=lambda x: -x["n_12h"])
    chord_order = [tr["tissue"] for tr in chord_ranked]

    print("\n  Zhu 2017 ranking (eigenvalue/pencil):")
    for i, t in enumerate(zhu_order, 1):
        name = ZHANG2014_TISSUES.get(t, t)
        print("    {}. {} ({})".format(i, t, name))

    print("\n  CHORD ranking (BHDT V2):")
    for i, tr in enumerate(chord_ranked, 1):
        print("    {}. {} ({}) — {} 12h genes ({:.1f}%)".format(
            i, tr["tissue"], tr["tissue_name"],
            tr["n_12h"], tr["pct_12h"]
        ))

    # Compute rank correlation
    chord_rank = {t: i for i, t in enumerate(chord_order)}
    zhu_rank = {t: i for i, t in enumerate(zhu_order)}
    common = set(chord_rank.keys()) & set(zhu_rank.keys())
    if len(common) >= 3:
        from scipy.stats import spearmanr
        chord_ranks = [chord_rank[t] for t in sorted(common)]
        zhu_ranks = [zhu_rank[t] for t in sorted(common)]
        rho, pval = spearmanr(chord_ranks, zhu_ranks)
        print("\n  Spearman rank correlation: rho={:.3f}, p={:.4f}".format(
            rho, pval))
        if rho > 0.5:
            print("  -> CHORD ranking is CONSISTENT with Zhu 2017")
        elif rho > 0:
            print("  -> CHORD ranking shows PARTIAL agreement with Zhu 2017")
        else:
            print("  -> CHORD ranking DIFFERS from Zhu 2017")


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
    print("\n--- Conserved 12h genes (>=3 tissues): {} ---".format(
        len(conserved)))
    for gene, tissues in sorted(conserved.items(), key=lambda x: -len(x[1])):
        is_known = " [Zhu2017]" if gene in set(KNOWN_12H_GENES_ZHU2017) else ""
        print("  {:<12} ({} tissues): {}{}".format(
            gene, len(tissues), ", ".join(sorted(tissues)), is_known))

    # Known 12h gene recovery
    known_set = set(KNOWN_12H_GENES_ZHU2017)
    print("\n--- Known 12h gene (Zhu 2017) recovery per tissue ---")
    for tr in tissue_results:
        found = tr["all_12h_genes"] & known_set
        print("  {:<6} {:<20}: {}/{} known 12h genes detected".format(
            tr["tissue"], tr["tissue_name"],
            len(found), len(known_set)))
        if found:
            print("         {}".format(", ".join(sorted(found)[:10])))

    # UPR/ER stress gene analysis
    print("\n--- UPR/ER stress genes across tissues ---")
    print("  {:<12}".format("Gene"), end="")
    for tr in tissue_results:
        print(" {:>6}".format(tr["tissue"]), end="")
    print()
    for gene in sorted(UPR_GENES):
        print("  {:<12}".format(gene), end="")
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
                marker = " {:>4}".format(cls[:4])
            print(" {:>6}".format(marker), end="")
        print()

    return conserved


def save_results(tissue_results, conserved,
                 output_path="zhang2014_multi_tissue_12h_atlas.csv"):
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
                "is_known_zhu2017": gene in set(KNOWN_12H_GENES_ZHU2017),
                "is_upr_gene": gene in UPR_GENES,
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("\nResults saved to {} ({} rows)".format(output_path, len(df)))
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 100)
    print("  CHORD Multi-Tissue 12h Rhythm Atlas")
    print("  Zhang et al. 2014 (PNAS) — 12 mouse tissues, 2h resolution, 48h")
    print("  Running BHDT V2 genome-wide on all 12 tissues")
    print("=" * 100)

    overall_start = time.time()
    tissue_results = []

    for tissue_code, tissue_name in ZHANG2014_TISSUES.items():
        result = analyze_tissue(tissue_code, tissue_name)
        if result is not None:
            tissue_results.append(result)

    if not tissue_results:
        print("ERROR: No tissues were successfully analyzed.")
        sys.exit(1)

    # Summary table
    print_summary_table(tissue_results)

    # Compare with Zhu 2017
    compare_with_zhu2017(tissue_results)

    # Cross-tissue analysis
    conserved = cross_tissue_analysis(tissue_results)

    # Save
    df = save_results(tissue_results, conserved)

    total_time = time.time() - overall_start
    print("\nTotal analysis time: {:.1f} minutes".format(total_time / 60))
    print("Done.")


if __name__ == "__main__":
    main()
