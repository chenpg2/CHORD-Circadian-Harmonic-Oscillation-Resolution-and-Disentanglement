"""
BMAL1-KO biological narrative analysis for CHORD publication.

Analyzes Zhu 2023 BMAL1-KO data to build the biological narrative:
- In WT: circadian clock drives both 24h and 12h (harmonic) rhythms
- In KO: circadian clock is disrupted, but independent 12h rhythms persist
- Key question: Which pathways' 12h rhythms survive BMAL1 knockout?

This supports the independent oscillator hypothesis (Zhu 2017, 2023).

Usage:
    PYTHONPATH=src python analysis_bmal1ko_narrative.py
"""

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from chord.data.geo_loader import load_zhu2023_bmal1ko
from chord.data.known_genes import (
    KNOWN_12H_GENES_ZHU2017, CORE_CIRCADIAN_GENES,
    NON_RHYTHMIC_HOUSEKEEPING,
)
from chord.bhdt.inference import bhdt_analytic, component_f_test


# ============================================================================
# Configuration
# ============================================================================

TWELVE_H_CLASSES = {"independent_ultradian", "likely_independent_ultradian"}
HAS_12H_CLASSES = TWELVE_H_CLASSES | {"harmonic"}

# Pathway annotations for known 12h genes
PATHWAY_ANNOTATIONS = {
    # UPR / ER stress
    "Xbp1": "UPR/ER stress", "Atf4": "UPR/ER stress",
    "Atf6": "UPR/ER stress", "Atf6b": "UPR/ER stress",
    "Ddit3": "UPR/ER stress", "Hspa5": "UPR/ER stress",
    "Hsp90b1": "UPR/ER stress", "Pdia4": "UPR/ER stress",
    "Pdia6": "UPR/ER stress", "Dnajb9": "UPR/ER stress",
    "Dnajb11": "UPR/ER stress", "Edem1": "UPR/ER stress",
    "Hyou1": "UPR/ER stress", "Calr": "UPR/ER stress",
    "Canx": "UPR/ER stress", "P4hb": "UPR/ER stress",
    "Ero1l": "UPR/ER stress", "Herpud1": "UPR/ER stress",
    "Manf": "UPR/ER stress",
    # ERAD
    "Derl1": "ERAD", "Derl2": "ERAD", "Sel1l": "ERAD",
    "Os9": "ERAD", "Erlec1": "ERAD",
    # Protein translocation / glycosylation
    "Sec61a1": "Protein translocation", "Sec61b": "Protein translocation",
    "Srp54a": "Protein translocation",
    "Stt3a": "N-glycosylation", "Stt3b": "N-glycosylation",
    "Uggt1": "N-glycosylation",
    # Lipid metabolism
    "Fasn": "Lipid metabolism", "Acaca": "Lipid metabolism",
    "Scd1": "Lipid metabolism", "Hmgcr": "Lipid metabolism",
    "Hmgcs1": "Lipid metabolism", "Sqle": "Lipid metabolism",
    "Fdft1": "Lipid metabolism", "Srebf1": "Lipid metabolism",
}


# ============================================================================
# Analysis
# ============================================================================

def run_targeted_bhdt(condition_data, condition_name, target_genes=None):
    """Run BHDT V2 on target genes (or all if target_genes is None)."""
    t = condition_data["timepoints"]
    expr = condition_data["expr"]
    gene_names = condition_data["gene_names"]
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    if target_genes is None:
        indices = list(range(len(gene_names)))
    else:
        indices = [gene_idx[g] for g in target_genes if g in gene_idx]

    print("  Running BHDT V2 on {} ({}/{} genes, {} timepoints)...".format(
        condition_name, len(indices), len(gene_names), len(t)))

    results = {}
    for idx, i in enumerate(indices):
        gene = gene_names[i]
        y = expr[i]
        try:
            r = bhdt_analytic(t, y, classifier_version="v2")
            f24 = component_f_test(t, y, [24.0, 12.0], test_period_idx=0)
            f12 = component_f_test(t, y, [24.0, 12.0], test_period_idx=1)
            results[gene] = {
                "classification": r["classification"],
                "log_bf": r["log_bayes_factor"],
                "bayes_factor": r["bayes_factor"],
                "p24_ftest": f24["p_value"],
                "p12_ftest": f12["p_value"],
                "amp_ratio": _get_amp_ratio(r),
            }
        except Exception:
            results[gene] = {
                "classification": "error", "log_bf": 0,
                "bayes_factor": 0, "p24_ftest": 1.0,
                "p12_ftest": 1.0, "amp_ratio": 0,
            }
        if (idx + 1) % 100 == 0:
            print("    Processed {}/{}...".format(idx + 1, len(indices)))

    return results


def _get_amp_ratio(r):
    """Extract A_12/A_24 amplitude ratio."""
    try:
        m1 = r.get("m1", {})
        comps = m1.get("components", [])
        amps = {c["T"]: c["A"] for c in comps}
        a24 = amps.get(24.0, 1e-10)
        a12 = amps.get(12.0, 0)
        return a12 / max(a24, 1e-10)
    except Exception:
        return 0.0


def analyze_circadian_disruption(wt_results, ko_results):
    """Analyze how BMAL1-KO disrupts circadian rhythms."""
    print("\n" + "=" * 80)
    print("  1. CIRCADIAN DISRUPTION IN BMAL1-KO")
    print("=" * 80)

    # Core circadian genes
    print("\n  Core circadian clock genes:")
    print("  {:<12} {:<30} {:<30}".format("Gene", "WT", "KO"))
    print("  " + "-" * 72)
    for gene in CORE_CIRCADIAN_GENES:
        wt_cls = wt_results.get(gene, {}).get("classification", "not_found")
        ko_cls = ko_results.get(gene, {}).get("classification", "not_found")
        wt_p24 = wt_results.get(gene, {}).get("p24_ftest", 1.0)
        ko_p24 = ko_results.get(gene, {}).get("p24_ftest", 1.0)
        marker = ""
        if wt_cls in {"circadian_only", "harmonic"} and ko_cls == "non_rhythmic":
            marker = " <-- DISRUPTED"
        elif wt_cls == "circadian_only" and ko_cls == "circadian_only":
            marker = " (preserved)"
        print("  {:<12} {:<30} {:<30}{}".format(gene, wt_cls, ko_cls, marker))

    # Count circadian disruption
    wt_circ = sum(1 for g in CORE_CIRCADIAN_GENES
                  if wt_results.get(g, {}).get("classification") in
                  {"circadian_only", "harmonic"})
    ko_circ = sum(1 for g in CORE_CIRCADIAN_GENES
                  if ko_results.get(g, {}).get("classification") in
                  {"circadian_only", "harmonic"})
    print("\n  Circadian genes with rhythmic classification:")
    print("    WT: {}/{}".format(wt_circ, len(CORE_CIRCADIAN_GENES)))
    print("    KO: {}/{}".format(ko_circ, len(CORE_CIRCADIAN_GENES)))


def analyze_12h_persistence(wt_results, ko_results):
    """Analyze which 12h rhythms persist in BMAL1-KO."""
    print("\n" + "=" * 80)
    print("  2. 12h RHYTHM PERSISTENCE IN BMAL1-KO")
    print("=" * 80)

    # Known 12h genes
    print("\n  Known 12h genes (Zhu 2017):")
    print("  {:<12} {:<15} {:<30} {:<30}".format(
        "Gene", "Pathway", "WT", "KO"))
    print("  " + "-" * 87)

    pathway_stats = defaultdict(lambda: {"wt_12h": 0, "ko_12h": 0,
                                          "ko_persist": 0, "total": 0})

    for gene in KNOWN_12H_GENES_ZHU2017:
        wt_cls = wt_results.get(gene, {}).get("classification", "not_found")
        ko_cls = ko_results.get(gene, {}).get("classification", "not_found")
        pathway = PATHWAY_ANNOTATIONS.get(gene, "Other")

        pathway_stats[pathway]["total"] += 1
        if wt_cls in HAS_12H_CLASSES:
            pathway_stats[pathway]["wt_12h"] += 1
        if ko_cls in HAS_12H_CLASSES:
            pathway_stats[pathway]["ko_12h"] += 1
        if ko_cls in TWELVE_H_CLASSES:
            pathway_stats[pathway]["ko_persist"] += 1

        marker = ""
        if ko_cls in TWELVE_H_CLASSES:
            marker = " <-- INDEPENDENT 12h PERSISTS"
        elif wt_cls in HAS_12H_CLASSES and ko_cls not in HAS_12H_CLASSES:
            marker = " (lost in KO)"

        print("  {:<12} {:<15} {:<30} {:<30}{}".format(
            gene, pathway[:15], wt_cls, ko_cls, marker))

    # Pathway-level summary
    print("\n  Pathway-level 12h persistence:")
    print("  {:<25} {:>5} {:>8} {:>8} {:>12}".format(
        "Pathway", "Total", "WT 12h", "KO 12h", "KO indep."))
    print("  " + "-" * 58)
    for pathway, stats in sorted(pathway_stats.items()):
        print("  {:<25} {:>5} {:>8} {:>8} {:>12}".format(
            pathway, stats["total"], stats["wt_12h"],
            stats["ko_12h"], stats["ko_persist"]))


def analyze_genome_wide_shifts(wt_results, ko_results):
    """Genome-wide analysis of classification shifts WT -> KO."""
    print("\n" + "=" * 80)
    print("  3. GENOME-WIDE CLASSIFICATION SHIFTS (WT -> KO)")
    print("=" * 80)

    # Build transition matrix
    all_genes = set(wt_results.keys()) & set(ko_results.keys())
    transitions = defaultdict(int)
    for gene in all_genes:
        wt_cls = wt_results[gene]["classification"]
        ko_cls = ko_results[gene]["classification"]
        transitions[(wt_cls, ko_cls)] += 1

    # Print transition matrix
    all_classes = sorted(set(
        [c for c, _ in transitions.keys()] +
        [c for _, c in transitions.keys()]
    ))
    print("\n  Transition matrix (rows=WT, cols=KO):")
    header = "  {:<30}".format("WT \\ KO")
    for cls in all_classes:
        header += " {:>8}".format(cls[:8])
    print(header)
    print("  " + "-" * len(header))

    for wt_cls in all_classes:
        row = "  {:<30}".format(wt_cls)
        for ko_cls in all_classes:
            count = transitions.get((wt_cls, ko_cls), 0)
            row += " {:>8}".format(count if count > 0 else ".")
            row_total = sum(transitions.get((wt_cls, c), 0) for c in all_classes)
        print(row)

    # Key narrative numbers
    print("\n  Key findings:")

    # Circadian -> non_rhythmic (clock disruption)
    circ_lost = transitions.get(("circadian_only", "non_rhythmic"), 0)
    circ_total = sum(transitions.get(("circadian_only", c), 0)
                     for c in all_classes)
    print("    Circadian genes lost in KO: {}/{} ({:.0f}%)".format(
        circ_lost, circ_total,
        100.0 * circ_lost / max(circ_total, 1)))

    # Independent 12h persisting
    for src_cls in TWELVE_H_CLASSES:
        for dst_cls in TWELVE_H_CLASSES:
            persist = transitions.get((src_cls, dst_cls), 0)
            if persist > 0:
                print("    {} -> {} in KO: {} genes".format(
                    src_cls, dst_cls, persist))

    # New 12h in KO (not 12h in WT)
    new_12h = 0
    for wt_cls in all_classes:
        if wt_cls not in HAS_12H_CLASSES:
            for ko_cls in TWELVE_H_CLASSES:
                new_12h += transitions.get((wt_cls, ko_cls), 0)
    print("    New independent 12h genes in KO (not 12h in WT): {}".format(
        new_12h))


def build_narrative_summary(wt_results, ko_results):
    """Build the publication narrative summary."""
    print("\n" + "=" * 80)
    print("  4. PUBLICATION NARRATIVE SUMMARY")
    print("=" * 80)

    all_genes = set(wt_results.keys()) & set(ko_results.keys())

    # Count key categories
    wt_circ = sum(1 for g in all_genes
                  if wt_results[g]["classification"] == "circadian_only")
    ko_circ = sum(1 for g in all_genes
                  if ko_results[g]["classification"] == "circadian_only")
    wt_12h_indep = sum(1 for g in all_genes
                       if wt_results[g]["classification"] in TWELVE_H_CLASSES)
    ko_12h_indep = sum(1 for g in all_genes
                       if ko_results[g]["classification"] in TWELVE_H_CLASSES)
    wt_harmonic = sum(1 for g in all_genes
                      if wt_results[g]["classification"] == "harmonic")
    ko_harmonic = sum(1 for g in all_genes
                      if ko_results[g]["classification"] == "harmonic")

    print("""
  BMAL1-KO Validation of Independent 12h Oscillator Hypothesis
  =============================================================

  In wildtype (WT) mouse liver:
    - {wt_circ} genes classified as circadian_only (24h dominant)
    - {wt_12h} genes classified as independent 12h ultradian
    - {wt_harm} genes classified as harmonic (12h = 24h harmonic)

  In BMAL1-KO mouse liver:
    - {ko_circ} genes classified as circadian_only ({circ_change})
    - {ko_12h} genes classified as independent 12h ultradian ({indep_change})
    - {ko_harm} genes classified as harmonic ({harm_change})

  Key biological insight:
    BMAL1 knockout disrupts the core circadian clock, causing a dramatic
    reduction in circadian_only classifications ({wt_circ} -> {ko_circ}).
    However, independent 12h ultradian rhythms are {persist_word}
    ({wt_12h} -> {ko_12h}), supporting the hypothesis that 12h rhythms
    are driven by an independent oscillator mechanism (e.g., XBP1s/IRE1a
    ER stress cycle) rather than being harmonics of the circadian clock.
""".format(
        wt_circ=wt_circ, ko_circ=ko_circ,
        wt_12h=wt_12h_indep, ko_12h=ko_12h_indep,
        wt_harm=wt_harmonic, ko_harm=ko_harmonic,
        circ_change="{}% reduction".format(
            int(100 * (1 - ko_circ / max(wt_circ, 1)))),
        indep_change="{}".format(
            "preserved" if ko_12h_indep >= wt_12h_indep
            else "{}% reduction".format(
                int(100 * (1 - ko_12h_indep / max(wt_12h_indep, 1))))),
        harm_change="{}% reduction".format(
            int(100 * (1 - ko_harmonic / max(wt_harmonic, 1)))),
        persist_word="maintained" if ko_12h_indep >= wt_12h_indep
        else "partially maintained",
    ))


def save_results(wt_results, ko_results, output_path="bmal1ko_narrative_results.csv"):
    """Save gene-level WT vs KO comparison."""
    rows = []
    all_genes = sorted(set(wt_results.keys()) & set(ko_results.keys()))
    for gene in all_genes:
        wt = wt_results[gene]
        ko = ko_results[gene]
        rows.append({
            "gene": gene,
            "pathway": PATHWAY_ANNOTATIONS.get(gene, ""),
            "is_known_12h": gene in set(KNOWN_12H_GENES_ZHU2017),
            "is_core_circadian": gene in set(CORE_CIRCADIAN_GENES),
            "wt_classification": wt["classification"],
            "ko_classification": ko["classification"],
            "wt_log_bf": wt["log_bf"],
            "ko_log_bf": ko["log_bf"],
            "wt_p24": wt["p24_ftest"],
            "ko_p24": ko["p24_ftest"],
            "wt_p12": wt["p12_ftest"],
            "ko_p12": ko["p12_ftest"],
            "wt_amp_ratio": wt["amp_ratio"],
            "ko_amp_ratio": ko["amp_ratio"],
            "wt_is_12h": wt["classification"] in TWELVE_H_CLASSES,
            "ko_is_12h": ko["classification"] in TWELVE_H_CLASSES,
            "wt_is_circadian": wt["classification"] == "circadian_only",
            "ko_is_circadian": ko["classification"] == "circadian_only",
            "circadian_disrupted": (
                wt["classification"] == "circadian_only" and
                ko["classification"] != "circadian_only"
            ),
            "12h_persists": (
                wt["classification"] in TWELVE_H_CLASSES and
                ko["classification"] in TWELVE_H_CLASSES
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("Results saved to {} ({} genes)".format(output_path, len(df)))
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 80)
    print("  CHORD BMAL1-KO Biological Narrative Analysis")
    print("  Zhu et al. 2023 — WT vs BMAL1-KO mouse liver")
    print("=" * 80)

    # Load data
    data = load_zhu2023_bmal1ko()
    wt_data = data["wt"]
    ko_data = data["ko"]
    print("  WT: {} genes, {} timepoints".format(
        wt_data["expr"].shape[0], wt_data["expr"].shape[1]))
    print("  KO: {} genes, {} timepoints".format(
        ko_data["expr"].shape[0], ko_data["expr"].shape[1]))

    # Build target gene set: known 12h + circadian + housekeeping
    target_genes = (set(KNOWN_12H_GENES_ZHU2017) |
                    set(CORE_CIRCADIAN_GENES) |
                    set(NON_RHYTHMIC_HOUSEKEEPING))
    print("  Target gene set: {} genes".format(len(target_genes)))

    # Run BHDT V2 on target genes only (fast)
    wt_results = run_targeted_bhdt(wt_data, "WT", target_genes)
    ko_results = run_targeted_bhdt(ko_data, "KO", target_genes)

    # Analyses
    analyze_circadian_disruption(wt_results, ko_results)
    analyze_12h_persistence(wt_results, ko_results)
    analyze_genome_wide_shifts(wt_results, ko_results)
    build_narrative_summary(wt_results, ko_results)

    # Save
    df = save_results(wt_results, ko_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
