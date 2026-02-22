"""Validate CHORD on Hughes 2009 mouse liver data (GSE11923, real GEO data).

Computes:
1. Detection rate of 12h signal in known 12h genes (Zhu 2017 list)
2. Core circadian gene classification accuracy
3. Housekeeping gene false positive rate
4. Genome-wide classification summary
"""
import numpy as np
import pandas as pd

from chord.data.geo_loader import load_hughes2009
from chord.bhdt.pipeline import run_bhdt
from chord.data.known_genes import (
    KNOWN_12H_GENES_ZHU2017,
    CORE_CIRCADIAN_GENES,
    NON_RHYTHMIC_HOUSEKEEPING,
)


def validate_hughes2009(cache_dir="~/.chord_cache", method="analytic",
                         n_jobs=1):
    """Run CHORD on Hughes 2009 and evaluate against known gene lists.

    Parameters
    ----------
    cache_dir : str
    method : str
        BHDT method: 'analytic', 'bootstrap', 'ensemble'
    n_jobs : int

    Returns
    -------
    dict with validation metrics
    """
    data = load_hughes2009(cache_dir=cache_dir, downsample_2h=True)

    # Build DataFrame with gene names as index
    expr_df = pd.DataFrame(
        data["expr"],
        index=data["gene_names"],
    )

    bhdt_results = run_bhdt(
        expr_df, data["timepoints"],
        method=method, n_jobs=n_jobs, verbose=True,
    )

    gene_cls = dict(zip(bhdt_results["gene"], bhdt_results["classification"]))

    has_12h_classes = {
        "independent_ultradian", "likely_independent_ultradian", "harmonic",
    }

    # 1. Known 12h gene detection rate
    known_found = [g for g in KNOWN_12H_GENES_ZHU2017 if g in gene_cls]
    known_detected = [g for g in known_found
                      if gene_cls[g] in has_12h_classes]

    # 2. Core circadian gene classification
    circ_found = [g for g in CORE_CIRCADIAN_GENES if g in gene_cls]
    circ_correct = [g for g in circ_found
                    if gene_cls[g] == "circadian_only"]

    # 3. Housekeeping false positive rate
    hk_found = [g for g in NON_RHYTHMIC_HOUSEKEEPING if g in gene_cls]
    hk_false_pos = [g for g in hk_found
                    if gene_cls[g] != "non_rhythmic"]

    return {
        "source": data["metadata"]["source"],
        "method": method,
        "n_genes": len(bhdt_results),
        # Known 12h genes
        "known_12h_detection_rate": (
            len(known_detected) / max(len(known_found), 1)
        ),
        "known_12h_found": len(known_found),
        "known_12h_detected": len(known_detected),
        "known_12h_details": {
            g: gene_cls.get(g, "not_found")
            for g in KNOWN_12H_GENES_ZHU2017
        },
        # Core circadian genes
        "core_circadian_correct_rate": (
            len(circ_correct) / max(len(circ_found), 1)
        ),
        "core_circadian_found": len(circ_found),
        "core_circadian_correct": len(circ_correct),
        "core_circadian_details": {
            g: gene_cls.get(g, "not_found")
            for g in CORE_CIRCADIAN_GENES
        },
        # Housekeeping false positives
        "housekeeping_false_positive_rate": (
            len(hk_false_pos) / max(len(hk_found), 1)
        ),
        "housekeeping_found": len(hk_found),
        "housekeeping_false_positives": len(hk_false_pos),
        # Overall
        "classification_counts": (
            bhdt_results["classification"].value_counts().to_dict()
        ),
        "bhdt_results": bhdt_results,
    }


if __name__ == "__main__":
    results = validate_hughes2009(n_jobs=4)
    print("\n=== Hughes 2009 Validation Results ===")
    print(f"Source: {results['source']}")
    print(f"Genes analyzed: {results['n_genes']}")
    print(f"Known 12h detection rate: "
          f"{results['known_12h_detection_rate']:.1%} "
          f"({results['known_12h_detected']}/{results['known_12h_found']})")
    print(f"Core circadian correct rate: "
          f"{results['core_circadian_correct_rate']:.1%} "
          f"({results['core_circadian_correct']}/{results['core_circadian_found']})")
    print(f"Housekeeping FPR: "
          f"{results['housekeeping_false_positive_rate']:.1%} "
          f"({results['housekeeping_false_positives']}/{results['housekeeping_found']})")
    print(f"\nClassification counts: {results['classification_counts']}")
