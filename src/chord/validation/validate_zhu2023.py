"""Validate CHORD on Zhu 2023 BMAL1 KO data (GSE171975, real GEO data).

The CRITICAL test: In BMAL1 KO mice, any 12h signal cannot be a harmonic
of 24h (since the circadian clock is abolished). CHORD should classify
KO 12h genes as 'independent_ultradian', not 'harmonic'.
"""
import numpy as np
import pandas as pd

from chord.data.geo_loader import load_zhu2023_bmal1ko
from chord.bhdt.pipeline import run_bhdt
from chord.data.known_genes import (
    KNOWN_12H_GENES_ZHU2017,
    CORE_CIRCADIAN_GENES,
    NON_RHYTHMIC_HOUSEKEEPING,
)


def validate_zhu2023(cache_dir="~/.chord_cache", method="analytic",
                      n_jobs=1):
    """Run CHORD on both WT and KO conditions and compare.

    Returns
    -------
    dict with validation metrics for both conditions
    """
    data = load_zhu2023_bmal1ko(cache_dir=cache_dir)

    # Build DataFrames with gene names as index
    wt_expr_df = pd.DataFrame(
        data["wt"]["expr"], index=data["wt"]["gene_names"],
    )
    ko_expr_df = pd.DataFrame(
        data["ko"]["expr"], index=data["ko"]["gene_names"],
    )

    wt_results = run_bhdt(
        wt_expr_df, data["wt"]["timepoints"],
        method=method, n_jobs=n_jobs, verbose=True,
    )
    ko_results = run_bhdt(
        ko_expr_df, data["ko"]["timepoints"],
        method=method, n_jobs=n_jobs, verbose=True,
    )

    has_12h_classes = {
        "independent_ultradian", "likely_independent_ultradian", "harmonic",
    }
    independent_classes = {
        "independent_ultradian", "likely_independent_ultradian",
    }

    wt_cls = dict(zip(wt_results["gene"], wt_results["classification"]))
    ko_cls = dict(zip(ko_results["gene"], ko_results["classification"]))

    # --- KO analysis ---
    # Known 12h genes in KO: should be independent (not harmonic)
    ko_12h_found = [g for g in KNOWN_12H_GENES_ZHU2017 if g in ko_cls]
    ko_12h_detected = [g for g in ko_12h_found
                       if ko_cls[g] in has_12h_classes]
    ko_12h_independent = [g for g in ko_12h_found
                          if ko_cls[g] in independent_classes]
    ko_12h_harmonic = [g for g in ko_12h_found
                       if ko_cls[g] == "harmonic"]

    # Core circadian genes in KO: should be non-rhythmic
    ko_circ_found = [g for g in CORE_CIRCADIAN_GENES if g in ko_cls]
    ko_circ_abolished = [g for g in ko_circ_found
                         if ko_cls[g] == "non_rhythmic"]

    # --- WT analysis ---
    wt_12h_found = [g for g in KNOWN_12H_GENES_ZHU2017 if g in wt_cls]
    wt_12h_detected = [g for g in wt_12h_found
                       if wt_cls[g] in has_12h_classes]
    wt_12h_independent = [g for g in wt_12h_found
                          if wt_cls[g] in independent_classes]
    wt_12h_harmonic = [g for g in wt_12h_found
                       if wt_cls[g] == "harmonic"]

    wt_circ_found = [g for g in CORE_CIRCADIAN_GENES if g in wt_cls]
    wt_circ_correct = [g for g in wt_circ_found
                       if wt_cls[g] == "circadian_only"]

    return {
        "source": data["wt"]["metadata"]["source"],
        "method": method,
        # KO metrics
        "ko_12h_detection_rate": (
            len(ko_12h_detected) / max(len(ko_12h_found), 1)
        ),
        "ko_12h_independent_rate": (
            len(ko_12h_independent) / max(len(ko_12h_found), 1)
        ),
        "ko_12h_harmonic_rate": (
            len(ko_12h_harmonic) / max(len(ko_12h_found), 1)
        ),
        "ko_circadian_abolished_rate": (
            len(ko_circ_abolished) / max(len(ko_circ_found), 1)
        ),
        "ko_12h_found": len(ko_12h_found),
        "ko_12h_detected": len(ko_12h_detected),
        "ko_12h_independent": len(ko_12h_independent),
        "ko_12h_harmonic": len(ko_12h_harmonic),
        # WT metrics
        "wt_12h_detection_rate": (
            len(wt_12h_detected) / max(len(wt_12h_found), 1)
        ),
        "wt_12h_independent_rate": (
            len(wt_12h_independent) / max(len(wt_12h_found), 1)
        ),
        "wt_12h_harmonic_rate": (
            len(wt_12h_harmonic) / max(len(wt_12h_found), 1)
        ),
        "wt_circadian_correct_rate": (
            len(wt_circ_correct) / max(len(wt_circ_found), 1)
        ),
        # Classification summaries
        "ko_classification_counts": (
            ko_results["classification"].value_counts().to_dict()
        ),
        "wt_classification_counts": (
            wt_results["classification"].value_counts().to_dict()
        ),
        # Detailed per-gene results
        "ko_known_12h_details": {
            g: ko_cls.get(g, "not_found")
            for g in KNOWN_12H_GENES_ZHU2017
        },
        "wt_known_12h_details": {
            g: wt_cls.get(g, "not_found")
            for g in KNOWN_12H_GENES_ZHU2017
        },
        # Raw results
        "ko_results": ko_results,
        "wt_results": wt_results,
    }


if __name__ == "__main__":
    results = validate_zhu2023(n_jobs=4)
    print("\n=== Zhu 2023 BMAL1 KO Validation Results ===")
    print(f"Source: {results['source']}")
    print(f"\n--- KO condition ---")
    print(f"12h detection rate: {results['ko_12h_detection_rate']:.1%} "
          f"({results['ko_12h_detected']}/{results['ko_12h_found']})")
    print(f"12h independent rate: {results['ko_12h_independent_rate']:.1%}")
    print(f"12h harmonic rate: {results['ko_12h_harmonic_rate']:.1%}")
    print(f"Circadian abolished: {results['ko_circadian_abolished_rate']:.1%}")
    print(f"\n--- WT condition ---")
    print(f"12h detection rate: {results['wt_12h_detection_rate']:.1%}")
    print(f"12h independent rate: {results['wt_12h_independent_rate']:.1%}")
    print(f"12h harmonic rate: {results['wt_12h_harmonic_rate']:.1%}")
    print(f"Circadian correct: {results['wt_circadian_correct_rate']:.1%}")
