"""Validate CHORD on Mure 2018 baboon liver data — cross-species validation.

Mure et al. (Science 2018) profiled 64 baboon (Papio anubis) tissues every
2h over 24h (GSE98965).  We use the liver tissue for cross-species comparison
with mouse liver (Hughes 2009 / Zhu 2023).

Key validation questions:
1. Can CHORD detect 12h rhythms in baboon liver?
2. Are the detected 12h genes enriched in ER stress / UPR pathways
   (consistent with mouse findings)?
3. Are core circadian genes correctly classified as circadian_only?
4. Are housekeeping genes correctly classified as non_rhythmic?
"""
import numpy as np
import pandas as pd

from chord.data.geo_loader import load_mure2018
from chord.bhdt.pipeline import run_bhdt
from chord.data.known_genes import (
    CORE_CIRCADIAN_GENES_PRIMATE,
    KNOWN_12H_GENES_PRIMATE,
    NON_RHYTHMIC_HOUSEKEEPING_PRIMATE,
)


def validate_mure2018(cache_dir="~/.chord_cache", tissue="LIV",
                       method="analytic", n_jobs=1, min_fpkm=1.0):
    """Run CHORD on Mure 2018 baboon tissue and evaluate.

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    tissue : str
        Tissue abbreviation (default 'LIV' for liver).
    method : str
        BHDT method: 'analytic', 'bootstrap', 'ensemble'.
    n_jobs : int
        Number of parallel jobs.
    min_fpkm : float
        Minimum mean FPKM to keep a gene.

    Returns
    -------
    dict with validation metrics
    """
    # Load real data (no synthetic fallback)
    data = load_mure2018(cache_dir=cache_dir, tissue=tissue,
                          min_fpkm=min_fpkm)

    # Build a DataFrame with gene names as index so run_bhdt uses them
    import pandas as pd
    expr_df = pd.DataFrame(
        data["expr"],
        index=data["gene_names"],
        columns=[f"ZT{h:02d}" for h in range(0, 24, 2)],
    )

    # Run BHDT
    bhdt_results = run_bhdt(
        expr_df, data["timepoints"],
        method=method, n_jobs=n_jobs, verbose=True,
    )

    # Build gene -> classification map
    gene_cls = dict(zip(bhdt_results["gene"], bhdt_results["classification"]))

    has_12h_classes = {
        "independent_ultradian", "likely_independent_ultradian", "harmonic",
    }

    # --- 1. Known 12h gene detection ---
    known_12h_found = [g for g in KNOWN_12H_GENES_PRIMATE if g in gene_cls]
    known_12h_detected = [g for g in known_12h_found
                          if gene_cls[g] in has_12h_classes]
    known_12h_independent = [g for g in known_12h_found
                             if gene_cls[g] in {
                                 "independent_ultradian",
                                 "likely_independent_ultradian",
                             }]
    known_12h_harmonic = [g for g in known_12h_found
                          if gene_cls[g] == "harmonic"]

    # --- 2. Core circadian gene classification ---
    circ_found = [g for g in CORE_CIRCADIAN_GENES_PRIMATE if g in gene_cls]
    circ_correct = [g for g in circ_found
                    if gene_cls[g] == "circadian_only"]

    # --- 3. Housekeeping false positive rate ---
    hk_found = [g for g in NON_RHYTHMIC_HOUSEKEEPING_PRIMATE if g in gene_cls]
    hk_false_pos = [g for g in hk_found
                    if gene_cls[g] != "non_rhythmic"]

    # --- 4. Genome-wide 12h detection summary ---
    all_12h_genes = [g for g, c in gene_cls.items() if c in has_12h_classes]

    # --- 5. Cross-species overlap with mouse 12h genes ---
    # Convert mouse known 12h genes to primate symbols for comparison
    from chord.data.known_genes import (
        KNOWN_12H_GENES_ZHU2017,
        mouse_to_primate_symbol,
    )
    mouse_12h_as_primate = {mouse_to_primate_symbol(g)
                            for g in KNOWN_12H_GENES_ZHU2017}
    # Genes detected as 12h in baboon that are also known 12h in mouse
    cross_species_overlap = [g for g in all_12h_genes
                             if g in mouse_12h_as_primate]

    return {
        "source": data["metadata"]["source"],
        "organism": data["metadata"]["organism"],
        "tissue": tissue,
        "method": method,
        "n_genes_analyzed": len(bhdt_results),
        # Known 12h genes
        "known_12h_detection_rate": (
            len(known_12h_detected) / max(len(known_12h_found), 1)
        ),
        "known_12h_found": len(known_12h_found),
        "known_12h_detected": len(known_12h_detected),
        "known_12h_independent": len(known_12h_independent),
        "known_12h_harmonic": len(known_12h_harmonic),
        "known_12h_details": {
            g: gene_cls.get(g, "not_found")
            for g in KNOWN_12H_GENES_PRIMATE
        },
        # Core circadian genes
        "core_circadian_correct_rate": (
            len(circ_correct) / max(len(circ_found), 1)
        ),
        "core_circadian_found": len(circ_found),
        "core_circadian_correct": len(circ_correct),
        "core_circadian_details": {
            g: gene_cls.get(g, "not_found")
            for g in CORE_CIRCADIAN_GENES_PRIMATE
        },
        # Housekeeping false positives
        "housekeeping_false_positive_rate": (
            len(hk_false_pos) / max(len(hk_found), 1)
        ),
        "housekeeping_found": len(hk_found),
        "housekeeping_false_positives": len(hk_false_pos),
        # Genome-wide
        "total_12h_genes_detected": len(all_12h_genes),
        "pct_12h_genes": len(all_12h_genes) / max(len(bhdt_results), 1),
        # Cross-species
        "cross_species_12h_overlap": len(cross_species_overlap),
        "cross_species_12h_overlap_genes": cross_species_overlap,
        # Classification summary
        "classification_counts": (
            bhdt_results["classification"].value_counts().to_dict()
        ),
        # Raw results
        "bhdt_results": bhdt_results,
    }


if __name__ == "__main__":
    results = validate_mure2018(n_jobs=4)
    print("\n=== Mure 2018 Baboon Liver Validation Results ===")
    print(f"Source: {results['source']}")
    print(f"Organism: {results['organism']}, Tissue: {results['tissue']}")
    print(f"Genes analyzed: {results['n_genes_analyzed']}")
    print(f"\n--- Known 12h genes ---")
    print(f"Detection rate: {results['known_12h_detection_rate']:.1%} "
          f"({results['known_12h_detected']}/{results['known_12h_found']})")
    print(f"  Independent: {results['known_12h_independent']}")
    print(f"  Harmonic: {results['known_12h_harmonic']}")
    print(f"\n--- Core circadian genes ---")
    print(f"Correct rate: {results['core_circadian_correct_rate']:.1%} "
          f"({results['core_circadian_correct']}/{results['core_circadian_found']})")
    print(f"\n--- Housekeeping genes ---")
    print(f"False positive rate: {results['housekeeping_false_positive_rate']:.1%} "
          f"({results['housekeeping_false_positives']}/{results['housekeeping_found']})")
    print(f"\n--- Genome-wide ---")
    print(f"Total 12h genes: {results['total_12h_genes_detected']} "
          f"({results['pct_12h_genes']:.1%})")
    print(f"\n--- Cross-species overlap ---")
    print(f"Mouse-baboon 12h overlap: {results['cross_species_12h_overlap']} genes")
    if results["cross_species_12h_overlap_genes"]:
        print(f"  Genes: {', '.join(results['cross_species_12h_overlap_genes'][:10])}")
    print(f"\nClassification counts: {results['classification_counts']}")
