"""Zhu 2024 human peripheral blood data loader (GSE220120).

Zhu et al. (2024) profiled buffy coat RNA-seq from 3 healthy subjects
sampled every 2h for 48h, identifying 653 ~12h genes and 5453 ~24h genes.

Loading order:
  Tier 1 -- NPZ cache (fast, local)
  Tier 2 -- GEOparse download (requires network + GEOparse)
  Fail   -- RuntimeError (no synthetic fallback)
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np


def load_zhu2024(cache_dir="~/.chord_cache", subject="all"):
    """Load Zhu 2024 human peripheral blood data (GSE220120).

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    subject : str
        'S1', 'S2', 'S3', or 'all' (averaged across subjects).

    Returns
    -------
    dict with keys:
        expr : ndarray (genes x timepoints)
        timepoints : ndarray of hours
        gene_names : list of str
        metadata : dict
    """
    valid_subjects = ("S1", "S2", "S3", "all")
    if subject not in valid_subjects:
        raise ValueError(
            "subject must be one of {}, got '{}'".format(valid_subjects, subject)
        )

    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / "zhu2024_{}.npz".format(subject)

    expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: try loading from NPZ cache ---
    if npz_file.exists():
        data = np.load(str(npz_file), allow_pickle=True)
        expr = data["expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: try GEOparse ---
    if expr is None:
        try:
            import GEOparse
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "Zhu 2024 (GSE220120) data not cached and GEOparse is not "
                "installed.  Install GEOparse (`pip install GEOparse`) and "
                "re-run, or manually place the NPZ cache at: " + str(npz_file)
            )

        try:
            gse = GEOparse.get_GEO("GSE220120", destdir=str(cache_path))

            # --- Parse sample metadata ---
            sample_meta = gse.phenotype_data

            # Build mapping: sample_id -> (subject, hour)
            sample_info = {}
            for sid in sample_meta.index:
                title = sample_meta.loc[sid, "title"]
                # Expected title patterns like "S1_0h", "S2_24h", etc.
                m = re.search(r'(S[123])[\s_]+(\d+)\s*h', title, re.IGNORECASE)
                if m:
                    subj = m.group(1).upper()
                    hour = float(m.group(2))
                    sample_info[sid] = {"subject": subj, "hour": hour}
                else:
                    # Try alternative patterns
                    m = re.search(r'[Ss]ubject\s*(\d+).*?(\d+)\s*h', title)
                    if m:
                        subj = "S" + m.group(1)
                        hour = float(m.group(2))
                        sample_info[sid] = {"subject": subj, "hour": hour}

            if not sample_info:
                raise ValueError(
                    "Could not parse subject/hour from sample titles in GSE220120"
                )

            # Get expression matrix
            pivot = gse.pivot_samples("VALUE")

            # Filter samples by subject
            if subject == "all":
                subjects_to_use = ["S1", "S2", "S3"]
            else:
                subjects_to_use = [subject]

            # Group samples by hour, averaging across requested subjects
            hour_samples = {}
            for sid, info in sample_info.items():
                if info["subject"] in subjects_to_use and sid in pivot.columns:
                    h = info["hour"]
                    if h not in hour_samples:
                        hour_samples[h] = []
                    hour_samples[h].append(sid)

            if not hour_samples:
                raise ValueError(
                    "No samples found for subject(s) {} in GSE220120".format(
                        subjects_to_use)
                )

            sorted_hours = sorted(hour_samples.keys())
            timepoints_arr = np.array(sorted_hours, dtype=np.float64)

            # Average expression across replicates at each timepoint
            avg_expr = np.column_stack([
                pivot[hour_samples[h]].mean(axis=1).values
                for h in sorted_hours
            ]).astype(np.float64)

            # Map probe IDs to gene symbols
            gpl = list(gse.gpls.values())[0]
            gene_col = None
            for col_name in ["Gene Symbol", "GENE_SYMBOL", "gene_symbol",
                             "Symbol", "SYMBOL", "Gene symbol"]:
                if col_name in gpl.table.columns:
                    gene_col = col_name
                    break

            if gene_col is not None:
                probe2gene = dict(
                    zip(gpl.table["ID"], gpl.table[gene_col])
                )
                probe_ids = pivot.index.tolist()
                gene_symbols = [
                    str(probe2gene.get(pid, "")).split("///")[0].strip()
                    for pid in probe_ids
                ]
                has_symbol = [
                    bool(g) and g != "nan" and g != ""
                    for g in gene_symbols
                ]
                avg_expr = avg_expr[has_symbol]
                gene_symbols = [
                    g for g, h in zip(gene_symbols, has_symbol) if h
                ]

                # Deduplicate: keep probe with highest mean expression
                df_dedup = pd.DataFrame({
                    "gene": gene_symbols,
                    "mean_expr": avg_expr.mean(axis=1),
                    "idx": range(len(gene_symbols)),
                })
                best_idx = (
                    df_dedup.groupby("gene")["mean_expr"]
                    .idxmax().values.astype(int)
                )
                avg_expr = avg_expr[best_idx]
                gene_names_arr = [gene_symbols[i] for i in best_idx]
            else:
                gene_names_arr = pivot.index.tolist()

            # Filter bottom 10% by mean expression
            means = avg_expr.mean(axis=1)
            threshold = np.percentile(means, 10)
            keep = means >= threshold
            avg_expr = avg_expr[keep]
            gene_names_arr = [
                g for g, k in zip(gene_names_arr, keep) if k
            ]

            expr = avg_expr
            timepoints = timepoints_arr
            gene_names = gene_names_arr
            source = "GEOparse"

            # Cache to NPZ
            np.savez(
                str(npz_file),
                expr=expr,
                timepoints=timepoints,
                gene_names=np.array(gene_names, dtype=object),
                source=np.array("GEOparse"),
            )
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                "Failed to download/parse Zhu 2024 (GSE220120): {}".format(e)
            )

    metadata = {
        "dataset": "Zhu 2024",
        "geo_accession": "GSE220120",
        "organism": "Homo sapiens",
        "tissue": "peripheral blood (buffy coat)",
        "platform": "RNA-seq",
        "source": source,
        "n_genes": expr.shape[0],
        "n_timepoints": expr.shape[1],
        "n_subjects": 3 if subject == "all" else 1,
        "subject": subject,
        "sampling_interval_h": 2,
        "total_duration_h": 48,
        "reference": "Zhu et al. (2024)",
    }

    return {
        "expr": expr,
        "timepoints": timepoints,
        "gene_names": gene_names,
        "metadata": metadata,
    }
