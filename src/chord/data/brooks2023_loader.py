"""Brooks 2023 meta-analysis data loader.

Loads pre-processed TPM expression matrices from Brooks et al. 2023
(Journal of Biological Rhythms), which aggregated ~57 mouse liver
circadian RNA-seq studies from GEO.

Data source: Zenodo record 7760579 (Supplemental File S1).
Local path: public_data/brooks2023/supplemental/
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# Representative studies selected for CHORD benchmark
# Criteria: >= 8 timepoints/cycle, >= 2 replicates, no special diet/treatment
BROOKS2023_BENCHMARK_STUDIES = {
    "Sinturel17A": {"tp_per_cycle": 12, "reps": 4, "n_samples": 48, "light": "LD",
                    "description": "Sinturel 2017 Ad Lib (GSE73552)"},
    "Morton20":    {"tp_per_cycle": 8,  "reps": "5-6", "n_samples": 77, "light": "LD",
                    "description": "Morton 2020 Liver (GSE151565)"},
    "Pan20":       {"tp_per_cycle": 12, "reps": 2, "n_samples": 48, "light": "DD",
                    "description": "Pan 2020 XBP1 flox WT (GSE130890)"},
    "Guan20":      {"tp_per_cycle": 8,  "reps": 3, "n_samples": 24, "light": "LD",
                    "description": "Guan 2020 Rev-erb WT (GSE143524)"},
    "Stubblefield18": {"tp_per_cycle": 8, "reps": 4, "n_samples": 32, "light": "LD",
                       "description": "Stubblefield 2018 (GSE105413)"},
}


def _find_brooks2023_dir():
    """Locate the Brooks 2023 supplemental data directory."""
    candidates = [
        Path(__file__).resolve().parents[4] / "public_data" / "brooks2023" / "supplemental",
        Path(os.path.expanduser("~")) / "public_data" / "brooks2023" / "supplemental",
        Path("/home/data2/fangcong2/ovary_aging/public_data/brooks2023/supplemental"),
    ]
    for p in candidates:
        if p.exists() and (p / "tpm.by_sample.txt.gz").exists():
            return p
    raise FileNotFoundError(
        "Brooks 2023 supplemental data not found. Expected at one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nDownload from Zenodo record 7760579 and extract S1.zip."
    )


def load_brooks2023(study_id=None, data_dir=None, min_tpm=1.0):
    """Load mouse liver time-series from Brooks 2023 meta-analysis.

    Parameters
    ----------
    study_id : str or None
        Specific study ID (e.g. 'Sinturel17A'). If None, loads all
        benchmark studies and returns a dict of dicts.
    data_dir : str or Path or None
        Path to the supplemental/ directory. Auto-detected if None.
    min_tpm : float
        Minimum mean TPM to keep a gene. Default 1.0.

    Returns
    -------
    dict (if study_id given) with keys: expr, timepoints, gene_names, metadata
    dict-of-dicts (if study_id is None) keyed by study_id
    """
    if data_dir is None:
        data_dir = _find_brooks2023_dir()
    else:
        data_dir = Path(data_dir)

    # Load sample metadata
    sample_meta = pd.read_csv(data_dir / "sample_metadata.txt", sep="\t")

    # Determine which studies to load
    if study_id is not None:
        study_ids = [study_id]
    else:
        study_ids = list(BROOKS2023_BENCHMARK_STUDIES.keys())

    # Load TPM matrix (lazy: only columns we need)
    needed_samples = set()
    for sid in study_ids:
        study_samples = sample_meta[
            (sample_meta["study"] == sid) & (sample_meta["outlier"] == False)
        ]["sample"].tolist()
        needed_samples.update(study_samples)

    usecols = ["Name", "Symbol"] + list(needed_samples)
    tpm_df = pd.read_csv(
        data_dir / "tpm.by_sample.txt.gz",
        sep="\t", compression="gzip", usecols=usecols,
    )

    results = {}
    for sid in study_ids:
        study_samples_df = sample_meta[
            (sample_meta["study"] == sid) & (sample_meta["outlier"] == False)
        ].copy()

        if len(study_samples_df) == 0:
            warnings.warn(f"No non-outlier samples for study {sid}")
            continue

        sample_ids = study_samples_df["sample"].tolist()
        times = study_samples_df["time"].values.astype(float)

        # Get unique timepoints and average replicates
        unique_times = np.sort(np.unique(times))
        gene_symbols = tpm_df["Symbol"].values
        raw_expr = tpm_df[sample_ids].values.astype(np.float64)

        # Average replicates per timepoint
        avg_expr = np.column_stack([
            raw_expr[:, times == t].mean(axis=1) for t in unique_times
        ])

        # Filter: keep genes with symbol and sufficient expression
        has_symbol = pd.notna(gene_symbols) & (gene_symbols != "")
        mean_tpm = avg_expr.mean(axis=1)
        keep = has_symbol & (mean_tpm >= min_tpm)

        filtered_expr = avg_expr[keep]
        filtered_genes = gene_symbols[keep].tolist()

        # Deduplicate: keep gene with highest mean expression
        gene_mean = pd.DataFrame({
            "gene": filtered_genes,
            "mean": filtered_expr.mean(axis=1),
            "idx": range(len(filtered_genes)),
        })
        best_idx = gene_mean.groupby("gene")["mean"].idxmax().values.astype(int)
        final_expr = filtered_expr[best_idx]
        final_genes = [filtered_genes[i] for i in best_idx]

        info = BROOKS2023_BENCHMARK_STUDIES.get(sid, {})
        metadata = {
            "dataset": f"Brooks2023_{sid}",
            "study_id": sid,
            "organism": "Mus musculus",
            "tissue": "liver",
            "platform": "RNA-seq (various)",
            "source": "Brooks2023_Zenodo",
            "n_genes": final_expr.shape[0],
            "n_timepoints": final_expr.shape[1],
            "n_samples_raw": len(sample_ids),
            "light_condition": info.get("light", "unknown"),
            "description": info.get("description", sid),
            "reference": "Brooks et al., J Biol Rhythms 38:538-550 (2023)",
        }

        results[sid] = {
            "expr": final_expr,
            "timepoints": unique_times,
            "gene_names": final_genes,
            "metadata": metadata,
        }

    if study_id is not None:
        return results.get(study_id, {})
    return results
