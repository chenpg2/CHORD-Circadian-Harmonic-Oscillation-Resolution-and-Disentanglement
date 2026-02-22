#!/usr/bin/env python3
"""
run_bhdt_ovary.py — Genome-wide BHDT analysis on the ovary aging dataset.

Two-stage pipeline:
  Stage 1: Analytic mode (~7ms/gene) on all genes for quick classification.
  Stage 2: Ensemble mode (analytic+bootstrap, ~3s/gene) on ambiguous/independent
           candidates for refined classification.

Compares BHDT results with existing RAIN/Cosinor/Lomb-Scargle/Pencil results
and performs differential rhythm analysis between Young and Old groups.

Usage:
    python run_bhdt_ovary.py
    python run_bhdt_ovary.py --skip-ensemble
    python run_bhdt_ovary.py --n-jobs 8 --ensemble-threshold 0.1
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the chord package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../analysis/
_SRC_DIR = _SCRIPT_DIR.parent.parent                   # .../src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from chord.bhdt.pipeline import run_bhdt  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===================================================================
# Data loading helpers
# ===================================================================

def load_mean_matrix(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a mean expression matrix CSV.

    Returns
    -------
    expr : pd.DataFrame
        Genes (rows) x timepoints (columns), gene_id as index.
    timepoints : np.ndarray
        Numeric timepoints parsed from column names.
    """
    df = pd.read_csv(path, index_col=0)
    timepoints = np.array([float(c) for c in df.columns])
    log.info("Loaded %s: %d genes x %d timepoints (ZT %s..%s)",
             Path(path).name, df.shape[0], df.shape[1],
             timepoints[0], timepoints[-1])
    return df, timepoints


def load_gene_name_map(rhythm_cls_path: str) -> dict:
    """Extract gene_id -> gene_name mapping from a rhythm_classification file."""
    df = pd.read_csv(rhythm_cls_path, usecols=["gene_id", "gene_name"])
    return dict(zip(df["gene_id"], df["gene_name"]))


def load_rhythm_classification(path: str) -> pd.DataFrame:
    """Load existing rhythm classification results."""
    df = pd.read_csv(path)
    log.info("Loaded rhythm classification: %s (%d genes)", Path(path).name, len(df))
    return df


def load_12h_consensus(path: str) -> pd.DataFrame:
    """Load existing 12h consensus results."""
    df = pd.read_csv(path)
    log.info("Loaded 12h consensus: %s (%d genes)", Path(path).name, len(df))
    return df


# ===================================================================
# Stage 1: Analytic genome-wide screen
# ===================================================================

def run_stage1(expr: pd.DataFrame, timepoints: np.ndarray,
               n_jobs: int, label: str) -> pd.DataFrame:
    """Run BHDT analytic mode on all genes.

    Parameters
    ----------
    expr : DataFrame (genes x timepoints)
    timepoints : array of hours
    n_jobs : parallel workers
    label : "Young" or "Old" for logging

    Returns
    -------
    DataFrame with BHDT analytic results.
    """
    log.info("=== Stage 1 [%s]: analytic mode on %d genes ===", label, len(expr))
    t0 = time.time()
    results = run_bhdt(expr, timepoints, method="analytic", n_jobs=n_jobs, verbose=True)
    elapsed = time.time() - t0
    log.info("Stage 1 [%s] completed in %.1f s (%.1f ms/gene)",
             label, elapsed, 1000 * elapsed / len(expr))

    # Summary counts
    counts = results["classification"].value_counts()
    for cls, n in counts.items():
        log.info("  %s: %d genes (%.1f%%)", cls, n, 100 * n / len(results))

    return results


# ===================================================================
# Stage 2: Ensemble refinement on candidates
# ===================================================================

def run_stage2(expr: pd.DataFrame, timepoints: np.ndarray,
               stage1: pd.DataFrame, n_jobs: int, label: str,
               fdr_threshold: float = 0.05) -> pd.DataFrame:
    """Run BHDT ensemble mode on candidate genes from Stage 1.

    Candidates are genes classified as 'independent' or 'ambiguous' by the
    analytic screen, plus genes with significant 12h F-test (FDR < threshold).
    """
    mask_cls = stage1["classification"].isin(["independent", "ambiguous"])
    mask_fdr = stage1["f_test_12h_fdr"] < fdr_threshold
    candidates = stage1.loc[mask_cls | mask_fdr, "gene"].values
    candidate_set = set(candidates)

    n_cand = len(candidate_set)
    if n_cand == 0:
        log.info("Stage 2 [%s]: no candidates — skipping ensemble.", label)
        return pd.DataFrame()

    log.info("=== Stage 2 [%s]: ensemble mode on %d candidates ===", label, n_cand)
    expr_sub = expr.loc[expr.index.isin(candidate_set)]

    t0 = time.time()
    results = run_bhdt(expr_sub, timepoints, method="ensemble",
                       n_jobs=n_jobs, verbose=True)
    elapsed = time.time() - t0
    log.info("Stage 2 [%s] completed in %.1f s (%.1f s/gene)",
             label, elapsed, elapsed / max(n_cand, 1))

    counts = results["classification"].value_counts()
    for cls, n in counts.items():
        log.info("  %s: %d genes (%.1f%%)", cls, n, 100 * n / len(results))

    return results



# ===================================================================
# Comparison with existing methods
# ===================================================================

def compare_with_existing(bhdt_results: pd.DataFrame,
                          rhythm_cls: pd.DataFrame,
                          consensus_12h: pd.DataFrame,
                          label: str) -> pd.DataFrame:
    """Compare BHDT classifications with existing RAIN/Cosinor/LS/Pencil results.

    Returns a per-gene comparison DataFrame with concordance flags.
    """
    log.info("Comparing BHDT vs existing methods [%s]...", label)

    # Build a unified comparison frame
    comp = bhdt_results[["gene", "classification", "f_test_12h_fdr"]].copy()
    comp = comp.rename(columns={
        "classification": "bhdt_class",
        "f_test_12h_fdr": "bhdt_fdr_12h",
    })

    # Merge existing rhythm classification
    rc = rhythm_cls[["gene_id", "category", "cos_fdr_12h", "rain_12h_bh", "gene_name"]].copy()
    rc = rc.rename(columns={"gene_id": "gene", "category": "existing_category"})
    comp = comp.merge(rc, on="gene", how="left")

    # Merge 12h consensus
    c12 = consensus_12h[["gene_id", "consensus", "n_methods", "best_period"]].copy()
    c12 = c12.rename(columns={"gene_id": "gene", "consensus": "consensus_12h"})
    comp = comp.merge(c12, on="gene", how="left")

    # --- Concordance logic ---
    # BHDT says 12h rhythm: classification is 'independent' (true 12h)
    # or f_test_12h significant
    comp["bhdt_12h"] = (
        (comp["bhdt_class"] == "independent") |
        (comp["bhdt_fdr_12h"] < 0.05)
    )

    # Existing methods say 12h rhythm
    comp["existing_12h"] = (
        comp["existing_category"].isin(["ultradian_12h", "12h_rhythmic"]) |
        comp["consensus_12h"].isin(["consensus", "majority", "strong"])
    )

    # Agreement categories
    comp["concordance"] = "TN"  # default: both say no
    comp.loc[comp["bhdt_12h"] & comp["existing_12h"], "concordance"] = "TP"
    comp.loc[comp["bhdt_12h"] & ~comp["existing_12h"], "concordance"] = "BHDT_only"
    comp.loc[~comp["bhdt_12h"] & comp["existing_12h"], "concordance"] = "existing_only"

    # Log summary
    conc_counts = comp["concordance"].value_counts()
    log.info("Concordance [%s]:", label)
    for cat, n in conc_counts.items():
        log.info("  %s: %d genes", cat, n)

    total_agree = conc_counts.get("TP", 0) + conc_counts.get("TN", 0)
    log.info("  Overall agreement: %d / %d (%.1f%%)",
             total_agree, len(comp), 100 * total_agree / len(comp))

    return comp


# ===================================================================
# Differential rhythm analysis (Young vs Old)
# ===================================================================

def differential_rhythm(young_res: pd.DataFrame, old_res: pd.DataFrame,
                        gene_name_map: dict) -> pd.DataFrame:
    """Compare BHDT results between Young and Old to find rhythm changes.

    Categories:
      - lost_12h: independent in Young, not in Old
      - gained_12h: independent in Old, not in Young
      - harmonic_to_independent: harmonic in Young -> independent in Old
      - independent_to_harmonic: independent in Young -> harmonic in Old
      - stable_independent: independent in both
      - stable_harmonic: harmonic in both
      - other_change: any other classification change
      - stable_other: same classification, neither independent nor harmonic
    """
    log.info("Running differential rhythm analysis (Young vs Old)...")

    y = young_res[["gene", "classification", "f_test_12h_fdr",
                    "log_bayes_factor", "A_12", "A_24"]].copy()
    o = old_res[["gene", "classification", "f_test_12h_fdr",
                  "log_bayes_factor", "A_12", "A_24"]].copy()

    diff = y.merge(o, on="gene", suffixes=("_young", "_old"))

    # Assign differential category
    cy = diff["classification_young"]
    co = diff["classification_old"]

    # Order matters: np.select picks the first True condition per row,
    # so place specific transitions before the broader lost/gained catches.
    conditions = [
        (cy == "independent") & (co == "independent"),   # stable first
        (cy == "harmonic") & (co == "harmonic"),
        (cy == "harmonic") & (co == "independent"),      # specific transitions
        (cy == "independent") & (co == "harmonic"),
        (cy == "independent") & (co != "independent"),   # broad lost/gained
        (cy != "independent") & (co == "independent"),
        (cy == co),                                      # same non-ind/harm class
    ]
    choices = [
        "stable_independent",
        "stable_harmonic",
        "harmonic_to_independent",
        "independent_to_harmonic",
        "lost_12h",
        "gained_12h",
        "stable_other",
    ]
    diff["diff_category"] = np.select(conditions, choices, default="other_change")

    # Add gene names
    diff["gene_name"] = diff["gene"].map(gene_name_map).fillna("")

    # Log-fold-change of 12h amplitude
    diff["A_12_log2fc"] = np.log2(
        (diff["A_12_old"] + 1e-6) / (diff["A_12_young"] + 1e-6)
    )

    # Summary
    cat_counts = diff["diff_category"].value_counts()
    log.info("Differential rhythm categories:")
    for cat, n in cat_counts.items():
        log.info("  %s: %d genes", cat, n)

    return diff



# ===================================================================
# Summary report
# ===================================================================

def write_summary(output_dir: Path,
                  young_s1: pd.DataFrame, old_s1: pd.DataFrame,
                  young_s2: pd.DataFrame, old_s2: pd.DataFrame,
                  diff: pd.DataFrame,
                  comp_young: pd.DataFrame, comp_old: pd.DataFrame,
                  elapsed_total: float):
    """Write a human-readable summary of key findings."""
    path = output_dir / "bhdt_summary.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("BHDT Genome-Wide Analysis — Ovary Aging Dataset")
    lines.append("=" * 70)
    lines.append(f"Total runtime: {elapsed_total:.1f} s")
    lines.append("")

    for label, s1, s2 in [("Young", young_s1, young_s2),
                           ("Old", old_s1, old_s2)]:
        lines.append(f"--- {label} group ---")
        lines.append(f"  Total genes analysed: {len(s1)}")
        counts = s1["classification"].value_counts()
        for cls in ["harmonic", "independent", "ambiguous"]:
            n = counts.get(cls, 0)
            lines.append(f"  Stage 1 {cls}: {n} ({100*n/len(s1):.1f}%)")
        n_sig = (s1["f_test_12h_fdr"] < 0.05).sum()
        lines.append(f"  12h F-test significant (FDR<0.05): {n_sig}")
        if len(s2) > 0:
            lines.append(f"  Stage 2 ensemble candidates: {len(s2)}")
            e_counts = s2["classification"].value_counts()
            for cls in ["harmonic", "independent", "ambiguous"]:
                n = e_counts.get(cls, 0)
                lines.append(f"    Ensemble {cls}: {n}")
        lines.append("")

    # Concordance
    for label, comp in [("Young", comp_young), ("Old", comp_old)]:
        lines.append(f"--- Concordance with existing methods [{label}] ---")
        conc = comp["concordance"].value_counts()
        for cat in ["TP", "TN", "BHDT_only", "existing_only"]:
            n = conc.get(cat, 0)
            lines.append(f"  {cat}: {n}")
        total_agree = conc.get("TP", 0) + conc.get("TN", 0)
        lines.append(f"  Agreement: {total_agree}/{len(comp)} "
                      f"({100*total_agree/len(comp):.1f}%)")
        lines.append("")

    # Differential
    lines.append("--- Differential rhythm (Young vs Old) ---")
    cat_counts = diff["diff_category"].value_counts()
    for cat in ["lost_12h", "gained_12h", "harmonic_to_independent",
                "independent_to_harmonic", "stable_independent",
                "stable_harmonic", "other_change", "stable_other"]:
        n = cat_counts.get(cat, 0)
        lines.append(f"  {cat}: {n}")

    # Top lost/gained genes
    for cat_label, cat_val in [("Lost 12h (top 10)", "lost_12h"),
                                ("Gained 12h (top 10)", "gained_12h")]:
        sub = diff[diff["diff_category"] == cat_val].copy()
        if len(sub) > 0:
            sub = sub.sort_values("A_12_log2fc",
                                  ascending=(cat_val == "lost_12h"))
            lines.append(f"\n  {cat_label}:")
            for _, row in sub.head(10).iterrows():
                lines.append(f"    {row['gene_name']:20s} "
                             f"({row['gene']})  "
                             f"A12_log2fc={row['A_12_log2fc']:+.2f}")

    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    path.write_text(text)
    log.info("Summary written to %s", path)
    # Also print to stdout
    print(text)



# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run BHDT genome-wide analysis on ovary aging dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/home/data2/fangcong2/ovary_aging/",
        help="Root data directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Default: {data-dir}/06rhythm/chord/",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel workers (-1 = all cores).",
    )
    parser.add_argument(
        "--skip-ensemble", action="store_true",
        help="Skip Stage 2 (ensemble refinement).",
    )
    parser.add_argument(
        "--ensemble-threshold", type=float, default=0.05,
        help="FDR threshold for selecting ensemble candidates.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rhythm_dir = data_dir / "06rhythm"
    output_dir = Path(args.output_dir) if args.output_dir else rhythm_dir / "chord"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Data dir:   %s", data_dir)
    log.info("Output dir: %s", output_dir)
    log.info("n_jobs:     %d", args.n_jobs)
    log.info("Ensemble:   %s (threshold=%.3f)",
             "SKIP" if args.skip_ensemble else "ON", args.ensemble_threshold)

    t_total = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info("Loading expression matrices...")
    expr_young, timepoints = load_mean_matrix(rhythm_dir / "mean_matrix_Young.csv")
    expr_old, _ = load_mean_matrix(rhythm_dir / "mean_matrix_Old.csv")

    log.info("Loading gene name mappings...")
    gene_name_map = load_gene_name_map(rhythm_dir / "rhythm_classification_Young.csv")
    # Supplement with Old in case there are extra genes
    gene_name_map.update(
        load_gene_name_map(rhythm_dir / "rhythm_classification_Old.csv")
    )

    log.info("Loading existing rhythm results for comparison...")
    rc_young = load_rhythm_classification(rhythm_dir / "rhythm_classification_Young.csv")
    rc_old = load_rhythm_classification(rhythm_dir / "rhythm_classification_Old.csv")
    c12_young = load_12h_consensus(rhythm_dir / "rhythm_12h_consensus_Young.csv")
    c12_old = load_12h_consensus(rhythm_dir / "rhythm_12h_consensus_Old.csv")

    # ------------------------------------------------------------------
    # 2. Stage 1 — Analytic genome-wide screen
    # ------------------------------------------------------------------
    young_s1 = run_stage1(expr_young, timepoints, args.n_jobs, "Young")
    old_s1 = run_stage1(expr_old, timepoints, args.n_jobs, "Old")

    # Annotate with gene names
    young_s1["gene_name"] = young_s1["gene"].map(gene_name_map).fillna("")
    old_s1["gene_name"] = old_s1["gene"].map(gene_name_map).fillna("")

    # Save Stage 1 results
    young_s1.to_csv(output_dir / "bhdt_results_Young.csv", index=False)
    old_s1.to_csv(output_dir / "bhdt_results_Old.csv", index=False)
    log.info("Saved Stage 1 results.")

    # ------------------------------------------------------------------
    # 3. Stage 2 — Ensemble refinement (optional)
    # ------------------------------------------------------------------
    young_s2 = pd.DataFrame()
    old_s2 = pd.DataFrame()

    if not args.skip_ensemble:
        young_s2 = run_stage2(expr_young, timepoints, young_s1,
                              args.n_jobs, "Young", args.ensemble_threshold)
        old_s2 = run_stage2(expr_old, timepoints, old_s1,
                            args.n_jobs, "Old", args.ensemble_threshold)

        if len(young_s2) > 0:
            young_s2["gene_name"] = young_s2["gene"].map(gene_name_map).fillna("")
            young_s2.to_csv(output_dir / "bhdt_ensemble_Young.csv", index=False)
        if len(old_s2) > 0:
            old_s2["gene_name"] = old_s2["gene"].map(gene_name_map).fillna("")
            old_s2.to_csv(output_dir / "bhdt_ensemble_Old.csv", index=False)
        log.info("Saved Stage 2 ensemble results.")
    else:
        log.info("Stage 2 skipped (--skip-ensemble).")

    # ------------------------------------------------------------------
    # 4. Comparison with existing methods
    # ------------------------------------------------------------------
    comp_young = compare_with_existing(young_s1, rc_young, c12_young, "Young")
    comp_old = compare_with_existing(old_s1, rc_old, c12_old, "Old")

    # Combine into one file with a group column
    comp_young["group"] = "Young"
    comp_old["group"] = "Old"
    comp_all = pd.concat([comp_young, comp_old], ignore_index=True)
    comp_all.to_csv(output_dir / "bhdt_vs_existing_comparison.csv", index=False)
    log.info("Saved comparison results.")

    # ------------------------------------------------------------------
    # 5. Differential rhythm analysis
    # ------------------------------------------------------------------
    diff = differential_rhythm(young_s1, old_s1, gene_name_map)
    diff.to_csv(output_dir / "bhdt_diff_rhythm.csv", index=False)
    log.info("Saved differential rhythm results.")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_total
    write_summary(output_dir, young_s1, old_s1, young_s2, old_s2,
                  diff, comp_young, comp_old, elapsed_total)

    log.info("All done in %.1f s.", elapsed_total)


if __name__ == "__main__":
    main()
