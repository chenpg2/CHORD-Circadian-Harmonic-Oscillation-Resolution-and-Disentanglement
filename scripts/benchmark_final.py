#!/usr/bin/env python3
"""
CHORD Benchmark — Publication Pipeline for iMeta.

Compares CHORD against 6 classical rhythm detection methods across
multiple real GEO datasets and synthetic data. Produces publication-ready
CSV results and figures.

Usage:
    python benchmark_final.py --mode all --output results/benchmark/
    python benchmark_final.py --mode synthetic --reps 50
    python benchmark_final.py --mode real --datasets hughes2009,zhu2023_wt
"""

import sys
import os
import time
import argparse
import warnings
import traceback
from pathlib import Path

# Ensure chord package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from chord package
# ---------------------------------------------------------------------------
from chord.bhdt.classifier import classify_gene, batch_classify, CHORDConfig
from chord.benchmarks.wrappers import (
    lomb_scargle, cosinor, harmonic_regression,
    jtk_cycle, rain, pencil_method,
)
from chord.data.geo_loader import (
    load_hughes2009, load_zhu2023_bmal1ko, load_mure2018,
    load_zhang2014, load_scp2ko_2015,
)
from chord.data.zhu2024_loader import load_zhu2024
from chord.data.known_genes import (
    KNOWN_12H_GENES_ZHU2017, CORE_CIRCADIAN_GENES,
    NON_RHYTHMIC_HOUSEKEEPING,
    KNOWN_12H_GENES_PRIMATE, CORE_CIRCADIAN_GENES_PRIMATE,
    NON_RHYTHMIC_HOUSEKEEPING_PRIMATE,
    KNOWN_12H_GENES_ZHU2024, CORE_CIRCADIAN_GENES_HUMAN,
    NON_RHYTHMIC_HOUSEKEEPING_HUMAN,
    CONSERVED_12H_GENES_CROSS_SPECIES,
)
from chord.benchmarks.metrics import roc_auc
from chord.simulation.generator import generate_all_scenarios


# ============================================================================
# Dataset Registry
# ============================================================================

def _load_hughes2009_1h(cache_dir):
    d = load_hughes2009(cache_dir=cache_dir, downsample_2h=False)
    return d["expr"], d["timepoints"], d["gene_names"], "mouse"

def _load_hughes2009_2h(cache_dir):
    d = load_hughes2009(cache_dir=cache_dir, downsample_2h=True)
    return d["expr"], d["timepoints"], d["gene_names"], "mouse"

def _load_zhang2014_liver(cache_dir):
    d = load_zhang2014(cache_dir=cache_dir, tissue="Liv")
    return d["expr"], d["timepoints"], d["gene_names"], "mouse"

def _load_zhu2023_wt(cache_dir):
    d = load_zhu2023_bmal1ko(cache_dir=cache_dir)
    wt = d["wt"]
    return wt["expr"], wt["timepoints"], wt["gene_names"], "mouse"

def _load_zhu2023_ko(cache_dir):
    d = load_zhu2023_bmal1ko(cache_dir=cache_dir)
    ko = d["ko"]
    return ko["expr"], ko["timepoints"], ko["gene_names"], "mouse"

def _load_mure2018_liver(cache_dir):
    d = load_mure2018(cache_dir=cache_dir, tissue="LIV")
    return d["expr"], d["timepoints"], d["gene_names"], "primate"

def _load_zhu2024(cache_dir):
    d = load_zhu2024(cache_dir=cache_dir)
    return d["expr"], d["timepoints"], d["gene_names"], "human"

def _load_scp2ko_wt(cache_dir):
    d = load_scp2ko_2015(cache_dir=cache_dir, genotype="wt")
    return d["expr"], d["timepoints"], d["gene_names"], "mouse"

def _load_scp2ko_ko(cache_dir):
    d = load_scp2ko_2015(cache_dir=cache_dir, genotype="ko")
    return d["expr"], d["timepoints"], d["gene_names"], "mouse"


DATASET_REGISTRY = {
    "hughes2009":      {"loader": _load_hughes2009_1h,   "organism": "mouse",   "tissue": "liver",  "resolution": "1h",  "n_tp": 48, "description": "Hughes 2009 mouse liver 1h"},
    "hughes2009_2h":   {"loader": _load_hughes2009_2h,   "organism": "mouse",   "tissue": "liver",  "resolution": "2h",  "n_tp": 24, "description": "Hughes 2009 mouse liver 2h"},
    "zhang2014_liver": {"loader": _load_zhang2014_liver, "organism": "mouse",   "tissue": "liver",  "resolution": "2h",  "n_tp": 24, "description": "Zhang 2014 mouse liver"},
    "zhu2023_wt":      {"loader": _load_zhu2023_wt,      "organism": "mouse",   "tissue": "liver",  "resolution": "4h",  "n_tp": 12, "description": "Zhu 2023 WT liver"},
    "zhu2023_ko":      {"loader": _load_zhu2023_ko,      "organism": "mouse",   "tissue": "liver",  "resolution": "4h",  "n_tp": 12, "description": "Zhu 2023 BMAL1-KO liver"},
    "mure2018_liver":  {"loader": _load_mure2018_liver,  "organism": "primate", "tissue": "liver",  "resolution": "2h",  "n_tp": 12, "description": "Mure 2018 baboon liver"},
    "zhu2024":         {"loader": _load_zhu2024,          "organism": "human",   "tissue": "blood",  "resolution": "2h",  "n_tp": 24, "description": "Zhu 2024 human blood"},
    "scp2ko_wt":       {"loader": _load_scp2ko_wt,       "organism": "mouse",   "tissue": "liver",  "resolution": "2h",  "n_tp": 12, "description": "Scp2 KO study WT liver"},
    "scp2ko_ko":       {"loader": _load_scp2ko_ko,       "organism": "mouse",   "tissue": "liver",  "resolution": "2h",  "n_tp": 12, "description": "Scp2 KO study KO liver"},
}


def load_dataset(name, cache_dir="~/.chord_cache"):
    """Load a dataset by registry name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[name]
    print(f"  Loading {info['description']}...")
    expr, timepoints, gene_names, organism = info["loader"](cache_dir)
    print(f"    -> {expr.shape[0]} genes x {expr.shape[1]} timepoints")
    return expr, timepoints, gene_names, organism


# ============================================================================
# Ground Truth
# ============================================================================

def get_ground_truth(gene_names, organism):
    """Assign ground truth labels based on organism-specific known gene lists.

    Returns dict: {gene_name: {"is_12h": bool, "is_circadian": bool,
                                "is_negative": bool, "category": str}}
    """
    if organism == "mouse":
        pos_set = set(KNOWN_12H_GENES_ZHU2017)
        circ_set = set(CORE_CIRCADIAN_GENES)
        neg_set = set(NON_RHYTHMIC_HOUSEKEEPING)
    elif organism == "primate":
        pos_set = set(KNOWN_12H_GENES_PRIMATE)
        circ_set = set(CORE_CIRCADIAN_GENES_PRIMATE)
        neg_set = set(NON_RHYTHMIC_HOUSEKEEPING_PRIMATE)
    elif organism == "human":
        pos_set = set(KNOWN_12H_GENES_ZHU2024)
        circ_set = set(CORE_CIRCADIAN_GENES_HUMAN)
        neg_set = set(NON_RHYTHMIC_HOUSEKEEPING_HUMAN)
    else:
        return {}

    gt = {}
    for g in gene_names:
        g_upper = g.upper()
        entry = {"is_12h": False, "is_circadian": False,
                 "is_negative": False, "category": "unknown"}
        if g in pos_set or g_upper in {x.upper() for x in pos_set}:
            entry.update(is_12h=True, category="known_12h")
        elif g in circ_set or g_upper in {x.upper() for x in circ_set}:
            entry.update(is_circadian=True, category="circadian")
        elif g in neg_set or g_upper in {x.upper() for x in neg_set}:
            entry.update(is_negative=True, category="housekeeping")
        gt[g] = entry
    return gt


# ============================================================================
# Method Wrappers (unified interface)
# ============================================================================

def run_chord(t, y, config=None):
    """Run CHORD on a single gene, return unified result dict."""
    try:
        result = classify_gene(t, y, config=config)
        cls = result.get("classification", "non_rhythmic")
        # Tier-1 detection: Stage 1 passed = 12h component detected
        stage1_passed = result.get("stage1_passed", False)
        # Tier-2 disentanglement: independent vs harmonic
        is_independent = cls in ("independent_ultradian", "likely_independent_ultradian")
        return {
            "p_value": result.get("stage1_p_detect", 1.0),
            "period_estimate": 12.0,
            "amplitude_estimate": result.get("amp_ratio", 0.0),
            "method_name": "CHORD",
            "has_12h": stage1_passed,
            "classification": cls,
            "is_independent": is_independent,
            "confidence": result.get("confidence", 0.0),
            "evidence_score": result.get("evidence_score", 0.0),
            "stage1_passed": stage1_passed,
        }
    except Exception as e:
        return {
            "p_value": 1.0, "period_estimate": np.nan,
            "amplitude_estimate": 0.0, "method_name": "CHORD",
            "has_12h": False, "classification": "error",
            "is_independent": False,
            "confidence": 0.0, "evidence_score": 0.0,
            "stage1_passed": False, "error": str(e),
        }


def run_classical_method(method_func, method_name, t, y, **kwargs):
    """Run a classical method and return unified result dict."""
    try:
        result = method_func(t, y, **kwargs)
        p = result.get("p_value", 1.0)
        period = result.get("period_estimate", np.nan)
        has_12h = (p < 0.05) and (10.0 < period < 14.0) if np.isfinite(period) else False
        return {
            "p_value": p,
            "period_estimate": period,
            "amplitude_estimate": result.get("amplitude_estimate", 0.0),
            "method_name": method_name,
            "has_12h": has_12h,
            "classification": "detected_12h" if has_12h else "not_detected",
            "confidence": 1.0 - p if np.isfinite(p) else 0.0,
            "evidence_score": 0.0,
            "stage1_passed": has_12h,
        }
    except Exception as e:
        return {
            "p_value": 1.0, "period_estimate": np.nan,
            "amplitude_estimate": 0.0, "method_name": method_name,
            "has_12h": False, "classification": "error",
            "confidence": 0.0, "evidence_score": 0.0,
            "stage1_passed": False, "error": str(e),
        }


METHOD_REGISTRY = {
    "CHORD":            {"func": run_chord,     "type": "detect+disentangle"},
    "Lomb_Scargle":        {"func": lambda t, y: run_classical_method(lomb_scargle, "Lomb_Scargle", t, y),        "type": "detect_only"},
    "Cosinor":             {"func": lambda t, y: run_classical_method(cosinor, "Cosinor", t, y),                  "type": "detect_only"},
    "Harmonic_Regression": {"func": lambda t, y: run_classical_method(harmonic_regression, "Harmonic_Regression", t, y), "type": "detect_only"},
    "JTK_CYCLE":           {"func": lambda t, y: run_classical_method(jtk_cycle, "JTK_CYCLE", t, y),             "type": "detect_only"},
    "RAIN":                {"func": lambda t, y: run_classical_method(rain, "RAIN", t, y),                        "type": "detect_only"},
    "Pencil":              {"func": lambda t, y: run_classical_method(pencil_method, "Pencil", t, y),             "type": "detect_only"},
}


# ============================================================================
# Batch Execution Engine
# ============================================================================

def run_all_methods_on_dataset(dataset_name, expr, timepoints, gene_names,
                                organism, methods=None, cache_dir="~/.chord_cache",
                                verbose=True, gt_only=False, max_genes=None):
    """Run all methods on all genes in a dataset.

    Parameters
    ----------
    gt_only : bool
        If True, only run on genes with ground truth labels.
    max_genes : int or None
        If set, limit to this many genes (for quick testing).

    Returns pd.DataFrame with per-gene per-method results.
    """
    # Filter to ground truth genes if requested
    if gt_only:
        gt = get_ground_truth(gene_names, organism)
        gt_indices = [i for i, g in enumerate(gene_names) if gt.get(g, {}).get("category", "unknown") != "unknown"]
        if gt_indices:
            expr = expr[gt_indices]
            gene_names = [gene_names[i] for i in gt_indices]
            if verbose:
                print(f"    (gt_only: filtered to {len(gene_names)} ground truth genes)")

    if max_genes and len(gene_names) > max_genes:
        expr = expr[:max_genes]
        gene_names = gene_names[:max_genes]
        if verbose:
            print(f"    (max_genes: limited to {max_genes} genes)")
    if methods is None:
        methods = list(METHOD_REGISTRY.keys())

    results = []
    n_genes = len(gene_names)

    for method_name in methods:
        method_info = METHOD_REGISTRY[method_name]
        method_func = method_info["func"]

        if verbose:
            print(f"  Running {method_name} on {dataset_name} ({n_genes} genes)...")

        t_start = time.time()
        for i in range(n_genes):
            gene = gene_names[i]
            y = expr[i, :]

            # Skip genes with all NaN or zero variance
            if np.all(np.isnan(y)) or np.nanstd(y) < 1e-10:
                results.append({
                    "gene": gene, "method": method_name, "dataset": dataset_name,
                    "organism": organism, "p_value": 1.0, "period_estimate": np.nan,
                    "has_12h": False, "classification": "skipped",
                    "confidence": 0.0, "evidence_score": 0.0,
                    "wall_time_ms": 0.0,
                })
                continue

            t0 = time.perf_counter()
            if method_name == "CHORD":
                res = method_func(timepoints, y)
            else:
                res = method_func(timepoints, y)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            results.append({
                "gene": gene,
                "method": method_name,
                "dataset": dataset_name,
                "organism": organism,
                "p_value": res.get("p_value", 1.0),
                "period_estimate": res.get("period_estimate", np.nan),
                "has_12h": res.get("has_12h", False),
                "classification": res.get("classification", "unknown"),
                "confidence": res.get("confidence", 0.0),
                "evidence_score": res.get("evidence_score", 0.0),
                "wall_time_ms": elapsed_ms,
            })

            if verbose and (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(f"    {method_name}: {i+1}/{n_genes} genes ({rate:.0f} genes/s)")

        if verbose:
            elapsed = time.time() - t_start
            print(f"    {method_name}: done in {elapsed:.1f}s")

    return pd.DataFrame(results)


# ============================================================================
# Metric Computation
# ============================================================================

def compute_tier1_metrics(results_df, cache_dir="~/.chord_cache"):
    """Compute Tier-1 binary 12h detection metrics per method x dataset.

    Positive: known 12h genes. Negative: circadian + housekeeping genes.
    """
    rows = []
    for (dataset, method), grp in results_df.groupby(["dataset", "method"]):
        organism = grp["organism"].iloc[0]
        gt = get_ground_truth(grp["gene"].tolist(), organism)

        # Filter to genes with ground truth
        gt_genes = {g for g, v in gt.items() if v["category"] != "unknown"}
        sub = grp[grp["gene"].isin(gt_genes)].copy()
        if len(sub) == 0:
            continue

        y_true = []
        y_pred = []
        y_scores = []
        for _, row in sub.iterrows():
            g = row["gene"]
            info = gt[g]
            label = 1 if info["is_12h"] else 0
            pred = 1 if row["has_12h"] else 0
            y_true.append(label)
            y_pred.append(pred)
            y_scores.append(row["confidence"])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # ROC AUC
        try:
            auc_result = roc_auc(y_true, y_scores)
            auc_val = auc_result["auc"]
        except Exception:
            auc_val = np.nan

        rows.append({
            "dataset": dataset, "method": method,
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "f1": f1, "roc_auc": auc_val,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_positive": n_pos, "n_negative": n_neg,
        })

    return pd.DataFrame(rows)


def compute_recovery_metrics(results_df):
    """Compute recovery rate of known gene lists per method x dataset."""
    rows = []
    for (dataset, method), grp in results_df.groupby(["dataset", "method"]):
        organism = grp["organism"].iloc[0]
        gt = get_ground_truth(grp["gene"].tolist(), organism)

        for category in ["known_12h", "circadian", "housekeeping"]:
            cat_genes = [g for g, v in gt.items() if v["category"] == category]
            if not cat_genes:
                continue
            sub = grp[grp["gene"].isin(cat_genes)]
            n_total = len(cat_genes)
            n_found = len(sub)
            n_detected = sub["has_12h"].sum() if len(sub) > 0 else 0
            recovery = n_detected / n_total if n_total > 0 else 0.0

            rows.append({
                "dataset": dataset, "method": method,
                "category": category, "n_total": n_total,
                "n_in_data": n_found, "n_detected_12h": int(n_detected),
                "recovery_rate": recovery,
            })

    return pd.DataFrame(rows)


def compute_cross_species_metrics(results_df):
    """Compute cross-species consistency for conserved 12h genes."""
    conserved = set(CONSERVED_12H_GENES_CROSS_SPECIES)
    rows = []

    for method in results_df["method"].unique():
        method_df = results_df[results_df["method"] == method]
        detected_per_dataset = {}

        for dataset in method_df["dataset"].unique():
            ds_df = method_df[method_df["dataset"] == dataset]
            # Match conserved genes (case-insensitive)
            gene_upper = {g.upper(): g for g in ds_df["gene"].tolist()}
            detected = set()
            for cg in conserved:
                if cg.upper() in gene_upper:
                    real_name = gene_upper[cg.upper()]
                    row = ds_df[ds_df["gene"] == real_name]
                    if len(row) > 0 and row.iloc[0]["has_12h"]:
                        detected.add(cg)
            detected_per_dataset[dataset] = detected

        # Compute pairwise Jaccard
        datasets = list(detected_per_dataset.keys())
        if len(datasets) < 2:
            continue

        all_detected = set()
        for d in detected_per_dataset.values():
            all_detected |= d

        # Overall recovery across all datasets
        n_detected_any = len(all_detected)
        n_detected_all = len(set.intersection(*detected_per_dataset.values())) if detected_per_dataset else 0

        rows.append({
            "method": method,
            "n_conserved_genes": len(conserved),
            "n_detected_any_dataset": n_detected_any,
            "n_detected_all_datasets": n_detected_all,
            "recovery_any": n_detected_any / len(conserved),
            "recovery_all": n_detected_all / len(conserved),
            "n_datasets": len(datasets),
        })

    return pd.DataFrame(rows)


def compute_efficiency_metrics(results_df):
    """Aggregate wall_time_ms per method across datasets."""
    rows = []
    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method]
        times = sub["wall_time_ms"].values
        times = times[times > 0]  # exclude skipped
        if len(times) == 0:
            continue
        rows.append({
            "method": method,
            "mean_time_ms": np.mean(times),
            "median_time_ms": np.median(times),
            "std_time_ms": np.std(times),
            "p95_time_ms": np.percentile(times, 95),
            "total_genes": len(times),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Synthetic Benchmark
# ============================================================================

def run_synthetic_benchmark(n_reps=50, n_timepoints=48, sampling_hours=1.0,
                             methods=None, seed=42, verbose=True):
    """Run all methods on 15 synthetic scenarios x n_reps replicates."""
    if methods is None:
        methods = list(METHOD_REGISTRY.keys())

    t = np.arange(0, n_timepoints * sampling_hours, sampling_hours)
    all_results = []

    for rep in range(n_reps):
        if verbose and (rep + 1) % 10 == 0:
            print(f"  Synthetic rep {rep+1}/{n_reps}")

        scenarios = generate_all_scenarios(t, seed=seed + rep, n_replicates=1)

        for scenario in scenarios:
            y = scenario["y"]
            truth = scenario["truth"]
            scenario_id = truth.get("scenario_id", scenario.get("scenario_id", scenario.get("name", "unknown")))

            has_ind_12h = truth.get("has_independent_12h", False)
            has_harm_12h = truth.get("has_harmonic_12h", False)

            # Ground truth class
            if has_ind_12h and not has_harm_12h:
                gt_class = "independent_ultradian"
            elif has_harm_12h and not has_ind_12h:
                gt_class = "harmonic"
            elif has_ind_12h and has_harm_12h:
                gt_class = "mixed"
            else:
                gt_class = "non_rhythmic"

            for method_name in methods:
                method_func = METHOD_REGISTRY[method_name]["func"]

                t0 = time.perf_counter()
                if method_name == "CHORD":
                    res = method_func(t, y)
                else:
                    res = method_func(t, y)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                all_results.append({
                    "scenario": scenario_id,
                    "replicate": rep,
                    "method": method_name,
                    "gt_has_12h": has_ind_12h,
                    "gt_has_harmonic": has_harm_12h,
                    "gt_class": gt_class,
                    "p_value": res.get("p_value", 1.0),
                    "has_12h": res.get("has_12h", False),
                    "classification": res.get("classification", "unknown"),
                    "confidence": res.get("confidence", 0.0),
                    "wall_time_ms": elapsed_ms,
                })

    return pd.DataFrame(all_results)


def compute_synthetic_metrics(synth_df):
    """Compute Tier-1 and Tier-2 metrics on synthetic data.

    Tier-1 (detection): Can the method detect ANY 12h component?
        - For classical methods: p < 0.05 with period near 12h
        - For CHORD: stage1_passed (CCT-fused detection gate)
        - Ground truth positive = has_independent_12h OR has_harmonic_12h
        - All methods evaluated on ALL scenarios (fair comparison)

    Tier-2 (disentanglement): Can CHORD distinguish independent from harmonic?
        - Only CHORD provides this; classical methods cannot disentangle.
        - Ground truth positive = has_independent_12h (only)
        - Evaluated on genes with any 12h component.
    """
    rows = []

    for method in synth_df["method"].unique():
        sub = synth_df[synth_df["method"] == method]

        # --- Tier-1: Binary 12h detection (all scenarios, fair for all) ---
        # Ground truth: any 12h component present (independent OR harmonic)
        y_true = (sub["gt_has_12h"] | sub["gt_has_harmonic"]).astype(int).values
        y_pred = sub["has_12h"].astype(int).values
        y_scores = sub["confidence"].values

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos > 0 and n_neg > 0:
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
            try:
                auc_val = roc_auc(y_true, y_scores)["auc"]
            except Exception:
                auc_val = np.nan
        else:
            sens = spec = prec = f1 = auc_val = np.nan

        rows.append({
            "method": method, "tier": "Tier-1 (detection)",
            "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "roc_auc": auc_val,
        })

        # --- Tier-2: Disentanglement (CHORD only) ---
        if method == "CHORD":
            # Among genes with any 12h component, can CHORD tell independent
            # from harmonic?
            has_any_12h = sub["gt_has_12h"] | sub["gt_has_harmonic"]
            disent_sub = sub[has_any_12h]
            if len(disent_sub) > 0:
                y_true_d = disent_sub["gt_has_12h"].astype(int).values
                if "is_independent" in disent_sub.columns:
                    y_pred_d = disent_sub["is_independent"].astype(int).values
                else:
                    y_pred_d = disent_sub["classification"].isin(
                        ["independent_ultradian", "likely_independent_ultradian"]
                    ).astype(int).values
                y_scores_d = disent_sub["confidence"].values

                n_pos_d = y_true_d.sum()
                n_neg_d = len(y_true_d) - n_pos_d
                if n_pos_d > 0 and n_neg_d > 0:
                    tp_d = ((y_pred_d == 1) & (y_true_d == 1)).sum()
                    fp_d = ((y_pred_d == 1) & (y_true_d == 0)).sum()
                    fn_d = ((y_pred_d == 0) & (y_true_d == 1)).sum()
                    tn_d = ((y_pred_d == 0) & (y_true_d == 0)).sum()
                    sens_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
                    spec_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0
                    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
                    f1_d = 2 * prec_d * sens_d / (prec_d + sens_d) if (prec_d + sens_d) > 0 else 0
                    try:
                        auc_d = roc_auc(y_true_d, y_scores_d)["auc"]
                    except Exception:
                        auc_d = np.nan
                else:
                    sens_d = spec_d = prec_d = f1_d = auc_d = np.nan

                rows.append({
                    "method": method, "tier": "Tier-2 (disentangle)",
                    "sensitivity": sens_d, "specificity": spec_d,
                    "precision": prec_d, "f1": f1_d, "roc_auc": auc_d,
                })

    return pd.DataFrame(rows)


# ============================================================================
# Robustness Experiments
# ============================================================================

def run_downsampling_experiment(cache_dir="~/.chord_cache", methods=None, verbose=True):
    """Test robustness by downsampling Hughes2009 (48 tp @ 1h)."""
    if methods is None:
        methods = list(METHOD_REGISTRY.keys())

    # Load full-resolution data
    d = load_hughes2009(cache_dir=cache_dir, downsample_2h=False)
    expr_full, tp_full = d["expr"], d["timepoints"]
    gene_names = d["gene_names"]

    # Downsampling configs: (step, label, n_timepoints)
    configs = [
        (1, "48tp_1h", 48),
        (2, "24tp_2h", 24),
        (3, "16tp_3h", 16),
        (4, "12tp_4h", 12),
        (6, "8tp_6h",  8),
    ]

    all_results = []
    for step, label, expected_n in configs:
        idx = np.arange(0, len(tp_full), step)
        expr_ds = expr_full[:, idx]
        tp_ds = tp_full[idx]

        if verbose:
            print(f"  Downsampling: {label} ({len(tp_ds)} timepoints)")

        df = run_all_methods_on_dataset(
            f"downsample_{label}", expr_ds, tp_ds, gene_names,
            "mouse", methods=methods, verbose=verbose,
        )
        df["downsample_config"] = label
        df["n_timepoints"] = len(tp_ds)
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def run_noise_experiment(n_reps=20, methods=None, seed=42, verbose=True):
    """Test robustness by varying SNR on synthetic independent superposition."""
    if methods is None:
        methods = list(METHOD_REGISTRY.keys())

    snr_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    t = np.arange(0, 48, 1.0)
    rng = np.random.RandomState(seed)

    all_results = []
    for snr in snr_levels:
        if verbose:
            print(f"  Noise experiment: SNR={snr}")

        for rep in range(n_reps):
            # Generate independent superposition signal
            y_clean = (np.cos(2 * np.pi * t / 24.0) +
                       0.5 * np.cos(2 * np.pi * t / 11.8 + 0.7))
            noise_std = np.std(y_clean) / snr
            y = y_clean + rng.normal(0, noise_std, len(t))

            for method_name in methods:
                method_func = METHOD_REGISTRY[method_name]["func"]
                res = method_func(t, y)
                all_results.append({
                    "snr": snr, "replicate": rep, "method": method_name,
                    "has_12h": res.get("has_12h", False),
                    "p_value": res.get("p_value", 1.0),
                    "confidence": res.get("confidence", 0.0),
                })

    return pd.DataFrame(all_results)


# ============================================================================
# Brooks 2023 Integration
# ============================================================================

def _register_brooks2023_datasets():
    """Dynamically register Brooks 2023 datasets into DATASET_REGISTRY."""
    from chord.data.brooks2023_loader import load_brooks2023, BROOKS2023_BENCHMARK_STUDIES

    for study_id in BROOKS2023_BENCHMARK_STUDIES:
        key = f"brooks2023_{study_id}"
        if key not in DATASET_REGISTRY:
            def make_loader(sid):
                def loader(cache_dir):
                    d = load_brooks2023(study_id=sid)
                    return d["expr"], d["timepoints"], d["gene_names"], "mouse"
                return loader

            info = BROOKS2023_BENCHMARK_STUDIES[study_id]
            DATASET_REGISTRY[key] = {
                "loader": make_loader(study_id),
                "organism": "mouse",
                "tissue": "liver",
                "resolution": f"{24 // info['tp_per_cycle']}h",
                "n_tp": info["tp_per_cycle"],
                "description": f"Brooks2023 {info['description']}",
            }


# ============================================================================
# Main Orchestration
# ============================================================================

def run_real_benchmark(datasets=None, methods=None, cache_dir="~/.chord_cache",
                       verbose=True, gt_only=False, max_genes=None):
    """Run benchmark on real GEO datasets."""
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())

    all_results = []
    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"  WARNING: Unknown dataset '{ds_name}', skipping.")
            continue
        try:
            expr, tp, genes, organism = load_dataset(ds_name, cache_dir)
            df = run_all_methods_on_dataset(
                ds_name, expr, tp, genes, organism,
                methods=methods, verbose=verbose,
                gt_only=gt_only, max_genes=max_genes,
            )
            all_results.append(df)
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            traceback.print_exc()
            continue

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def print_summary_table(tier1_df, title="Tier-1 Metrics Summary"):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if len(tier1_df) == 0:
        print("  (no results)")
        return

    pivot = tier1_df.pivot_table(
        index="method", columns="dataset", values="f1", aggfunc="first"
    )
    # Add mean column
    pivot["MEAN"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("MEAN", ascending=False)

    print(pivot.round(3).to_string())
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CHORD Benchmark Pipeline for iMeta publication"
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "real", "robustness", "all"],
        default="all", help="Benchmark mode"
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (default: all registered)"
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated method names (default: all)"
    )
    parser.add_argument(
        "--reps", type=int, default=50,
        help="Number of synthetic replicates (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="results/benchmark/",
        help="Output directory"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="~/.chord_cache",
        help="GEO data cache directory"
    )
    parser.add_argument(
        "--no-brooks", action="store_true",
        help="Skip Brooks 2023 datasets"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 2 reps, subset of datasets"
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Only run on ground truth genes (much faster for real data)"
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit number of genes per dataset (for testing)"
    )
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register Brooks 2023 datasets
    if not args.no_brooks:
        try:
            _register_brooks2023_datasets()
            print(f"Registered Brooks 2023 datasets: {[k for k in DATASET_REGISTRY if k.startswith('brooks')]}")
        except Exception as e:
            print(f"WARNING: Could not register Brooks 2023 datasets: {e}")

    # Parse method list
    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]

    # Parse dataset list
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    if args.quick:
        args.reps = 2
        if datasets is None:
            datasets = ["hughes2009_2h", "zhu2023_wt"]

    print(f"\nCHORD Benchmark Pipeline")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {output_dir}")
    print(f"  Datasets: {datasets or 'all'}")
    print(f"  Methods: {methods or 'all'}")
    print(f"  Synthetic reps: {args.reps}")
    print()

    # ---- Synthetic Benchmark ----
    if args.mode in ("synthetic", "all"):
        print("=" * 60)
        print("PHASE 1: Synthetic Benchmark")
        print("=" * 60)
        synth_results = run_synthetic_benchmark(
            n_reps=args.reps, methods=methods, verbose=True,
        )
        synth_results.to_csv(output_dir / "synthetic_results.csv", index=False)
        print(f"  Saved: {output_dir / 'synthetic_results.csv'}")

        synth_metrics = compute_synthetic_metrics(synth_results)
        synth_metrics.to_csv(output_dir / "synthetic_metrics.csv", index=False)
        print(f"  Saved: {output_dir / 'synthetic_metrics.csv'}")

        print("\nSynthetic Benchmark Summary:")
        print(synth_metrics.to_string(index=False))

    # ---- Real Data Benchmark ----
    if args.mode in ("real", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: Real Data Benchmark")
        print("=" * 60)
        real_results = run_real_benchmark(
            datasets=datasets, methods=methods,
            cache_dir=args.cache_dir, verbose=True,
            gt_only=args.gt_only, max_genes=args.max_genes,
        )
        if len(real_results) > 0:
            real_results.to_csv(output_dir / "real_results.csv", index=False)
            print(f"  Saved: {output_dir / 'real_results.csv'}")

            # Compute metrics
            tier1 = compute_tier1_metrics(real_results)
            tier1.to_csv(output_dir / "tier1_metrics.csv", index=False)
            print_summary_table(tier1, "Tier-1: Binary 12h Detection")

            recovery = compute_recovery_metrics(real_results)
            recovery.to_csv(output_dir / "recovery_metrics.csv", index=False)
            print(f"  Saved: {output_dir / 'recovery_metrics.csv'}")

            cross_sp = compute_cross_species_metrics(real_results)
            cross_sp.to_csv(output_dir / "cross_species_metrics.csv", index=False)
            print(f"  Saved: {output_dir / 'cross_species_metrics.csv'}")

            efficiency = compute_efficiency_metrics(real_results)
            efficiency.to_csv(output_dir / "efficiency_metrics.csv", index=False)
            print(f"  Saved: {output_dir / 'efficiency_metrics.csv'}")

            print("\nEfficiency Summary:")
            print(efficiency.to_string(index=False))

    # ---- Robustness Experiments ----
    if args.mode in ("robustness", "all"):
        print("\n" + "=" * 60)
        print("PHASE 3: Robustness Experiments")
        print("=" * 60)

        print("\n--- Downsampling Experiment ---")
        ds_results = run_downsampling_experiment(
            cache_dir=args.cache_dir, methods=methods, verbose=True,
        )
        ds_results.to_csv(output_dir / "downsampling_results.csv", index=False)
        print(f"  Saved: {output_dir / 'downsampling_results.csv'}")

        print("\n--- Noise Experiment ---")
        noise_results = run_noise_experiment(
            n_reps=min(args.reps, 20), methods=methods, verbose=True,
        )
        noise_results.to_csv(output_dir / "noise_results.csv", index=False)
        print(f"  Saved: {output_dir / 'noise_results.csv'}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
