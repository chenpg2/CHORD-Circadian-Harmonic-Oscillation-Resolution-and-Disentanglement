"""
Systematic benchmark runner: CHORD (BHDT + PINOD) vs classical methods.

Supports two modes:
  1. Synthetic benchmark — 15 scenarios x N replicates (default)
  2. Real-data benchmark — public GEO datasets with known gene lists as
     ground truth (Hughes 2009, Zhu 2023 BMAL1 KO, Mure 2018 baboon)

Usage
-----
    python -m chord.benchmarks.run_benchmark              # synthetic
    python -m chord.benchmarks.run_benchmark --real-data   # real GEO data
"""

import time
import warnings
import numpy as np
import pandas as pd

from chord.simulation.generator import (
    generate_scenario,
    generate_all_scenarios,
    SCENARIO_NAMES,
)
from chord.benchmarks.wrappers import lomb_scargle, cosinor, harmonic_regression
from chord.benchmarks.metrics import (
    classification_metrics,
    harmonic_disentangle_accuracy,
)
from chord.bhdt.inference import bhdt_analytic


# ============================================================================
# Helper: convert classical method output to a classification label
# ============================================================================

_PERIOD_TOL = 2.0  # hours tolerance for period matching


def _classify_classical(method_result, method_name):
    """Convert a classical method result dict to a classification label.

    Labels follow the BHDT convention:
        'independent_ultradian'  — significant ~12 h signal detected
        'circadian_only'         — significant ~24 h signal, no 12 h
        'harmonic'               — (classical methods cannot distinguish this)
        'non_rhythmic'           — nothing significant

    Parameters
    ----------
    method_result : dict
        Output from one of the wrapper functions.
    method_name : str
        One of 'lomb_scargle', 'cosinor', 'harmonic_regression'.

    Returns
    -------
    dict
        predicted_class, predicted_has_12h, period_12h_est, p_value
    """
    out = {
        "predicted_class": "non_rhythmic",
        "predicted_has_12h": False,
        "period_12h_est": np.nan,
        "p_value": method_result.get("p_value", np.nan),
    }

    if method_name == "lomb_scargle":
        p = method_result["p_value"]
        best_T = method_result["period_estimate"]
        if p < 0.05:
            if abs(best_T - 12.0) < _PERIOD_TOL:
                out["predicted_class"] = "independent_ultradian"
                out["predicted_has_12h"] = True
                out["period_12h_est"] = best_T
            elif abs(best_T - 24.0) < _PERIOD_TOL:
                out["predicted_class"] = "circadian_only"
            else:
                # Significant but at an unexpected period
                out["predicted_class"] = "circadian_only"
        return out

    if method_name == "cosinor":
        # Cosinor is run at period=12 h specifically
        p = method_result["p_value"]
        if p < 0.05:
            out["predicted_class"] = "independent_ultradian"
            out["predicted_has_12h"] = True
            out["period_12h_est"] = method_result["period_estimate"]
        return out

    if method_name == "harmonic_regression":
        components = method_result.get("components", [])
        sig_24 = False
        sig_12 = False
        period_12_est = np.nan
        for comp in components:
            T = comp["period"]
            if comp["p_value"] < 0.05:
                if abs(T - 24.0) < _PERIOD_TOL:
                    sig_24 = True
                elif abs(T - 12.0) < _PERIOD_TOL:
                    sig_12 = True
                    period_12_est = T
        if sig_12:
            out["predicted_class"] = "independent_ultradian"
            out["predicted_has_12h"] = True
            out["period_12h_est"] = period_12_est
        elif sig_24:
            out["predicted_class"] = "circadian_only"
        return out

    return out


# ============================================================================
# Helper: classify BHDT output
# ============================================================================

def _classify_bhdt(bhdt_result):
    """Extract classification from bhdt_analytic output.

    Returns
    -------
    dict
        predicted_class, predicted_has_12h, period_12h_est, log_bf
    """
    cls = bhdt_result["classification"]
    has_12h = cls in (
        "independent_ultradian",
        "likely_independent_ultradian",
    )
    period_est = np.nan
    if has_12h:
        dev = bhdt_result.get("period_deviation", {})
        period_est = dev.get("T_12_fitted", 12.0)
    return {
        "predicted_class": cls,
        "predicted_has_12h": has_12h,
        "period_12h_est": period_est,
        "log_bf": bhdt_result["log_bayes_factor"],
    }


# ============================================================================
# Derive ground-truth classification label from truth dict
# ============================================================================

def _true_class(truth):
    """Map a scenario truth dict to a classification label.

    Returns
    -------
    str
        'independent_ultradian', 'harmonic', 'circadian_only', or 'non_rhythmic'
    """
    if truth["has_independent_12h"]:
        return "independent_ultradian"
    if truth["has_harmonic_12h"]:
        return "harmonic"
    # Check if any oscillator is circadian
    for osc in truth.get("oscillators", []):
        T = osc.get("T", osc.get("T_start", 0))
        if 20 <= T <= 28:
            return "circadian_only"
    return "non_rhythmic"


# ============================================================================
# Main benchmark runner
# ============================================================================

_DEFAULT_METHODS = ["bhdt", "lomb_scargle", "cosinor_12h", "harmonic_regression"]


def run_benchmark(
    n_replicates=10,
    seed=42,
    methods=None,
    include_pinod=False,
    verbose=True,
):
    """Run the full benchmark across all 12 scenarios.

    Parameters
    ----------
    n_replicates : int
        Number of noise replicates per scenario.
    seed : int
        Base random seed.
    methods : list of str, optional
        Methods to run. Default: bhdt, lomb_scargle, cosinor_12h,
        harmonic_regression.
    include_pinod : bool
        If True, also run PINOD (slow). Default False.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per (scenario, replicate, method) with columns:
        scenario_id, scenario_name, replicate, method, predicted_class,
        true_class, true_has_12h, predicted_has_12h, period_12h_est,
        p_value, log_bf, runtime_s.
    """
    if methods is None:
        methods = list(_DEFAULT_METHODS)
    if include_pinod and "pinod" not in methods:
        methods.append("pinod")

    if verbose:
        print(f"Benchmark: {len(methods)} methods x 12 scenarios x "
              f"{n_replicates} replicates")
        print(f"Methods: {methods}")

    # Generate all data upfront
    all_data = generate_all_scenarios(seed=seed, n_replicates=n_replicates)

    rows = []
    total = len(all_data) * len(methods)
    done = 0

    for data in all_data:
        t = data["t"]
        y = data["y"]
        truth = data["truth"]
        sid = truth["scenario_id"]
        rep = truth["replicate"]
        sname = SCENARIO_NAMES.get(sid, truth.get("scenario", "unknown"))
        tc = _true_class(truth)
        true_has_12h = truth["has_independent_12h"]

        for method in methods:
            t0 = time.time()
            row = {
                "scenario_id": sid,
                "scenario_name": sname,
                "replicate": rep,
                "method": method,
                "true_class": tc,
                "true_has_12h": true_has_12h,
                "predicted_class": "error",
                "predicted_has_12h": False,
                "period_12h_est": np.nan,
                "p_value": np.nan,
                "log_bf": np.nan,
                "runtime_s": np.nan,
            }

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if method == "bhdt":
                        res = bhdt_analytic(t, y)
                        cls = _classify_bhdt(res)
                        row["predicted_class"] = cls["predicted_class"]
                        row["predicted_has_12h"] = cls["predicted_has_12h"]
                        row["period_12h_est"] = cls["period_12h_est"]
                        row["log_bf"] = cls["log_bf"]

                    elif method == "lomb_scargle":
                        res = lomb_scargle(t, y)
                        cls = _classify_classical(res, "lomb_scargle")
                        row.update(cls)

                    elif method == "cosinor_12h":
                        res = cosinor(t, y, period=12.0)
                        cls = _classify_classical(res, "cosinor")
                        row.update(cls)

                    elif method == "harmonic_regression":
                        res = harmonic_regression(t, y, periods=[24, 12, 8])
                        cls = _classify_classical(res, "harmonic_regression")
                        row.update(cls)

                    elif method == "pinod":
                        # Optional: PINOD is slow, skip by default
                        try:
                            from chord.pinod.decompose import decompose
                            from chord.pinod.analysis import classify_gene_pinod
                            res = decompose(y.reshape(1, -1), t)
                            analysis = res.iloc[0].to_dict()
                            cls_res = classify_gene_pinod(analysis)
                            row["predicted_class"] = cls_res.get(
                                "classification", "error"
                            )
                            row["predicted_has_12h"] = cls_res.get(
                                "has_independent_12h", False
                            )
                            row["p_value"] = cls_res.get("p_value", np.nan)
                        except ImportError:
                            row["predicted_class"] = "not_available"

            except Exception as e:
                row["predicted_class"] = "error"
                if verbose:
                    print(f"  ERROR scenario {sid} rep {rep} {method}: {e}")

            row["runtime_s"] = time.time() - t0
            rows.append(row)

            done += 1
            if verbose and done % 48 == 0:
                print(f"  Progress: {done}/{total} "
                      f"({100 * done / total:.0f}%)")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Benchmark complete: {len(df)} results collected.")
    return df


# ============================================================================
# Summary statistics
# ============================================================================

def summarize_benchmark(results_df):
    """Compute summary metrics from benchmark results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_benchmark().

    Returns
    -------
    dict with keys:
        per_method : dict of {method: {accuracy, sensitivity, specificity, ...}}
        per_scenario : pd.DataFrame (scenario x method accuracy)
        overall : dict of aggregate stats
    """
    df = results_df.copy()
    methods = sorted(df["method"].unique())

    # --- Per-method 12h detection metrics ---
    per_method = {}
    for m in methods:
        mdf = df[df["method"] == m]
        y_true = mdf["true_has_12h"].astype(int).values
        y_pred = mdf["predicted_has_12h"].astype(int).values
        hda = harmonic_disentangle_accuracy(y_true, y_pred)

        # Multi-class accuracy
        cls_met = classification_metrics(
            mdf["true_class"].values, mdf["predicted_class"].values
        )

        per_method[m] = {
            "accuracy_12h": hda["accuracy"],
            "sensitivity_12h": hda["sensitivity"],
            "specificity_12h": hda["specificity"],
            "tp": hda["tp"],
            "tn": hda["tn"],
            "fp": hda["fp"],
            "fn": hda["fn"],
            "multiclass_accuracy": cls_met["accuracy"],
            "macro_f1": cls_met["macro_f1"],
            "mean_runtime_s": float(mdf["runtime_s"].mean()),
        }

    # --- Per-scenario accuracy breakdown ---
    scenario_rows = []
    for sid in sorted(df["scenario_id"].unique()):
        sdf = df[df["scenario_id"] == sid]
        sname = sdf["scenario_name"].iloc[0]
        row = {"scenario_id": sid, "scenario_name": sname}
        for m in methods:
            mdf = sdf[sdf["method"] == m]
            if len(mdf) > 0:
                correct = (
                    mdf["predicted_has_12h"] == mdf["true_has_12h"]
                ).sum()
                row[f"{m}_12h_acc"] = correct / len(mdf)
            else:
                row[f"{m}_12h_acc"] = np.nan
        scenario_rows.append(row)
    per_scenario = pd.DataFrame(scenario_rows)

    # --- Overall ---
    overall = {}
    for m in methods:
        overall[m] = per_method[m]["accuracy_12h"]

    # --- Tiered reporting (fairness correction) ---
    # Scenarios where the 12h component is a harmonic artifact, not independent.
    # Classical methods cannot distinguish harmonic from independent 12h, so
    # including these scenarios in their binary accuracy is unfair.
    _HARMONIC_SCENARIO_NAMES = {"sawtooth_harmonic", "peaked_harmonic"}

    classical_methods = {"lomb_scargle", "cosinor_12h", "harmonic_regression"}

    # Tier-1: binary 12h detection — fair comparison across all methods.
    # For classical methods, exclude the harmonic-ambiguous scenarios.
    # For BHDT, include all scenarios.
    tier1 = {}
    for m in methods:
        mdf = df[df["method"] == m]
        if m in classical_methods:
            mdf = mdf[~mdf["scenario_name"].isin(_HARMONIC_SCENARIO_NAMES)]
        y_true = mdf["true_has_12h"].astype(int).values
        y_pred = mdf["predicted_has_12h"].astype(int).values
        hda = harmonic_disentangle_accuracy(y_true, y_pred)
        tier1[m] = {
            "accuracy": hda["accuracy"],
            "sensitivity": hda["sensitivity"],
            "specificity": hda["specificity"],
            "n_samples": len(mdf),
        }

    # Tier-2: harmonic disentanglement — only methods that can distinguish
    # harmonic from independent 12h (currently only BHDT).
    tier2 = {}
    for m in methods:
        if m in classical_methods:
            continue  # classical methods cannot do 4-class disentanglement
        mdf = df[df["method"] == m]
        cls_met = classification_metrics(
            mdf["true_class"].values, mdf["predicted_class"].values
        )
        tier2[m] = {
            "accuracy": cls_met["accuracy"],
            "macro_f1": cls_met["macro_f1"],
            "confusion_matrix": cls_met["confusion_matrix"],
            "labels": cls_met["labels"],
            "n_samples": len(mdf),
        }

    return {
        "per_method": per_method,
        "per_scenario": per_scenario,
        "overall": overall,
        "tier1_binary_12h": tier1,
        "tier2_disentangle": tier2,
    }


# ============================================================================
# Real-data benchmark
# ============================================================================

_REAL_DATA_DATASETS = {
    "hughes2009": {
        "description": "Hughes 2009 mouse liver microarray (GSE11923)",
        "organism": "mouse",
    },
    "zhu2023_wt": {
        "description": "Zhu 2023 WT mouse liver RNA-seq (GSE171975)",
        "organism": "mouse",
    },
    "zhu2023_ko": {
        "description": "Zhu 2023 BMAL1 KO mouse liver RNA-seq (GSE171975)",
        "organism": "mouse",
    },
    "mure2018": {
        "description": "Mure 2018 baboon liver RNA-seq (GSE98965)",
        "organism": "primate",
    },
}


def _assign_ground_truth(gene_name, organism="mouse"):
    """Assign ground-truth class for a gene based on known gene lists.

    Returns
    -------
    tuple (true_class, true_has_12h)
        true_class : str or None
            'known_12h', 'core_circadian', 'housekeeping', or None (unknown)
        true_has_12h : bool or None
            True for known 12h genes, False for circadian/housekeeping,
            None for genes without ground truth.
    """
    from chord.data.known_genes import (
        KNOWN_12H_GENES_ZHU2017,
        CORE_CIRCADIAN_GENES,
        NON_RHYTHMIC_HOUSEKEEPING,
        KNOWN_12H_GENES_PRIMATE,
        CORE_CIRCADIAN_GENES_PRIMATE,
        NON_RHYTHMIC_HOUSEKEEPING_PRIMATE,
    )

    if organism == "mouse":
        if gene_name in KNOWN_12H_GENES_ZHU2017:
            return "known_12h", True
        if gene_name in CORE_CIRCADIAN_GENES:
            return "core_circadian", False
        if gene_name in NON_RHYTHMIC_HOUSEKEEPING:
            return "housekeeping", False
    elif organism == "primate":
        if gene_name in KNOWN_12H_GENES_PRIMATE:
            return "known_12h", True
        if gene_name in CORE_CIRCADIAN_GENES_PRIMATE:
            return "core_circadian", False
        if gene_name in NON_RHYTHMIC_HOUSEKEEPING_PRIMATE:
            return "housekeeping", False

    return None, None


def run_real_data_benchmark(
    datasets=None,
    methods=None,
    cache_dir="~/.chord_cache",
    verbose=True,
    classifier_version="v2",
):
    """Run benchmark on real public GEO datasets using known gene lists.

    For each dataset, runs all methods on genes that have ground-truth
    labels (known 12h genes, core circadian genes, housekeeping genes).
    This produces a DataFrame with the same schema as run_benchmark()
    so the same summary/plotting functions can be reused.

    Parameters
    ----------
    datasets : list of str, optional
        Dataset keys to include. Default: all available.
        Options: 'hughes2009', 'zhu2023_wt', 'zhu2023_ko', 'mure2018'
    methods : list of str, optional
        Methods to run. Default: bhdt, lomb_scargle, cosinor_12h,
        harmonic_regression.
    cache_dir : str
        Directory for caching downloaded GEO data.
    verbose : bool
        Print progress.
    classifier_version : str
        "v1" for original hard-gate classifier, "v2" for soft-gate (default).

    Returns
    -------
    pd.DataFrame
        One row per (dataset, gene, method) with columns:
        dataset, gene_name, method, predicted_class, predicted_has_12h,
        true_class, true_has_12h, p_value, log_bf, runtime_s,
        scenario_id, scenario_name, replicate.
    """
    if datasets is None:
        datasets = list(_REAL_DATA_DATASETS.keys())
    if methods is None:
        methods = list(_DEFAULT_METHODS)

    rows = []

    for ds_key in datasets:
        if ds_key not in _REAL_DATA_DATASETS:
            raise ValueError(
                f"Unknown dataset '{ds_key}'. "
                f"Available: {list(_REAL_DATA_DATASETS.keys())}"
            )

        ds_info = _REAL_DATA_DATASETS[ds_key]
        organism = ds_info["organism"]

        if verbose:
            print(f"\n--- Loading {ds_info['description']} ---")

        # Load data
        try:
            expr, timepoints, gene_names = _load_dataset(
                ds_key, cache_dir
            )
        except Exception as e:
            if verbose:
                print(f"  SKIP {ds_key}: {e}")
            continue

        if verbose:
            print(f"  Loaded {len(gene_names)} genes, "
                  f"{len(timepoints)} timepoints")

        # Filter to genes with ground truth
        gt_genes = {}
        for gname in gene_names:
            tc, th = _assign_ground_truth(gname, organism)
            if tc is not None:
                gt_genes[gname] = (tc, th)

        if not gt_genes:
            if verbose:
                print(f"  No ground-truth genes found in {ds_key}, skipping")
            continue

        if verbose:
            n_12h = sum(1 for _, (_, h) in gt_genes.items() if h)
            n_circ = sum(
                1 for _, (c, _) in gt_genes.items()
                if c == "core_circadian"
            )
            n_hk = sum(
                1 for _, (c, _) in gt_genes.items()
                if c == "housekeeping"
            )
            print(f"  Ground-truth genes: {len(gt_genes)} "
                  f"(12h: {n_12h}, circadian: {n_circ}, "
                  f"housekeeping: {n_hk})")

        # Build gene name -> row index mapping
        name2idx = {g: i for i, g in enumerate(gene_names)}

        # Assign a synthetic scenario_id for compatibility with summarize
        ds_scenario_id = (
            100 + list(_REAL_DATA_DATASETS.keys()).index(ds_key)
        )

        # Run each method on each ground-truth gene
        total = len(gt_genes) * len(methods)
        done = 0

        for gname, (true_cls, true_has_12h) in gt_genes.items():
            idx = name2idx[gname]
            y = expr[idx]
            t = timepoints

            for method in methods:
                t0 = time.time()
                row = {
                    "dataset": ds_key,
                    "gene_name": gname,
                    "scenario_id": ds_scenario_id,
                    "scenario_name": ds_key,
                    "replicate": 0,
                    "method": method,
                    "true_class": true_cls,
                    "true_has_12h": true_has_12h,
                    "predicted_class": "error",
                    "predicted_has_12h": False,
                    "period_12h_est": np.nan,
                    "p_value": np.nan,
                    "log_bf": np.nan,
                    "runtime_s": np.nan,
                }

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        if method == "bhdt":
                            res = bhdt_analytic(t, y, classifier_version=classifier_version)
                            cls = _classify_bhdt(res)
                            row["predicted_class"] = cls["predicted_class"]
                            row["predicted_has_12h"] = cls[
                                "predicted_has_12h"
                            ]
                            row["period_12h_est"] = cls["period_12h_est"]
                            row["log_bf"] = cls["log_bf"]

                        elif method == "lomb_scargle":
                            res = lomb_scargle(t, y)
                            cls = _classify_classical(res, "lomb_scargle")
                            row.update(cls)

                        elif method == "cosinor_12h":
                            res = cosinor(t, y, period=12.0)
                            cls = _classify_classical(res, "cosinor")
                            row.update(cls)

                        elif method == "harmonic_regression":
                            res = harmonic_regression(
                                t, y, periods=[24, 12, 8]
                            )
                            cls = _classify_classical(
                                res, "harmonic_regression"
                            )
                            row.update(cls)

                except Exception as e:
                    row["predicted_class"] = "error"
                    if verbose:
                        print(
                            f"  ERROR {ds_key}/{gname}/{method}: {e}"
                        )

                row["runtime_s"] = time.time() - t0
                rows.append(row)

                done += 1
                if verbose and done % 100 == 0:
                    print(f"  Progress: {done}/{total} "
                          f"({100 * done / total:.0f}%)")

    df = pd.DataFrame(rows)
    if verbose:
        print(
            f"\nReal-data benchmark complete: {len(df)} results collected."
        )
    return df


def _load_dataset(ds_key, cache_dir):
    """Load a dataset and return (expr, timepoints, gene_names).

    Returns
    -------
    tuple (expr: ndarray, timepoints: ndarray, gene_names: list)
    """
    from chord.data.geo_loader import (
        load_hughes2009,
        load_zhu2023_bmal1ko,
        load_mure2018,
    )

    if ds_key == "hughes2009":
        data = load_hughes2009(cache_dir=cache_dir, downsample_2h=True)
        return data["expr"], data["timepoints"], data["gene_names"]

    elif ds_key == "zhu2023_wt":
        data = load_zhu2023_bmal1ko(cache_dir=cache_dir)
        d = data["wt"]
        return d["expr"], d["timepoints"], d["gene_names"]

    elif ds_key == "zhu2023_ko":
        data = load_zhu2023_bmal1ko(cache_dir=cache_dir)
        d = data["ko"]
        return d["expr"], d["timepoints"], d["gene_names"]

    elif ds_key == "mure2018":
        data = load_mure2018(cache_dir=cache_dir, tissue="LIV")
        return data["expr"], data["timepoints"], data["gene_names"]

    else:
        raise ValueError(f"Unknown dataset: {ds_key}")


def summarize_real_data_benchmark(results_df):
    """Compute summary metrics from real-data benchmark results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_real_data_benchmark().

    Returns
    -------
    dict with keys:
        per_dataset : dict of {dataset: {method: metrics}}
        per_gene_class : dict of {true_class: {method: metrics}}
        overall : dict of {method: overall_accuracy}
        concordance : dict of cross-method agreement stats
    """
    df = results_df.copy()
    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())

    # --- Per-dataset, per-method metrics ---
    per_dataset = {}
    for ds in datasets:
        per_dataset[ds] = {}
        ddf = df[df["dataset"] == ds]
        for m in methods:
            mdf = ddf[ddf["method"] == m]
            if len(mdf) == 0:
                continue
            y_true = mdf["true_has_12h"].astype(int).values
            y_pred = mdf["predicted_has_12h"].astype(int).values
            hda = harmonic_disentangle_accuracy(y_true, y_pred)
            per_dataset[ds][m] = {
                "accuracy": hda["accuracy"],
                "sensitivity": hda["sensitivity"],
                "specificity": hda["specificity"],
                "tp": hda["tp"],
                "tn": hda["tn"],
                "fp": hda["fp"],
                "fn": hda["fn"],
                "n_genes": len(mdf),
                "mean_runtime_s": float(mdf["runtime_s"].mean()),
            }

    # --- Per gene class (known_12h, core_circadian, housekeeping) ---
    per_gene_class = {}
    for tc in sorted(df["true_class"].unique()):
        per_gene_class[tc] = {}
        tcdf = df[df["true_class"] == tc]
        for m in methods:
            mdf = tcdf[tcdf["method"] == m]
            if len(mdf) == 0:
                continue
            y_true = mdf["true_has_12h"].astype(int).values
            y_pred = mdf["predicted_has_12h"].astype(int).values
            hda = harmonic_disentangle_accuracy(y_true, y_pred)
            per_gene_class[tc][m] = {
                "accuracy": hda["accuracy"],
                "sensitivity": hda["sensitivity"],
                "specificity": hda["specificity"],
                "n_genes": len(mdf),
            }

    # --- Overall per-method ---
    overall = {}
    for m in methods:
        mdf = df[df["method"] == m]
        if len(mdf) == 0:
            continue
        y_true = mdf["true_has_12h"].astype(int).values
        y_pred = mdf["predicted_has_12h"].astype(int).values
        hda = harmonic_disentangle_accuracy(y_true, y_pred)
        overall[m] = {
            "accuracy": hda["accuracy"],
            "sensitivity": hda["sensitivity"],
            "specificity": hda["specificity"],
            "n_genes": len(mdf),
        }

    # --- Cross-method concordance (BHDT vs each classical method) ---
    concordance = {}
    if "bhdt" in methods:
        bhdt_df = df[df["method"] == "bhdt"].set_index(
            ["dataset", "gene_name"]
        )
        for m in methods:
            if m == "bhdt":
                continue
            m_df = df[df["method"] == m].set_index(
                ["dataset", "gene_name"]
            )
            common = bhdt_df.index.intersection(m_df.index)
            if len(common) == 0:
                continue
            agree = (
                bhdt_df.loc[common, "predicted_has_12h"].values
                == m_df.loc[common, "predicted_has_12h"].values
            ).sum()
            concordance[f"bhdt_vs_{m}"] = {
                "agreement_rate": agree / len(common),
                "n_common": len(common),
            }

    return {
        "per_dataset": per_dataset,
        "per_gene_class": per_gene_class,
        "overall": overall,
        "concordance": concordance,
    }


def _print_real_data_summary(summary):
    """Pretty-print real-data benchmark summary."""
    print("\n" + "=" * 70)
    print("Real-Data Benchmark Summary")
    print("=" * 70)

    # Overall
    print("\n--- Overall 12h Detection Accuracy ---")
    header = (
        f"{'Method':<25s} {'Acc':>8s} {'Sens':>8s} "
        f"{'Spec':>8s} {'N':>6s}"
    )
    print(header)
    print("-" * len(header))
    for m, stats in sorted(summary["overall"].items()):
        print(f"{m:<25s} {stats['accuracy']:>8.3f} "
              f"{stats['sensitivity']:>8.3f} "
              f"{stats['specificity']:>8.3f} "
              f"{stats['n_genes']:>6d}")

    # Per dataset
    for ds, ds_stats in sorted(summary["per_dataset"].items()):
        print(f"\n--- {ds} ---")
        for m, stats in sorted(ds_stats.items()):
            print(f"  {m:<23s} acc={stats['accuracy']:.3f} "
                  f"sens={stats['sensitivity']:.3f} "
                  f"spec={stats['specificity']:.3f} "
                  f"(n={stats['n_genes']})")

    # Per gene class
    print("\n--- Per Gene Class ---")
    for tc, tc_stats in sorted(summary["per_gene_class"].items()):
        print(f"  {tc}:")
        for m, stats in sorted(tc_stats.items()):
            print(f"    {m:<23s} acc={stats['accuracy']:.3f} "
                  f"(n={stats['n_genes']})")

    # Concordance
    if summary["concordance"]:
        print("\n--- Cross-Method Concordance ---")
        for pair, stats in sorted(summary["concordance"].items()):
            print(f"  {pair}: {stats['agreement_rate']:.3f} "
                  f"(n={stats['n_common']})")

    print("=" * 70)


# ============================================================================
# Pretty-print helpers
# ============================================================================

def _print_method_table(summary):
    """Print per-method summary as a formatted table."""
    pm = summary["per_method"]
    header = (
        f"{'Method':<25s} {'12h Acc':>8s} {'Sens':>8s} "
        f"{'Spec':>8s} {'F1':>8s} {'Time(s)':>8s}"
    )
    print("\n" + "=" * len(header))
    print("Per-Method Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m, stats in sorted(pm.items()):
        print(
            f"{m:<25s} {stats['accuracy_12h']:>8.3f} "
            f"{stats['sensitivity_12h']:>8.3f} "
            f"{stats['specificity_12h']:>8.3f} "
            f"{stats['macro_f1']:>8.3f} "
            f"{stats['mean_runtime_s']:>8.4f}"
        )
    print("=" * len(header))


def _print_scenario_table(summary):
    """Print per-scenario accuracy breakdown."""
    ps = summary["per_scenario"]
    acc_cols = [c for c in ps.columns if c.endswith("_12h_acc")]
    method_names = [c.replace("_12h_acc", "") for c in acc_cols]

    col_w = 10
    header = f"{'ID':>3s} {'Scenario':<28s}"
    for mn in method_names:
        header += f" {mn:>{col_w}s}"
    print("\n" + "=" * len(header))
    print("Per-Scenario 12h Detection Accuracy")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for _, row in ps.iterrows():
        line = f"{int(row['scenario_id']):>3d} {row['scenario_name']:<28s}"
        for c in acc_cols:
            val = row[c]
            if np.isnan(val):
                line += f" {'N/A':>{col_w}s}"
            else:
                line += f" {val:>{col_w}.3f}"
        print(line)
    print("=" * len(header))


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CHORD benchmark: compare BHDT vs classical methods"
    )
    parser.add_argument(
        "--reps", type=int, default=10,
        help="Number of replicates per scenario (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--pinod", action="store_true",
        help="Include PINOD (slow)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results DataFrame to CSV",
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="Run real-data benchmark using public GEO datasets",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Datasets for real-data benchmark (default: all). "
             "Options: hughes2009, zhu2023_wt, zhu2023_ko, mure2018",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="~/.chord_cache",
        help="Cache directory for GEO data (default: ~/.chord_cache)",
    )
    args = parser.parse_args()

    print("CHORD Benchmark Runner")
    print("=" * 50)

    if args.real_data:
        results = run_real_data_benchmark(
            datasets=args.datasets,
            cache_dir=args.cache_dir,
            verbose=True,
        )
        summary = summarize_real_data_benchmark(results)
        _print_real_data_summary(summary)
    else:
        results = run_benchmark(
            n_replicates=args.reps,
            seed=args.seed,
            include_pinod=args.pinod,
            verbose=True,
        )
        summary = summarize_benchmark(results)
        _print_method_table(summary)
        _print_scenario_table(summary)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    else:
        print("\nTip: use --output results.csv to save full results.")
