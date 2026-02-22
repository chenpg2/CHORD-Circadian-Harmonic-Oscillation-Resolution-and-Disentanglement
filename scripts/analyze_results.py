#!/usr/bin/env python3
"""
CHORD Benchmark — Comprehensive Analysis & Figure Generator.

Generates:
  1. Version evolution comparison (v6→v9 synthetic)
  2. Per-scenario performance heatmap (v9)
  3. Real data analysis (when results available)
  4. Cross-method comparison on real data

Usage:
    python analyze_results.py [--real]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# Constants
# ============================================================================

SINGLE_COL = 3.5
DOUBLE_COL = 7.0

# Wong 2011 colorblind-friendly palette
PALETTE = {
    "CHORD":                "#D55E00",
    "Lomb_Scargle":         "#0072B2",
    "Cosinor":              "#009E73",
    "Harmonic_Regression":  "#CC79A7",
    "JTK_CYCLE":            "#E69F00",
    "RAIN":                 "#56B4E9",
    "Pencil":               "#999999",
}

VERSION_COLORS = {
    "v6": "#56B4E9",   # sky blue
    "v7": "#009E73",   # green
    "v8": "#E69F00",   # orange
    "v9": "#D55E00",   # vermilion (highlight)
}

SCENARIO_NAMES = {
    1: "pure_circadian",
    2: "pure_ultradian",
    3: "indep_superposition",
    4: "sawtooth_harmonic",
    5: "peaked_harmonic",
    6: "indep_multi_ultradian",
    7: "damped_ultradian",
    8: "asymmetric_ultradian",
    9: "pure_noise",
    10: "trend_noise",
    11: "low_snr_ultradian",
    12: "drifting_ultradian",
    13: "square_wave_harmonic",
    14: "bimodal_circadian",
    15: "pulse_circadian",
}

SCENARIO_GROUND_TRUTH = {
    1:  {"has_12h": False, "has_harmonic": False, "type": "non_rhythmic"},
    2:  {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    3:  {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    4:  {"has_12h": False, "has_harmonic": True,  "type": "harmonic"},
    5:  {"has_12h": False, "has_harmonic": True,  "type": "harmonic"},
    6:  {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    7:  {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    8:  {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    9:  {"has_12h": False, "has_harmonic": False, "type": "non_rhythmic"},
    10: {"has_12h": False, "has_harmonic": False, "type": "non_rhythmic"},
    11: {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    12: {"has_12h": True,  "has_harmonic": False, "type": "independent"},
    13: {"has_12h": False, "has_harmonic": True,  "type": "harmonic"},
    14: {"has_12h": False, "has_harmonic": True,  "type": "harmonic"},
    15: {"has_12h": False, "has_harmonic": True,  "type": "harmonic"},
}

BASE_DIR = Path("scripts/chord/results")
OUTPUT_DIR = BASE_DIR / "analysis_figures"


def _setup_style():
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    })


def _save(fig, stem, formats=("pdf", "png")):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(stem.with_suffix(f".{fmt}")))


def _panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


# ============================================================================
# Load data
# ============================================================================

def load_version_data():
    """Load synthetic results for all versions."""
    versions = {}
    paths = {
        "v6": BASE_DIR / "benchmark" / "synthetic_results.csv",
        "v7": BASE_DIR / "benchmark_v7" / "synthetic_results.csv",
        "v8": BASE_DIR / "benchmark_v8" / "synthetic_results.csv",
        "v9": BASE_DIR / "benchmark_v9" / "synthetic_results.csv",
    }
    for v, p in paths.items():
        if p.exists():
            df = pd.read_csv(p)
            # Only keep CHORD rows
            df = df[df["method"] == "CHORD"].copy()
            df["version"] = v
            versions[v] = df
            print(f"  Loaded {v}: {len(df)} rows")
    return versions


def compute_per_scenario_metrics(df):
    """Compute detection and disentanglement metrics per scenario."""
    results = []
    for sc in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == sc]
        gt = SCENARIO_GROUND_TRUTH.get(sc, {})
        gt_has_12h = gt.get("has_12h", False)
        gt_has_harmonic = gt.get("has_harmonic", False)
        gt_type = gt.get("type", "unknown")
        n = len(sub)

        # Detection: did we detect 12h?
        detected = sub["has_12h"].astype(bool).sum()
        detection_rate = detected / n if n > 0 else 0

        # Classification accuracy
        correct = 0
        wrong = 0
        ambig = 0
        for _, row in sub.iterrows():
            cls = row["classification"]
            if gt_type == "non_rhythmic":
                if cls in ("non_rhythmic", "circadian_only"):
                    correct += 1
                elif cls == "ambiguous":
                    ambig += 1
                else:
                    wrong += 1
            elif gt_type == "independent":
                if cls in ("independent_ultradian", "likely_independent_ultradian"):
                    correct += 1
                elif cls == "ambiguous":
                    ambig += 1
                else:
                    wrong += 1
            elif gt_type == "harmonic":
                if cls in ("harmonic", "likely_harmonic"):
                    correct += 1
                elif cls == "ambiguous":
                    ambig += 1
                else:
                    wrong += 1

        results.append({
            "scenario": sc,
            "name": SCENARIO_NAMES.get(sc, f"sc_{sc}"),
            "gt_type": gt_type,
            "n": n,
            "detected": detected,
            "detection_rate": detection_rate,
            "correct": correct,
            "wrong": wrong,
            "ambiguous": ambig,
            "accuracy": correct / n if n > 0 else 0,
            "accuracy_of_detected": correct / detected if detected > 0 else 0,
        })
    return pd.DataFrame(results)


def compute_tier_metrics(df):
    """Compute Tier-1 and Tier-2 metrics."""
    # Tier-1: detection
    gt_positive = df["gt_has_12h"].astype(bool) | df["gt_has_harmonic"].astype(bool)
    pred_positive = df["has_12h"].astype(bool)

    tp = (pred_positive & gt_positive).sum()
    fp = (pred_positive & ~gt_positive).sum()
    fn = (~pred_positive & gt_positive).sum()
    tn = (~pred_positive & ~gt_positive).sum()

    t1_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    t1_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    t1_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    t1_f1 = 2 * t1_prec * t1_sens / (t1_prec + t1_sens) if (t1_prec + t1_sens) > 0 else 0

    # Tier-2: disentanglement (among detected genes with 12h component)
    has_any_12h = gt_positive
    detected = pred_positive & has_any_12h
    det_df = df[detected].copy()

    if len(det_df) > 0:
        gt_indep = det_df["gt_has_12h"].astype(bool) & ~det_df["gt_has_harmonic"].astype(bool)
        pred_indep = det_df["classification"].isin(["independent_ultradian", "likely_independent_ultradian"])

        t2_tp = (pred_indep & gt_indep).sum()
        t2_fp = (pred_indep & ~gt_indep).sum()
        t2_fn = (~pred_indep & gt_indep).sum()
        t2_tn = (~pred_indep & ~gt_indep).sum()

        t2_sens = t2_tp / (t2_tp + t2_fn) if (t2_tp + t2_fn) > 0 else 0
        t2_spec = t2_tn / (t2_tn + t2_fp) if (t2_tn + t2_fp) > 0 else 0
        t2_prec = t2_tp / (t2_tp + t2_fp) if (t2_tp + t2_fp) > 0 else 0
        t2_f1 = 2 * t2_prec * t2_sens / (t2_prec + t2_sens) if (t2_prec + t2_sens) > 0 else 0
    else:
        t2_sens = t2_spec = t2_prec = t2_f1 = 0

    return {
        "tier1": {"sensitivity": t1_sens, "specificity": t1_spec, "precision": t1_prec, "f1": t1_f1},
        "tier2": {"sensitivity": t2_sens, "specificity": t2_spec, "precision": t2_prec, "f1": t2_f1},
    }


# ============================================================================
# Figure A: Version Evolution (v6→v9)
# ============================================================================

def fig_version_evolution(versions, output_dir):
    """Bar chart comparing v6→v9 Tier-1 and Tier-2 metrics."""
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2))

    version_order = ["v6", "v7", "v8", "v9"]
    metrics_data = {"tier1": [], "tier2": []}

    for v in version_order:
        if v not in versions:
            continue
        m = compute_tier_metrics(versions[v])
        metrics_data["tier1"].append(m["tier1"])
        metrics_data["tier2"].append(m["tier2"])

    available_versions = [v for v in version_order if v in versions]

    for ax_idx, (tier, title) in enumerate([
        ("tier1", "Stage 1: 12h Detection"),
        ("tier2", "Stage 2: Disentanglement"),
    ]):
        ax = axes[ax_idx]
        metric_names = ["sensitivity", "specificity", "precision", "f1"]
        metric_labels = ["Sens", "Spec", "Prec", "F1"]
        x = np.arange(len(metric_names))
        width = 0.18
        n_versions = len(available_versions)

        for i, v in enumerate(available_versions):
            vals = [metrics_data[tier][i][m] for m in metric_names]
            offset = (i - n_versions / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                         color=VERSION_COLORS[v], label=v,
                         edgecolor="white", linewidth=0.3)
            # Value labels on bars
            for bar, val in zip(bars, vals):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        if ax_idx == 0:
            ax.set_ylabel("Score")
        _panel_label(ax, chr(ord("a") + ax_idx))

    plt.tight_layout(w_pad=2.0)
    _save(fig, output_dir / "fig_version_evolution")
    plt.close(fig)
    print("  Version evolution figure saved")


# ============================================================================
# Figure B: Per-Scenario Heatmap (v9)
# ============================================================================

def fig_scenario_heatmap(df, output_dir, version="v9"):
    """Heatmap showing detection rate and classification accuracy per scenario."""
    _setup_style()

    sc_metrics = compute_per_scenario_metrics(df)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 1.1, 4.5),
                              gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})

    scenarios = sc_metrics["scenario"].values
    names = sc_metrics["name"].values
    types = sc_metrics["gt_type"].values

    # Color-code scenario names by type
    type_colors = {"non_rhythmic": "#999999", "independent": "#D55E00", "harmonic": "#0072B2"}

    # Panel A: Detection rate
    ax = axes[0]
    det_rates = sc_metrics["detection_rate"].values
    colors = ["#D55E00" if t in ("independent",) else "#0072B2" if t == "harmonic" else "#999999"
              for t in types]
    y_pos = np.arange(len(scenarios))
    bars = ax.barh(y_pos, det_rates, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"S{s}: {n}" for s, n in zip(scenarios, names)], fontsize=5.5)
    ax.set_xlabel("Detection rate")
    ax.set_title("12h Detection", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    for bar, val in zip(bars, det_rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=5.5)
    _panel_label(ax, "a", x=-0.35)

    # Panel B: Classification accuracy (of detected)
    ax = axes[1]
    acc = sc_metrics["accuracy"].values
    bars = ax.barh(y_pos, acc, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Classification accuracy")
    ax.set_title("Correct Classification", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    for bar, val in zip(bars, acc):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=5.5)
    _panel_label(ax, "b", x=-0.15)

    # Panel C: Error breakdown (wrong vs ambiguous)
    ax = axes[2]
    wrong = sc_metrics["wrong"].values
    ambig = sc_metrics["ambiguous"].values
    n_total = sc_metrics["n"].values

    ax.barh(y_pos, wrong, color="#E63946", edgecolor="white", linewidth=0.3,
            height=0.7, label="Wrong")
    ax.barh(y_pos, ambig, left=wrong, color="#F4A261", edgecolor="white",
            linewidth=0.3, height=0.7, label="Ambiguous")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Count (of 50)")
    ax.set_title("Errors", fontsize=8)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")

    for i, (w, a) in enumerate(zip(wrong, ambig)):
        if w + a > 0:
            ax.text(w + a + 0.3, i, f"{w}+{a}", va="center", fontsize=5)
    _panel_label(ax, "c", x=-0.15)

    plt.tight_layout(w_pad=1.5)
    _save(fig, output_dir / f"fig_scenario_heatmap_{version}")
    plt.close(fig)
    print(f"  Scenario heatmap ({version}) saved")


# ============================================================================
# Figure C: Version Comparison Per-Scenario (v6 vs v9)
# ============================================================================

def fig_version_scenario_comparison(versions, output_dir):
    """Side-by-side per-scenario accuracy for v6 vs v9."""
    _setup_style()

    if "v6" not in versions or "v9" not in versions:
        print("  Skipping version comparison (need v6 and v9)")
        return

    sc_v6 = compute_per_scenario_metrics(versions["v6"])
    sc_v9 = compute_per_scenario_metrics(versions["v9"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))

    scenarios = sc_v9["scenario"].values
    names = sc_v9["name"].values
    y_pos = np.arange(len(scenarios))
    height = 0.35

    acc_v6 = sc_v6.set_index("scenario")["accuracy"]
    acc_v9 = sc_v9.set_index("scenario")["accuracy"]

    bars1 = ax.barh(y_pos - height/2, [acc_v6.get(s, 0) for s in scenarios],
                    height, color=VERSION_COLORS["v6"], label="v6",
                    edgecolor="white", linewidth=0.3)
    bars2 = ax.barh(y_pos + height/2, [acc_v9.get(s, 0) for s in scenarios],
                    height, color=VERSION_COLORS["v9"], label="v9",
                    edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"S{s}: {n}" for s, n in zip(scenarios, names)], fontsize=6)
    ax.set_xlabel("Classification accuracy")
    ax.set_title("v6 → v9 Improvement Per Scenario", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # Annotate improvements
    for i, s in enumerate(scenarios):
        v6_val = acc_v6.get(s, 0)
        v9_val = acc_v9.get(s, 0)
        diff = v9_val - v6_val
        if abs(diff) > 0.01:
            color = "#009E73" if diff > 0 else "#E63946"
            ax.text(max(v6_val, v9_val) + 0.02, i,
                    f"{diff:+.0%}", va="center", fontsize=5.5, color=color,
                    fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir / "fig_version_scenario_comparison")
    plt.close(fig)
    print("  Version scenario comparison saved")


# ============================================================================
# Figure D: Tier-2 Evolution Detail
# ============================================================================

def fig_tier2_evolution(versions, output_dir):
    """Line chart showing Tier-2 metrics evolution across versions."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, SINGLE_COL))

    version_order = ["v6", "v7", "v8", "v9"]
    available = [v for v in version_order if v in versions]

    metrics = ["sensitivity", "specificity", "precision", "f1"]
    metric_colors = {"sensitivity": "#D55E00", "specificity": "#0072B2",
                     "precision": "#009E73", "f1": "#CC79A7"}
    metric_markers = {"sensitivity": "o", "specificity": "s",
                      "precision": "^", "f1": "D"}

    for metric in metrics:
        vals = []
        for v in available:
            m = compute_tier_metrics(versions[v])
            vals.append(m["tier2"][metric])
        ax.plot(range(len(available)), vals,
                color=metric_colors[metric],
                marker=metric_markers[metric],
                label=metric.capitalize(),
                linewidth=1.5, markersize=5)

        # Annotate last point
        ax.text(len(available) - 1 + 0.1, vals[-1],
                f"{vals[-1]:.3f}", fontsize=5.5, va="center")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available)
    ax.set_xlabel("Version")
    ax.set_ylabel("Score")
    ax.set_title("Tier-2 Disentanglement Evolution", fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    plt.tight_layout()
    _save(fig, output_dir / "fig_tier2_evolution")
    plt.close(fig)
    print("  Tier-2 evolution figure saved")


# ============================================================================
# Text Summary
# ============================================================================

def print_summary(versions):
    """Print comprehensive text summary."""
    print("\n" + "=" * 70)
    print("CHORD Benchmark — Comprehensive Summary")
    print("=" * 70)

    for v in ["v6", "v7", "v8", "v9"]:
        if v not in versions:
            continue
        m = compute_tier_metrics(versions[v])
        print(f"\n--- {v.upper()} ---")
        print(f"  Tier-1: Sens={m['tier1']['sensitivity']:.3f}  "
              f"Spec={m['tier1']['specificity']:.3f}  "
              f"Prec={m['tier1']['precision']:.3f}  "
              f"F1={m['tier1']['f1']:.3f}")
        print(f"  Tier-2: Sens={m['tier2']['sensitivity']:.3f}  "
              f"Spec={m['tier2']['specificity']:.3f}  "
              f"Prec={m['tier2']['precision']:.3f}  "
              f"F1={m['tier2']['f1']:.3f}")

    # Per-scenario for v9
    if "v9" in versions:
        print("\n--- V9 Per-Scenario ---")
        sc = compute_per_scenario_metrics(versions["v9"])
        print(f"{'Sc':>3} {'Name':<25} {'Type':<14} {'Det%':>5} {'Acc%':>5} {'Wrong':>5} {'Ambig':>5}")
        print("-" * 70)
        for _, row in sc.iterrows():
            print(f"{row['scenario']:3d} {row['name']:<25} {row['gt_type']:<14} "
                  f"{row['detection_rate']:5.0%} {row['accuracy']:5.0%} "
                  f"{row['wrong']:5.0f} {row['ambiguous']:5.0f}")

    # Improvement summary
    if "v6" in versions and "v9" in versions:
        m6 = compute_tier_metrics(versions["v6"])
        m9 = compute_tier_metrics(versions["v9"])
        print("\n--- v6 → v9 Improvement ---")
        for tier in ["tier1", "tier2"]:
            print(f"  {tier}:")
            for metric in ["sensitivity", "specificity", "precision", "f1"]:
                diff = m9[tier][metric] - m6[tier][metric]
                print(f"    {metric}: {m6[tier][metric]:.3f} → {m9[tier][metric]:.3f} ({diff:+.3f})")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    print("Loading version data...")
    versions = load_version_data()

    if not versions:
        print("ERROR: No synthetic results found!")
        sys.exit(1)

    # Text summary
    print_summary(versions)

    # Generate figures
    print("\nGenerating figures...")

    # Fig A: Version evolution
    if len(versions) > 1:
        fig_version_evolution(versions, OUTPUT_DIR)

    # Fig B: Per-scenario heatmap (v9)
    if "v9" in versions:
        fig_scenario_heatmap(versions["v9"], OUTPUT_DIR, "v9")

    # Fig C: Version comparison per scenario
    if "v6" in versions and "v9" in versions:
        fig_version_scenario_comparison(versions, OUTPUT_DIR)

    # Fig D: Tier-2 evolution
    if len(versions) > 1:
        fig_tier2_evolution(versions, OUTPUT_DIR)

    # Check for real data results
    for v, dirname in [("v6", "benchmark"), ("v7", "benchmark_v7"),
                       ("v8", "benchmark_v8"), ("v9", "benchmark_v9")]:
        real_path = BASE_DIR / dirname / "real_results.csv"
        if real_path.exists():
            print(f"\n  Real data available for {v}: {real_path}")
        else:
            print(f"\n  Real data NOT YET available for {v}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
