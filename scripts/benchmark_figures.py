#!/usr/bin/env python3
"""
CHORD Benchmark — Publication Figure Generator.

Reads CSV results from benchmark_final.py and generates
publication-ready figures for iMeta.

Usage:
    python benchmark_figures.py --input results/benchmark/ --output results/benchmark/figures/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Style
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

METHOD_COLORS = {
    "CHORD": "#E63946",
    "Lomb_Scargle": "#457B9D",
    "Cosinor": "#2A9D8F",
    "Harmonic_Regression": "#E9C46A",
    "JTK_CYCLE": "#F4A261",
    "RAIN": "#264653",
    "Pencil": "#A8DADC",
}

METHOD_ORDER = ["CHORD", "Lomb_Scargle", "Cosinor", "Harmonic_Regression",
                "JTK_CYCLE", "RAIN", "Pencil"]


def fig1_synthetic_roc(synth_df, output_dir):
    """Figure 1: Synthetic data ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (title, use_fair) in enumerate([
        ("A) All scenarios", False),
        ("B) Fair comparison (excl. harmonic)", True),
    ]):
        ax = axes[ax_idx]
        for method in METHOD_ORDER:
            sub = synth_df[synth_df["method"] == method].copy()
            if use_fair and method != "CHORD":
                sub = sub[~sub["gt_has_harmonic"]]

            y_true = sub["gt_has_12h"].astype(int).values
            y_scores = sub["confidence"].values

            if len(np.unique(y_true)) < 2:
                continue

            # Compute ROC manually
            thresholds = np.sort(np.unique(y_scores))[::-1]
            tpr_list, fpr_list = [0.0], [0.0]
            for thresh in thresholds:
                y_pred = (y_scores >= thresh).astype(int)
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            tpr_list.append(1.0)
            fpr_list.append(1.0)

            auc_val = np.trapz(tpr_list, fpr_list)
            color = METHOD_COLORS.get(method, "#999999")
            ax.plot(fpr_list, tpr_list, color=color, linewidth=2,
                    label=f"{method} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_synthetic_roc.pdf")
    fig.savefig(output_dir / "fig1_synthetic_roc.png")
    plt.close(fig)
    print("  Fig 1: Synthetic ROC curves saved")


def fig2_real_heatmap(tier1_df, output_dir):
    """Figure 2: Real data performance heatmap (F1 scores)."""
    if len(tier1_df) == 0:
        print("  Fig 2: Skipped (no tier1 data)")
        return

    pivot = tier1_df.pivot_table(
        index="method", columns="dataset", values="f1", aggfunc="first"
    )
    # Reorder methods
    ordered = [m for m in METHOD_ORDER if m in pivot.index]
    pivot = pivot.loc[ordered]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
    ax.set_title("Tier-1: Binary 12h Detection (F1 Score)")
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_real_heatmap.pdf")
    fig.savefig(output_dir / "fig2_real_heatmap.png")
    plt.close(fig)
    print("  Fig 2: Real data heatmap saved")


def fig3_recovery(recovery_df, output_dir):
    """Figure 3: Known gene recovery bar chart."""
    if len(recovery_df) == 0:
        print("  Fig 3: Skipped (no recovery data)")
        return

    # Average across datasets
    avg = recovery_df.groupby(["method", "category"])["recovery_rate"].mean().reset_index()

    categories = ["known_12h", "circadian", "housekeeping"]
    cat_labels = {"known_12h": "Known 12h genes\n(should be HIGH)",
                  "circadian": "Circadian genes\n(should be LOW)",
                  "housekeeping": "Housekeeping genes\n(should be LOW)"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, cat in zip(axes, categories):
        cat_data = avg[avg["category"] == cat]
        methods = [m for m in METHOD_ORDER if m in cat_data["method"].values]
        values = [cat_data[cat_data["method"] == m]["recovery_rate"].values[0]
                  if m in cat_data["method"].values else 0 for m in methods]
        colors = [METHOD_COLORS.get(m, "#999") for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Detection Rate" if ax == axes[0] else "")
        ax.set_title(cat_labels.get(cat, cat))
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_recovery.pdf")
    fig.savefig(output_dir / "fig3_recovery.png")
    plt.close(fig)
    print("  Fig 3: Recovery bar chart saved")


def fig4_efficiency(efficiency_df, output_dir):
    """Figure 4: Computational efficiency bar chart."""
    if len(efficiency_df) == 0:
        print("  Fig 4: Skipped (no efficiency data)")
        return

    methods = [m for m in METHOD_ORDER if m in efficiency_df["method"].values]
    medians = [efficiency_df[efficiency_df["method"] == m]["median_time_ms"].values[0]
               for m in methods]
    colors = [METHOD_COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(methods)), medians, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Median Time per Gene (ms)")
    ax.set_title("Computational Efficiency")
    ax.set_yscale("log")

    # Add value labels
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_efficiency.pdf")
    fig.savefig(output_dir / "fig4_efficiency.png")
    plt.close(fig)
    print("  Fig 4: Efficiency bar chart saved")


def fig5_robustness(ds_df, noise_df, output_dir):
    """Figure 5: Robustness plots (downsampling + noise)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Downsampling
    if ds_df is not None and len(ds_df) > 0:
        ax = axes[0]
        # Compute F1 per method x config
        from benchmark_final import get_ground_truth, compute_tier1_metrics
        for method in METHOD_ORDER:
            sub = ds_df[ds_df["method"] == method]
            configs = sub["downsample_config"].unique()
            n_tps = []
            f1s = []
            for cfg in sorted(configs):
                cfg_sub = sub[sub["downsample_config"] == cfg]
                n_tp = cfg_sub["n_timepoints"].iloc[0]
                # Simple F1 from has_12h vs ground truth
                gt = get_ground_truth(cfg_sub["gene"].tolist(), "mouse")
                gt_genes = {g for g, v in gt.items() if v["category"] != "unknown"}
                gt_sub = cfg_sub[cfg_sub["gene"].isin(gt_genes)]
                if len(gt_sub) == 0:
                    continue
                y_true = np.array([1 if gt[g]["is_12h"] else 0 for g in gt_sub["gene"]])
                y_pred = gt_sub["has_12h"].astype(int).values
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
                n_tps.append(n_tp)
                f1s.append(f1)

            if n_tps:
                color = METHOD_COLORS.get(method, "#999")
                ax.plot(n_tps, f1s, "o-", color=color, label=method, linewidth=2, markersize=5)

        ax.set_xlabel("Number of Timepoints")
        ax.set_ylabel("F1 Score")
        ax.set_title("A) Temporal Resolution Robustness")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-0.05, 1.05)

    # Panel B: Noise
    if noise_df is not None and len(noise_df) > 0:
        ax = axes[1]
        for method in METHOD_ORDER:
            sub = noise_df[noise_df["method"] == method]
            snr_vals = sorted(sub["snr"].unique())
            detection_rates = []
            for snr in snr_vals:
                snr_sub = sub[sub["snr"] == snr]
                rate = snr_sub["has_12h"].mean()
                detection_rates.append(rate)

            color = METHOD_COLORS.get(method, "#999")
            ax.plot(snr_vals, detection_rates, "o-", color=color,
                    label=method, linewidth=2, markersize=5)

        ax.set_xlabel("Signal-to-Noise Ratio")
        ax.set_ylabel("Detection Rate")
        ax.set_title("B) Noise Robustness")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_robustness.pdf")
    fig.savefig(output_dir / "fig5_robustness.png")
    plt.close(fig)
    print("  Fig 5: Robustness plots saved")


def generate_all_figures(input_dir, output_dir):
    """Generate all publication figures from benchmark results."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating publication figures...")

    # Fig 1: Synthetic ROC
    synth_file = input_dir / "synthetic_results.csv"
    if synth_file.exists():
        synth_df = pd.read_csv(synth_file)
        fig1_synthetic_roc(synth_df, output_dir)

    # Fig 2: Real data heatmap
    tier1_file = input_dir / "tier1_metrics.csv"
    if tier1_file.exists():
        tier1_df = pd.read_csv(tier1_file)
        fig2_real_heatmap(tier1_df, output_dir)

    # Fig 3: Recovery
    recovery_file = input_dir / "recovery_metrics.csv"
    if recovery_file.exists():
        recovery_df = pd.read_csv(recovery_file)
        fig3_recovery(recovery_df, output_dir)

    # Fig 4: Efficiency
    eff_file = input_dir / "efficiency_metrics.csv"
    if eff_file.exists():
        eff_df = pd.read_csv(eff_file)
        fig4_efficiency(eff_df, output_dir)

    # Fig 5: Robustness
    ds_file = input_dir / "downsampling_results.csv"
    noise_file = input_dir / "noise_results.csv"
    ds_df = pd.read_csv(ds_file) if ds_file.exists() else None
    noise_df = pd.read_csv(noise_file) if noise_file.exists() else None
    if ds_df is not None or noise_df is not None:
        fig5_robustness(ds_df, noise_df, output_dir)

    print("All figures generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--input", type=str, default="results/benchmark/",
                        help="Input directory with CSV results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: input/figures/)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(args.input) / "figures"
    generate_all_figures(args.input, output)


if __name__ == "__main__":
    main()
