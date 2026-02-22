#!/usr/bin/env python3
"""
CHORD Benchmark — Publication-Quality Figure Generator for iMeta.

Reads CSV results from the full benchmark pipeline and generates
publication-ready figures following Nature Methods / iMeta standards.

Usage:
    python benchmark_pub_figures.py --input results/benchmark/ --output results/benchmark/pub_figures/

Style: Wong 2011 colorblind-friendly palette, Arial/Helvetica fonts,
       PDF fonttype 42 (editable text), 300 DPI.
"""

import argparse
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

SINGLE_COL = 3.5   # inches — single column width
DOUBLE_COL = 7.0   # inches — double column width
MAX_HEIGHT = 9.0

# Colorblind-friendly palette (Wong 2011, Nature Methods)
PALETTE = {
    "CHORD":                "#D55E00",  # vermilion — highlight
    "Lomb_Scargle":         "#0072B2",  # blue
    "Cosinor":              "#009E73",  # green
    "Harmonic_Regression":  "#CC79A7",  # pink
    "JTK_CYCLE":            "#E69F00",  # orange
    "RAIN":                 "#56B4E9",  # sky blue
    "Pencil":               "#999999",  # grey
}

METHOD_ORDER = ["CHORD", "Cosinor", "Harmonic_Regression",
                "Lomb_Scargle", "JTK_CYCLE", "RAIN", "Pencil"]

METHOD_DISPLAY = {
    "CHORD":                "CHORD",
    "Lomb_Scargle":         "Lomb\u2013Scargle",
    "Cosinor":              "Cosinor",
    "Harmonic_Regression":  "Harmonic Reg.",
    "JTK_CYCLE":            "JTK_CYCLE",
    "RAIN":                 "RAIN",
    "Pencil":               "PENCIL",
}

DATASET_DISPLAY = {
    "hughes2009_2h":   "Hughes 2009\n(mouse liver, 2h)",
    "zhang2014_liver": "Zhang 2014\n(mouse liver, 2h)",
    "zhu2023_wt":      "Zhu 2023 WT\n(mouse liver, 4h)",
    "zhu2023_ko":      "Zhu 2023 KO\n(mouse liver, 4h)",
    "mure2018_liver":  "Mure 2018\n(baboon liver, 2h)",
}


def _setup_style():
    """Set matplotlib rcParams for publication-quality output."""
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


def _label(m):
    return METHOD_DISPLAY.get(m, m)


def _color(m):
    return PALETTE.get(m, "#999999")


def _save(fig, stem, formats=("pdf", "png")):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(stem.with_suffix(f".{fmt}")))


def _panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a, b, c, ...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")



# ============================================================================
# Figure 1: Synthetic ROC Curves (2 panels)
# ============================================================================

def fig1_synthetic_roc(synth_df, output_dir):
    """Synthetic data ROC curves — all scenarios vs fair comparison."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2))

    for ax_idx, (label, title, use_fair) in enumerate([
        ("a", "All synthetic scenarios", False),
        ("b", "Fair comparison (excl. harmonic)", True),
    ]):
        ax = axes[ax_idx]
        for method in METHOD_ORDER:
            sub = synth_df[synth_df["method"] == method].copy()
            if use_fair and method != "CHORD":
                sub = sub[~sub["gt_has_harmonic"].astype(bool)]

            y_true = (sub["gt_has_12h"] | sub["gt_has_harmonic"]).astype(int).values
            # Use -log10(p_value) as detection score for ALL methods.
            # CHORD's "confidence" is a Stage-2 disentanglement score [-1,1],
            # NOT a detection score — using it for ROC is incorrect.
            pvals = np.clip(sub["p_value"].values.astype(float), 1e-300, 1.0)
            y_scores = -np.log10(pvals)

            if len(np.unique(y_true)) < 2:
                continue

            # Compute ROC manually (no sklearn dependency)
            sorted_idx = np.argsort(-y_scores)
            y_true_sorted = y_true[sorted_idx]
            y_scores_sorted = y_scores[sorted_idx]

            tpr_list, fpr_list = [0.0], [0.0]
            tp, fp = 0, 0
            n_pos = y_true.sum()
            n_neg = len(y_true) - n_pos

            for i in range(len(y_true_sorted)):
                if y_true_sorted[i] == 1:
                    tp += 1
                else:
                    fp += 1
                # Only add point when score changes or at end
                if i == len(y_true_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                    tpr_list.append(tp / n_pos if n_pos > 0 else 0)
                    fpr_list.append(fp / n_neg if n_neg > 0 else 0)

            # Compute AUC via trapezoidal rule
            auc_val = np.abs(np.trapz(tpr_list, fpr_list))

            ax.plot(fpr_list, tpr_list, color=_color(method), linewidth=1.5,
                    label=f"{_label(method)} (AUC = {auc_val:.2f})")

        ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title, fontsize=8)
        ax.legend(loc="lower right", frameon=False, fontsize=6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        _panel_label(ax, label)

    plt.tight_layout(w_pad=2.0)
    _save(fig, output_dir / "fig1_synthetic_roc")
    plt.close(fig)
    print("  Fig 1: Synthetic ROC curves")


# ============================================================================
# Figure 2: Real Data Performance Heatmap
# ============================================================================

def fig2_real_heatmap(tier1_df, output_dir):
    """Heatmap of F1 scores: methods x datasets."""
    _setup_style()
    if len(tier1_df) == 0:
        print("  Fig 2: Skipped (no data)")
        return

    pivot = tier1_df.pivot_table(
        index="method", columns="dataset", values="f1", aggfunc="first"
    )
    ordered = [m for m in METHOD_ORDER if m in pivot.index]
    pivot = pivot.loc[ordered]

    # Reorder columns
    ds_order = ["hughes2009_2h", "zhang2014_liver", "zhu2023_wt", "zhu2023_ko", "mure2018_liver"]
    ds_order = [d for d in ds_order if d in pivot.columns]
    pivot = pivot[ds_order]

    n_methods = len(pivot.index)
    n_datasets = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, SINGLE_COL * 0.9))

    # Custom colormap: white -> light orange -> vermilion
    cmap = LinearSegmentedColormap.from_list(
        "chord_heat", ["#FFFFFF", "#FEE0B6", "#F4A261", "#D55E00"], N=256
    )

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=0.7)

    ax.set_xticks(range(n_datasets))
    ds_labels = [DATASET_DISPLAY.get(d, d) for d in pivot.columns]
    ax.set_xticklabels(ds_labels, rotation=0, ha="center", fontsize=6)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([_label(m) for m in pivot.index], fontsize=7)

    # Annotate cells
    for i in range(n_methods):
        for j in range(n_datasets):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if val > 0.4 else "black"
                weight = "bold" if val > 0.3 else "normal"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("F1 Score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("12h rhythm detection performance (F1 score, Stage 1 for CHORD)", fontsize=9)
    _panel_label(ax, "", x=-0.18)

    plt.tight_layout()
    _save(fig, output_dir / "fig2_real_heatmap")
    plt.close(fig)
    print("  Fig 2: Real data heatmap")



# ============================================================================
# Figure 3: Known Gene Recovery (3-panel bar chart)
# ============================================================================

def fig3_recovery(recovery_df, output_dir):
    """Grouped bar chart: recovery rate by gene category."""
    _setup_style()
    if len(recovery_df) == 0:
        print("  Fig 3: Skipped (no data)")
        return

    # Average across datasets
    avg = recovery_df.groupby(["method", "category"])["recovery_rate"].mean().reset_index()

    categories = ["known_12h", "circadian", "housekeeping"]
    cat_titles = {
        "known_12h":     "Known 12h genes\n(target: high)",
        "circadian":     "Circadian-only genes\n(target: low)",
        "housekeeping":  "Housekeeping genes\n(target: low)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8), sharey=True)

    for ax, cat, panel_lbl in zip(axes, categories, ["a", "b", "c"]):
        cat_data = avg[avg["category"] == cat]
        methods = [m for m in METHOD_ORDER if m in cat_data["method"].values]
        values = []
        for m in methods:
            v = cat_data[cat_data["method"] == m]["recovery_rate"].values
            values.append(v[0] if len(v) > 0 else 0)
        colors = [_color(m) for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors,
                      edgecolor="white", linewidth=0.4, width=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods], rotation=45,
                           ha="right", fontsize=6)
        ax.set_title(cat_titles.get(cat, cat), fontsize=7)
        ax.set_ylim(0, 1.08)

        if cat == "known_12h":
            ax.set_ylabel("Detection rate")
        else:
            # Reference line for false positive threshold
            ax.axhline(y=0.05, color="0.5", ls="--", lw=0.5, alpha=0.7)

        # Value labels
        for bar, val in zip(bars, values):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)

        _panel_label(ax, panel_lbl)

    plt.tight_layout(w_pad=1.0)
    _save(fig, output_dir / "fig3_recovery")
    plt.close(fig)
    print("  Fig 3: Known gene recovery")


# ============================================================================
# Figure 4: Cross-Species Consistency
# ============================================================================

def fig4_cross_species(cross_df, recovery_df, output_dir):
    """Cross-species conserved gene recovery."""
    _setup_style()
    if len(cross_df) == 0:
        print("  Fig 4: Skipped (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # Panel A: Recovery of conserved genes (any dataset)
    ax = axes[0]
    methods = [m for m in METHOD_ORDER if m in cross_df["method"].values]
    recovery_any = [cross_df[cross_df["method"] == m]["recovery_any"].values[0]
                    for m in methods]
    colors = [_color(m) for m in methods]

    bars = ax.bar(range(len(methods)), recovery_any, color=colors,
                  edgecolor="white", linewidth=0.4, width=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=45,
                       ha="right", fontsize=6)
    ax.set_ylabel("Recovery rate")
    ax.set_title("Conserved 12h genes\n(detected in \u22651 dataset)", fontsize=7)
    ax.set_ylim(0, 1.08)

    for bar, val in zip(bars, recovery_any):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=6)
    _panel_label(ax, "a")

    # Panel B: Per-dataset recovery for CHORD vs best classical
    ax = axes[1]
    if len(recovery_df) > 0:
        chord_rec = recovery_df[
            (recovery_df["method"] == "CHORD") &
            (recovery_df["category"] == "known_12h")
        ][["dataset", "recovery_rate"]].set_index("dataset")

        # Best classical per dataset
        classical = recovery_df[
            (recovery_df["method"] != "CHORD") &
            (recovery_df["category"] == "known_12h")
        ]
        best_classical = classical.groupby("dataset")["recovery_rate"].max().to_frame("recovery_rate")

        datasets = sorted(set(chord_rec.index) & set(best_classical.index))
        ds_labels = [DATASET_DISPLAY.get(d, d).replace("\n", " ") for d in datasets]

        x = np.arange(len(datasets))
        w = 0.35
        chord_vals = [chord_rec.loc[d, "recovery_rate"] for d in datasets]
        best_vals = [best_classical.loc[d, "recovery_rate"] for d in datasets]

        ax.bar(x - w/2, chord_vals, w, color=_color("CHORD"),
               edgecolor="white", linewidth=0.4, label="CHORD")
        ax.bar(x + w/2, best_vals, w, color="#0072B2",
               edgecolor="white", linewidth=0.4, label="Best classical")

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=5.5)
        ax.set_ylabel("12h gene recovery")
        ax.set_title("CHORD vs best classical\n(per dataset)", fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.legend(frameon=False, fontsize=6, loc="upper right")

    _panel_label(ax, "b")

    plt.tight_layout(w_pad=2.0)
    _save(fig, output_dir / "fig4_cross_species")
    plt.close(fig)
    print("  Fig 4: Cross-species consistency")



# ============================================================================
# Figure 5: Robustness (Downsampling + Noise)
# ============================================================================

def _compute_f1_from_downsampling(ds_df, method, config, known_12h_genes, negative_genes):
    """Compute F1 for a method at a specific downsampling config."""
    sub = ds_df[(ds_df["method"] == method) & (ds_df["downsample_config"] == config)]
    if len(sub) == 0:
        return np.nan

    # Filter to ground truth genes only
    gt_genes = set(known_12h_genes) | set(negative_genes)
    sub_gt = sub[sub["gene"].isin(gt_genes)]
    if len(sub_gt) == 0:
        return np.nan

    y_true = np.array([1 if g in known_12h_genes else 0 for g in sub_gt["gene"]])
    y_pred = sub_gt["has_12h"].astype(int).values

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    return f1


def fig5_robustness(ds_df, noise_df, output_dir, known_12h_genes=None, negative_genes=None):
    """Robustness: downsampling + noise experiments."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # Panel A: Downsampling
    ax = axes[0]
    if ds_df is not None and len(ds_df) > 0:
        configs = sorted(ds_df["downsample_config"].unique())
        tp_map = {}
        for cfg in configs:
            n_tp = ds_df[ds_df["downsample_config"] == cfg]["n_timepoints"].iloc[0]
            tp_map[cfg] = n_tp

        for method in METHOD_ORDER:
            n_tps = []
            f1s = []
            for cfg in configs:
                n_tp = tp_map[cfg]
                if known_12h_genes is not None and negative_genes is not None:
                    f1 = _compute_f1_from_downsampling(
                        ds_df, method, cfg, known_12h_genes, negative_genes
                    )
                else:
                    # Fallback: use has_12h detection rate on all genes
                    sub = ds_df[(ds_df["method"] == method) & (ds_df["downsample_config"] == cfg)]
                    f1 = sub["has_12h"].mean() if len(sub) > 0 else 0
                n_tps.append(n_tp)
                f1s.append(f1)

            if n_tps:
                ax.plot(n_tps, f1s, "o-", color=_color(method),
                        label=_label(method), linewidth=1.2, markersize=4)

        ax.set_xlabel("Number of timepoints")
        ax.set_ylabel("Detection rate")
        ax.set_title("Temporal resolution robustness", fontsize=8)
        ax.legend(fontsize=5.5, loc="upper left", frameon=False, ncol=2)
        ax.set_ylim(-0.02, None)
    _panel_label(ax, "a")

    # Panel B: Noise
    ax = axes[1]
    if noise_df is not None and len(noise_df) > 0:
        for method in METHOD_ORDER:
            sub = noise_df[noise_df["method"] == method]
            snr_vals = sorted(sub["snr"].unique())
            detection_rates = []
            for snr in snr_vals:
                snr_sub = sub[sub["snr"] == snr]
                rate = snr_sub["has_12h"].mean()
                detection_rates.append(rate)

            ax.plot(snr_vals, detection_rates, "o-", color=_color(method),
                    label=_label(method), linewidth=1.2, markersize=4)

        ax.set_xlabel("Signal-to-noise ratio (SNR)")
        ax.set_ylabel("Detection rate")
        ax.set_title("Noise robustness", fontsize=8)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False, ncol=2)
        ax.set_ylim(-0.05, 1.08)
    _panel_label(ax, "b")

    plt.tight_layout(w_pad=2.0)
    _save(fig, output_dir / "fig5_robustness")
    plt.close(fig)
    print("  Fig 5: Robustness plots")


# ============================================================================
# Figure 6: Computational Efficiency
# ============================================================================

def fig6_efficiency(eff_df, output_dir):
    """Computational efficiency bar chart (log scale)."""
    _setup_style()
    if len(eff_df) == 0:
        print("  Fig 6: Skipped (no data)")
        return

    methods = [m for m in METHOD_ORDER if m in eff_df["method"].values]
    medians = [eff_df[eff_df["method"] == m]["median_time_ms"].values[0]
               for m in methods]
    colors = [_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))

    bars = ax.bar(range(len(methods)), medians, color=colors,
                  edgecolor="white", linewidth=0.4, width=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=45,
                       ha="right", fontsize=6)
    ax.set_ylabel("Median time per gene (ms)")
    ax.set_title("Computational efficiency", fontsize=9)
    ax.set_yscale("log")

    # Value labels
    for bar, val in zip(bars, medians):
        y_pos = bar.get_height() * 1.3
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.1f}" if val >= 1 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=6)

    # Add speed annotation
    ax.axhline(y=1.0, color="0.7", ls=":", lw=0.5)
    ax.text(len(methods) - 0.5, 1.0, "1 ms", fontsize=5, color="0.5",
            ha="right", va="bottom")

    plt.tight_layout()
    _save(fig, output_dir / "fig6_efficiency")
    plt.close(fig)
    print("  Fig 6: Computational efficiency")



# ============================================================================
# Supplementary Figure S1: Synthetic Benchmark Summary Table
# ============================================================================

def figS1_synthetic_summary(synth_metrics_df, output_dir):
    """Table-style figure of synthetic benchmark metrics."""
    _setup_style()
    if len(synth_metrics_df) == 0:
        print("  Fig S1: Skipped (no data)")
        return

    # Filter to Tier-1 only for the main comparison
    t1 = synth_metrics_df[synth_metrics_df["tier"] == "Tier-1 (detection)"].copy()
    t1 = t1.sort_values("f1", ascending=False)

    methods = t1["method"].tolist()
    metrics = ["sensitivity", "specificity", "precision", "f1", "roc_auc"]
    metric_labels = ["Sensitivity", "Specificity", "Precision", "F1", "ROC AUC"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, SINGLE_COL * 0.7))
    ax.axis("off")

    data = []
    for _, row in t1.iterrows():
        data.append([_label(row["method"])] + [f"{row[m]:.3f}" for m in metrics])

    table = ax.table(
        cellText=data,
        colLabels=["Method"] + metric_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(metrics) + 1):
        cell = table[0, j]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold")

    # Highlight CHORD row
    for j in range(len(metrics) + 1):
        for i, row in enumerate(data):
            if "CHORD" in row[0]:
                table[i + 1, j].set_facecolor("#FFF3E0")

    ax.set_title("Synthetic Benchmark: Tier-1 Detection Metrics (50 replicates)",
                 fontsize=8, pad=20)

    plt.tight_layout()
    _save(fig, output_dir / "figS1_synthetic_summary")
    plt.close(fig)
    print("  Fig S1: Synthetic summary table")


# ============================================================================
# Supplementary Figure S2: BMAL1 KO Validation
# ============================================================================

def figS2_bmal1ko(real_df, output_dir):
    """Compare WT vs KO classification distributions for CHORD."""
    _setup_style()
    if real_df is None or len(real_df) == 0:
        print("  Fig S2: Skipped (no data)")
        return

    chord_wt = real_df[(real_df["method"] == "CHORD") & (real_df["dataset"] == "zhu2023_wt")]
    chord_ko = real_df[(real_df["method"] == "CHORD") & (real_df["dataset"] == "zhu2023_ko")]

    if len(chord_wt) == 0 or len(chord_ko) == 0:
        print("  Fig S2: Skipped (no WT/KO data)")
        return

    classes = ["independent_ultradian", "likely_independent_ultradian",
               "harmonic", "circadian_only", "non_rhythmic", "ambiguous"]
    class_labels = ["Independent\n12h", "Likely\nindependent", "Harmonic",
                    "Circadian\nonly", "Non-\nrhythmic", "Ambiguous"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), sharey=True)

    for ax, (df_sub, title, panel) in zip(axes, [
        (chord_wt, "Wild-type (WT)", "a"),
        (chord_ko, "BMAL1-KO", "b"),
    ]):
        counts = []
        for cls in classes:
            n = (df_sub["classification"] == cls).sum()
            counts.append(n)

        total = sum(counts)
        fracs = [c / total if total > 0 else 0 for c in counts]

        # Color: warm for 12h-related, cool for others
        cls_colors = ["#D55E00", "#E69F00", "#CC79A7", "#0072B2", "#999999", "#56B4E9"]

        bars = ax.barh(range(len(classes)), fracs, color=cls_colors,
                       edgecolor="white", linewidth=0.4, height=0.7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(class_labels, fontsize=6)
        ax.set_xlabel("Fraction of genes")
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, max(fracs) * 1.3 if fracs else 1)

        for bar, frac, count in zip(bars, fracs, counts):
            if frac > 0.01:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{count} ({frac:.0%})", va="center", fontsize=5.5)

        _panel_label(ax, panel)

    fig.suptitle("CHORD classification: WT vs BMAL1-KO", fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir / "figS2_bmal1ko")
    plt.close(fig)
    print("  Fig S2: BMAL1 KO validation")


# ============================================================================
# Supplementary Figure S3: Per-Dataset Sensitivity/Specificity
# ============================================================================

def figS3_per_dataset(tier1_df, output_dir):
    """Per-dataset sensitivity and specificity for all methods."""
    _setup_style()
    if len(tier1_df) == 0:
        print("  Fig S3: Skipped (no data)")
        return

    datasets = sorted(tier1_df["dataset"].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(2, n_ds, figsize=(DOUBLE_COL, 4.0),
                             sharex=True, sharey="row")
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for j, ds in enumerate(datasets):
        ds_data = tier1_df[tier1_df["dataset"] == ds]
        methods = [m for m in METHOD_ORDER if m in ds_data["method"].values]

        for row_idx, (metric, ylabel) in enumerate([
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
        ]):
            ax = axes[row_idx, j]
            vals = [ds_data[ds_data["method"] == m][metric].values[0]
                    if m in ds_data["method"].values else 0 for m in methods]
            colors = [_color(m) for m in methods]

            ax.bar(range(len(methods)), vals, color=colors,
                   edgecolor="white", linewidth=0.3, width=0.7)
            ax.set_ylim(0, 1.08)

            if row_idx == 1:
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([_label(m) for m in methods],
                                   rotation=90, ha="center", fontsize=5)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(ylabel, fontsize=7)

            if row_idx == 0:
                ds_label = DATASET_DISPLAY.get(ds, ds).replace("\n", " ")
                ax.set_title(ds_label, fontsize=6)

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    _save(fig, output_dir / "figS3_per_dataset")
    plt.close(fig)
    print("  Fig S3: Per-dataset metrics")



# ============================================================================
# Master Summary Panel (composite figure)
# ============================================================================

def fig_summary_panel(synth_df, tier1_df, eff_df, cross_df, output_dir):
    """4-panel summary figure for the main text."""
    _setup_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.45,
                           left=0.10, right=0.95, top=0.94, bottom=0.10)

    # (a) Synthetic ROC — fair comparison only
    ax_a = fig.add_subplot(gs[0, 0])
    if synth_df is not None and len(synth_df) > 0:
        for method in METHOD_ORDER:
            sub = synth_df[synth_df["method"] == method].copy()
            if method != "CHORD":
                sub = sub[~sub["gt_has_harmonic"].astype(bool)]

            y_true = (sub["gt_has_12h"] | sub["gt_has_harmonic"]).astype(int).values
            pvals = np.clip(sub["p_value"].values.astype(float), 1e-300, 1.0)
            y_scores = -np.log10(pvals)

            if len(np.unique(y_true)) < 2:
                continue

            sorted_idx = np.argsort(-y_scores)
            y_true_sorted = y_true[sorted_idx]
            y_scores_sorted = y_scores[sorted_idx]

            tpr_list, fpr_list = [0.0], [0.0]
            tp, fp = 0, 0
            n_pos = y_true.sum()
            n_neg = len(y_true) - n_pos

            for i in range(len(y_true_sorted)):
                if y_true_sorted[i] == 1:
                    tp += 1
                else:
                    fp += 1
                if i == len(y_true_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                    tpr_list.append(tp / n_pos if n_pos > 0 else 0)
                    fpr_list.append(fp / n_neg if n_neg > 0 else 0)

            auc_val = np.abs(np.trapz(tpr_list, fpr_list))
            ax_a.plot(fpr_list, tpr_list, color=_color(method), linewidth=1.0,
                      label=f"{_label(method)} ({auc_val:.2f})")

        ax_a.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.5)
        ax_a.set_xlabel("False positive rate")
        ax_a.set_ylabel("True positive rate")
        ax_a.set_title("Synthetic ROC (fair)", fontsize=8)
        ax_a.legend(loc="lower right", frameon=False, fontsize=5)
        ax_a.set_xlim(-0.02, 1.02)
        ax_a.set_ylim(-0.02, 1.02)
        ax_a.set_aspect("equal")
    _panel_label(ax_a, "a")

    # (b) Real data heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    if tier1_df is not None and len(tier1_df) > 0:
        pivot = tier1_df.pivot_table(
            index="method", columns="dataset", values="f1", aggfunc="first"
        )
        ordered = [m for m in METHOD_ORDER if m in pivot.index]
        pivot = pivot.loc[ordered]
        ds_order = ["hughes2009_2h", "zhang2014_liver", "zhu2023_wt",
                     "zhu2023_ko", "mure2018_liver"]
        ds_order = [d for d in ds_order if d in pivot.columns]
        pivot = pivot[ds_order]

        cmap = LinearSegmentedColormap.from_list(
            "heat2", ["#FFFFFF", "#FEE0B6", "#F4A261", "#D55E00"], N=256
        )
        im = ax_b.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=0.7)

        ax_b.set_xticks(range(len(pivot.columns)))
        short_ds = {
            "hughes2009_2h": "Hughes",
            "zhang2014_liver": "Zhang",
            "zhu2023_wt": "Zhu WT",
            "zhu2023_ko": "Zhu KO",
            "mure2018_liver": "Mure",
        }
        ax_b.set_xticklabels([short_ds.get(d, d) for d in pivot.columns],
                             rotation=45, ha="right", fontsize=6)
        ax_b.set_yticks(range(len(pivot.index)))
        ax_b.set_yticklabels([_label(m) for m in pivot.index], fontsize=6)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    color = "white" if val > 0.4 else "black"
                    ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                              color=color, fontsize=5.5)

        ax_b.set_title("Real data F1", fontsize=8)
    _panel_label(ax_b, "b")

    # (c) Cross-species recovery
    ax_c = fig.add_subplot(gs[1, 0])
    if cross_df is not None and len(cross_df) > 0:
        methods = [m for m in METHOD_ORDER if m in cross_df["method"].values]
        recovery = [cross_df[cross_df["method"] == m]["recovery_any"].values[0]
                    for m in methods]
        colors = [_color(m) for m in methods]
        ax_c.bar(range(len(methods)), recovery, color=colors,
                 edgecolor="white", linewidth=0.3, width=0.7)
        ax_c.set_xticks(range(len(methods)))
        ax_c.set_xticklabels([_label(m) for m in methods], rotation=45,
                             ha="right", fontsize=5.5)
        ax_c.set_ylabel("Recovery rate")
        ax_c.set_title("Conserved 12h genes", fontsize=8)
        ax_c.set_ylim(0, 1.08)
    _panel_label(ax_c, "c")

    # (d) Efficiency
    ax_d = fig.add_subplot(gs[1, 1])
    if eff_df is not None and len(eff_df) > 0:
        methods = [m for m in METHOD_ORDER if m in eff_df["method"].values]
        medians = [eff_df[eff_df["method"] == m]["median_time_ms"].values[0]
                   for m in methods]
        colors = [_color(m) for m in methods]
        ax_d.bar(range(len(methods)), medians, color=colors,
                 edgecolor="white", linewidth=0.3, width=0.7)
        ax_d.set_xticks(range(len(methods)))
        ax_d.set_xticklabels([_label(m) for m in methods], rotation=45,
                             ha="right", fontsize=5.5)
        ax_d.set_ylabel("Time per gene (ms)")
        ax_d.set_title("Computational cost", fontsize=8)
        ax_d.set_yscale("log")
    _panel_label(ax_d, "d")

    _save(fig, output_dir / "fig_summary_panel")
    plt.close(fig)
    print("  Summary panel: 4-panel overview")



# ============================================================================
# Main orchestration
# ============================================================================

def generate_all_figures(input_dir, output_dir):
    """Generate all publication figures from benchmark results."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating publication figures")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load all available data
    def _load(name):
        p = input_dir / name
        if p.exists():
            df = pd.read_csv(p)
            print(f"  Loaded {name}: {len(df)} rows")
            return df
        print(f"  Missing: {name}")
        return None

    synth_df = _load("synthetic_results.csv")
    synth_metrics = _load("synthetic_metrics.csv")
    real_df = _load("real_results.csv")
    tier1_df = _load("tier1_metrics.csv")
    recovery_df = _load("recovery_metrics.csv")
    cross_df = _load("cross_species_metrics.csv")
    eff_df = _load("efficiency_metrics.csv")
    ds_df = _load("downsampling_results.csv")
    noise_df = _load("noise_results.csv")

    print(f"\n--- Main Figures ---")

    # Fig 1: Synthetic ROC
    if synth_df is not None:
        fig1_synthetic_roc(synth_df, output_dir)

    # Fig 2: Real data heatmap
    if tier1_df is not None:
        fig2_real_heatmap(tier1_df, output_dir)

    # Fig 3: Known gene recovery
    if recovery_df is not None:
        fig3_recovery(recovery_df, output_dir)

    # Fig 4: Cross-species
    if cross_df is not None:
        fig4_cross_species(cross_df, recovery_df, output_dir)

    # Fig 5: Robustness
    if ds_df is not None or noise_df is not None:
        fig5_robustness(ds_df, noise_df, output_dir)

    # Fig 6: Efficiency
    if eff_df is not None:
        fig6_efficiency(eff_df, output_dir)

    print(f"\n--- Supplementary Figures ---")

    # Fig S1: Synthetic summary table
    if synth_metrics is not None:
        figS1_synthetic_summary(synth_metrics, output_dir)

    # Fig S2: BMAL1 KO
    if real_df is not None:
        figS2_bmal1ko(real_df, output_dir)

    # Fig S3: Per-dataset metrics
    if tier1_df is not None:
        figS3_per_dataset(tier1_df, output_dir)

    # Summary panel
    fig_summary_panel(synth_df, tier1_df, eff_df, cross_df, output_dir)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="CHORD Benchmark — Publication Figure Generator"
    )
    parser.add_argument("--input", type=str, default="results/benchmark/",
                        help="Input directory with CSV results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: input/pub_figures/)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(args.input) / "pub_figures"
    generate_all_figures(args.input, output)


if __name__ == "__main__":
    main()
