"""
Publication-quality benchmark visualisation for CHORD.

Generates Nature Aging–level figures comparing CHORD (BHDT + PINOD)
against classical rhythm-detection methods.

Usage
-----
    from chord.benchmarks.plot_benchmark import plot_benchmark_summary
    plot_benchmark_summary(results_df, output_dir="figures/")
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from chord.benchmarks.metrics import (
    classification_metrics,
    harmonic_disentangle_accuracy,
    roc_auc,
)

# ============================================================================
# Constants
# ============================================================================

# Journal dimensions (inches)
SINGLE_COL = 3.5
DOUBLE_COL = 7.0
MAX_HEIGHT = 9.0

# Colorblind-friendly palette (Wong 2011, Nature Methods)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]

METHOD_DISPLAY = {
    "bhdt": "CHORD (BHDT)",
    "lomb_scargle": "Lomb–Scargle",
    "cosinor_12h": "Cosinor (12 h)",
    "harmonic_regression": "Harmonic Reg.",
    "pinod": "PINOD",
}


# ============================================================================
# Style helper
# ============================================================================

def _setup_style():
    """Set matplotlib rcParams for publication-quality output."""
    matplotlib.rcdefaults()
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        # Lines / markers
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        # Axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid
        "axes.grid": False,
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
        # Figure
        "figure.dpi": 150,
    })


def _method_label(m):
    """Return a display-friendly method name."""
    return METHOD_DISPLAY.get(m, m)


def _save_fig(fig, path_stem, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(str(path_stem.with_suffix(f".{fmt}")))


# ============================================================================
# 1. Accuracy heatmap
# ============================================================================

def plot_accuracy_heatmap(results_df, methods=None, scenarios=None, ax=None):
    """Heatmap of per-scenario accuracy for each method.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_benchmark()``.
    methods : list of str, optional
        Methods to include (column order).
    scenarios : list of str, optional
        Scenario names to include (row order).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for subplot integration.

    Returns
    -------
    fig, ax
    """
    _setup_style()
    df = results_df.copy()
    df["correct"] = df["predicted_has_12h"] == df["true_has_12h"]

    if methods is None:
        methods = sorted(df["method"].unique())
    if scenarios is None:
        scenarios = sorted(df["scenario_name"].unique())

    # Build accuracy matrix (rows=scenarios, cols=methods)
    acc = np.full((len(scenarios), len(methods)), np.nan)
    for i, scen in enumerate(scenarios):
        for j, meth in enumerate(methods):
            mask = (df["scenario_name"] == scen) & (df["method"] == meth)
            sub = df.loc[mask]
            if len(sub) > 0:
                acc[i, j] = sub["correct"].mean() * 100.0

    own_fig = ax is None
    if own_fig:
        height = max(2.0, 0.3 * len(scenarios) + 0.8)
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL, min(height, MAX_HEIGHT))
        )
    else:
        fig = ax.figure

    # Red-to-green diverging colourmap
    cmap = LinearSegmentedColormap.from_list(
        "acc_rg", ["#d73027", "#fee08b", "#1a9850"], N=256
    )

    im = ax.imshow(acc, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    # Annotate cells
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            val = acc[i, j]
            if np.isnan(val):
                txt = "–"
            else:
                txt = f"{val:.0f}"
            colour = "white" if val < 40 or val > 85 else "black"
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=6, color=colour,
            )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [_method_label(m) for m in methods], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("Method")
    ax.set_ylabel("Scenario")
    ax.set_title("12 h detection accuracy (%)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ============================================================================
# 2. ROC curves
# ============================================================================

def plot_roc_curves(results_df, methods=None, ax=None):
    """ROC curves for binary 12 h detection (rhythmic vs non-rhythmic).

    For BHDT the score is ``log_bf``; for classical methods the score is
    ``-log10(p_value)`` (higher = more confident positive).

    Parameters
    ----------
    results_df : pd.DataFrame
    methods : list of str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    _setup_style()
    df = results_df.copy()

    if methods is None:
        methods = sorted(df["method"].unique())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    else:
        fig = ax.figure

    for idx, meth in enumerate(methods):
        mdf = df[df["method"] == meth].copy()
        y_true = mdf["true_has_12h"].astype(int).values

        # Build a continuous score
        if "log_bf" in mdf.columns and meth == "bhdt":
            scores = mdf["log_bf"].fillna(-999).values.astype(float)
        elif "p_value" in mdf.columns:
            pvals = mdf["p_value"].fillna(1.0).values.astype(float)
            pvals = np.clip(pvals, 1e-300, 1.0)
            scores = -np.log10(pvals)
        else:
            scores = mdf["predicted_has_12h"].astype(float).values

        roc = roc_auc(y_true, scores)
        colour = PALETTE[idx % len(PALETTE)]
        label = f"{_method_label(meth)} (AUC={roc['auc']:.2f})"
        ax.plot(roc["fpr"], roc["tpr"], color=colour, label=label, lw=1.2)

    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — 12 h rhythm detection")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    if own_fig:
        fig.tight_layout()
    return fig, ax



# ============================================================================
# 3. Confusion matrix
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion matrix",
                          ax=None):
    """Single confusion-matrix heatmap with counts and percentages.

    Parameters
    ----------
    y_true, y_pred : array-like
    labels : list of str, optional
    title : str
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    _setup_style()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    n = len(labels)
    lab2i = {l: i for i, l in enumerate(labels)}

    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        i, j = lab2i.get(yt), lab2i.get(yp)
        if i is not None and j is not None:
            cm[i, j] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    ) * 100.0

    own_fig = ax is None
    if own_fig:
        side = max(2.0, 0.55 * n + 0.6)
        fig, ax = plt.subplots(figsize=(side, side))
    else:
        fig = ax.figure

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")

    for i in range(n):
        for j in range(n):
            txt = f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)"
            colour = "white" if cm_pct[i, j] > 60 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color=colour)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ============================================================================
# 4. Method comparison bar chart
# ============================================================================

def _bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    """Return (mean, ci_low, ci_high) via bootstrap."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lo), float(hi)


def plot_method_comparison_bar(summary_dict, metric="accuracy", ax=None):
    """Grouped bar chart comparing methods on a single metric.

    Parameters
    ----------
    summary_dict : dict
        ``{method_name: array_of_per_replicate_values}``.
    metric : str
        Label for the y-axis.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    _setup_style()
    methods = list(summary_dict.keys())
    means, lows, highs = [], [], []
    for m in methods:
        mu, lo, hi = _bootstrap_ci(summary_dict[m])
        means.append(mu)
        lows.append(mu - lo)
        highs.append(hi - mu)

    own_fig = ax is None
    if own_fig:
        width = max(SINGLE_COL, 0.7 * len(methods) + 0.8)
        fig, ax = plt.subplots(figsize=(min(width, DOUBLE_COL), 2.5))
    else:
        fig = ax.figure

    x = np.arange(len(methods))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(methods))]
    bars = ax.bar(
        x, means, yerr=[lows, highs], width=0.6,
        color=colours, edgecolor="white", linewidth=0.4,
        capsize=2, error_kw={"lw": 0.8},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=30,
                       ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Method comparison — {metric.replace('_', ' ')}")
    ax.set_ylim(0, min(1.15 * max(means) if means else 1.0, 1.05))

    # Value labels above bars
    for bar, mu in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{mu:.2f}", ha="center", va="bottom", fontsize=6,
        )

    if own_fig:
        fig.tight_layout()
    return fig, ax



# ============================================================================
# 5. Parameter estimation scatter
# ============================================================================

def plot_parameter_estimation(true_params, est_params, param_name="period",
                              ax=None):
    """Scatter of estimated vs true parameter values with identity line.

    Parameters
    ----------
    true_params : array-like
        Ground-truth values.
    est_params : array-like
        Estimated values.
    param_name : str
        One of 'period', 'amplitude', 'phase'.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    _setup_style()
    true_arr = np.asarray(true_params, dtype=float)
    est_arr = np.asarray(est_params, dtype=float)

    # Drop NaN pairs
    valid = ~(np.isnan(true_arr) | np.isnan(est_arr))
    true_arr = true_arr[valid]
    est_arr = est_arr[valid]

    rmse = float(np.sqrt(np.mean((est_arr - true_arr) ** 2))) if len(true_arr) > 0 else np.nan

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    else:
        fig = ax.figure

    ax.scatter(true_arr, est_arr, s=12, alpha=0.6, color=PALETTE[0],
               edgecolors="none", zorder=2)

    # Identity line
    lo = min(true_arr.min(), est_arr.min()) if len(true_arr) > 0 else 0
    hi = max(true_arr.max(), est_arr.max()) if len(true_arr) > 0 else 1
    margin = 0.05 * (hi - lo) if hi > lo else 0.5
    lims = (lo - margin, hi + margin)
    ax.plot(lims, lims, ls="--", color="0.5", lw=0.6, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    units = {"period": " (h)", "amplitude": "", "phase": " (rad)"}
    unit = units.get(param_name, "")
    ax.set_xlabel(f"True {param_name}{unit}")
    ax.set_ylabel(f"Estimated {param_name}{unit}")
    ax.set_title(f"Parameter recovery — {param_name}")
    ax.set_aspect("equal")

    # RMSE annotation
    ax.text(
        0.05, 0.95, f"RMSE = {rmse:.3f}",
        transform=ax.transAxes, fontsize=7,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.4),
    )

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ============================================================================
# 6. Harmonic disentanglement stacked bar
# ============================================================================

def plot_harmonic_disentangle(results_df, ax=None):
    """Stacked bar: correct/incorrect for harmonic vs independent genes.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``true_class`` and ``predicted_has_12h`` columns.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    _setup_style()
    df = results_df.copy()
    methods = sorted(df["method"].unique())

    # Separate harmonic (true 12 h from harmonic of 24 h) and independent
    harmonic_mask = df["true_class"] == "harmonic"
    independent_mask = df["true_class"] == "independent_ultradian"

    categories = ["Harmonic\n(should be neg.)", "Independent\n(should be pos.)"]
    masks = [harmonic_mask, independent_mask]
    # For harmonic genes, correct = predicted_has_12h is False
    # For independent genes, correct = predicted_has_12h is True
    correct_fn = [
        lambda sub: ~sub["predicted_has_12h"],
        lambda sub: sub["predicted_has_12h"].astype(bool),
    ]

    own_fig = ax is None
    if own_fig:
        width = max(SINGLE_COL, 0.6 * len(methods) * 2 + 1.0)
        fig, ax = plt.subplots(figsize=(min(width, DOUBLE_COL), 2.8))
    else:
        fig = ax.figure

    n_groups = len(categories)
    n_methods = len(methods)
    bar_w = 0.8 / n_methods
    x_base = np.arange(n_groups)

    for mi, meth in enumerate(methods):
        correct_vals = []
        incorrect_vals = []
        for ci, (mask, cfn) in enumerate(zip(masks, correct_fn)):
            sub = df.loc[mask & (df["method"] == meth)]
            if len(sub) == 0:
                correct_vals.append(0)
                incorrect_vals.append(0)
                continue
            c = cfn(sub).sum()
            correct_vals.append(c)
            incorrect_vals.append(len(sub) - c)

        offset = (mi - n_methods / 2 + 0.5) * bar_w
        colour = PALETTE[mi % len(PALETTE)]
        ax.bar(
            x_base + offset, correct_vals, bar_w,
            color=colour, edgecolor="white", linewidth=0.3,
            label=_method_label(meth) if mi < len(methods) else None,
        )
        ax.bar(
            x_base + offset, incorrect_vals, bar_w,
            bottom=correct_vals,
            color=colour, alpha=0.3, edgecolor="white", linewidth=0.3,
            hatch="//",
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count")
    ax.set_title("Harmonic disentanglement")
    ax.legend(loc="upper right", frameon=False, ncol=2)

    # Custom legend note
    ax.text(
        0.98, 0.02,
        "solid = correct, hatched = incorrect",
        transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom",
        style="italic", color="0.4",
    )

    if own_fig:
        fig.tight_layout()
    return fig, ax



# ============================================================================
# 7. Master summary function
# ============================================================================

def plot_benchmark_summary(results_df, output_dir="figures/benchmark"):
    """Generate all benchmark plots and save to *output_dir*.

    Produces:
        accuracy_heatmap.{pdf,png}
        roc_curves.{pdf,png}
        confusion_matrix_<method>.{pdf,png}   (one per method)
        method_comparison.{pdf,png}
        parameter_estimation_<param>.{pdf,png}
        harmonic_disentangle.{pdf,png}
        summary_panel.{pdf,png}               (multi-panel overview)

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_benchmark()``.
    output_dir : str or Path
        Directory for saved figures.

    Returns
    -------
    list of Path
        Paths to all generated files.
    """
    _setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    df = results_df.copy()
    methods = sorted(df["method"].unique())

    # --- 1. Accuracy heatmap ---
    fig, _ = plot_accuracy_heatmap(df, methods=methods)
    stem = out / "accuracy_heatmap"
    _save_fig(fig, stem)
    saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    plt.close(fig)

    # --- 2. ROC curves ---
    fig, _ = plot_roc_curves(df, methods=methods)
    stem = out / "roc_curves"
    _save_fig(fig, stem)
    saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    plt.close(fig)

    # --- 3. Confusion matrices (one per method) ---
    labels = sorted(set(df["true_class"]) | set(df["predicted_class"]))
    for meth in methods:
        mdf = df[df["method"] == meth]
        fig, _ = plot_confusion_matrix(
            mdf["true_class"].values,
            mdf["predicted_class"].values,
            labels=labels,
            title=f"{_method_label(meth)}",
        )
        stem = out / f"confusion_matrix_{meth}"
        _save_fig(fig, stem)
        saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
        plt.close(fig)

    # --- 4. Method comparison bar ---
    acc_dict = {}
    for meth in methods:
        mdf = df[df["method"] == meth]
        per_rep = (
            mdf.groupby("replicate")
            .apply(lambda g: (g["predicted_has_12h"] == g["true_has_12h"]).mean())
        )
        acc_dict[meth] = per_rep.values

    fig, _ = plot_method_comparison_bar(acc_dict, metric="accuracy")
    stem = out / "method_comparison"
    _save_fig(fig, stem)
    saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    plt.close(fig)

    # --- 5. Parameter estimation (if period estimates exist) ---
    has_period = df["period_12h_est"].notna().any()
    if has_period:
        for meth in methods:
            mdf = df[df["method"] == meth]
            valid = mdf["period_12h_est"].notna() & mdf["true_has_12h"]
            if valid.sum() < 3:
                continue
            sub = mdf.loc[valid]
            # True period is 12 h for independent ultradian genes
            true_p = np.full(len(sub), 12.0)
            fig, _ = plot_parameter_estimation(
                true_p, sub["period_12h_est"].values, param_name="period"
            )
            fig.axes[0].set_title(f"Period recovery — {_method_label(meth)}")
            stem = out / f"param_period_{meth}"
            _save_fig(fig, stem)
            saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
            plt.close(fig)

    # --- 6. Harmonic disentanglement ---
    if df["true_class"].isin(["harmonic", "independent_ultradian"]).any():
        fig, _ = plot_harmonic_disentangle(df)
        stem = out / "harmonic_disentangle"
        _save_fig(fig, stem)
        saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
        plt.close(fig)

    # --- 7. Multi-panel summary figure ---
    fig = _make_summary_panel(df, methods)
    stem = out / "summary_panel"
    _save_fig(fig, stem)
    saved.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    plt.close(fig)

    return saved


# ============================================================================
# Multi-panel summary (internal)
# ============================================================================

def _make_summary_panel(df, methods):
    """Create a 2x2 multi-panel overview figure.

    Panels:
        (a) Accuracy heatmap
        (b) ROC curves
        (c) Method comparison bar
        (d) Harmonic disentanglement
    """
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.85))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.40,
        left=0.10, right=0.95, top=0.95, bottom=0.08,
    )

    # (a) Heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    plot_accuracy_heatmap(df, methods=methods, ax=ax_a)
    ax_a.set_title("a", fontsize=9, fontweight="bold", loc="left", pad=4)

    # (b) ROC
    ax_b = fig.add_subplot(gs[0, 1])
    plot_roc_curves(df, methods=methods, ax=ax_b)
    ax_b.set_title("b", fontsize=9, fontweight="bold", loc="left", pad=4)

    # (c) Bar chart
    ax_c = fig.add_subplot(gs[1, 0])
    acc_dict = {}
    for meth in methods:
        mdf = df[df["method"] == meth]
        per_rep = (
            mdf.groupby("replicate")
            .apply(lambda g: (g["predicted_has_12h"] == g["true_has_12h"]).mean())
        )
        acc_dict[meth] = per_rep.values
    plot_method_comparison_bar(acc_dict, metric="accuracy", ax=ax_c)
    ax_c.set_title("c", fontsize=9, fontweight="bold", loc="left", pad=4)

    # (d) Harmonic disentangle
    ax_d = fig.add_subplot(gs[1, 1])
    if df["true_class"].isin(["harmonic", "independent_ultradian"]).any():
        plot_harmonic_disentangle(df, ax=ax_d)
    else:
        ax_d.text(0.5, 0.5, "No harmonic\nscenarios", ha="center",
                  va="center", transform=ax_d.transAxes, fontsize=8,
                  color="0.5")
        ax_d.set_axis_off()
    ax_d.set_title("d", fontsize=9, fontweight="bold", loc="left", pad=4)

    return fig


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate benchmark figures for CHORD."
    )
    parser.add_argument(
        "csv", type=str,
        help="Path to benchmark results CSV (output of run_benchmark).",
    )
    parser.add_argument(
        "--outdir", type=str, default="figures/benchmark",
        help="Output directory for figures (default: figures/benchmark).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    saved = plot_benchmark_summary(df, output_dir=args.outdir)
    print(f"Saved {len(saved)} files to {args.outdir}/")
    for p in saved:
        print(f"  {p}")
