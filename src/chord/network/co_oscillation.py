"""
Cross-gene co-oscillation network module for CHORD.

Discovers groups of genes sharing similar oscillatory patterns
(co-oscillation modules) and uses module-level information to
refine ambiguous single-gene rhythm classifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------

_FEATURE_COLS = ["A_24", "A_12", "A_8", "phi_24", "phi_12", "phi_8",
                 "log_bayes_factor", "T_12_fitted"]


def build_oscillation_features(
    bhdt_results_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract oscillation feature vectors from BHDT results.

    Parameters
    ----------
    bhdt_results_df : pd.DataFrame
        Output of ``chord.bhdt.pipeline.run_bhdt``.

    Returns
    -------
    features : np.ndarray, shape (n_genes, n_features)
        Feature matrix with NaN values imputed to column medians.
    gene_names : np.ndarray of str
        Gene names aligned with rows of *features*.
    """
    df = bhdt_results_df.copy()

    # Ensure expected columns exist; fill missing ones with NaN
    for col in _FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    features = df[_FEATURE_COLS].values.astype(np.float64)
    gene_names = df["gene"].values.astype(str)

    # Impute NaN with column median (robust to outliers)
    for j in range(features.shape[1]):
        col = features[:, j]
        mask = np.isfinite(col)
        if mask.any():
            median_val = np.median(col[mask])
            col[~mask] = median_val
        else:
            col[:] = 0.0  # all-NaN column → zero

    return features, gene_names


# ---------------------------------------------------------------------------
# 2. Module discovery
# ---------------------------------------------------------------------------

def _auto_n_modules(X: np.ndarray, method: str, max_k: int = 20) -> int:
    """Choose *n_modules* by maximising silhouette score."""
    n = X.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2

    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        if method == "hierarchical":
            Z = linkage(X, method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
        else:  # spectral
            from sklearn.cluster import SpectralClustering
            labels = SpectralClustering(
                n_clusters=k, affinity="precomputed", random_state=42,
                assign_labels="kmeans",
            ).fit_predict(_correlation_affinity(X))

        # silhouette requires >= 2 unique labels
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score

    return best_k


def _correlation_affinity(X: np.ndarray) -> np.ndarray:
    """Build a non-negative affinity matrix from Pearson correlations."""
    C = np.corrcoef(X)
    # Shift from [-1, 1] to [0, 1]
    A = (C + 1.0) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def discover_modules(
    features: np.ndarray,
    gene_names: np.ndarray,
    method: str = "hierarchical",
    n_modules: Optional[int] = None,
    min_module_size: int = 5,
) -> Dict[int, List[str]]:
    """Cluster genes into co-oscillation modules.

    Parameters
    ----------
    features : np.ndarray, shape (n_genes, n_features)
        Feature matrix (output of :func:`build_oscillation_features`).
    gene_names : np.ndarray of str
        Gene names aligned with rows of *features*.
    method : {'hierarchical', 'spectral'}
        Clustering algorithm.
    n_modules : int or None
        Number of modules.  If *None*, chosen automatically via
        silhouette score.
    min_module_size : int
        Modules smaller than this are merged into the nearest larger
        module.

    Returns
    -------
    dict
        ``{module_id: [gene_name, ...]}``
    """
    if method not in ("hierarchical", "spectral"):
        raise ValueError(f"method must be 'hierarchical' or 'spectral', got '{method}'")

    n_genes = features.shape[0]
    if n_genes < 3:
        return {0: list(gene_names)}

    # Standardise features (zero mean, unit variance)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Determine k
    k = n_modules if n_modules is not None else _auto_n_modules(X, method)
    k = max(2, min(k, n_genes - 1))

    # Cluster
    if method == "hierarchical":
        Z = linkage(X, method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
    else:
        from sklearn.cluster import SpectralClustering
        A = _correlation_affinity(X)
        labels = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=42,
            assign_labels="kmeans",
        ).fit_predict(A)
        labels = labels + 1  # 1-indexed to match fcluster convention

    # Build module dict
    modules: Dict[int, List[str]] = {}
    for label, gname in zip(labels, gene_names):
        modules.setdefault(int(label), []).append(gname)

    # Merge small modules into nearest large module
    modules = _merge_small_modules(modules, features, gene_names, labels,
                                   min_module_size)

    return modules


def _merge_small_modules(
    modules: Dict[int, List[str]],
    features: np.ndarray,
    gene_names: np.ndarray,
    labels: np.ndarray,
    min_module_size: int,
) -> Dict[int, List[str]]:
    """Merge modules with fewer than *min_module_size* genes into the
    nearest larger module (by centroid Euclidean distance)."""
    name_to_idx = {g: i for i, g in enumerate(gene_names)}

    large = {mid: genes for mid, genes in modules.items()
             if len(genes) >= min_module_size}
    small = {mid: genes for mid, genes in modules.items()
             if len(genes) < min_module_size}

    if not large:
        # All modules are small — return as-is to avoid losing data
        return modules

    # Compute centroids of large modules
    large_centroids = {}
    for mid, genes in large.items():
        idxs = [name_to_idx[g] for g in genes]
        large_centroids[mid] = features[idxs].mean(axis=0)

    for _small_mid, genes in small.items():
        idxs = [name_to_idx[g] for g in genes]
        centroid = features[idxs].mean(axis=0)
        # Find nearest large module
        best_mid, best_dist = None, np.inf
        for mid, lc in large_centroids.items():
            d = np.linalg.norm(centroid - lc)
            if d < best_dist:
                best_mid, best_dist = mid, d
        large[best_mid].extend(genes)

    # Re-index modules starting from 0
    return {i: genes for i, genes in enumerate(large.values())}
# ---------------------------------------------------------------------------
# 3. Phase coherence (Rayleigh test)
# ---------------------------------------------------------------------------

def module_phase_coherence(
    phases: np.ndarray,
    module_assignments: Dict[int, List[str]],
    gene_names: Optional[np.ndarray] = None,
    gene_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[int, dict]:
    """Compute Rayleigh test statistics for phase coherence per module.

    Parameters
    ----------
    phases : np.ndarray, shape (n_genes,)
        Phase values in radians (e.g. ``phi_12`` column).
    module_assignments : dict
        ``{module_id: [gene_name, ...]}`` from :func:`discover_modules`.
    gene_names : np.ndarray, optional
        Gene names aligned with *phases*.  Required if *gene_to_idx* is
        not provided.
    gene_to_idx : dict, optional
        Pre-built ``{gene_name: row_index}`` mapping.

    Returns
    -------
    dict
        ``{module_id: {mean_phase, resultant_length, rayleigh_p, n_genes}}``
    """
    if gene_to_idx is None:
        if gene_names is None:
            raise ValueError("Provide gene_names or gene_to_idx")
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    results: Dict[int, dict] = {}
    for mid, genes in module_assignments.items():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idxs:
            continue
        phi = phases[idxs]
        phi = phi[np.isfinite(phi)]
        n = len(phi)
        if n == 0:
            results[mid] = dict(mean_phase=np.nan, resultant_length=0.0,
                                rayleigh_p=1.0, n_genes=0)
            continue

        # Mean resultant length  R = |mean(exp(i*phi))|
        C = np.mean(np.cos(phi))
        S = np.mean(np.sin(phi))
        R = np.sqrt(C ** 2 + S ** 2)
        mean_phase = np.arctan2(S, C) % (2 * np.pi)

        # Rayleigh test: p ≈ exp(-n * R^2)  (large-sample approximation)
        # More precise: p = exp(sqrt(1 + 4n + 4(n^2 - R_bar^2)) - (1 + 2n))
        nR2 = n * R ** 2
        rayleigh_p = np.exp(
            np.sqrt(1 + 4 * n + 4 * (n ** 2 - nR2)) - (1 + 2 * n)
        )
        rayleigh_p = min(rayleigh_p, 1.0)

        results[mid] = dict(
            mean_phase=float(mean_phase),
            resultant_length=float(R),
            rayleigh_p=float(rayleigh_p),
            n_genes=n,
        )

    return results


# ---------------------------------------------------------------------------
# 4. Rhythm class enrichment (Fisher exact test)
# ---------------------------------------------------------------------------

def module_rhythm_enrichment(
    classifications: np.ndarray,
    module_assignments: Dict[int, List[str]],
    gene_names: Optional[np.ndarray] = None,
    gene_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[int, dict]:
    """Test whether each module is enriched for a specific rhythm class.

    Uses Fisher's exact test (one-sided, greater) for each rhythm class
    within each module.

    Parameters
    ----------
    classifications : np.ndarray of str, shape (n_genes,)
        Classification labels (e.g. ``classification`` column).
    module_assignments : dict
        ``{module_id: [gene_name, ...]}``
    gene_names : np.ndarray, optional
        Gene names aligned with *classifications*.
    gene_to_idx : dict, optional
        Pre-built mapping.

    Returns
    -------
    dict
        ``{module_id: {dominant_class, enrichment_p, odds_ratio, n_genes}}``
    """
    if gene_to_idx is None:
        if gene_names is None:
            raise ValueError("Provide gene_names or gene_to_idx")
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    all_classes = np.unique(classifications[classifications != None])  # noqa: E711
    total_n = len(classifications)

    results: Dict[int, dict] = {}
    for mid, genes in module_assignments.items():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idxs:
            continue
        mod_classes = classifications[idxs]
        n_mod = len(idxs)

        best_class, best_p, best_or = "none", 1.0, 1.0
        for cls in all_classes:
            # 2x2 contingency table
            a = np.sum(mod_classes == cls)          # in module & class
            b = n_mod - a                           # in module & not class
            c = np.sum(classifications == cls) - a  # not in module & class
            d = total_n - n_mod - c                 # not in module & not class

            # Ensure non-negative counts
            c = max(c, 0)
            d = max(d, 0)

            table = np.array([[a, b], [c, d]])
            try:
                odds_ratio, p_val = fisher_exact(table, alternative="greater")
            except ValueError:
                odds_ratio, p_val = 1.0, 1.0

            if p_val < best_p:
                best_class = str(cls)
                best_p = float(p_val)
                best_or = float(odds_ratio) if np.isfinite(odds_ratio) else np.inf

        results[mid] = dict(
            dominant_class=best_class,
            enrichment_p=best_p,
            odds_ratio=best_or,
            n_genes=n_mod,
        )

    return results
# ---------------------------------------------------------------------------
# 5. Refine ambiguous classifications using module context
# ---------------------------------------------------------------------------

def refine_classification_with_modules(
    bhdt_results_df: pd.DataFrame,
    modules: Dict[int, List[str]],
    module_stats: Dict[int, dict],
    enrichment_p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Refine ambiguous single-gene classifications using module context.

    If a gene is classified as ``"ambiguous"`` but belongs to a module
    strongly enriched for a specific rhythm class, reclassify it as
    ``"likely_<dominant_class>"``.  Confident classifications are never
    overridden.

    Parameters
    ----------
    bhdt_results_df : pd.DataFrame
        Original BHDT results (must contain ``gene`` and ``classification``).
    modules : dict
        ``{module_id: [gene_name, ...]}``
    module_stats : dict
        Output of :func:`module_rhythm_enrichment`.
    enrichment_p_threshold : float
        Only refine if the module enrichment p-value is below this
        threshold.

    Returns
    -------
    pd.DataFrame
        Copy of *bhdt_results_df* with an added ``refined_classification``
        column and a ``module_id`` column.
    """
    df = bhdt_results_df.copy()

    # Build gene → module_id mapping
    gene_to_module: Dict[str, int] = {}
    for mid, genes in modules.items():
        for g in genes:
            gene_to_module[g] = mid

    df["module_id"] = df["gene"].map(gene_to_module)
    df["refined_classification"] = df["classification"].copy()

    for mid, stats in module_stats.items():
        if stats["enrichment_p"] >= enrichment_p_threshold:
            continue
        dominant = stats["dominant_class"]
        if dominant in ("ambiguous", "none"):
            continue

        mask = (
            (df["module_id"] == mid)
            & (df["classification"] == "ambiguous")
        )
        df.loc[mask, "refined_classification"] = f"likely_{dominant}"

    return df


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def plot_module_network(
    modules: Dict[int, List[str]],
    module_stats: Dict[int, dict],
    phase_coherence: Optional[Dict[int, dict]] = None,
    output_path: Optional[str] = None,
):
    """Visualise the co-oscillation network.

    Nodes represent modules (sized by gene count, coloured by dominant
    rhythm class).  Edges connect modules whose mean phases are
    correlated (cosine similarity > 0.5).

    Parameters
    ----------
    modules : dict
        ``{module_id: [gene_name, ...]}``
    module_stats : dict
        Output of :func:`module_rhythm_enrichment`.
    phase_coherence : dict, optional
        Output of :func:`module_phase_coherence`.  If provided, edges
        are drawn between modules with similar mean phases.
    output_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colour map for rhythm classes
    class_colours = {
        "independent_ultradian": "#e74c3c",
        "harmonic_only": "#3498db",
        "ambiguous": "#95a5a6",
        "circadian_dominant": "#2ecc71",
        "likely_independent_ultradian": "#e67e22",
        "likely_harmonic_only": "#1abc9c",
    }
    default_colour = "#bdc3c7"

    mids = sorted(modules.keys())
    n = len(mids)

    # Node positions — circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {mid: (np.cos(a), np.sin(a)) for mid, a in zip(mids, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))

    # Edges (phase similarity)
    if phase_coherence is not None and len(phase_coherence) > 1:
        for i, m1 in enumerate(mids):
            for m2 in mids[i + 1:]:
                if m1 not in phase_coherence or m2 not in phase_coherence:
                    continue
                p1 = phase_coherence[m1].get("mean_phase", np.nan)
                p2 = phase_coherence[m2].get("mean_phase", np.nan)
                if np.isnan(p1) or np.isnan(p2):
                    continue
                sim = np.cos(p1 - p2)
                if sim > 0.5:
                    x = [pos[m1][0], pos[m2][0]]
                    y = [pos[m1][1], pos[m2][1]]
                    ax.plot(x, y, "-", color="#cccccc",
                            linewidth=1 + 2 * sim, alpha=0.6, zorder=1)

    # Nodes
    for mid in mids:
        x, y = pos[mid]
        n_genes = len(modules[mid])
        stats = module_stats.get(mid, {})
        dom_class = stats.get("dominant_class", "none")
        colour = class_colours.get(dom_class, default_colour)
        size = 200 + 40 * n_genes  # scale node size

        ax.scatter(x, y, s=size, c=colour, edgecolors="black",
                   linewidths=0.8, zorder=2, alpha=0.85)
        ax.annotate(f"M{mid}\n({n_genes})", (x, y),
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", zorder=3)

    # Legend
    handles = []
    for cls, col in class_colours.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=col, markersize=8,
                                  label=cls.replace("_", " ")))
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.9, title="Rhythm class")

    ax.set_title("Co-oscillation Module Network", fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
