"""
Hierarchical Bayesian parameter sharing for CHORD.

Uses co-oscillation modules (groups of genes with similar rhythm patterns)
to share oscillator parameters via hierarchical priors, improving detection
power for individual genes through empirical Bayes shrinkage.

This implements Task 3.4 from the CHORD design plan.  The approach is
James-Stein / empirical Bayes — fast, no MCMC needed:

1. Estimate module-level hyperparameters from BHDT point estimates.
2. Shrink gene-level estimates toward the module mean.
3. Reclassify ambiguous genes using module-level prior information.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple

from scipy.stats import vonmises

from chord.network.co_oscillation import (
    build_oscillation_features,
    discover_modules,
    module_phase_coherence,
    module_rhythm_enrichment,
)


# ---------------------------------------------------------------------------
# Circular statistics helpers
# ---------------------------------------------------------------------------

def _circular_mean(angles: np.ndarray) -> float:
    """Compute circular mean of angles in radians."""
    S = np.nanmean(np.sin(angles))
    C = np.nanmean(np.cos(angles))
    return float(np.arctan2(S, C))


def _circular_concentration(angles: np.ndarray) -> float:
    """Estimate Von Mises concentration parameter kappa via ML.

    Uses the approximation from Mardia & Jupp (2000):
        R_bar = mean resultant length
        kappa ≈ R_bar * (2 - R_bar^2) / (1 - R_bar^2)   for R_bar < 0.85
        kappa ≈ 1 / (2*(1 - R_bar) - (1 - R_bar)^2)     for R_bar >= 0.85
    """
    S = np.nanmean(np.sin(angles))
    C = np.nanmean(np.cos(angles))
    R_bar = np.sqrt(S ** 2 + C ** 2)

    if R_bar < 1e-10:
        return 0.0
    if R_bar >= 0.85:
        kappa = 1.0 / (2.0 * (1.0 - R_bar) - (1.0 - R_bar) ** 2)
    else:
        kappa = R_bar * (2.0 - R_bar ** 2) / (1.0 - R_bar ** 2)

    return float(min(kappa, 500.0))  # cap to avoid numerical issues


# ---------------------------------------------------------------------------
# 1. Estimate module hyperparameters
# ---------------------------------------------------------------------------

def estimate_module_hyperparams(
    bhdt_df: pd.DataFrame,
    module_labels: Dict[int, list],
    min_module_size: int = 5,
) -> Dict[int, dict]:
    """Estimate hierarchical hyperparameters for each co-oscillation module.

    Parameters
    ----------
    bhdt_df : pd.DataFrame
        BHDT results with columns: gene, A_12, A_24, phi_12, phi_24,
        classification, f_test_12h_significant, log_bayes_factor.
    module_labels : dict
        ``{module_id: [gene_name, ...]}`` from ``discover_modules``.
    min_module_size : int
        Modules smaller than this are skipped (hyperparams set to None).

    Returns
    -------
    dict
        ``module_id -> {n_genes, mu_A_12, sigma_A_12, mu_A_24, sigma_A_24,
        mu_phi_12, kappa_phi_12, mu_phi_24, kappa_phi_24,
        prop_independent, prop_rhythmic}``
        Modules below *min_module_size* map to ``None``.
    """
    gene_to_row = {g: i for i, g in enumerate(bhdt_df["gene"].values)}

    A_12 = bhdt_df["A_12"].values.astype(float)
    A_24 = bhdt_df["A_24"].values.astype(float)
    phi_12 = bhdt_df["phi_12"].values.astype(float)
    phi_24 = bhdt_df["phi_24"].values.astype(float)
    classifications = bhdt_df["classification"].values.astype(str)

    # f_test_12h_significant may be bool or 0/1
    if "f_test_12h_significant" in bhdt_df.columns:
        rhythmic_flag = bhdt_df["f_test_12h_significant"].values.astype(bool)
    else:
        rhythmic_flag = np.zeros(len(bhdt_df), dtype=bool)

    hyperparams: Dict[int, Optional[dict]] = {}

    for mid, genes in module_labels.items():
        idxs = [gene_to_row[g] for g in genes if g in gene_to_row]
        n = len(idxs)

        if n < min_module_size:
            hyperparams[mid] = None
            continue

        idx = np.array(idxs)

        # Amplitude hyperparameters (normal model)
        a12 = A_12[idx]
        a24 = A_24[idx]
        valid_a12 = a12[np.isfinite(a12)]
        valid_a24 = a24[np.isfinite(a24)]

        mu_a12 = float(np.mean(valid_a12)) if len(valid_a12) > 0 else 0.0
        sigma_a12 = float(np.std(valid_a12, ddof=1)) if len(valid_a12) > 1 else 1.0
        mu_a24 = float(np.mean(valid_a24)) if len(valid_a24) > 0 else 0.0
        sigma_a24 = float(np.std(valid_a24, ddof=1)) if len(valid_a24) > 1 else 1.0

        # Ensure sigma > 0 for shrinkage denominator
        sigma_a12 = max(sigma_a12, 1e-8)
        sigma_a24 = max(sigma_a24, 1e-8)

        # Phase hyperparameters (Von Mises model)
        p12 = phi_12[idx]
        p24 = phi_24[idx]
        valid_p12 = p12[np.isfinite(p12)]
        valid_p24 = p24[np.isfinite(p24)]

        mu_p12 = _circular_mean(valid_p12) if len(valid_p12) > 0 else 0.0
        kappa_p12 = _circular_concentration(valid_p12) if len(valid_p12) > 1 else 0.0
        mu_p24 = _circular_mean(valid_p24) if len(valid_p24) > 0 else 0.0
        kappa_p24 = _circular_concentration(valid_p24) if len(valid_p24) > 1 else 0.0

        # Classification proportions
        cls = classifications[idx]
        prop_indep = float(np.mean(
            (cls == "independent_ultradian") | (cls == "likely_independent_ultradian")
        ))
        prop_rhythmic = float(np.mean(rhythmic_flag[idx]))

        hyperparams[mid] = {
            "n_genes": n,
            "mu_A_12": mu_a12,
            "sigma_A_12": sigma_a12,
            "mu_A_24": mu_a24,
            "sigma_A_24": sigma_a24,
            "mu_phi_12": mu_p12,
            "kappa_phi_12": kappa_p12,
            "mu_phi_24": mu_p24,
            "kappa_phi_24": kappa_p24,
            "prop_independent": prop_indep,
            "prop_rhythmic": prop_rhythmic,
        }

    return hyperparams


# ---------------------------------------------------------------------------
# 2. Empirical Bayes shrinkage
# ---------------------------------------------------------------------------

def _shrinkage_weight(sigma_module: float, sigma_gene: float) -> float:
    """James-Stein shrinkage weight: w = sigma_module^2 / (sigma_module^2 + sigma_gene^2).

    w → 1 when gene noise is small relative to module spread (keep gene estimate).
    w → 0 when gene noise dominates (shrink toward module mean).
    """
    s2_mod = sigma_module ** 2
    s2_gene = sigma_gene ** 2
    denom = s2_mod + s2_gene
    if denom < 1e-16:
        return 1.0
    return s2_mod / denom


def _circular_shrinkage(phi_gene: float, mu_module: float,
                         kappa_module: float, kappa_gene: float) -> float:
    """Shrink a circular estimate toward the module mean.

    Combines the gene and module Von Mises distributions:
        posterior direction ≈ atan2(kappa_gene*sin(phi_gene) + kappa_module*sin(mu_module),
                                    kappa_gene*cos(phi_gene) + kappa_module*cos(mu_module))
    """
    S = kappa_gene * np.sin(phi_gene) + kappa_module * np.sin(mu_module)
    C = kappa_gene * np.cos(phi_gene) + kappa_module * np.cos(mu_module)
    return float(np.arctan2(S, C))


def shrink_parameters(
    bhdt_df: pd.DataFrame,
    module_labels: Dict[int, list],
    hyperparams: Dict[int, Optional[dict]],
    shrinkage_strength: float = 1.0,
) -> pd.DataFrame:
    """Apply empirical Bayes shrinkage to gene-level parameter estimates.

    For each gene in a valid module, the shrunk amplitude is:
        A_shrunk = w * A_gene + (1 - w) * mu_module
    where w is the James-Stein shrinkage weight, scaled by
    *shrinkage_strength* (higher = more shrinkage toward module mean).

    Phase shrinkage uses Von Mises posterior combination.

    Parameters
    ----------
    bhdt_df : pd.DataFrame
        BHDT results.
    module_labels : dict
        ``{module_id: [gene_name, ...]}``
    hyperparams : dict
        Output of :func:`estimate_module_hyperparams`.
    shrinkage_strength : float
        Multiplier on the module prior precision.  1.0 = standard
        empirical Bayes; >1 = stronger shrinkage; <1 = weaker.

    Returns
    -------
    pd.DataFrame
        Copy of *bhdt_df* with added columns: A_12_shrunk, A_24_shrunk,
        phi_12_shrunk, phi_24_shrunk, shrinkage_weight_A12,
        shrinkage_weight_phi12.
    """
    df = bhdt_df.copy()
    n = len(df)

    # Initialise output columns with original values
    df["A_12_shrunk"] = df["A_12"].values.astype(float)
    df["A_24_shrunk"] = df["A_24"].values.astype(float)
    df["phi_12_shrunk"] = df["phi_12"].values.astype(float)
    df["phi_24_shrunk"] = df["phi_24"].values.astype(float)
    df["shrinkage_weight_A12"] = 1.0
    df["shrinkage_weight_phi12"] = 1.0

    gene_to_row = {g: i for i, g in enumerate(df["gene"].values)}

    # Estimate per-gene noise from residual (use T_12_deviation if available)
    if "T_12_deviation" in df.columns:
        gene_sigma = df["T_12_deviation"].values.astype(float)
        gene_sigma = np.where(np.isfinite(gene_sigma) & (gene_sigma > 0),
                              gene_sigma, np.nan)
    else:
        gene_sigma = np.full(n, np.nan)

    # Fallback: median gene sigma across dataset
    valid_sigma = gene_sigma[np.isfinite(gene_sigma)]
    fallback_sigma = float(np.median(valid_sigma)) if len(valid_sigma) > 0 else 1.0

    for mid, genes in module_labels.items():
        hp = hyperparams.get(mid)
        if hp is None:
            continue  # skip small modules

        for g in genes:
            if g not in gene_to_row:
                continue
            i = gene_to_row[g]

            a12 = df.at[df.index[i], "A_12"]
            a24 = df.at[df.index[i], "A_24"]
            p12 = df.at[df.index[i], "phi_12"]
            p24 = df.at[df.index[i], "phi_24"]

            # Skip genes with NaN values
            if not np.isfinite(a12) or not np.isfinite(a24):
                continue

            # Gene-level noise estimate
            sg = gene_sigma[i] if np.isfinite(gene_sigma[i]) else fallback_sigma

            # --- Amplitude shrinkage ---
            # Effective module sigma scaled by shrinkage_strength
            eff_sigma_12 = hp["sigma_A_12"] / max(shrinkage_strength, 1e-8)
            eff_sigma_24 = hp["sigma_A_24"] / max(shrinkage_strength, 1e-8)

            w12 = _shrinkage_weight(eff_sigma_12, sg)
            w24 = _shrinkage_weight(eff_sigma_24, sg)

            df.iat[i, df.columns.get_loc("A_12_shrunk")] = (
                w12 * a12 + (1.0 - w12) * hp["mu_A_12"]
            )
            df.iat[i, df.columns.get_loc("A_24_shrunk")] = (
                w24 * a24 + (1.0 - w24) * hp["mu_A_24"]
            )
            df.iat[i, df.columns.get_loc("shrinkage_weight_A12")] = w12

            # --- Phase shrinkage ---
            if np.isfinite(p12) and np.isfinite(p24):
                # Gene-level kappa: inversely related to noise
                kappa_gene = max(1.0 / (sg + 1e-8), 0.1)
                kappa_mod_12 = hp["kappa_phi_12"] * shrinkage_strength
                kappa_mod_24 = hp["kappa_phi_24"] * shrinkage_strength

                df.iat[i, df.columns.get_loc("phi_12_shrunk")] = (
                    _circular_shrinkage(p12, hp["mu_phi_12"],
                                        kappa_mod_12, kappa_gene)
                )
                df.iat[i, df.columns.get_loc("phi_24_shrunk")] = (
                    _circular_shrinkage(p24, hp["mu_phi_24"],
                                        kappa_mod_24, kappa_gene)
                )
                # Phase shrinkage weight: proportion of gene contribution
                total_kappa = kappa_gene + kappa_mod_12
                df.iat[i, df.columns.get_loc("shrinkage_weight_phi12")] = (
                    kappa_gene / total_kappa if total_kappa > 0 else 1.0
                )

    return df


# ---------------------------------------------------------------------------
# 3. Hierarchical reclassification
# ---------------------------------------------------------------------------

def hierarchical_reclassify(
    bhdt_df: pd.DataFrame,
    module_labels: Dict[int, list],
    hyperparams: Dict[int, Optional[dict]],
    prior_weight: float = 0.3,
) -> pd.DataFrame:
    """Reclassify ambiguous genes using module-level prior information.

    For genes classified as ``'ambiguous'``:
    - If module has >60% independent genes → ``'likely_independent'``
    - If module has >60% harmonic genes → ``'likely_harmonic'``
    - Confidence is adjusted by *prior_weight* and module coherence.

    Genes that are not ambiguous, or whose module has no clear majority,
    keep their original classification.

    Parameters
    ----------
    bhdt_df : pd.DataFrame
        BHDT results (must contain ``gene``, ``classification``).
    module_labels : dict
        ``{module_id: [gene_name, ...]}``
    hyperparams : dict
        Output of :func:`estimate_module_hyperparams`.
    prior_weight : float
        Weight given to the module prior (0 = ignore module, 1 = fully
        trust module).  Default 0.3 is conservative.

    Returns
    -------
    pd.DataFrame
        Copy with added ``hierarchical_classification`` column.
    """
    df = bhdt_df.copy()
    df["hierarchical_classification"] = df["classification"].copy()

    gene_to_module: Dict[str, int] = {}
    for mid, genes in module_labels.items():
        for g in genes:
            gene_to_module[g] = mid

    classifications = df["classification"].values.astype(str)
    gene_names = df["gene"].values.astype(str)

    for i in range(len(df)):
        if classifications[i] != "ambiguous":
            continue

        g = gene_names[i]
        mid = gene_to_module.get(g)
        if mid is None:
            continue

        hp = hyperparams.get(mid)
        if hp is None:
            continue

        # Compute module class proportions (excluding ambiguous genes)
        module_genes = module_labels[mid]
        mod_idxs = [j for j, gn in enumerate(gene_names) if gn in set(module_genes)]
        mod_cls = classifications[mod_idxs]
        non_ambig = mod_cls[mod_cls != "ambiguous"]

        if len(non_ambig) == 0:
            continue  # all ambiguous — no information to borrow

        # Count independent vs harmonic
        n_indep = np.sum(
            (non_ambig == "independent_ultradian")
            | (non_ambig == "likely_independent_ultradian")
        )
        n_harmonic = np.sum(
            (non_ambig == "harmonic")
            | (non_ambig == "likely_harmonic")
        )
        n_total = len(non_ambig)

        prop_indep = n_indep / n_total
        prop_harmonic = n_harmonic / n_total

        # Apply threshold with prior_weight scaling
        threshold = 0.6 * (1.0 - prior_weight) + 0.4 * prior_weight
        # At prior_weight=0.3: threshold = 0.6*0.7 + 0.4*0.3 = 0.54
        # More permissive with higher prior_weight

        if prop_indep > threshold:
            df.iat[i, df.columns.get_loc("hierarchical_classification")] = (
                "likely_independent"
            )
        elif prop_harmonic > threshold:
            df.iat[i, df.columns.get_loc("hierarchical_classification")] = (
                "likely_harmonic"
            )

    return df


# ---------------------------------------------------------------------------
# 4. End-to-end pipeline
# ---------------------------------------------------------------------------

def run_hierarchical(
    bhdt_df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    min_module_size: int = 5,
    shrinkage_strength: float = 1.0,
    prior_weight: float = 0.3,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """End-to-end hierarchical Bayesian refinement pipeline.

    Steps:
        1. Extract oscillation features from BHDT results.
        2. Discover co-oscillation modules via clustering.
        3. Estimate module-level hyperparameters.
        4. Apply empirical Bayes shrinkage to gene parameters.
        5. Reclassify ambiguous genes using module priors.

    Parameters
    ----------
    bhdt_df : pd.DataFrame
        BHDT results DataFrame.
    n_clusters : int or None
        Number of co-oscillation modules.  ``None`` = auto-select.
    min_module_size : int
        Minimum genes per module for shrinkage to apply.
    shrinkage_strength : float
        Controls how strongly genes are pulled toward module mean.
    prior_weight : float
        Weight for module prior in reclassification (0–1).
    verbose : bool
        Print progress messages.

    Returns
    -------
    (updated_df, module_info)
        *updated_df* has shrunk parameters and hierarchical classifications.
        *module_info* is a dict with keys: ``modules``, ``hyperparams``,
        ``phase_coherence``, ``enrichment``.
    """
    if len(bhdt_df) < 3:
        warnings.warn("Too few genes for hierarchical analysis; returning input unchanged.")
        return bhdt_df.copy(), {}

    # Step 1: Extract features
    if verbose:
        print("[hierarchical] Extracting oscillation features ...")
    features, gene_names = build_oscillation_features(bhdt_df)

    # Step 2: Discover modules
    if verbose:
        print("[hierarchical] Discovering co-oscillation modules ...")
    modules = discover_modules(
        features, gene_names,
        method="hierarchical",
        n_modules=n_clusters,
        min_module_size=min_module_size,
    )
    n_mod = len(modules)
    if verbose:
        sizes = [len(g) for g in modules.values()]
        print(f"  Found {n_mod} modules (sizes: {sorted(sizes, reverse=True)})")

    # Step 3: Estimate hyperparameters
    if verbose:
        print("[hierarchical] Estimating module hyperparameters ...")
    hyperparams = estimate_module_hyperparams(
        bhdt_df, modules, min_module_size=min_module_size,
    )

    # Step 4: Shrinkage
    if verbose:
        print("[hierarchical] Applying empirical Bayes shrinkage ...")
    df = shrink_parameters(
        bhdt_df, modules, hyperparams,
        shrinkage_strength=shrinkage_strength,
    )

    # Step 5: Reclassify
    if verbose:
        print("[hierarchical] Reclassifying ambiguous genes ...")
    df = hierarchical_reclassify(
        df, modules, hyperparams, prior_weight=prior_weight,
    )

    # Attach module IDs
    gene_to_module = {}
    for mid, genes in modules.items():
        for g in genes:
            gene_to_module[g] = mid
    df["module_id"] = df["gene"].map(gene_to_module)

    # Compute auxiliary module-level statistics
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    phi_12 = bhdt_df["phi_12"].values.astype(float)
    phase_coh = module_phase_coherence(
        phi_12, modules, gene_names=gene_names, gene_to_idx=gene_to_idx,
    )
    enrichment = module_rhythm_enrichment(
        bhdt_df["classification"].values.astype(str),
        modules, gene_names=gene_names, gene_to_idx=gene_to_idx,
    )

    # Summary
    if verbose:
        n_ambig_before = (bhdt_df["classification"] == "ambiguous").sum()
        n_ambig_after = (df["hierarchical_classification"] == "ambiguous").sum()
        n_reclassified = n_ambig_before - n_ambig_after
        print(f"  Reclassified {n_reclassified}/{n_ambig_before} ambiguous genes")
        print("[hierarchical] Done.")

    module_info = {
        "modules": modules,
        "hyperparams": hyperparams,
        "phase_coherence": phase_coh,
        "enrichment": enrichment,
    }

    return df, module_info
