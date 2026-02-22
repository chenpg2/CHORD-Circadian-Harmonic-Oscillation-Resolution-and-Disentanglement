"""
Weight learner for CHORD ensemble integration.

Learns optimal weights for combining BHDT and PINOD results using
synthetic data with known ground truth from chord.simulation.generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Ground-truth label mapping                                         #
# ------------------------------------------------------------------ #

def _ground_truth_label(truth_dict: Dict[str, Any]) -> str:
    """Convert generator truth dict to a canonical classification label.

    Parameters
    ----------
    truth_dict : dict
        The ``truth`` field from a generator result, containing at least
        ``has_independent_12h``, ``has_harmonic_12h``, and ``oscillators``.

    Returns
    -------
    str
        One of: ``independent_ultradian``, ``harmonic``,
        ``circadian_only``, ``non_rhythmic``.
    """
    has_ind = truth_dict.get("has_independent_12h", False)
    has_harm = truth_dict.get("has_harmonic_12h", False)
    oscillators = truth_dict.get("oscillators", [])

    if has_ind and not has_harm:
        return "independent_ultradian"
    if not has_ind and has_harm:
        return "harmonic"
    if not has_ind and not has_harm:
        if len(oscillators) > 0:
            return "circadian_only"
        return "non_rhythmic"
    # Both flags set — unusual, treat as independent
    return "independent_ultradian"


# ------------------------------------------------------------------ #
#  Normalise method-specific labels to the 4-class vocabulary         #
# ------------------------------------------------------------------ #

_BHDT_MAP = {
    "independent_ultradian": "independent_ultradian",
    "likely_independent_ultradian": "independent_ultradian",
    "harmonic": "harmonic",
    "circadian_only": "circadian_only",
    "non_rhythmic": "non_rhythmic",
    "ambiguous": "ambiguous",
}

_PINOD_MAP = {
    "independent_ultradian": "independent_ultradian",
    "harmonic": "harmonic",
    "circadian_only": "circadian_only",
    "non_rhythmic": "non_rhythmic",
    "damped_ultradian": "independent_ultradian",
    "multi_ultradian": "independent_ultradian",
    "ambiguous": "ambiguous",
}


def _normalise_label(label: str, source: str) -> str:
    """Map a method-specific label to the 4-class vocabulary."""
    mapping = _BHDT_MAP if source == "bhdt" else _PINOD_MAP
    return mapping.get(str(label), "ambiguous")


# ------------------------------------------------------------------ #
#  Weighted voting for a single gene                                  #
# ------------------------------------------------------------------ #

def _vote_single(
    bhdt_cls: str,
    pinod_cls: str,
    bhdt_conf: float,
    pinod_conf: float,
    w_bhdt: float,
    w_pinod: float,
) -> str:
    """Return consensus label for one gene via weighted confidence voting."""
    bhdt_norm = _normalise_label(bhdt_cls, "bhdt")
    pinod_norm = _normalise_label(pinod_cls, "pinod")

    # Both agree
    if bhdt_norm == pinod_norm:
        return bhdt_norm

    # One is ambiguous — defer to the other
    if bhdt_norm == "ambiguous":
        return pinod_norm
    if pinod_norm == "ambiguous":
        return bhdt_norm

    # Weighted confidence decides
    bhdt_score = w_bhdt * (bhdt_conf if np.isfinite(bhdt_conf) else 0.5)
    pinod_score = w_pinod * (pinod_conf if np.isfinite(pinod_conf) else 0.5)
    return pinod_norm if pinod_score >= bhdt_score else bhdt_norm


# ------------------------------------------------------------------ #
#  Evaluate a weight pair                                             #
# ------------------------------------------------------------------ #

def _evaluate_weights(
    bhdt_cls: List[str],
    pinod_cls: List[str],
    bhdt_conf: np.ndarray,
    pinod_conf: np.ndarray,
    true_labels: List[str],
    w_bhdt: float,
    w_pinod: float,
) -> Dict[str, Any]:
    """Evaluate a (w_bhdt, w_pinod) pair on labelled data.

    Parameters
    ----------
    bhdt_cls, pinod_cls : list of str
        Per-gene classification strings from each method.
    bhdt_conf, pinod_conf : array-like
        Per-gene confidence values in [0, 1].
    true_labels : list of str
        Ground-truth labels (4-class vocabulary).
    w_bhdt, w_pinod : float
        Voting weights (should sum to 1).

    Returns
    -------
    dict with ``accuracy``, ``f1_macro``, ``per_class`` metrics.
    """
    n = len(true_labels)
    preds = [
        _vote_single(bhdt_cls[i], pinod_cls[i],
                      float(bhdt_conf[i]), float(pinod_conf[i]),
                      w_bhdt, w_pinod)
        for i in range(n)
    ]

    # Overall accuracy
    correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
    accuracy = correct / n if n > 0 else 0.0

    # Per-class precision / recall / F1
    classes = sorted(set(true_labels))
    per_class = {}
    f1_values = []
    for cls in classes:
        tp = sum(1 for p, t in zip(preds, true_labels) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, true_labels) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, true_labels) if p != cls and t == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": sum(1 for t in true_labels if t == cls)}
        f1_values.append(f1)

    f1_macro = float(np.mean(f1_values)) if f1_values else 0.0

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "per_class": per_class,
        "predictions": preds,
    }


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def learn_weights(
    n_genes: int = 200,
    seed: int = 42,
    bhdt_results: Optional[pd.DataFrame] = None,
    pinod_results: Optional[pd.DataFrame] = None,
    genome_data: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Learn optimal ensemble weights from synthetic data.

    Parameters
    ----------
    n_genes : int
        Number of synthetic genes to generate.
    seed : int
        Random seed for reproducibility.
    bhdt_results : pd.DataFrame, optional
        Pre-computed BHDT results (avoids re-running).
    pinod_results : pd.DataFrame, optional
        Pre-computed PINOD results (avoids re-running).
    genome_data : dict, optional
        Pre-generated synthetic data from ``generate_genome_like()``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        ``optimal_weights`` : dict with ``bhdt`` and ``pinod`` floats
        ``best_accuracy`` : float
        ``best_f1`` : float
        ``grid_results`` : list of dicts (one per weight pair tested)
        ``confusion`` : dict mapping (true, pred) to count
        ``per_class`` : dict of per-class metrics at optimal weights
    """
    from chord.simulation.generator import generate_genome_like

    # --- Generate or reuse synthetic data ----------------------------- #
    if genome_data is None:
        if verbose:
            print(f"Generating synthetic genome ({n_genes} genes, seed={seed})...")
        genome_data = generate_genome_like(n_genes=n_genes, seed=seed)

    expr = genome_data["expr"]
    t = genome_data["t"]
    truths = genome_data["truth"]

    # Ground-truth labels
    true_labels = [_ground_truth_label(tr) for tr in truths]

    # --- Run BHDT if needed ------------------------------------------- #
    if bhdt_results is None:
        if verbose:
            print("Running BHDT pipeline...")
        from chord.bhdt.pipeline import run_bhdt
        bhdt_results = run_bhdt(expr, t, method="analytic",
                                n_jobs=-1, verbose=verbose)

    # --- Run PINOD if needed ------------------------------------------ #
    if pinod_results is None:
        if verbose:
            print("Running PINOD decomposition (this may take a while)...")
        from chord.pinod.decompose import decompose
        pinod_results = decompose(expr, t, n_epochs=300, verbose=verbose)

    # --- Extract per-gene classifications and confidences ------------- #
    bhdt_cls = bhdt_results["classification"].tolist()
    pinod_cls = pinod_results["classification"].tolist()

    # BHDT confidence: sigmoid of log Bayes factor
    lbf = bhdt_results.get("log_bayes_factor", pd.Series(np.full(len(bhdt_cls), 0.0)))
    bhdt_conf = 1.0 / (1.0 + np.exp(-lbf.values.astype(float)))

    pinod_conf = pinod_results.get(
        "confidence", pd.Series(np.full(len(pinod_cls), 0.5))
    ).values.astype(float)

    # --- Grid search over weight pairs -------------------------------- #
    grid_steps = np.arange(0.1, 1.0, 0.1)
    grid_results = []
    best = {"accuracy": -1.0, "w_bhdt": 0.5, "w_pinod": 0.5}

    for w_b in grid_steps:
        w_p = round(1.0 - w_b, 2)
        if w_p < 0.05:
            continue
        metrics = _evaluate_weights(
            bhdt_cls, pinod_cls, bhdt_conf, pinod_conf,
            true_labels, w_b, w_p,
        )
        entry = {"w_bhdt": round(w_b, 2), "w_pinod": w_p,
                 "accuracy": metrics["accuracy"],
                 "f1_macro": metrics["f1_macro"]}
        grid_results.append(entry)

        if verbose:
            print(f"  w_bhdt={w_b:.1f}  w_pinod={w_p:.1f}  "
                  f"acc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}")

        if metrics["accuracy"] > best["accuracy"]:
            best.update({"accuracy": metrics["accuracy"],
                         "f1": metrics["f1_macro"],
                         "w_bhdt": round(w_b, 2),
                         "w_pinod": w_p,
                         "per_class": metrics["per_class"],
                         "predictions": metrics["predictions"]})

    # --- Build confusion matrix at optimal weights -------------------- #
    confusion: Dict[tuple[str, str], int] = {}
    if "predictions" in best:
        for t_lbl, p_lbl in zip(true_labels, best["predictions"]):
            key = (t_lbl, p_lbl)
            confusion[key] = confusion.get(key, 0) + 1

    if verbose:
        print(f"\nOptimal: w_bhdt={best['w_bhdt']:.2f}, "
              f"w_pinod={best['w_pinod']:.2f}, "
              f"accuracy={best['accuracy']:.3f}, "
              f"F1={best.get('f1', 0):.3f}")

    return {
        "optimal_weights": {"bhdt": best["w_bhdt"], "pinod": best["w_pinod"]},
        "best_accuracy": best["accuracy"],
        "best_f1": best.get("f1", 0.0),
        "grid_results": grid_results,
        "confusion": confusion,
        "per_class": best.get("per_class", {}),
    }
