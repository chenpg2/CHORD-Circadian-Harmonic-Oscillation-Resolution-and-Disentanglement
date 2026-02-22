"""
Evaluation metrics for rhythm detection benchmarks.

All metrics are implemented with numpy/scipy/pandas only (no sklearn).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def classification_metrics(y_true, y_pred, labels=None):
    """Compute classification metrics without sklearn.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Ordered label set. Inferred from data if None.

    Returns
    -------
    dict
        accuracy, per_class (dict of precision/recall/f1 per label),
        macro_f1, confusion_matrix (2-D array), labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    n_labels = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Confusion matrix: rows = true, cols = pred
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        i = label_to_idx.get(yt)
        j = label_to_idx.get(yp)
        if i is not None and j is not None:
            cm[i, j] += 1

    accuracy = float(np.trace(cm)) / max(cm.sum(), 1)

    per_class = {}
    f1_values = []
    for idx, lab in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "labels": labels,
    }


def harmonic_disentangle_accuracy(y_true_is_independent, y_pred_is_independent):
    """Binary accuracy for harmonic-vs-independent classification.

    Parameters
    ----------
    y_true_is_independent : array-like of bool/int
        Ground truth: 1 if independent harmonic, 0 if dependent.
    y_pred_is_independent : array-like of bool/int
        Predicted labels.

    Returns
    -------
    dict
        accuracy, sensitivity (recall of positive=independent),
        specificity (recall of negative=dependent).
    """
    yt = np.asarray(y_true_is_independent, dtype=int)
    yp = np.asarray(y_pred_is_independent, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _circular_distance(a, b):
    """Signed circular distance in radians, mapped to [-pi, pi]."""
    d = (a - b) % (2.0 * np.pi)
    d = np.where(d > np.pi, d - 2.0 * np.pi, d)
    return d


def parameter_estimation_error(true_params, est_params):
    """Compute per-gene parameter estimation errors.

    Parameters
    ----------
    true_params : pd.DataFrame or dict of arrays
        Must contain columns/keys: 'period', 'amplitude', 'phase'.
        Each row corresponds to one gene.
    est_params : pd.DataFrame or dict of arrays
        Same structure as true_params.

    Returns
    -------
    dict
        per_gene : pd.DataFrame with columns period_error,
            relative_amplitude_error, circular_phase_error.
        summary : dict with mean/median/std for each error type.
    """
    true_df = pd.DataFrame(true_params)
    est_df = pd.DataFrame(est_params)

    period_error = np.abs(est_df["period"].values - true_df["period"].values)

    true_amp = true_df["amplitude"].values.astype(float)
    est_amp = est_df["amplitude"].values.astype(float)
    # Avoid division by zero
    safe_amp = np.where(true_amp > 0, true_amp, 1.0)
    relative_amp_error = np.abs(est_amp - true_amp) / safe_amp

    true_phase = true_df["phase"].values.astype(float)
    est_phase = est_df["phase"].values.astype(float)
    circ_phase_error = np.abs(_circular_distance(est_phase, true_phase))

    per_gene = pd.DataFrame(
        {
            "period_error": period_error,
            "relative_amplitude_error": relative_amp_error,
            "circular_phase_error": circ_phase_error,
        }
    )

    summary = {}
    for col in per_gene.columns:
        vals = per_gene[col].values
        summary[col] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }

    return {
        "per_gene": per_gene,
        "summary": summary,
    }


def roc_auc(y_true_binary, scores):
    """Compute ROC AUC without sklearn.

    Uses the trapezoidal rule on the empirical ROC curve.

    Parameters
    ----------
    y_true_binary : array-like of {0, 1}
        True binary labels.
    scores : array-like of float
        Predicted scores (higher = more likely positive).

    Returns
    -------
    dict
        auc, fpr (array), tpr (array), thresholds (array).
    """
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)

    n_pos = np.sum(y_true_binary == 1)
    n_neg = np.sum(y_true_binary == 0)

    if n_pos == 0 or n_neg == 0:
        return {
            "auc": float("nan"),
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "thresholds": np.array([]),
        }

    # Sort by descending score
    desc_idx = np.argsort(-scores)
    y_sorted = y_true_binary[desc_idx]
    scores_sorted = scores[desc_idx]

    # Unique thresholds (descending)
    distinct_vals, first_idx = np.unique(scores_sorted[::-1], return_index=True)
    # Reverse to get descending order
    thresholds = distinct_vals[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]

    for thresh in thresholds:
        predicted_pos = scores >= thresh
        tp = np.sum((predicted_pos) & (y_true_binary == 1))
        fp = np.sum((predicted_pos) & (y_true_binary == 0))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Sort by fpr for proper trapezoidal integration
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    tpr_arr = tpr_arr[sort_idx]

    auc = float(np.trapz(tpr_arr, fpr_arr))

    return {
        "auc": auc,
        "fpr": fpr_arr,
        "tpr": tpr_arr,
        "thresholds": thresholds,
    }
