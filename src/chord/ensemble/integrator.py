"""
Ensemble integration module for CHORD rhythm detection.

Combines results from BHDT (Bayesian Harmonic Disentanglement Test) and
PINOD (Physics-Informed Neural ODE) to produce a consensus classification
for each gene via weighted voting.

Default weights favour PINOD (0.6) over BHDT (0.4) because PINOD resolves
the harmonic-vs-independent ambiguity that limited BHDT in Phase 1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Classification vocabulary shared by BHDT and PINOD
# ---------------------------------------------------------------------------
VALID_CLASSIFICATIONS = frozenset({
    "independent_ultradian",
    "likely_independent_ultradian",
    "harmonic",
    "circadian_only",
    "non_rhythmic",
    "ambiguous",
    "damped_ultradian",
    "multi_ultradian",
})

# Default method weights
_DEFAULT_W_BHDT = 0.4
_DEFAULT_W_PINOD = 0.6

# BHDT columns that carry oscillator / test parameters
_BHDT_PARAM_COLS = [
    "log_bayes_factor",
    "bayes_factor",
    "f_test_24h_pvalue",
    "f_test_12h_pvalue",
]

# PINOD columns that carry oscillator / fit parameters
_PINOD_PARAM_COLS = [
    "confidence",
    "reconstruction_r2",
    "T_0",
    "gamma_0",
    "amp_0",
    "T_1",
    "gamma_1",
    "amp_1",
]


# ===================================================================== #
#  Public API                                                            #
# ===================================================================== #

def integrate_results(
    bhdt_df: pd.DataFrame,
    pinod_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_vote",
) -> pd.DataFrame:
    """Merge BHDT and PINOD results and produce a consensus classification.

    Parameters
    ----------
    bhdt_df : pd.DataFrame
        Output of ``chord.bhdt.pipeline.run_bhdt()``.  Must contain at least
        ``gene`` and ``classification`` columns.
    pinod_df : pd.DataFrame
        Output of ``chord.pinod.decompose.decompose()``.  Must contain at
        least ``gene``, ``classification``, and ``confidence`` columns.
    weights : dict, optional
        ``{"bhdt": float, "pinod": float}`` voting weights.  Defaults to
        ``{"bhdt": 0.4, "pinod": 0.6}``.
    method : str
        Ensemble strategy.  Currently only ``"weighted_vote"`` is supported.

    Returns
    -------
    pd.DataFrame
        One row per gene with columns:

        * ``gene``
        * ``consensus_classification``
        * ``consensus_confidence``
        * ``bhdt_classification``
        * ``pinod_classification``
        * ``agreement``  (bool — True when both methods agree)
        * ``review_flag``  (bool — True when the gene needs manual review)
        * merged parameter columns from both methods
    """
    if method != "weighted_vote":
        raise ValueError(
            f"Unknown ensemble method {method!r}. "
            "Only 'weighted_vote' is currently supported."
        )

    # Resolve weights
    w_bhdt, w_pinod = _resolve_weights(weights)

    # ---- Merge on gene name ----------------------------------------- #
    merged = pd.merge(
        bhdt_df,
        pinod_df,
        on="gene",
        how="outer",
        suffixes=("_bhdt", "_pinod"),
    )

    # Normalise classification column names after merge
    merged = _normalise_classification_cols(merged, bhdt_df, pinod_df)

    # ---- Vectorised confidence computation --------------------------- #
    # BHDT confidence: σ(|log_bf|)
    lbf_col = (
        "log_bayes_factor_bhdt" if "log_bayes_factor_bhdt" in merged.columns
        else "log_bayes_factor"
    )
    if lbf_col in merged.columns:
        lbf_vals = pd.to_numeric(merged[lbf_col], errors="coerce").fillna(0.0)
        merged["_bhdt_conf"] = 1.0 / (1.0 + np.exp(-np.abs(lbf_vals)))
    else:
        merged["_bhdt_conf"] = 0.5

    # PINOD confidence
    if "pinod_confidence" in merged.columns:
        merged["_pinod_conf"] = pd.to_numeric(
            merged["pinod_confidence"], errors="coerce"
        ).fillna(0.5)
    elif "confidence" in merged.columns:
        merged["_pinod_conf"] = pd.to_numeric(
            merged["confidence"], errors="coerce"
        ).fillna(0.5)
    else:
        merged["_pinod_conf"] = 0.5

    # ---- Row-wise consensus via apply (faster than iterrows) --------- #
    bhdt_cols_set = set(bhdt_df.columns)
    pinod_cols_set = set(pinod_df.columns)

    def _process_row(row):
        bhdt_cls = row.get("bhdt_classification", np.nan)
        pinod_cls = row.get("pinod_classification", np.nan)
        bhdt_conf = row["_bhdt_conf"]
        pinod_conf = row["_pinod_conf"]

        cons_cls, cons_conf, agree, flag = _weighted_vote(
            bhdt_cls, pinod_cls, bhdt_conf, pinod_conf, w_bhdt, w_pinod,
        )

        params = _merge_parameters(row, bhdt_df.columns, pinod_df.columns)

        return pd.Series({
            "gene": row["gene"],
            "consensus_classification": cons_cls,
            "consensus_confidence": round(cons_conf, 4),
            "bhdt_classification": bhdt_cls if pd.notna(bhdt_cls) else None,
            "pinod_classification": pinod_cls if pd.notna(pinod_cls) else None,
            "agreement": agree,
            "review_flag": flag,
            **params,
        })

    result = merged.apply(_process_row, axis=1)
    # Drop internal helper columns
    result = result.drop(columns=["_bhdt_conf", "_pinod_conf"], errors="ignore")
    return result


# ===================================================================== #
#  Internal helpers                                                      #
# ===================================================================== #

def _resolve_weights(
    weights: Optional[Dict[str, float]],
) -> Tuple[float, float]:
    """Return (w_bhdt, w_pinod), normalised so they sum to 1."""
    if weights is None:
        return _DEFAULT_W_BHDT, _DEFAULT_W_PINOD
    w_b = float(weights.get("bhdt", _DEFAULT_W_BHDT))
    w_p = float(weights.get("pinod", _DEFAULT_W_PINOD))
    total = w_b + w_p
    if total == 0:
        return 0.5, 0.5
    return w_b / total, w_p / total


def _normalise_classification_cols(
    merged: pd.DataFrame,
    bhdt_df: pd.DataFrame,
    pinod_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure we have ``bhdt_classification`` and ``pinod_classification``."""
    # After an outer merge with suffixes, the classification column may
    # appear as classification_bhdt / classification_pinod or stay as-is.
    if "bhdt_classification" not in merged.columns:
        if "classification_bhdt" in merged.columns:
            merged = merged.rename(
                columns={"classification_bhdt": "bhdt_classification"}
            )
        elif "classification" in merged.columns:
            merged = merged.rename(
                columns={"classification": "bhdt_classification"}
            )

    if "pinod_classification" not in merged.columns:
        if "classification_pinod" in merged.columns:
            merged = merged.rename(
                columns={"classification_pinod": "pinod_classification"}
            )

    # Confidence column from PINOD
    if "pinod_confidence" not in merged.columns:
        if "confidence_pinod" in merged.columns:
            merged = merged.rename(
                columns={"confidence_pinod": "pinod_confidence"}
            )
        elif "confidence" in merged.columns:
            merged = merged.rename(columns={"confidence": "pinod_confidence"})

    return merged


def _canonical_class(label: str) -> str:
    """Map a classification label to its canonical group for cross-method comparison.

    BHDT and PINOD use overlapping but non-identical label vocabularies.
    This function collapses subtypes into canonical groups so that e.g.
    BHDT's ``likely_independent_ultradian`` and PINOD's ``damped_ultradian``
    are recognised as agreeing (both map to ``"independent"``).
    """
    _CANONICAL_MAP = {
        "independent_ultradian": "independent",
        "likely_independent_ultradian": "independent",
        "damped_ultradian": "independent",
        "multi_ultradian": "independent",
        "harmonic": "harmonic",
        "circadian_only": "circadian",
        "non_rhythmic": "non_rhythmic",
        "ambiguous": "ambiguous",
    }
    return _CANONICAL_MAP.get(label, label)


def _bhdt_confidence(row: pd.Series) -> float:
    """Derive a [0, 1] confidence from BHDT's log Bayes factor.

    Uses a sigmoid mapping on the *absolute* value: ``conf = 1 / (1 + exp(-|log_bf|))``.
    This ensures strong evidence in *either* direction (harmonic or independent)
    yields high confidence.  The direction is already encoded in the
    classification label, so confidence should reflect evidence strength only.

    Falls back to 0.5 (uninformative) when the value is missing.
    """
    lbf = row.get("log_bayes_factor_bhdt", np.nan)
    if pd.isna(lbf):
        lbf = row.get("log_bayes_factor", np.nan)
    if pd.isna(lbf):
        return 0.5
    return float(1.0 / (1.0 + np.exp(-abs(float(lbf)))))


def _weighted_vote(
    bhdt_cls: str | None,
    pinod_cls: str | None,
    bhdt_conf: float,
    pinod_conf: float,
    w_bhdt: float,
    w_pinod: float,
) -> Tuple[str, float, bool, bool]:
    """Core voting logic for a single gene.

    Returns
    -------
    consensus_cls : str
    consensus_conf : float
    agreement : bool
    review_flag : bool
    """
    bhdt_missing = pd.isna(bhdt_cls) or bhdt_cls is None
    pinod_missing = pd.isna(pinod_cls) or pinod_cls is None

    # --- Only one method produced a result ----------------------------- #
    if bhdt_missing and pinod_missing:
        return "ambiguous", 0.0, False, True
    if bhdt_missing:
        return str(pinod_cls), float(pinod_conf), False, True
    if pinod_missing:
        return str(bhdt_cls), float(bhdt_conf), False, True

    bhdt_cls = str(bhdt_cls)
    pinod_cls = str(pinod_cls)

    # Safe-guard confidence values
    bhdt_conf = float(bhdt_conf) if not pd.isna(bhdt_conf) else 0.5
    pinod_conf = float(pinod_conf) if not pd.isna(pinod_conf) else 0.5

    # --- Both agree (exact match) --------------------------------------- #
    if bhdt_cls == pinod_cls:
        conf = w_bhdt * bhdt_conf + w_pinod * pinod_conf
        return bhdt_cls, conf, True, False

    # --- Canonical agreement ------------------------------------------- #
    # BHDT and PINOD use different label vocabularies.  When labels differ
    # but belong to the same canonical group (e.g. likely_independent_ultradian
    # vs damped_ultradian → both "independent"), treat as agreement and
    # prefer the more specific PINOD label (higher default weight).
    if _canonical_class(bhdt_cls) == _canonical_class(pinod_cls):
        conf = w_bhdt * bhdt_conf + w_pinod * pinod_conf
        # Prefer the label from the higher-weighted method
        chosen = pinod_cls if w_pinod >= w_bhdt else bhdt_cls
        return chosen, conf, True, False

    # --- BHDT non_rhythmic protection ---------------------------------- #
    #     BHDT's non_rhythmic call is based on F-test (reliable).  PINOD
    #     can overfit noise with its flexible ODE, so trust BHDT here.
    if bhdt_cls == "non_rhythmic" and _canonical_class(pinod_cls) != "non_rhythmic":
        # Only override BHDT if PINOD is very confident AND has good fit
        if pinod_conf > 0.9:
            return pinod_cls, pinod_conf * 0.8, False, True
        return bhdt_cls, bhdt_conf, False, True

    # --- PINOD non_rhythmic protection -------------------------------- #
    #     Conversely, if PINOD says non_rhythmic (low R²) but BHDT found
    #     a significant signal, trust BHDT.
    if _canonical_class(pinod_cls) == "non_rhythmic" and bhdt_cls != "non_rhythmic":
        if bhdt_conf > 0.6:
            return bhdt_cls, bhdt_conf, False, True
        return pinod_cls, pinod_conf, False, True

    # --- One side is ambiguous → defer to the other -------------------- #
    if bhdt_cls == "ambiguous":
        return pinod_cls, pinod_conf, False, False
    if pinod_cls == "ambiguous":
        return bhdt_cls, bhdt_conf, False, False

    # --- Special rule: harmonic vs independent disagreement -------------- #
    #     When one method says harmonic and the other says independent,
    #     this is a fundamental disagreement about the nature of the 12h
    #     signal.  Neither method should automatically override the other.
    #     Flag for review and pick the higher-weighted confidence.
    bhdt_canon = _canonical_class(bhdt_cls)
    pinod_canon = _canonical_class(pinod_cls)
    if (bhdt_canon == "harmonic" and pinod_canon == "independent") or \
       (bhdt_canon == "independent" and pinod_canon == "harmonic"):
        bhdt_score = w_bhdt * bhdt_conf
        pinod_score = w_pinod * pinod_conf
        if pinod_score >= bhdt_score:
            return pinod_cls, pinod_score, False, True
        return bhdt_cls, bhdt_score, False, True

    # --- Special rule: BHDT harmonic detection ------------------------- #
    #     If BHDT says harmonic and PINOD says circadian_only, trust BHDT
    #     (BHDT's harmonic call is based on waveform analysis).
    if bhdt_cls == "harmonic" and _canonical_class(pinod_cls) == "circadian":
        return bhdt_cls, bhdt_conf, False, True

    # --- General disagreement: pick higher weighted confidence --------- #
    bhdt_score = w_bhdt * bhdt_conf
    pinod_score = w_pinod * pinod_conf

    if pinod_score >= bhdt_score:
        return pinod_cls, pinod_score, False, True
    return bhdt_cls, bhdt_score, False, True


def _merge_parameters(
    row: pd.Series,
    bhdt_columns: pd.Index,
    pinod_columns: pd.Index,
) -> Dict[str, object]:
    """Collect oscillator / test parameters from both methods into one dict.

    Columns are prefixed with ``bhdt_`` or ``pinod_`` to avoid collisions.
    """
    params: Dict[str, object] = {}

    for col in _BHDT_PARAM_COLS:
        # After merge the column may have a _bhdt suffix
        for candidate in (f"{col}_bhdt", col):
            if candidate in row.index:
                val = row[candidate]
                key = f"bhdt_{col}" if not col.startswith("bhdt_") else col
                params[key] = val if pd.notna(val) else None
                break

    for col in _PINOD_PARAM_COLS:
        for candidate in (f"{col}_pinod", col):
            if candidate in row.index:
                val = row[candidate]
                key = f"pinod_{col}" if not col.startswith("pinod_") else col
                params[key] = val if pd.notna(val) else None
                break

    return params
