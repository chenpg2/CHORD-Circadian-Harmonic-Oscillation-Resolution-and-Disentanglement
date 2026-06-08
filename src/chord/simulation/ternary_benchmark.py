"""Honest ternary (A/B/C) benchmark for CHORD Stage-2 redesign.

The legacy benchmark (`generator.py` scenarios 1-15) treats 12-h disentanglement
as binary (harmonic vs independent) and its harmonic scenarios are all "easy"
ideal waveforms, yielding artefactually perfect specificity (see
plan/CHORD_attack_matrix.md). This module supplies the *honest* substrate:

  * the ternary generative taxonomy (L3 synthesis cluster C0):
      A — harmonic of a non-sinusoidal 24-h waveform   (circadian, not autonomous)
      B — independent / autonomous 12-h oscillator
      C — intersection of two anti-phase 24-h processes (circadian, not autonomous)
  * the DEGENERATE Class-B cases the legacy set omits:
      - exact T=12.000 h (removes the period-deviation crutch)
      - phi_12 ~ 2*phi_24 (the genuinely under-determined near-harmonic phase)
  * a coexistence case (A harmonics + a weak autonomous B) that breaks the
    single-label assumption.

It reuses the validated generators in `generator.py` and adds only the missing
Class-B degenerate variants. The labelled set returned by
`build_ternary_benchmark` is the common AUC yardstick for the redesign.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from chord.simulation.generator import (
    _default_timepoints,
    _make_rng,
    # Class A — harmonics of a single non-sinusoidal 24-h waveform
    sawtooth_harmonic,
    peaked_harmonic,
    square_wave_harmonic,
    bimodal_circadian,
    pulse_circadian,
    # Class B — independent oscillators
    pure_ultradian,
    independent_superposition,
    independent_multi_ultradian,
    damped_ultradian,
    asymmetric_ultradian,
    # Class C — intersection of two anti-phase 24-h processes
    intersection_harmonic,
    # negative controls
    pure_circadian,
    pure_noise,
    trend_noise,
)

Result = Dict[str, Any]

__all__ = [
    "independent_exact_12h",
    "independent_near_harmonic_phase",
    "mixed_harmonic_and_independent",
    "ternary_class_of",
    "build_ternary_benchmark",
]


# ---------------------------------------------------------------------------
# Degenerate Class-B scenarios the legacy benchmark omits
# ---------------------------------------------------------------------------
def independent_exact_12h(
    t: Optional[np.ndarray] = None,
    A_24: float = 2.0,
    A_12: float = 1.5,
    phi_24: float = 0.0,
    phi_12: Optional[float] = None,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """Class B with T_12 == 12.000 h exactly and a free random phase.

    Removes the period-deviation "crutch" (Dimension B): a true independent
    oscillator need not sit off 12 h, so any discriminator that leans on
    |T_12 - 12| fails here. Discrimination must come from phase freedom.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    if phi_12 is None:
        phi_12 = rng.uniform(0, 2 * np.pi)
    w = 2 * np.pi / 24.0
    y_clean = M + A_24 * np.cos(w * t - phi_24) + A_12 * np.cos(2 * w * t - phi_12)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "independent_exact_12h",
            "oscillators": [
                {"T": 24.0, "A": A_24, "phi": phi_24, "type": "circadian"},
                {"T": 12.0, "A": A_12, "phi": float(phi_12),
                 "type": "independent_ultradian"},
            ],
            "M": M, "noise_sd": noise_sd,
            "class_12h": "B_independent", "degenerate": "exact_T12",
            "has_independent_12h": True, "has_harmonic_12h": False,
        },
    }


def independent_near_harmonic_phase(
    t: Optional[np.ndarray] = None,
    A_24: float = 2.0,
    A_12: float = 1.5,
    phi_24: float = 0.0,
    T_12: float = 12.0,
    phase_jitter: float = 0.25,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """Class B whose 12-h phase happens to sit near the harmonic prediction.

    phi_12 ~ 2*phi_24 + U(-jitter, jitter). This is the genuinely
    under-determined region (identifiability floor): an autonomous oscillator
    that, by coincidence, looks phase-locked to the 24-h fundamental. Expected
    AUC near chance for any single-series discriminator — the honest benchmark
    must contain it so the method can output "unidentifiable" rather than guess.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    w = 2 * np.pi / 24.0
    phi_12 = 2.0 * phi_24 + rng.uniform(-phase_jitter, phase_jitter)
    y_clean = M + A_24 * np.cos(w * t - phi_24) + A_12 * np.cos(2 * np.pi * t / T_12 - phi_12)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "independent_near_harmonic_phase",
            "oscillators": [
                {"T": 24.0, "A": A_24, "phi": phi_24, "type": "circadian"},
                {"T": T_12, "A": A_12, "phi": float(phi_12),
                 "type": "independent_ultradian"},
            ],
            "M": M, "noise_sd": noise_sd,
            "class_12h": "B_independent", "degenerate": "near_harmonic_phase",
            "has_independent_12h": True, "has_harmonic_12h": False,
        },
    }


def mixed_harmonic_and_independent(
    t: Optional[np.ndarray] = None,
    A_24: float = 2.0,
    peak_width: float = 0.25,
    A_12_indep: float = 0.8,
    T_12_indep: float = 11.7,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """Coexistence: a non-sinusoidal 24-h waveform (Class-A harmonics) PLUS a
    weak independent free-phase 12-h oscillator (Class B).

    Breaks the single-label assumption: an autonomous 12-h rhythm IS present on
    top of genuine harmonics. Ground-truth class is B (autonomous 12-h present),
    but it is a hard case because the harmonic 12-h energy partially masks it.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    w = 2 * np.pi / 24.0
    # Class-A part: peaked 24-h waveform (Fourier harmonics at 12, 8, 6 h)
    decay = -np.log(np.clip(peak_width, 0.01, 0.99))
    harmonic_part = np.zeros_like(t, dtype=np.float64)
    for k in range(1, 6):
        harmonic_part += A_24 * np.exp(-decay * (k - 1)) * np.cos(k * w * t)
    # Class-B part: weak independent free-phase 12-h
    phi_12 = rng.uniform(0, 2 * np.pi)
    indep_part = A_12_indep * np.cos(2 * np.pi * t / T_12_indep - phi_12)
    y_clean = M + harmonic_part + indep_part
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "mixed_harmonic_and_independent",
            "oscillators": [
                {"T": 24.0, "A": A_24, "type": "circadian_peaked"},
                {"T": T_12_indep, "A": A_12_indep, "phi": float(phi_12),
                 "type": "independent_ultradian"},
            ],
            "M": M, "noise_sd": noise_sd,
            "class_12h": "B_independent", "degenerate": "mixed_A_plus_B",
            "has_independent_12h": True, "has_harmonic_12h": True,
        },
    }


# ---------------------------------------------------------------------------
# Ternary class labelling + balanced benchmark builder
# ---------------------------------------------------------------------------
# Per-class generator pools (function, parameter-range dict). Ranges are sampled
# uniformly per gene to diversify the labelled set.
_CLASS_A: List[Tuple[Callable, Dict[str, Tuple[float, float]]]] = [
    (sawtooth_harmonic, {"A": (1.0, 4.0)}),
    (peaked_harmonic, {"A": (1.0, 4.0), "peak_width": (0.1, 0.5)}),
    (square_wave_harmonic, {"A": (1.0, 3.0), "duty_cycle": (0.3, 0.7)}),
    (bimodal_circadian, {"A_24": (1.0, 3.0), "A_12": (0.5, 2.0)}),
    (pulse_circadian, {"A": (1.5, 4.0), "pulse_width": (2.0, 6.0)}),
]
_CLASS_B: List[Tuple[Callable, Dict[str, Tuple[float, float]]]] = [
    (pure_ultradian, {"A": (0.8, 3.0)}),
    (independent_superposition, {"A_12": (0.8, 3.0), "T_12": (11.0, 13.0)}),
    (independent_exact_12h, {"A_12": (0.8, 3.0)}),
    (independent_near_harmonic_phase, {"A_12": (0.8, 3.0)}),
    (independent_multi_ultradian, {"A_12": (0.8, 2.0), "A_8": (0.5, 1.5)}),
    (damped_ultradian, {"A": (1.0, 3.0), "gamma": (0.01, 0.05)}),
    (asymmetric_ultradian, {"A": (0.8, 2.5)}),
    (mixed_harmonic_and_independent, {"A_24": (1.0, 3.0), "A_12_indep": (0.6, 1.5)}),
]
# Class C's induced 12-h component is intrinsically weak (~12% of the 24-h
# amplitude — that weakness is the defining feature). Bias toward the strong-
# induction regime and a larger target amplitude so the genuine 12-h survives the
# normalisation; build_ternary_benchmark additionally rejection-samples on the
# realised 12-h SNR so every C gene carries a *detectable* 12-h component.
_CLASS_C: List[Tuple[Callable, Dict[str, Tuple[float, float]]]] = [
    (intersection_harmonic, {"A": (2.5, 4.5), "A_s": (0.7, 0.95), "A_d": (0.7, 0.95),
                              "d0": (0.12, 0.22), "antiphase_jitter": (0.0, 0.6)}),
]
_CLASS_NONE: List[Tuple[Callable, Dict[str, Tuple[float, float]]]] = [
    (pure_circadian, {"A": (1.0, 4.0)}),
    (pure_noise, {"noise_sd": (0.5, 2.0)}),
    (trend_noise, {"slope": (-0.05, 0.05)}),
]

_CLASS_POOLS: Dict[str, List[Tuple[Callable, Dict[str, Tuple[float, float]]]]] = {
    "A": _CLASS_A, "B": _CLASS_B, "C": _CLASS_C, "none": _CLASS_NONE,
}


def _amplitude_at(y: np.ndarray, t: np.ndarray, T: float) -> float:
    """Least-squares amplitude of a period-T sinusoid in y (noise floor aside)."""
    w = 2.0 * np.pi / T
    X = np.column_stack([np.cos(w * t), np.sin(w * t)])
    b = np.linalg.lstsq(X, np.asarray(y) - np.mean(y), rcond=None)[0]
    return float(np.hypot(b[0], b[1]))


def ternary_class_of(truth: Dict[str, Any]) -> str:
    """Map a scenario truth dict to its ternary label: 'A', 'B', 'C', or 'none'.

    Prefers the explicit ``class_12h`` field; otherwise infers from the legacy
    ``has_independent_12h`` / ``has_harmonic_12h`` flags.
    """
    cls = truth.get("class_12h")
    if cls:
        return cls.split("_")[0]  # "B_independent" -> "B", "C_intersection" -> "C"
    if truth.get("has_independent_12h"):
        return "B"
    if truth.get("has_harmonic_12h"):
        return "A"
    return "none"


def build_ternary_benchmark(
    n_per_class: int = 100,
    t: Optional[np.ndarray] = None,
    classes: Tuple[str, ...] = ("A", "B", "C"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a balanced, labelled A/B/C(/none) benchmark.

    Parameters
    ----------
    n_per_class : int
        Number of genes per requested class.
    t : array, optional
        Time grid. Default 2-h sampling over 48 h.
    classes : tuple of str
        Subset of {"A", "B", "C", "none"} to include.
    seed : int
        Base random seed.

    Returns
    -------
    dict with keys:
        expr   : (n_genes, len(t)) expression matrix
        t      : time grid
        labels : list of str — ternary class per gene
        truth  : list of dict — per-gene scenario truth
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)

    rows: List[np.ndarray] = []
    labels: List[str] = []
    truths: List[Dict[str, Any]] = []

    # Every 12-h-bearing class (A/B/C) is rejection-sampled so its realised 12-h
    # component clears a detectability floor (SNR_12 = A_12 / noise_sd): the
    # benchmark tests *discrimination given a detectable 12-h*, not detection.
    # 'none' is exempt — it must stay 12-h-free. C needs more tries (weak induction).
    snr12_floor = 1.0
    gene_idx = 0
    for cls in classes:
        pool = _CLASS_POOLS[cls]
        needs_12h = cls in ("A", "B", "C")
        attempts = 40 if cls == "C" else (10 if needs_12h else 1)
        for _ in range(n_per_class):
            fn, ranges = pool[gene_idx % len(pool)]
            chosen: Optional[Result] = None
            best_snr = -1.0
            for _ in range(attempts):
                # Sample pool-specific params first, then fill M / noise_sd only if
                # the pool did not specify them (so e.g. pure_noise keeps its range).
                kwargs: Dict[str, float] = {
                    pname: float(rng.uniform(lo, hi)) for pname, (lo, hi) in ranges.items()
                }
                kwargs.setdefault("M", float(rng.uniform(3.0, 10.0)))
                kwargs.setdefault("noise_sd", float(rng.uniform(0.3, 0.7)))
                res = fn(t=t, seed=(seed * 1000 + gene_idx) % (2 ** 32 - 1), **kwargs)
                if not needs_12h:
                    chosen = res
                    break
                snr12 = _amplitude_at(res["y_clean"], res["t"], 12.0) / max(kwargs["noise_sd"], 1e-9)
                if snr12 > best_snr:
                    best_snr, chosen = snr12, res
                if snr12 >= snr12_floor:
                    break
            assert chosen is not None
            rows.append(chosen["y"])
            labels.append(ternary_class_of(chosen["truth"]))
            truths.append(chosen["truth"])
            gene_idx += 1

    return {
        "expr": np.asarray(rows),
        "t": np.asarray(t, dtype=np.float64),
        "labels": labels,
        "truth": truths,
    }
