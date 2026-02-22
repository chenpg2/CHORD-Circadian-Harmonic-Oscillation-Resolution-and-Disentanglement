"""
Synthetic data generator for CHORD algorithm validation.

Provides 12 benchmark scenarios covering independent ultradian oscillations,
circadian harmonics, damped signals, asymmetric waveforms, and negative controls.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Type alias
Result = Dict[str, Any]


def _default_timepoints() -> np.ndarray:
    """Default: 2-hour sampling over 48 hours (25 points)."""
    return np.arange(0, 48, 2, dtype=np.float64)


def _make_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Create a reproducible random state."""
    return np.random.RandomState(seed)


# ============================================================================
# Scenario 1: Pure 24-h circadian cosine
# ============================================================================
def pure_circadian(
    t: Optional[np.ndarray] = None,
    A: float = 2.0,
    T: float = 24.0,
    phi: float = 0.0,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """Pure 24-h cosine oscillation.

    Parameters
    ----------
    t : array, optional
        Time points in hours. Default 0, 2, ..., 46.
    A : float
        Amplitude.
    T : float
        Period in hours.
    phi : float
        Phase in radians.
    M : float
        MESOR (baseline).
    noise_sd : float
        Gaussian noise standard deviation.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys t, y, y_clean, truth
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    y_clean = M + A * np.cos(2 * np.pi * t / T - phi)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t,
        "y": y,
        "y_clean": y_clean,
        "truth": {
            "scenario": "pure_circadian",
            "oscillators": [{"T": T, "A": A, "phi": phi, "type": "circadian"}],
            "M": M,
            "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 2: Pure 12-h ultradian cosine
# ============================================================================
def pure_ultradian(
    t: Optional[np.ndarray] = None,
    A: float = 1.5,
    T: float = 12.0,
    phi: float = 0.0,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """Pure 12-h ultradian cosine oscillation."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    y_clean = M + A * np.cos(2 * np.pi * t / T - phi)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t,
        "y": y,
        "y_clean": y_clean,
        "truth": {
            "scenario": "pure_ultradian",
            "oscillators": [{"T": T, "A": A, "phi": phi, "type": "independent_ultradian"}],
            "M": M,
            "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 3: 24-h + independent 12-h superposition
# ============================================================================
def independent_superposition(
    t: Optional[np.ndarray] = None,
    A_24: float = 2.0,
    T_24: float = 24.0,
    phi_24: float = 0.0,
    A_12: float = 1.5,
    T_12: float = 11.8,
    phi_12: Optional[float] = None,
    M: float = 5.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> Result:
    """24-h circadian + independent 12-h ultradian superposition.

    Note: T_12 defaults to 11.8 (not exactly 12.0) to test period estimation
    and to distinguish from a perfect harmonic of 24 h.

    phi_12 is randomized uniformly in [0, 2*pi) by default to ensure the
    benchmark tests the full range of phase relationships between the 12-h
    and 24-h components.  A fixed value near 2*phi_24 would make the signal
    look harmonic, biasing the benchmark.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    if phi_12 is None:
        phi_12 = rng.uniform(0, 2 * np.pi)
    comp_24 = A_24 * np.cos(2 * np.pi * t / T_24 - phi_24)
    comp_12 = A_12 * np.cos(2 * np.pi * t / T_12 - phi_12)
    y_clean = M + comp_24 + comp_12
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t,
        "y": y,
        "y_clean": y_clean,
        "truth": {
            "scenario": "independent_superposition",
            "oscillators": [
                {"T": T_24, "A": A_24, "phi": phi_24, "type": "circadian"},
                {"T": T_12, "A": A_12, "phi": phi_12, "type": "independent_ultradian"},
            ],
            "M": M,
            "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 4: 24-h sawtooth wave (generates mathematical 12-h harmonic)
# ============================================================================
def sawtooth_harmonic(
    t=None, A=2.0, T=24.0, M=5.0, noise_sd=0.5, seed=None,
):
    """24-h sawtooth wave whose Fourier decomposition contains a 12-h harmonic.

    The 12-h component here is NOT an independent oscillator but a mathematical
    artifact of the non-sinusoidal waveform. BHDT should classify this as
    harmonic (H0), not independent (H1).
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    from scipy.signal import sawtooth as _saw
    y_clean = M + A * _saw(2 * np.pi * t / T)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "sawtooth_harmonic",
            "oscillators": [{"T": T, "A": A, "phi": 0.0, "type": "circadian_nonsinusoidal"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": True,
        },
    }


# ============================================================================
# Scenario 5: 24-h peaked wave (generates 12-h + 8-h harmonics)
# ============================================================================
def peaked_harmonic(
    t=None, A=2.0, T=24.0, peak_width=0.2, M=5.0, noise_sd=0.5, seed=None,
):
    """24-h peaked/spiked waveform generating multiple harmonics.

    Uses a raised-cosine peak with controllable width. Narrower peaks produce
    stronger higher harmonics (12 h, 8 h, 6 h).
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    phase = (t % T) / T  # 0..1
    # Fourier series truncation for a peaked wave
    decay_rate = -np.log(np.clip(peak_width, 0.01, 0.99))
    y_clean = M + 0.0
    for k in range(1, 6):
        amp_k = A * np.exp(-decay_rate * (k - 1))
        y_clean = y_clean + amp_k * np.cos(k * 2 * np.pi * t / T)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "peaked_harmonic",
            "oscillators": [{"T": T, "A": A, "phi": 0.0, "type": "circadian_peaked"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": True,
        },
    }


# ============================================================================
# Scenario 6: Independent 12-h + 8-h multi-ultradian
# ============================================================================
def independent_multi_ultradian(
    t=None,
    A_12=1.5, T_12=12.0, phi_12=0.0,
    A_8=1.0, T_8=8.0, phi_8=0.3,
    M=5.0, noise_sd=0.5, seed=None,
):
    """Two independent ultradian oscillators (12 h and 8 h)."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    comp_12 = A_12 * np.cos(2 * np.pi * t / T_12 - phi_12)
    comp_8 = A_8 * np.cos(2 * np.pi * t / T_8 - phi_8)
    y_clean = M + comp_12 + comp_8
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "independent_multi_ultradian",
            "oscillators": [
                {"T": T_12, "A": A_12, "phi": phi_12, "type": "independent_ultradian"},
                {"T": T_8, "A": A_8, "phi": phi_8, "type": "independent_ultradian"},
            ],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 7: Damped 12-h ultradian
# ============================================================================
def damped_ultradian(
    t=None, A=2.0, T=12.0, gamma=0.02, phi=0.0, M=5.0, noise_sd=0.5, seed=None,
):
    """Exponentially damped 12-h oscillation: A * exp(-gamma*t) * cos(...)."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    envelope = A * np.exp(-gamma * t)
    y_clean = M + envelope * np.cos(2 * np.pi * t / T - phi)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "damped_ultradian",
            "oscillators": [{"T": T, "A": A, "phi": phi, "gamma": gamma, "type": "damped_ultradian"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 8: Asymmetric 12-h ultradian (unequal rise/fall)
# ============================================================================
def asymmetric_ultradian(
    t=None, A=1.5, T=12.0, rise_fraction=0.3, M=5.0, noise_sd=0.5, seed=None,
):
    """12-h oscillation with asymmetric waveform (fast rise, slow fall).

    rise_fraction: fraction of the period spent rising (0.5 = symmetric cosine).
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    phase = (t % T) / T  # 0..1 within each cycle
    y_clean = np.zeros_like(t, dtype=np.float64)
    for i, p in enumerate(phase):
        if p < rise_fraction:
            y_clean[i] = M + A * (p / rise_fraction)
        else:
            y_clean[i] = M + A * (1.0 - (p - rise_fraction) / (1.0 - rise_fraction))
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "asymmetric_ultradian",
            "oscillators": [{"T": T, "A": A, "phi": 0.0, "rise_fraction": rise_fraction, "type": "asymmetric_ultradian"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 9: Pure noise (negative control)
# ============================================================================
def pure_noise(t=None, M=5.0, noise_sd=1.0, seed=None):
    """Gaussian noise only. No rhythmic signal. Negative control."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    y = M + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": np.full_like(t, M),
        "truth": {
            "scenario": "pure_noise",
            "oscillators": [],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 10: Linear trend + noise (negative control)
# ============================================================================
def trend_noise(t=None, slope=0.05, M=5.0, noise_sd=0.5, seed=None):
    """Linear trend plus noise. No rhythmic signal. Negative control."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    y_clean = M + slope * t
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "trend_noise",
            "oscillators": [],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 11: Low-SNR 12-h ultradian
# ============================================================================
def low_snr_ultradian(t=None, A=0.5, T=12.0, phi=0.0, M=5.0, noise_sd=1.0, seed=None):
    """Low signal-to-noise ratio 12-h oscillation (SNR ~ 0.5)."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    y_clean = M + A * np.cos(2 * np.pi * t / T - phi)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "low_snr_ultradian",
            "oscillators": [{"T": T, "A": A, "phi": phi, "type": "independent_ultradian"}],
            "M": M, "noise_sd": noise_sd,
            "snr": A / noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 12: Drifting-period 12-h ultradian
# ============================================================================
def drifting_ultradian(
    t=None, A=1.5, T_start=11.0, T_end=13.0, M=5.0, noise_sd=0.5, seed=None,
):
    """12-h oscillation whose period drifts linearly from T_start to T_end."""
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    T_inst = T_start + (T_end - T_start) * t / t[-1]  # instantaneous period
    # Integrate instantaneous frequency to get phase
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    inst_freq = 1.0 / T_inst
    cum_phase = np.cumsum(2 * np.pi * inst_freq * dt)
    y_clean = M + A * np.cos(cum_phase)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "drifting_ultradian",
            "oscillators": [{"T_start": T_start, "T_end": T_end, "A": A, "type": "drifting_ultradian"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": True,
            "has_harmonic_12h": False,
        },
    }


# ============================================================================
# Scenario 13: 24-h square wave (strong 12-h harmonic content)
# ============================================================================
def square_wave_harmonic(
    t=None, A=2.0, T=24.0, duty_cycle=0.35, M=5.0, noise_sd=0.5, seed=None,
):
    """24-h square wave with asymmetric duty cycle producing a 12-h harmonic.

    duty_cycle=0.35 gives a 2nd-harmonic (12 h) amplitude ~20% of the
    fundamental — enough to be detectable but clearly harmonic in origin.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    from scipy.signal import square as _square
    y_clean = M + A * _square(2 * np.pi * t / T, duty=duty_cycle)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "square_wave_harmonic",
            "oscillators": [{"T": T, "A": A, "phi": 0.0, "type": "circadian_square",
                             "duty_cycle": duty_cycle}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": True,
        },
    }


# ============================================================================
# Scenario 14: 24-h bimodal circadian (two unequal peaks per cycle)
# ============================================================================
def bimodal_circadian(
    t=None, A_24=2.0, A_12=0.8, T=24.0, M=5.0, noise_sd=0.5,
    seed=None,
):
    """24-h rhythm with a phase-locked 12-h harmonic producing two unequal peaks.

    Models a circadian waveform where the 12-h component is a true harmonic
    of the 24-h fundamental (phase = 2*phi_24).  The resulting waveform has
    two peaks per cycle with unequal amplitudes, common in temperature and
    activity rhythms.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    w = 2 * np.pi / T
    y_clean = M + A_24 * np.cos(w * t) + A_12 * np.cos(2 * w * t)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "bimodal_circadian",
            "oscillators": [{"T": T, "A_24": A_24, "A_12": A_12,
                             "type": "circadian_bimodal"}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": True,
        },
    }


# ============================================================================
# Scenario 15: 24-h pulse circadian (narrow Gaussian pulse per cycle)
# ============================================================================
def pulse_circadian(
    t=None, A=3.0, T=24.0, pulse_width=4.0, M=5.0, noise_sd=0.5, seed=None,
):
    """24-h pulsatile rhythm — narrow Gaussian pulse produces rich harmonic spectrum.

    Narrower pulse_width (relative to T) generates stronger higher harmonics.
    This mimics pulsatile hormone release patterns.
    """
    if t is None:
        t = _default_timepoints()
    rng = _make_rng(seed)
    phase = (t % T) / T  # 0..1
    sigma = pulse_width / T
    y_clean = M + A * np.exp(-0.5 * ((phase - 0.5) / max(sigma, 0.01)) ** 2)
    y = y_clean + rng.normal(0, noise_sd, len(t))
    return {
        "t": t, "y": y, "y_clean": y_clean,
        "truth": {
            "scenario": "pulse_circadian",
            "oscillators": [{"T": T, "A": A, "phi": 0.0, "type": "circadian_pulse",
                             "pulse_width": pulse_width}],
            "M": M, "noise_sd": noise_sd,
            "has_independent_12h": False,
            "has_harmonic_12h": True,
        },
    }


# ============================================================================
# Dispatchers
# ============================================================================
_SCENARIOS = {
    1: pure_circadian,
    2: pure_ultradian,
    3: independent_superposition,
    4: sawtooth_harmonic,
    5: peaked_harmonic,
    6: independent_multi_ultradian,
    7: damped_ultradian,
    8: asymmetric_ultradian,
    9: pure_noise,
    10: trend_noise,
    11: low_snr_ultradian,
    12: drifting_ultradian,
    13: square_wave_harmonic,
    14: bimodal_circadian,
    15: pulse_circadian,
}

SCENARIO_NAMES = {k: v.__name__ for k, v in _SCENARIOS.items()}


def generate_scenario(scenario_id, t=None, seed=None, **kwargs):
    """Generate a single synthetic scenario by ID (1-15).

    Parameters
    ----------
    scenario_id : int
        Scenario number (1-15).
    t : array, optional
        Time points.
    seed : int, optional
        Random seed.
    **kwargs
        Passed to the scenario function.

    Returns
    -------
    dict with keys t, y, y_clean, truth
    """
    if scenario_id not in _SCENARIOS:
        raise ValueError(
            "scenario_id must be 1-%d, got %d" % (max(_SCENARIOS), scenario_id)
        )
    return _SCENARIOS[scenario_id](t=t, seed=seed, **kwargs)


def generate_all_scenarios(t=None, seed=42, n_replicates=1):
    """Generate all 15 scenarios.

    Parameters
    ----------
    t : array, optional
        Time points.
    seed : int
        Base random seed. Each scenario gets seed + scenario_id.
    n_replicates : int
        Number of replicates per scenario.

    Returns
    -------
    list of dicts
    """
    results = []
    for sid in _SCENARIOS:
        for rep in range(n_replicates):
            s = seed + sid * 100 + rep if seed is not None else None
            r = generate_scenario(sid, t=t, seed=s)
            r["truth"]["replicate"] = rep
            r["truth"]["scenario_id"] = sid
            results.append(r)
    return results


def generate_genome_like(n_genes=1000, t=None, seed=42, composition=None):
    """Generate a realistic gene-expression-like matrix.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    t : array, optional
        Time points.
    seed : int
        Random seed.
    composition : dict, optional
        Fraction of genes per category. Default:
        {noise: 0.30, circadian: 0.25, independent_12h: 0.20,
         harmonic_12h: 0.10, ultradian_8h: 0.05, damped: 0.05, other: 0.05}

    Returns
    -------
    dict with keys:
        expr : np.ndarray of shape (n_genes, len(t))
        t : np.ndarray
        labels : list of str (gene category)
        truth : list of dicts (per-gene ground truth)
    """
    if t is None:
        t = _default_timepoints()
    if composition is None:
        composition = {
            "noise": 0.30,
            "circadian": 0.25,
            "independent_12h": 0.20,
            "harmonic_12h": 0.10,
            "ultradian_8h": 0.05,
            "damped": 0.05,
            "other": 0.05,
        }
    rng = _make_rng(seed)

    # Map categories to scenario functions with randomised parameters.
    # harmonic_12h uses a list of generators to diversify waveform types.
    _harmonic_generators = [
        (sawtooth_harmonic, {"A": (1.0, 4.0)}),
        (peaked_harmonic, {"A": (1.0, 4.0), "peak_width": (0.1, 0.5)}),
        (square_wave_harmonic, {"A": (1.0, 3.0), "duty_cycle": (0.3, 0.7)}),
        (bimodal_circadian, {"A1": (1.0, 3.0), "A2": (0.5, 2.0), "phase_gap": (4.0, 8.0)}),
        (pulse_circadian, {"A": (1.5, 4.0), "pulse_width": (2.0, 6.0)}),
    ]
    cat_to_fn = {
        "noise": (pure_noise, {}),
        "circadian": (pure_circadian, {"A": (1.0, 4.0), "T": (22.0, 26.0)}),
        "independent_12h": (independent_superposition, {"A_12": (0.8, 3.0), "T_12": (11.0, 13.0)}),
        "harmonic_12h": None,  # handled specially below
        "ultradian_8h": (independent_multi_ultradian, {"A_12": (0.8, 2.0), "A_8": (0.5, 1.5)}),
        "damped": (damped_ultradian, {"A": (1.0, 3.0), "gamma": (0.01, 0.05)}),
        "other": (asymmetric_ultradian, {"A": (0.8, 2.5)}),
    }

    expr = np.zeros((n_genes, len(t)))
    labels = []
    truths = []

    gene_idx = 0
    for cat, frac in composition.items():
        n_cat = int(round(frac * n_genes))
        for _ in range(n_cat):
            if gene_idx >= n_genes:
                break
            kw = {}
            if cat == "harmonic_12h":
                # Cycle through diverse harmonic waveform types
                hg_idx = gene_idx % len(_harmonic_generators)
                fn, param_ranges = _harmonic_generators[hg_idx]
            else:
                fn, param_ranges = cat_to_fn[cat]
            for pname, prange in param_ranges.items():
                kw[pname] = rng.uniform(prange[0], prange[1])
            kw["M"] = rng.uniform(3.0, 10.0)
            kw["noise_sd"] = rng.uniform(0.3, 1.5)
            s = seed * 1000 + gene_idx if seed is not None else None
            r = fn(t=t, seed=s, **kw)
            expr[gene_idx] = r["y"]
            labels.append(cat)
            truths.append(r["truth"])
            gene_idx += 1

    # Fill remaining genes with noise
    while gene_idx < n_genes:
        s = seed * 1000 + gene_idx if seed is not None else None
        r = pure_noise(t=t, seed=s, M=rng.uniform(3, 10), noise_sd=rng.uniform(0.5, 2.0))
        expr[gene_idx] = r["y"]
        labels.append("noise")
        truths.append(r["truth"])
        gene_idx += 1

    return {"expr": expr, "t": t, "labels": labels, "truth": truths}
