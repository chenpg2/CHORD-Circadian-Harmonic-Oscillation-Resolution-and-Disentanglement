"""Stage 2 redesign — ternary (A/B/C) disentanglement via calibrated orthogonal features.

Replaces the legacy 12-evidence additive score (full ensemble AUC 0.776 vs
E12-alone 0.820; see plan/CHORD_attack_matrix.md) with four orthogonal statistics
fed into a CALIBRATED multinomial-logistic model over the literature's ternary
generative taxonomy (plan/CHORD_deep_rebuild.md):

    Class A  harmonic of a non-sinusoidal 24-h waveform   (circadian-driven)
    Class B  autonomous independent 12-h oscillator        (its own clock)
    Class C  intersection of two anti-phase 24-h processes (circadian-driven)

This version addresses the Codex review of the rule-based first pass (P2):
  * statistics are made numerically robust (noise-aware floors, baseline-free
    twin symmetry, near-perfect-fit guard on the phase test);
  * the hand-thresholded rules are replaced by a class-weighted multinomial
    logistic model fitted on the honest benchmark (chord.simulation.ternary_benchmark),
    with an abstention band (-> 'ambiguous') for the under-determined region;
  * gates: no detectable 12 h -> 'none'; a 12 h with no 24 h fundamental cannot be
    a harmonic of anything -> autonomous Class B by construction.

The multinomial logistic is hand-rolled on scipy.optimize (no sklearn dependency).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f as f_dist

__all__ = [
    "phase_freedom_pvalue",
    "amplitude_ratio",
    "twin_peak_symmetry",
    "harmonic_decay_residual",
    "TernaryConfig",
    "TernaryModel",
    "extract_features",
    "fit_ternary_model",
    "get_default_model",
    "disentangle_ternary",
    "FEATURE_NAMES",
]

FEATURE_NAMES = ("neglog10_p_phase", "log_amp_ratio", "twin_symmetry",
                 "decay_residual", "log_snr12")


# ---------------------------------------------------------------------------
# Least-squares helpers
# ---------------------------------------------------------------------------
def _validate(t: np.ndarray, y: np.ndarray, min_n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if t.shape != y.shape:
        raise ValueError(f"t and y must have equal length, got {t.shape} vs {y.shape}")
    if t.size < min_n:
        raise ValueError(f"need at least {min_n} timepoints, got {t.size}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t and y must be finite (no NaN/inf)")
    return t, y


def _harmonic_fit(t: np.ndarray, y: np.ndarray, w: float, k: int):
    """k-harmonic OLS. Returns (amplitudes A_1..A_k, intercept, noise_sd)."""
    n = len(t)
    cols = [np.ones(n)]
    for j in range(1, k + 1):
        cols += [np.cos(j * w * t), np.sin(j * w * t)]
    x = np.column_stack(cols)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    amps = np.array([np.hypot(beta[1 + 2 * (j - 1)], beta[2 + 2 * (j - 1)])
                     for j in range(1, k + 1)])
    df = max(n - (1 + 2 * k), 1)
    noise_sd = float(np.sqrt(max(np.sum((y - x @ beta) ** 2) / df, 1e-12)))
    return amps, float(beta[0]), noise_sd


# ---------------------------------------------------------------------------
# The four orthogonal statistics (numerically robust)
# ---------------------------------------------------------------------------
def phase_freedom_pvalue(t: np.ndarray, y: np.ndarray, T_base: float = 24.0) -> float:
    """F-test of H0: phi_12 = 2*phi_24 (phase-locked) vs H1: phi_12 free.

    Small p -> 12-h phase is FREE -> autonomous-leaning (Class B).
    Large p -> phase-locked to the 24-h -> circadian-driven-leaning (Class A/C).

    Caveat (by design): this null only covers the g(cos theta) harmonic family
    (relative 2nd-harmonic phase 0 or pi). Sawtooth-type f(theta) harmonics have
    a free-looking phase here; the calibrated model leans on the decay-residual
    feature to catch them. Returns 0.5 when the 24-h component is below noise.
    """
    t, y = _validate(t, y)
    n = len(t); w = 2 * np.pi / T_base
    x24 = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)])
    b24 = np.linalg.lstsq(x24, y, rcond=None)[0]
    phi_24 = float(np.arctan2(b24[2], b24[1]))
    amp_24 = float(np.hypot(b24[1], b24[2]))
    resid_var = float(np.var(y - x24 @ b24))
    if amp_24 < 2.0 * np.sqrt(2.0 * resid_var / n):
        return 0.5
    tss = float(np.sum((y - np.mean(y)) ** 2))
    x_free = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t),
                              np.cos(2 * w * t), np.sin(2 * w * t)])
    rss_free = float(np.sum((y - x_free @ np.linalg.lstsq(x_free, y, rcond=None)[0]) ** 2))
    x_lock = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t),
                              np.cos(2 * w * t - 2 * phi_24)])
    rss_lock = float(np.sum((y - x_lock @ np.linalg.lstsq(x_lock, y, rcond=None)[0]) ** 2))
    df = n - 5
    # Guard near-perfect fits: floor rss_free at a tiny fraction of total variance
    # so two essentially-noiseless fits do not produce a spurious p≈0 from fp noise.
    rss_free = max(rss_free, 1e-9 * max(tss, 1e-12))
    if df <= 0:
        return 1.0
    f_stat = max(0.0, rss_lock - rss_free) / (rss_free / df)
    return float(1.0 - f_dist.cdf(f_stat, 1, df))


def amplitude_ratio(t: np.ndarray, y: np.ndarray, T_base: float = 24.0) -> float:
    """A_12 / A_24 from a 2-harmonic fit. High -> B, mid -> A, low -> C."""
    t, y = _validate(t, y)
    amps, _, _ = _harmonic_fit(t, y, 2 * np.pi / T_base, 2)
    return float(amps[1] / max(amps[0], 1e-9))


def twin_peak_symmetry(t: np.ndarray, y: np.ndarray, T_base: float = 24.0) -> float:
    """Symmetry of the two within-24h peaks of the (24h+12h) reconstruction.

    Heights are measured from the GLOBAL TROUGH (min-subtracted), so the metric is
    baseline/mesor-independent and bounded in [0, 1]: 1.0 = equal twin peaks
    (symmetric -> Class C), lower = unequal (asymmetric -> Class B). Returns 1.0
    when fewer than two peaks exist.
    """
    t, y = _validate(t, y)
    n = len(t); w = 2 * np.pi / T_base
    x = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    tt = np.linspace(0.0, T_base, 240, endpoint=False)
    yy = (beta[0] + beta[1] * np.cos(w * tt) + beta[2] * np.sin(w * tt)
          + beta[3] * np.cos(2 * w * tt) + beta[4] * np.sin(2 * w * tt))
    yy = yy - yy.min()  # baseline at the global trough -> all heights >= 0
    left = np.roll(yy, 1); right = np.roll(yy, -1)
    heights = yy[(yy > left) & (yy > right)]
    if heights.size < 2:
        return 1.0
    top2 = np.sort(heights)[::-1][:2]
    return float(np.clip(top2[1] / top2[0], 0.0, 1.0)) if top2[0] > 1e-12 else 1.0


def harmonic_decay_residual(t: np.ndarray, y: np.ndarray, T_base: float = 24.0,
                            k: int = 4, noise_sd: Optional[float] = None) -> float:
    """Is the 12-h component a 'bump' above the smooth harmonic-amplitude decay?

    Fits log(A_j) (floored at the NOISE amplitude, not 1e-9) ~ a + b*j through the
    estimable harmonics j in {1, 3, 4} (those with A_j > 2*noise), predicts
    log(A_2), returns residual = log(A_2) - log(A_2_pred).

    >> 0 : 12-h sticks above the decay -> extra component -> Class B.
    ~ 0  : 12-h lies on the decay -> harmonic of one waveform -> Class A.
    Returns 0.0 (neutral) when fewer than two trend harmonics are estimable, so an
    A waveform with only 24h+12h content is not spuriously scored B (Codex P1 fix).
    """
    t, y = _validate(t, y)
    n = len(t); w = 2 * np.pi / T_base
    if n < 2 * k + 3:  # not enough dof for a k-harmonic fit
        return 0.0
    amps, _, fit_noise = _harmonic_fit(t, y, w, k)
    floor = noise_sd if noise_sd is not None else fit_noise
    floor = max(float(floor), 1e-9)
    la = np.log(np.maximum(amps, floor))
    trend = [(j, amps[j - 1]) for j in (1, 3, 4)]
    valid = [(j, la[j - 1]) for j, a in trend if a > 2.0 * floor]
    if len(valid) < 2:
        return 0.0
    js = np.array([j for j, _ in valid], dtype=float)
    lv = np.array([v for _, v in valid])
    design = np.column_stack([np.ones(len(js)), js])
    coef = np.linalg.lstsq(design, lv, rcond=None)[0]
    return float(la[1] - (coef[0] + coef[1] * 2.0))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TernaryConfig:
    """Gates and abstention threshold for the ternary discriminator."""
    min_12h_snr: float = 1.0     # below this, no 12h to disentangle -> 'none'
    min_24h_snr: float = 2.0     # 12h present but no 24h fundamental -> autonomous B
    abstain_prob: float = 0.55   # max class prob below this -> 'ambiguous'
    # 24h-dominance prior: a strong 24h fundamental with a 12h dwarfed by it (low
    # A_12/A_24) is most parsimoniously a harmonic (A), so move mass B->A. Grounded
    # in P-0034 (harmonic-amplitude decay) and the XBP1-LKO interventional evidence
    # that A_12/A_24 — not phase-freedom — separates autonomous from driven 12h.
    dominance_snr24: float = 3.0     # only apply when the 24h is clearly strong
    dominance_amp_ratio: float = 0.5  # below this A_12/A_24, the 12h is "dwarfed"
    dominance_shift: float = 0.8     # max fraction of P(B) shifted to A


DEFAULT_TERNARY_CONFIG = TernaryConfig()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(t: np.ndarray, y: np.ndarray, T_base: float = 24.0,
                     noise_sd: Optional[float] = None) -> np.ndarray:
    """Five sane, transformed features (see FEATURE_NAMES) for the calibrated model."""
    t, y = _validate(t, y)
    p_phase = phase_freedom_pvalue(t, y, T_base)
    amp_r = amplitude_ratio(t, y, T_base)
    twin = twin_peak_symmetry(t, y, T_base)
    decay = harmonic_decay_residual(t, y, T_base, noise_sd=noise_sd)
    # snr_12 from a 2-harmonic fit
    amps, _, fit_noise = _harmonic_fit(t, y, 2 * np.pi / T_base, 2)
    nz = noise_sd if noise_sd is not None else fit_noise
    snr12 = amps[1] / max(float(nz), 1e-9)
    return np.array([
        float(np.clip(-np.log10(max(p_phase, 1e-12)), 0.0, 12.0)),
        float(np.log(max(amp_r, 1e-6))),
        float(twin),
        float(decay),
        float(np.log(max(snr12, 1e-6))),
    ])


# ---------------------------------------------------------------------------
# Hand-rolled class-weighted multinomial logistic regression (no sklearn)
# ---------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    # clip after max-subtraction so exp never overflows/underflows to inf/0-row
    z = np.clip(z - z.max(axis=1, keepdims=True), -700.0, 0.0)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_multinomial(x: np.ndarray, y_idx: np.ndarray, n_classes: int,
                     l2: float = 1.0, class_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Fit softmax regression. Returns weight matrix W of shape (n_classes, d+1)."""
    n, d = x.shape
    xb = np.column_stack([x, np.ones(n)])
    dd = d + 1
    w = np.ones(n) if class_weight is None else class_weight[y_idx]
    onehot = np.zeros((n, n_classes)); onehot[np.arange(n), y_idx] = 1.0

    def neg_ll(theta: np.ndarray) -> float:
        wm = theta.reshape(n_classes, dd)
        p = _softmax(xb @ wm.T)
        # no epsilon: _softmax clips logits to [-700, 0] so probabilities are
        # already bounded away from 0, and an epsilon would desync loss vs gradient.
        ll = -float(np.sum(w * np.log(p[np.arange(n), y_idx])))
        ll += 0.5 * l2 * float(np.sum(wm[:, :-1] ** 2))  # bias unregularised
        return ll

    def grad(theta: np.ndarray) -> np.ndarray:
        wm = theta.reshape(n_classes, dd)
        p = _softmax(xb @ wm.T)
        g = (w[:, None] * (p - onehot)).T @ xb
        greg = l2 * wm.copy(); greg[:, -1] = 0.0
        return (g + greg).ravel()

    # bound the weights so the line search cannot blow them up to inf; L2 keeps
    # the optimum well inside these bounds.
    bounds = [(-50.0, 50.0)] * (n_classes * dd)
    # numpy 2.x with the macOS Accelerate BLAS raises a spurious
    # divide/overflow/invalid fpe flag on matmul even for fully finite inputs
    # (verified). Inputs here are finite and bounded, so suppress that flag.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        res = minimize(neg_ll, np.zeros(n_classes * dd), jac=grad,
                       method="L-BFGS-B", bounds=bounds)
    w_out = res.x.reshape(n_classes, dd)
    if not np.all(np.isfinite(w_out)):
        raise FloatingPointError("multinomial fit produced non-finite weights")
    return w_out


@dataclass
class TernaryModel:
    """Fitted multinomial logistic over the 4 orthogonal features."""
    classes: List[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    weights: np.ndarray  # (n_classes, d+1)

    def predict_proba(self, feats: np.ndarray) -> Dict[str, float]:
        z = (feats - self.feat_mean) / self.feat_std
        xb = np.append(z, 1.0)
        # suppress the spurious numpy-2.x/Accelerate matmul fpe flag (finite inputs)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            p = _softmax((self.weights @ xb)[None, :])[0]
        return {c: float(pi) for c, pi in zip(self.classes, p)}


def fit_ternary_model(t: np.ndarray, expr: np.ndarray, labels: List[str],
                      T_base: float = 24.0, l2: float = 1.0,
                      config: Optional[TernaryConfig] = None) -> TernaryModel:
    """Fit the calibrated A/B/C model on a labelled benchmark.

    Only genes that pass both gates (detectable 12 h AND a 24 h fundamental) and
    carry an A/B/C label contribute — these are exactly the cases the model must
    disambiguate (no-24h B and no-12h 'none' are handled by gates at predict time).
    """
    cfg = config or DEFAULT_TERNARY_CONFIG
    if len(expr) != len(labels):
        raise ValueError(f"expr and labels length mismatch: {len(expr)} vs {len(labels)}")
    classes = ["A", "B", "C"]
    cidx = {c: i for i, c in enumerate(classes)}
    feats: List[np.ndarray] = []
    ys: List[int] = []
    for y, lab in zip(expr, labels):
        if lab not in cidx:
            continue
        tt, yy = _validate(t, y)
        amps, _, noise = _harmonic_fit(tt, yy, 2 * np.pi / T_base, 2)
        if amps[1] / max(noise, 1e-9) < cfg.min_12h_snr:      # gate 1
            continue
        if amps[0] / max(noise, 1e-9) < cfg.min_24h_snr:      # gate 2 (would be auto-B)
            continue
        feats.append(extract_features(tt, yy, T_base, noise_sd=noise))
        ys.append(cidx[lab])
    if not feats:
        raise ValueError("no gated A/B/C samples to fit on")
    x = np.vstack(feats)
    y_idx = np.array(ys)
    mean = x.mean(axis=0)
    std = x.std(axis=0); std[std < 1e-9] = 1.0
    counts = np.bincount(y_idx, minlength=len(classes)).astype(float)
    missing = [classes[i] for i in range(len(classes)) if counts[i] == 0]
    if missing:
        raise ValueError(f"no gated training examples for class(es): {missing}")
    cw = np.where(counts > 0, len(y_idx) / (len(classes) * np.maximum(counts, 1)), 0.0)
    w = _fit_multinomial((x - mean) / std, y_idx, len(classes), l2=l2, class_weight=cw)
    return TernaryModel(classes=classes, feat_mean=mean, feat_std=std, weights=w)


# Lazily-built default model, calibrated on a fixed-seed honest benchmark. Kept
# separate (distinct seed) from any evaluation benchmark to avoid train-on-test.
_DEFAULT_MODEL: Optional[TernaryModel] = None


def get_default_model() -> TernaryModel:
    """Return (building once) the default model fitted on a fixed honest benchmark."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        from chord.simulation.ternary_benchmark import build_ternary_benchmark
        bench = build_ternary_benchmark(n_per_class=200, classes=("A", "B", "C"),
                                        seed=8108)
        _DEFAULT_MODEL = fit_ternary_model(bench["t"], bench["expr"], bench["labels"])
    return _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Ternary disentangler
# ---------------------------------------------------------------------------
def disentangle_ternary(t: np.ndarray, y: np.ndarray, T_base: float = 24.0,
                        model: Optional[TernaryModel] = None,
                        config: Optional[TernaryConfig] = None) -> Dict[str, Any]:
    """Classify a gene's 12-h component as Class A / B / C (or ambiguous / none).

    Returns dict with:
      class           : 'A_harmonic' | 'B_independent' | 'C_intersection'
                        | 'ambiguous' | 'none'
      autonomy_score  : P(B), the binary autonomous-vs-driven axis (NaN for 'none').
                        Note: adjusted by the 24h-dominance decision prior below, so
                        it is a decision score, not a strictly calibrated posterior.
      proba           : {'A','B','C'} calibrated probabilities (None for gated cases)
      stats           : the raw statistics + SNRs
    """
    cfg = config or DEFAULT_TERNARY_CONFIG
    t, y = _validate(t, y)
    w = 2 * np.pi / T_base

    amps, _, noise = _harmonic_fit(t, y, w, 2)
    amp_24, amp_12 = float(amps[0]), float(amps[1])
    snr_24 = amp_24 / max(noise, 1e-9)
    snr_12 = amp_12 / max(noise, 1e-9)
    stats = {"amp_ratio": amp_12 / max(amp_24, 1e-9), "snr_12": snr_12, "snr_24": snr_24}

    # Gate 1: no detectable 12-h component -> nothing to disentangle.
    if snr_12 < cfg.min_12h_snr:
        return {"class": "none", "autonomy_score": float("nan"),
                "proba": None, "stats": stats}
    # Gate 2: a 12-h with no 24-h fundamental cannot be a harmonic -> autonomous B.
    if snr_24 < cfg.min_24h_snr:
        return {"class": "B_independent", "autonomy_score": 1.0,
                "proba": None, "stats": stats}

    mdl = model or get_default_model()
    feats = extract_features(t, y, T_base, noise_sd=noise)
    proba = mdl.predict_proba(feats)
    stats.update(dict(zip(FEATURE_NAMES, feats.tolist())))

    # 24h-dominance prior (a DECISION prior, not a recalibration — autonomy_score is
    # adjusted P(B) thereafter): for a strong-24h gene whose 12h is dwarfed by the
    # fundamental (low A_12/A_24), the 12h is most parsimoniously a harmonic, so move
    # mass B -> the driven pool {A,C}, graded by how dwarfed the 12h is.
    amp_ratio = amp_12 / max(amp_24, 1e-9)
    if snr_24 >= cfg.dominance_snr24 and amp_ratio < cfg.dominance_amp_ratio and "B" in proba:
        frac = float(np.clip(cfg.dominance_shift * (cfg.dominance_amp_ratio - amp_ratio)
                             / cfg.dominance_amp_ratio, 0.0, 1.0))
        shift = frac * proba["B"]
        proba["B"] -= shift
        driven = proba.get("A", 0.0) + proba.get("C", 0.0)
        if driven > 1e-12:                       # split across driven pool by A:C odds
            proba["A"] = proba.get("A", 0.0) + shift * proba.get("A", 0.0) / driven
            proba["C"] = proba.get("C", 0.0) + shift * proba.get("C", 0.0) / driven
        else:
            proba["A"] = proba.get("A", 0.0) + shift

    best = max(proba.items(), key=lambda kv: kv[1])[0]
    autonomy = proba.get("B", float("nan"))
    if proba[best] < cfg.abstain_prob:
        return {"class": "ambiguous", "autonomy_score": autonomy,
                "proba": proba, "stats": stats}
    cls = {"A": "A_harmonic", "B": "B_independent", "C": "C_intersection"}[best]
    return {"class": cls, "autonomy_score": autonomy, "proba": proba, "stats": stats}
