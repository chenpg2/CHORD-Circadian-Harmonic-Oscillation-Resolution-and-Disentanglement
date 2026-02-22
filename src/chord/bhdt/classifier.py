"""
CHORD Classifier: Detect-then-Disentangle two-stage architecture.

Stage 1 (Detection):
    Liberal multi-method fusion via Cauchy Combination Test.
    Maximizes recall for 12h rhythmic components.
    Methods: F-test + JTK_CYCLE + RAIN + Harmonic regression.

Stage 2 (Disentanglement):
    BHDT multi-evidence scoring with all evidence lines at full weight.
    9-line evidence scoring for robust harmonic disentanglement.
    7. Waveform-aware: bispectral weight reduced for non-sinusoidal signals
    8. Continuous confidence score output via tanh mapping
"""

import numpy as np
from scipy.stats import f as f_dist
from typing import Dict, List, Optional, Any

from chord.bhdt.detection.stage1_detector import stage1_detect
from chord.bhdt.models import (
    fit_harmonic_model,
    fit_independent_model,
    fit_independent_free_period,
)
from chord.bhdt.inference import (
    _bic_bayes_factor,
    _period_deviation_test,
    _residual_periodicity_test,
    _phase_coupling_score,
    _waveform_asymmetry_score,
    _harmonic_coherence_score,
)


# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

class CHORDConfig:
    """Configuration for CHORD classifier.

    All parameters have sensible defaults calibrated on 2h-sampled, 48h
    mouse liver/ovary RNA-seq data (N=24 timepoints, 4 replicates).
    Adjust according to your experimental design — see the User Manual
    (docs/CHORD_USER_MANUAL.md) for detailed guidance.

    Parameters
    ----------
    T_base : float
        Base circadian period in hours. Default 24.0.
        Change only for non-standard light/dark cycles (e.g., T22, T28
        protocols) or non-circadian ultradian studies.

    alpha_detect : float or None
        Stage 1 detection gate (CCT p-value threshold). Controls how
        many genes enter Stage 2 disentanglement.
        - None (default): auto-select based on sample size
          (0.10 for N>=48, 0.20 for N>=30, 0.50 for N<30)
        - Lower values (0.01-0.05): strict, fewer genes enter Stage 2,
          faster but may miss weak signals
        - Higher values (0.20-0.50): liberal, more genes enter Stage 2,
          slower but catches weak signals

    confidence_thresholds : tuple of 4 floats
        (independent_high, independent_low, harmonic_high, harmonic_low)
        Confidence score boundaries for classification.
        Default: (0.6, 0.3, -0.6, -0.3)
        - Tighter (e.g., 0.7/0.4/-0.7/-0.4): more conservative, fewer
          calls but higher precision
        - Looser (e.g., 0.5/0.2/-0.5/-0.2): more liberal, more calls
          but lower precision

    confidence_divisor : float
        Divisor in tanh(score / divisor) mapping. Controls how quickly
        evidence accumulates toward confident classification.
        Default: 6.0
        - Lower (4.0): evidence saturates faster, more decisive
        - Higher (8.0): requires more evidence, more conservative

    n_bootstrap : int
        Number of bootstrap replicates for parametric LRT.
        Default: 199. Higher values (999) give more stable p-values
        but increase runtime ~5x.

    bootstrap_trigger_small_n : float
        |log(BF)| threshold below which bootstrap is triggered for
        small samples (N < 40). Default: 6.0.
        Higher values trigger bootstrap more often (slower but more
        reliable for ambiguous cases).

    bootstrap_trigger_large_n : float
        |log(BF)| threshold for large samples (N >= 40). Default: 3.0.

    vmd_amp_gate_strong : float
        Minimum VMD 12h/24h amplitude ratio to trust VMD evidence when
        Stage 1 detection is strong (detection_strength > 2.0).
        Default: 0.10.

    vmd_amp_gate_default : float
        Minimum VMD amplitude ratio when detection is moderate.
        Default: 0.15.

    bf_small_n_threshold : int
        Sample size below which BIC-based Bayes Factor is downweighted.
        Default: 40.

    bf_small_n_weight : float
        Weight applied to negative (harmonic) BF evidence when N is
        below bf_small_n_threshold. Default: 0.5.

    period_dev_n_gate : int
        Minimum sample size to penalize period locked to harmonic.
        Default: 36.

    small_sample_aicc_threshold : int
        Sample size below which AICc is used instead of BIC for Bayes
        Factor computation. Default: 50.

    K_harmonics : int
        Number of harmonics in the M0 (harmonic) model. Default: 3
        (24h + 12h + 8h). Increase to 4 for tissues with strong 6h
        components.

    bispectral_nonsin_discount : float
        Discount factor for bispectral harmonic evidence when waveform
        is non-sinusoidal. Default: 0.5 (halve the evidence).

    amp_ratio_asymmetry_gate : float
        Maximum amplitude ratio for waveform asymmetry evidence to be
        considered. Default: 0.5.

    m0_r2_discount_threshold : float
        Minimum M0 R-squared for period deviation discount when
        waveform is asymmetric. Default: 0.5.
    """
    def __init__(self, **kwargs):
        self.T_base = kwargs.get("T_base", 24.0)
        self.alpha_detect = kwargs.get("alpha_detect", None)
        self.confidence_thresholds = kwargs.get("confidence_thresholds", (0.6, 0.3, -0.6, -0.3))
        self.confidence_divisor = kwargs.get("confidence_divisor", 4.0)
        self.n_bootstrap = kwargs.get("n_bootstrap", 199)
        self.bootstrap_trigger_small_n = kwargs.get("bootstrap_trigger_small_n", 6.0)
        self.bootstrap_trigger_large_n = kwargs.get("bootstrap_trigger_large_n", 3.0)
        self.vmd_amp_gate_strong = kwargs.get("vmd_amp_gate_strong", 0.10)
        self.vmd_amp_gate_default = kwargs.get("vmd_amp_gate_default", 0.15)
        self.bf_small_n_threshold = kwargs.get("bf_small_n_threshold", 40)
        self.bf_small_n_weight = kwargs.get("bf_small_n_weight", 0.5)
        self.small_sample_aicc_threshold = kwargs.get("small_sample_aicc_threshold", 50)
        self.K_harmonics = kwargs.get("K_harmonics", 3)
        self.bispectral_nonsin_discount = kwargs.get("bispectral_nonsin_discount", 0.5)
        self.amp_ratio_asymmetry_gate = kwargs.get("amp_ratio_asymmetry_gate", 0.5)
        self.m0_r2_discount_threshold = kwargs.get("m0_r2_discount_threshold", 0.5)
        self.period_dev_n_gate = kwargs.get("period_dev_n_gate", 36)

        # Evidence 10: 24h dominance gate
        self.h24_dominance_p_threshold = kwargs.get("h24_dominance_p_threshold", 0.001)
        self.h24_dominance_amp_ratio_max = kwargs.get("h24_dominance_amp_ratio_max", 0.35)
        self.h24_dominance_r2_min = kwargs.get("h24_dominance_r2_min", 0.6)
        self.h24_dominance_penalty = kwargs.get("h24_dominance_penalty", -3.0)

        # Evidence 12: F-test for phase freedom of the 12h component.
        # Tests H0: phi_12 = 2*phi_24 (harmonic) vs H1: phi_12 free.
        # Replaces the r2-based waveform_fit with a proper statistical test.
        self.waveform_fit_K = kwargs.get("waveform_fit_K", 4)
        self.phase_ftest_alpha_indep = kwargs.get("phase_ftest_alpha_indep", 0.01)
        self.phase_ftest_alpha_weak_indep = kwargs.get("phase_ftest_alpha_weak_indep", 0.05)
        self.phase_ftest_harmonic_p = kwargs.get("phase_ftest_harmonic_p", 0.30)
        self.phase_ftest_strong_harmonic_p = kwargs.get("phase_ftest_strong_harmonic_p", 0.50)
        self.phase_ftest_24h_snr_gate = kwargs.get("phase_ftest_24h_snr_gate", 2.0)

        # VMD-Hilbert r2 discount: suppress positive VMD evidence when
        # the K-harmonic 24h model fits well (VMD unreliable for non-sinusoidal)
        self.vmd_r2_discount_center = kwargs.get("vmd_r2_discount_center", 0.5)
        self.vmd_r2_discount_width = kwargs.get("vmd_r2_discount_width", 0.3)

        # Gate B: Residual 12h test after K-harmonic 24h removal
        self.gate_b_enabled = kwargs.get("gate_b_enabled", True)
        self.gate_b_alpha = kwargs.get("gate_b_alpha", 0.05)
        self.gate_b_K_max = kwargs.get("gate_b_K_max", 4)

        # Minimum number of methods with p < 0.10 to pass CCT gate
        self.min_methods_detect = kwargs.get("min_methods_detect", 2)

    @classmethod
    def for_tissue(cls, tissue_type="default", n_samples=None):
        """Create a config preset for common tissue types.

        Parameters
        ----------
        tissue_type : str
            One of: "default", "liver", "scn", "ovary", "heart",
            "fibroblast", "weak_rhythm"
        n_samples : int, optional
            Number of timepoints. If provided, auto-adjusts alpha_detect.

        Returns
        -------
        CHORDConfig
        """
        cfg = cls()

        # Sample-size adaptive alpha (tightened with Gate B)
        if n_samples is not None:
            if n_samples >= 48:
                cfg.alpha_detect = 0.05
            elif n_samples >= 30:
                cfg.alpha_detect = 0.10
            else:
                cfg.alpha_detect = 0.20

        tissue = tissue_type.lower().strip()

        if tissue in ("liver", "scn"):
            # Strong circadian tissues: default parameters work well
            pass

        elif tissue in ("ovary", "uterus", "oviduct", "reproductive"):
            # Reproductive tissues: 12h rhythms are biologically real
            # but often weaker than liver. Relax detection, keep
            # disentanglement strict.
            cfg.confidence_thresholds = (0.5, 0.25, -0.6, -0.3)
            cfg.vmd_amp_gate_default = 0.12
            if n_samples is not None and n_samples < 30:
                cfg.alpha_detect = 0.50
                cfg.bootstrap_trigger_small_n = 8.0

        elif tissue in ("heart", "muscle", "skeletal_muscle"):
            # Heart/muscle: moderate 12h rhythms, some mechanical
            # artifacts. Slightly more conservative.
            cfg.confidence_thresholds = (0.6, 0.35, -0.55, -0.25)

        elif tissue in ("fibroblast", "cell_line", "in_vitro"):
            # Cell lines: often noisy, weak rhythms, short time series.
            # Very liberal detection, rely on Stage 2.
            cfg.confidence_thresholds = (0.5, 0.2, -0.5, -0.2)
            cfg.vmd_amp_gate_default = 0.10
            cfg.bootstrap_trigger_small_n = 8.0
            cfg.bf_small_n_weight = 0.3

        elif tissue in ("weak_rhythm", "peripheral", "adipose", "kidney"):
            # Tissues with weak ultradian signals.
            cfg.confidence_thresholds = (0.5, 0.25, -0.6, -0.3)
            cfg.vmd_amp_gate_default = 0.12
            cfg.bispectral_nonsin_discount = 0.3

        return cfg


# Default config singleton
DEFAULT_CONFIG = CHORDConfig()


def _stage2_disentangle(t, y, T_base, detection_context, config=None):
    """Stage 2: BHDT disentanglement without F-test gating.

    All evidence lines participate at full weight. No w_12 compression,
    no 24h dominance penalty, no both_weak cap.

    Parameters
    ----------
    t : ndarray
        Time points.
    y : ndarray
        Expression values.
    T_base : float
        Base circadian period.
    detection_context : dict
        Output from Stage 1 (p_detect, waveform_hint, etc.)
    config : CHORDConfig, optional
        Configuration object. Uses DEFAULT_CONFIG if None.

    Returns
    -------
    dict with evidence_score, confidence, classification, evidence_details,
    and all model fit results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    ultradian_periods = [T_base, T_base / 2.0, T_base / 3.0]

    # --- Fit models (same as V2/V4) ---
    m0 = fit_harmonic_model(t, y, T_base=T_base, K=config.K_harmonics)
    m1 = fit_independent_model(t, y, periods=ultradian_periods)
    m1_free = fit_independent_free_period(t, y, period_inits=ultradian_periods)
    period_dev = _period_deviation_test(m1_free["fitted_periods"], T_base)

    # Bayes Factor (AICc for small samples)
    N = len(t)
    if N < config.small_sample_aicc_threshold and "aicc" in m0 and "aicc" in m1_free:
        log_bf = (m0["aicc"] - m1_free["aicc"]) / 2.0
    else:
        log_bf = _bic_bayes_factor(m0["bic"], m1_free["bic"])
    if not np.isfinite(log_bf):
        log_bf = _bic_bayes_factor(m0["bic"], m1_free["bic"])
    bf = np.exp(log_bf)

    # Amplitude ratio
    m1_amps = {c["T"]: c["A"] for c in m1["components"]}
    a_24 = m1_amps.get(24.0, 1e-10)
    a_12 = m1_amps.get(12.0, 0)
    amp_ratio = a_12 / max(a_24, 1e-10)

    # Waveform hint from Stage 1
    waveform_hint = detection_context.get("waveform_hint", "unknown")

    # Phase-locked K-harmonic 24h fit for waveform quality assessment.
    # Higher harmonics constrained to phase = k * phi_24 (1 DOF each, not 2).
    # Independent 12h with arbitrary phase cannot be captured by this model.
    # Used by Evidence 3 (amplitude_ratio), 5 (VMD discount), 12 (waveform_fit).
    w_base = 2.0 * np.pi / T_base
    X_24 = np.column_stack([np.ones(N), np.cos(w_base * t), np.sin(w_base * t)])
    beta_24 = np.linalg.lstsq(X_24, y, rcond=None)[0]
    phi_24 = np.arctan2(beta_24[2], beta_24[1])
    X_locked = np.column_stack([
        np.ones(N),
        np.cos(w_base * t), np.sin(w_base * t),
        np.cos(2 * w_base * t - 2 * phi_24),
        np.cos(3 * w_base * t - 3 * phi_24),
        np.cos(4 * w_base * t - 4 * phi_24),
    ])
    beta_locked = np.linalg.lstsq(X_locked, y, rcond=None)[0]
    rss_locked = float(np.sum((y - X_locked @ beta_locked) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    m0_r2_k4 = 1.0 - rss_locked / max(tss, 1e-10)

    # --- Evidence scoring (NO w_12, NO 24h penalty) ---
    evidence = {}
    score = 0.0

    # Evidence 1: Gate B phase freedom (asymmetric evidence)
    # Gate B tests phi_12 = 2*phi_24 constraint. High p → phase locked → harmonic.
    # Low p is ambiguous: could be independent OR non-sinusoidal harmonic
    # (sawtooth harmonics have phi_12 ≠ 2*phi_24 by Fourier series structure).
    # Therefore: only use high p as harmonic evidence, not low p as independent.
    gate_b = detection_context.get("gate_b_result")
    if gate_b is not None and "p_phase_freedom" in gate_b:
        p_phase = gate_b["p_phase_freedom"]
        if p_phase > 0.5:
            evidence["phase_freedom"] = -2.5
        elif p_phase > 0.3:
            evidence["phase_freedom"] = -1.5
        elif p_phase > 0.1:
            evidence["phase_freedom"] = -0.5
        else:
            evidence["phase_freedom"] = 0.0
    else:
        evidence["phase_freedom"] = 0.0
    score += evidence["phase_freedom"]

    # Evidence 2: Period deviation
    # Large deviation from 12.0h is weak evidence for independence (true
    # independent oscillators CAN have non-12h periods). But small deviation
    # is NOT harmonic evidence — both independent (T≈12h) and harmonic genes
    # can have fitted T≈12h. Non-sinusoidal harmonics can also show large
    # deviations (optimizer fits waveform shape, not just frequency).
    # Therefore: only mild positive evidence for large deviations, no penalty.
    # Two tiers: >5% = full evidence (+1.0), 2-5% = weak evidence (+0.5)
    # conditioned on m0_r2_k4 < 0.85 (when phase-locked harmonic model fits
    # well, the deviation is likely a fitting artifact, not a real frequency
    # difference).
    rel_dev = period_dev["relative_deviation"]
    if rel_dev > 0.05:
        evidence["period_deviation"] = 1.0
    elif rel_dev > 0.02 and m0_r2_k4 < 0.85:
        evidence["period_deviation"] = 0.5
    else:
        evidence["period_deviation"] = 0.0
    score += evidence["period_deviation"]

    # Evidence 3: Amplitude ratio — conditioned on waveform fit quality.
    # High amp_ratio + high r2_k4 = non-sinusoidal harmonic (peaked/sawtooth).
    # High amp_ratio + low r2_k4 = likely independent oscillator.
    if amp_ratio > 0.8:
        if m0_r2_k4 > 0.6:
            evidence["amplitude_ratio"] = 0.0
        else:
            evidence["amplitude_ratio"] = 3.0
    elif amp_ratio > 0.5:
        if m0_r2_k4 > 0.7:
            evidence["amplitude_ratio"] = 0.0
        else:
            evidence["amplitude_ratio"] = 1.0
    elif amp_ratio < 0.25:
        evidence["amplitude_ratio"] = -2.0
    elif amp_ratio < 0.4:
        evidence["amplitude_ratio"] = -1.0
    else:
        evidence["amplitude_ratio"] = 0.0
    score += evidence["amplitude_ratio"]

    # Evidence 4: Residual improvement — gated on m0_r2_k4.
    # When the phase-locked K=4 harmonic model fits very well, M1_free's
    # improvement comes from absorbing waveform shape (non-sinusoidal harmonics),
    # not from a genuinely different oscillator frequency.
    ri_raw = float(_residual_periodicity_test(m0, m1, m1_free))
    if ri_raw > 0 and m0_r2_k4 > 0.9:
        ri_raw = 0.0
    elif ri_raw > 0 and m0_r2_k4 > 0.8:
        ri_raw *= 0.5
    evidence["residual_improvement"] = ri_raw
    score += evidence["residual_improvement"]

    # Evidence 5: VMD-Hilbert IF coupling
    # VMD-Hilbert is unreliable at N<48 (edge effects dominate, segment
    # averaging insufficient for Hilbert transform stability)
    evidence["vmd_hilbert"] = 0.0
    vh_converged = False
    if N >= 48:
        try:
            from chord.bhdt.hilbert_if import vmd_hilbert_disentangle
            vh_result = vmd_hilbert_disentangle(t, y, T_base=T_base)
            vh_converged = vh_result.get("vmd_converged", False)

            amp_24_vmd = vh_result.get("mode_24_amplitude", 1e-10)
            amp_12_vmd = vh_result.get("mode_12_amplitude", 0.0)
            vh_amp_ratio = amp_12_vmd / max(amp_24_vmd, 1e-10)

            evidence_str = vh_result.get("classification_evidence", "")
            if "strong harmonic" in evidence_str:
                vh_score = -3.0
            elif "moderate harmonic" in evidence_str:
                vh_score = -2.0
            elif "weak harmonic" in evidence_str:
                vh_score = -1.0
            elif "strong independent" in evidence_str:
                vh_score = 2.0
            elif "moderate independent" in evidence_str:
                vh_score = 1.0
            else:
                vh_score = 0.0

            detection_strength = detection_context.get("detection_strength", 0.0)
            min_amp_gate = config.vmd_amp_gate_strong if detection_strength > 2.0 else config.vmd_amp_gate_default

            if vh_score > 0 and vh_converged:
                if vh_amp_ratio < min_amp_gate:
                    vh_score = 0.0
                else:
                    # Discount positive VMD when K-harmonic 24h model fits well
                    vmd_discount = max(0.0, 1.0 - (m0_r2_k4 - config.vmd_r2_discount_center) / config.vmd_r2_discount_width)
                    vh_score *= vmd_discount
            elif vh_score < 0 and vh_converged:
                pass

            evidence["vmd_hilbert"] = vh_score
        except Exception:
            pass
    score += evidence["vmd_hilbert"]

    # Evidence 6: Bispectral bicoherence
    # Bispectrum (third-order statistic) needs segment averaging and is
    # unreliable at N<96. Disable entirely for N<48, halve weight for N<96.
    evidence["bispectral"] = 0.0
    if N >= 48:
        try:
            from chord.bhdt.bispectral import bhct_evidence
            bhct = bhct_evidence(t, y, T_base=T_base)
            bic_val = bhct.get("bicoherence", 0.0)
            if bic_val > 0.7:
                bic_score = -2.0
            elif bic_val > 0.5:
                bic_score = -1.0
            elif bic_val < 0.3:
                bic_score = 1.0
            else:
                bic_score = 0.0

            if waveform_hint == "non_sinusoidal" and bic_score < 0:
                bic_score = bic_score * config.bispectral_nonsin_discount

            # Halve weight for 48 <= N < 96
            if N < 96:
                bic_score *= 0.5

            evidence["bispectral"] = bic_score
        except Exception:
            pass
    score += evidence["bispectral"]

    # Evidence 7: Phase coupling
    evidence["phase_coupling"] = float(_phase_coupling_score(m1, m1_free=m1_free))
    score += evidence["phase_coupling"]

    # Evidence 8: Bootstrap LRT
    # Only informative when the period optimizer found a real deviation
    # (otherwise M0≡M1_free and bootstrap p-values are random noise).
    # Gate: require period_dev > 1% OR Gate B p < 0.2 (ambiguous zone).
    evidence["bootstrap_lrt"] = 0.0
    has_period_signal = period_dev["relative_deviation"] > 0.01
    gate_b_ambiguous = (gate_b is not None
                        and 0.05 <= gate_b.get("p_phase_freedom", 1.0) <= 0.2)
    if has_period_signal or gate_b_ambiguous:
        try:
            from chord.bhdt.bootstrap import parametric_bootstrap_lrt
            boot = parametric_bootstrap_lrt(
                t, y, T_base=T_base, K_harmonics=config.K_harmonics,
                n_bootstrap=config.n_bootstrap,
                seed=hash(tuple(y[:5].tolist())) % (2**31),
                period_inits=ultradian_periods,
            )
            bp = boot["p_value"]
            if bp < 0.01:
                evidence["bootstrap_lrt"] = 2.0
            elif bp < 0.05:
                evidence["bootstrap_lrt"] = 1.0
            elif bp > 0.5:
                evidence["bootstrap_lrt"] = -1.0
        except Exception:
            pass
    # Gate bootstrap_lrt on m0_r2_k4 (same logic as residual_improvement):
    # high harmonic-model fit → LRT improvement is waveform shape, not independence.
    bl_raw = evidence["bootstrap_lrt"]
    if bl_raw > 0 and m0_r2_k4 > 0.9:
        bl_raw = 0.0
    elif bl_raw > 0 and m0_r2_k4 > 0.8:
        bl_raw *= 0.5
    evidence["bootstrap_lrt"] = bl_raw
    score += evidence["bootstrap_lrt"]

    # Evidence 9: Waveform asymmetry
    # Only count asymmetry evidence when the amplitude ratio is in the
    # harmonic-typical range. When amp_ratio > 0.5, the 12h component is
    # too strong to be a mere harmonic artifact regardless of waveform shape.
    if amp_ratio < config.amp_ratio_asymmetry_gate:
        asym_score = float(_waveform_asymmetry_score(t, y, m0))
        evidence["waveform_asymmetry"] = asym_score
        # Cross-check: if waveform is asymmetric AND M0 fits well,
        # the period deviation is likely a fitting artifact (the free-period
        # optimizer finds a slightly different period because the harmonic
        # component doesn't perfectly match a cosine). Discount period_deviation.
        if asym_score < 0 and evidence["period_deviation"] > 0:
            m0_r2 = 1.0 - m0["rss"] / max(np.var(y) * len(y), 1e-10)
            if m0_r2 > config.m0_r2_discount_threshold:
                # M0 fits well + asymmetric waveform → period deviation is artifact
                discount = min(evidence["period_deviation"], 2.0)
                evidence["period_deviation"] -= discount
                score -= discount
    else:
        evidence["waveform_asymmetry"] = 0.0
    score += evidence["waveform_asymmetry"]

    # Evidence 10: Harmonic coherence (sawtooth detection)
    # Tests if amplitude ratios A_12/A_24, A_8/A_24, A_6/A_24 follow the 1/n
    # decay pattern characteristic of sawtooth-like non-sinusoidal waveforms.
    # Gate B misses sawtooth because phi_12 ≠ 2*phi_24 for sawtooth Fourier
    # series, but the amplitude structure is still predictable.
    # When harmonic coherence is strong, also suppress VMD-Hilbert positive
    # scores (VMD decomposition of non-sinusoidal waveforms is unreliable).
    hc_result = _harmonic_coherence_score(t, y, T_base=T_base)
    evidence["harmonic_coherence"] = hc_result["score"]
    score += evidence["harmonic_coherence"]

    # When harmonic_coherence detects 1/n amplitude decay (sawtooth/non-sinusoidal),
    # suppress positive evidence from lines that are unreliable for such waveforms:
    # - VMD/bootstrap: VMD decomposition of non-sinusoidal waveforms is unreliable
    # - period_deviation: free-period optimizer fits waveform shape, not frequency
    # - residual_improvement: M1_free absorbs waveform shape via free periods
    # - amplitude_ratio: high amp_ratio is expected for non-sinusoidal harmonics
    _hc_detected = hc_result["score"] <= -1.0
    if _hc_detected:
        for key in ("vmd_hilbert", "bootstrap_lrt", "period_deviation",
                     "residual_improvement", "amplitude_ratio"):
            if evidence[key] > 0:
                score -= evidence[key]
                evidence[key] = 0.0

    # Evidence 11: 24h dominance gate (softened: linear decay)
    evidence["h24_dominance"] = 0.0
    p_24 = detection_context.get("p_24", 1.0)
    m0_r2 = 1.0 - m0["rss"] / max(np.var(y) * len(y), 1e-10)
    if (p_24 < config.h24_dominance_p_threshold
            and m0_r2 > config.h24_dominance_r2_min):
        if amp_ratio <= 0.20:
            evidence["h24_dominance"] = config.h24_dominance_penalty
        elif amp_ratio < 0.50:
            decay = (0.50 - amp_ratio) / 0.30
            evidence["h24_dominance"] = config.h24_dominance_penalty * decay
    score += evidence["h24_dominance"]

    # Evidence 12: F-test for phase freedom of the 12h component.
    # H0: phi_12 = 2*phi_24 (harmonic).  H1: phi_12 free (independent).
    # Compares M_locked (1-DOF 12h) vs M_free (2-DOF 12h).
    # Resolves all three failure modes of the old r2-based approach:
    #   FM1 (pure 12h): F≈0 because both models fit equally well.
    #   FM2 (near-harmonic phase): correctly returns "ambiguous".
    #   FM3 (non-sinusoidal 12h): K=3,4 terms cancel in RSS difference.
    amp_24_raw = float(np.sqrt(beta_24[1] ** 2 + beta_24[2] ** 2))
    resid_24_var = float(np.var(y - X_24 @ beta_24))
    noise_floor_24 = np.sqrt(2.0 * resid_24_var / N)

    evidence["waveform_fit"] = 0.0
    if amp_24_raw < config.phase_ftest_24h_snr_gate * noise_floor_24:
        evidence["waveform_fit"] = 1.0
    else:
        X_free_12 = np.column_stack([
            np.ones(N),
            np.cos(w_base * t), np.sin(w_base * t),
            np.cos(2 * w_base * t), np.sin(2 * w_base * t),
        ])
        beta_free_12 = np.linalg.lstsq(X_free_12, y, rcond=None)[0]
        rss_free_12 = float(np.sum((y - X_free_12 @ beta_free_12) ** 2))

        X_locked_12 = np.column_stack([
            np.ones(N),
            np.cos(w_base * t), np.sin(w_base * t),
            np.cos(2 * w_base * t - 2 * phi_24),
        ])
        beta_locked_12 = np.linalg.lstsq(X_locked_12, y, rcond=None)[0]
        rss_locked_12 = float(np.sum((y - X_locked_12 @ beta_locked_12) ** 2))

        df_resid = N - 5
        if df_resid > 0 and rss_free_12 > 0:
            f_phase = max(0.0, (rss_locked_12 - rss_free_12)) / (rss_free_12 / df_resid)
            p_phase_ftest = float(1.0 - f_dist.cdf(f_phase, 1, df_resid))
        else:
            f_phase = 0.0
            p_phase_ftest = 1.0

        if p_phase_ftest < config.phase_ftest_alpha_indep:
            evidence["waveform_fit"] = 2.0
        elif p_phase_ftest < config.phase_ftest_alpha_weak_indep:
            evidence["waveform_fit"] = 1.0
        elif p_phase_ftest > config.phase_ftest_strong_harmonic_p:
            evidence["waveform_fit"] = -3.0
        elif p_phase_ftest > config.phase_ftest_harmonic_p:
            evidence["waveform_fit"] = -1.5
    # Condition F-test on harmonic_coherence: when 1/n amplitude decay is
    # detected, the F-test "phase is free" result is expected (sawtooth
    # Fourier series property: phi_n ≠ n*phi_1) and is NOT independence evidence.
    if _hc_detected and evidence["waveform_fit"] > 0:
        evidence["waveform_fit"] = 0.0
    score += evidence["waveform_fit"]

    # --- Confidence and classification ---
    confidence = float(np.tanh(score / config.confidence_divisor))

    ct = config.confidence_thresholds
    if confidence >= ct[0]:
        classification = "independent_ultradian"
    elif confidence >= ct[1]:
        classification = "likely_independent_ultradian"
    elif confidence <= ct[2]:
        classification = "harmonic"
    elif confidence <= ct[3]:
        classification = "harmonic"
    else:
        classification = "ambiguous"

    return {
        "evidence_score": float(score),
        "confidence": confidence,
        "classification": classification,
        "evidence_details": evidence,
        "log_bayes_factor": float(log_bf),
        "bayes_factor": float(bf),
        "m0": m0,
        "m1": m1,
        "m1_free": m1_free,
        "period_deviation": period_dev,
        "amp_ratio": float(amp_ratio),
        "vmd_converged": vh_converged,
        "m0_r2_k4": float(m0_r2_k4),
    }


def classify_gene(t, y, T_base=24.0, alpha_detect=None, config=None):
    """CHORD two-stage classifier: detect then disentangle.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    y : array-like
        Expression values.
    T_base : float
        Base circadian period (default 24.0).
    alpha_detect : float or None
        Stage 1 gate threshold. If None (default), auto-selects based on
        sample size: 0.10 for N >= 48, 0.20 for N >= 30, 0.50 for N < 30.
        The liberal default for short series reflects the reduced statistical
        power of all detection methods at small N -- Stage 2 disentanglement
        controls false positives regardless of Stage 1 threshold.
    config : CHORDConfig, optional
        Configuration object. If None, uses DEFAULT_CONFIG.
        Use CHORDConfig.for_tissue() for tissue-specific presets.

    Returns
    -------
    dict with all Stage 1 and Stage 2 results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    T_base = config.T_base if T_base == 24.0 and config.T_base != 24.0 else T_base

    # Auto-select alpha based on sample size (tightened with Gate B)
    if alpha_detect is None and config.alpha_detect is None:
        N = len(t)
        if N >= 48:
            alpha_detect = 0.05
        elif N >= 30:
            alpha_detect = 0.10
        else:
            alpha_detect = 0.20
    elif alpha_detect is None:
        alpha_detect = config.alpha_detect

    # --- Stage 1: Detection (with Gate B) ---
    s1 = stage1_detect(
        t, y, T_base=T_base, alpha_detect=alpha_detect,
        gate_b_enabled=config.gate_b_enabled,
        alpha_residual=config.gate_b_alpha,
        K_max=config.gate_b_K_max,
    )

    # Min-methods gate: require ≥ min_methods_detect methods with p < 0.10
    if s1["passed"] and config.min_methods_detect > 1:
        p_vals = [s1["p_ftest"], s1["p_jtk"], s1["p_rain"], s1["p_harmreg"]]
        n_sig = sum(1 for p in p_vals if p < 0.10)
        if n_sig < config.min_methods_detect:
            s1 = dict(s1)
            s1["passed"] = False
            if s1["stage1_class"] == "has_12h":
                s1["stage1_class"] = "circadian_only" if s1["p_24"] < 0.05 else "non_rhythmic"

    result = {
        "stage1_passed": s1["passed"],
        "stage1_p_detect": s1["p_detect"],
        "stage1_p_ftest": s1["p_ftest"],
        "stage1_p_jtk": s1["p_jtk"],
        "stage1_p_rain": s1["p_rain"],
        "stage1_p_harmreg": s1["p_harmreg"],
        "stage1_p_24": s1["p_24"],
        "stage1_best_detector": s1["best_detector"],
        "stage1_waveform_hint": s1["waveform_hint"],
        "stage1_detection_strength": s1["detection_strength"],
        "stage1_gate_b": s1.get("gate_b_result"),
    }

    # If Stage 1 says non_rhythmic, output directly (no 12h component at all).
    # circadian_only genes (p_24 < 0.05 but CCT failed) skip Stage 2 —
    # they have no detectable 12h component to disentangle.
    if s1["stage1_class"] == "non_rhythmic":
        result.update({
            "classification": "non_rhythmic",
            "evidence_score": 0.0,
            "confidence": 0.0,
            "evidence_details": {},
        })
        return result

    if s1["stage1_class"] == "circadian_only":
        result.update({
            "classification": "circadian_only",
            "evidence_score": 0.0,
            "confidence": 0.0,
            "evidence_details": {},
        })
        return result

    # --- Stage 2: Disentanglement ---
    s2 = _stage2_disentangle(t, y, T_base, detection_context=s1, config=config)
    result.update(s2)

    return result


def batch_classify(t, Y_matrix, gene_names=None, T_base=24.0,
                      alpha_detect=None, config=None, verbose=True):
    """Run CHORD classifier on all genes in a matrix.

    Parameters
    ----------
    t : array-like
        Time points in hours.
    Y_matrix : array-like
        Expression matrix (n_genes, n_timepoints).
    gene_names : list of str, optional
        Gene names.
    T_base : float
        Base circadian period.
    alpha_detect : float
        Stage 1 gate threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict -- one per gene.
    """
    Y = np.asarray(Y_matrix, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    n_genes, n_tp = Y.shape

    if len(t) != n_tp:
        raise ValueError(
            "t length ({}) != Y_matrix columns ({})".format(len(t), n_tp)
        )

    if gene_names is None:
        gene_names = ["gene_{}".format(i) for i in range(n_genes)]

    results = []
    for i in range(n_genes):
        if verbose and (i % 100 == 0 or i == n_genes - 1):
            print("[CHORD {}/{}] {}".format(i + 1, n_genes, gene_names[i]))
        try:
            r = classify_gene(t, Y[i], T_base=T_base,
                                 alpha_detect=alpha_detect, config=config)
        except Exception as e:
            r = {
                "classification": "error",
                "stage1_passed": False,
                "evidence_score": 0.0,
                "confidence": 0.0,
                "evidence_details": {},
                "error": str(e),
            }
        r["gene_name"] = gene_names[i]
        results.append(r)

    return results
