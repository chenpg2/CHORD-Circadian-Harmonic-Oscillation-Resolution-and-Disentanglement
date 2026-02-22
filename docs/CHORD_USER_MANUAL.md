# CHORD User Manual

> Circadian Harmonic Oscillation Resolver & Disentangler — "Detect-then-Disentangle" Architecture

## 1. Overview

CHORD uses a two-stage architecture to classify 12-hour (ultradian) gene expression rhythms:

- **Stage 1 (Detection)**: Liberal multi-method fusion (F-test + JTK_CYCLE + RAIN + Harmonic regression) via Cauchy Combination Test (CCT). Maximizes recall.
- **Stage 2 (Disentanglement)**: 9-line evidence scoring to determine whether a detected 12h rhythm is an independent oscillator or a circadian harmonic artifact.

### Output Classifications

| Classification | Meaning |
|---|---|
| `independent_ultradian` | High-confidence independent 12h oscillator |
| `likely_independent_ultradian` | Moderate-confidence independent 12h oscillator |
| `harmonic` | 12h component is a harmonic artifact of 24h waveform |
| `circadian_only` | Only 24h rhythm detected, no 12h component |
| `non_rhythmic` | No significant rhythm detected |
| `ambiguous` | Insufficient evidence to classify |

---

## 2. Quick Start

### Default usage (recommended for most cases)

```python
from chord.bhdt.classifier import classify_gene, batch_classify
import numpy as np

t = np.arange(0, 48, 2.0)  # 2h sampling, 48h duration
result = classify_gene(t, y)
print(result["classification"])
```

### With tissue-specific preset

```python
from chord.bhdt.classifier import CHORDConfig, classify_gene

# For ovary tissue with 24 timepoints
config = CHORDConfig.for_tissue("ovary", n_samples=24)
result = classify_gene(t, y, config=config)
```

### With custom parameters

```python
from chord.bhdt.classifier import CHORDConfig, batch_classify

config = CHORDConfig(
    alpha_detect=0.20,
    confidence_thresholds=(0.5, 0.25, -0.6, -0.3),
    n_bootstrap=999,
)
results = batch_classify(t, Y_matrix, config=config)
```

### Batch analysis

```python
results = batch_classify(t, Y_matrix, gene_names=gene_list, config=config)

# Extract summary
for r in results:
    print(f"{r['gene_name']}: {r['classification']} "
          f"(confidence={r['confidence']:.2f})")
```

---

## 3. Configurable Parameters Reference

All parameters are set via the `CHORDConfig` class. Default values are calibrated on 2h-sampled, 48h mouse liver/ovary RNA-seq data (N=24 timepoints, 4 biological replicates per group per timepoint).

### 3.1 Stage 1: Detection Gate

| Parameter | Default | Range | Description |
|---|---|---|---|
| `alpha_detect` | `None` (auto) | 0.01 - 0.50 | CCT p-value threshold for Stage 1 gate |

**Auto-selection logic** (when `alpha_detect=None`):

| Sample Size (N) | Auto alpha | Rationale |
|---|---|---|
| N >= 48 | 0.10 | Sufficient power, standard threshold |
| 30 <= N < 48 | 0.20 | Reduced power, relax gate |
| N < 30 | 0.50 | Very low power, rely on Stage 2 for specificity |

**When to override**:
- Set `alpha_detect=0.05` if you want fewer genes entering Stage 2 (faster, more conservative)
- Set `alpha_detect=0.30` if you suspect many weak 12h signals (slower, more sensitive)

### 3.2 Stage 2: Classification Thresholds

| Parameter | Default | Range | Description |
|---|---|---|---|
| `confidence_thresholds` | `(0.6, 0.3, -0.6, -0.3)` | each in [-1, 1] | (indep_high, indep_low, harm_high, harm_low) |
| `confidence_divisor` | `6.0` | 3.0 - 10.0 | Divisor in tanh(score/divisor) mapping |

The confidence score is computed as `tanh(evidence_score / confidence_divisor)`, mapping the raw evidence score to [-1, 1].

**Classification rules**:
```
confidence >= indep_high  -> independent_ultradian
confidence >= indep_low   -> likely_independent_ultradian
confidence <= harm_high   -> harmonic
confidence <= harm_low    -> harmonic
otherwise                 -> ambiguous
```

**Tuning guidance**:

| Goal | confidence_thresholds | confidence_divisor |
|---|---|---|
| High precision (fewer false positives) | (0.7, 0.4, -0.7, -0.4) | 8.0 |
| Balanced (default) | (0.6, 0.3, -0.6, -0.3) | 6.0 |
| High recall (fewer false negatives) | (0.5, 0.2, -0.5, -0.2) | 4.0 |
| Exploratory / weak signals | (0.4, 0.15, -0.4, -0.15) | 4.0 |

### 3.3 Bootstrap Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `n_bootstrap` | `199` | 99 - 9999 | Number of bootstrap replicates |
| `bootstrap_trigger_small_n` | `6.0` | 2.0 - 10.0 | \|log(BF)\| trigger for N < bf_small_n_threshold |
| `bootstrap_trigger_large_n` | `3.0` | 1.0 - 6.0 | \|log(BF)\| trigger for N >= bf_small_n_threshold |

**Runtime impact**: Bootstrap is the most expensive step. Each gene triggers bootstrap only when the Bayes Factor is ambiguous (|log(BF)| below the trigger threshold).

| n_bootstrap | Relative runtime | p-value resolution |
|---|---|---|
| 99 | 0.5x | 0.01 |
| 199 (default) | 1x | 0.005 |
| 999 | 5x | 0.001 |
| 4999 | 25x | 0.0002 |

**Recommendation**: Use 199 for exploratory analysis, 999 for publication-quality results.

### 3.4 VMD-Hilbert Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `vmd_amp_gate_strong` | `0.10` | 0.05 - 0.20 | Min 12h/24h amplitude ratio when detection is strong |
| `vmd_amp_gate_default` | `0.15` | 0.08 - 0.30 | Min 12h/24h amplitude ratio otherwise |

These gates prevent VMD from contributing evidence when the 12h mode amplitude is negligible relative to 24h. Lower values trust VMD more aggressively.

### 3.5 Bayes Factor Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `bf_small_n_threshold` | `40` | 20 - 60 | N below which BF is downweighted |
| `bf_small_n_weight` | `0.5` | 0.1 - 1.0 | Weight for negative BF evidence at small N |
| `small_sample_aicc_threshold` | `50` | 30 - 80 | N below which AICc replaces BIC |

**Why downweight at small N**: BIC's log(N) penalty disproportionately penalizes the more complex M1-free model when N is small, creating a systematic bias toward the harmonic (M0) model. The downweighting compensates for this known limitation.

### 3.6 Model Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `K_harmonics` | `3` | 2 - 5 | Number of harmonics in M0 model |
| `period_dev_n_gate` | `36` | 20 - 48 | Min N to penalize period locked to harmonic |

`K_harmonics=3` means M0 models 24h + 12h + 8h components. Increase to 4 if your tissue has strong 6h rhythms.

### 3.7 Bispectral & Waveform Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `bispectral_nonsin_discount` | `0.5` | 0.0 - 1.0 | Discount for bispectral harmonic evidence on non-sinusoidal waveforms |
| `amp_ratio_asymmetry_gate` | `0.5` | 0.3 - 0.8 | Max amplitude ratio for asymmetry evidence |
| `m0_r2_discount_threshold` | `0.5` | 0.3 - 0.8 | Min M0 R-squared for period deviation discount |

---

## 4. Sample Size Recommendations

### 4.1 Minimum Requirements

| Timepoints (N) | Sampling | Duration | CHORD Capability |
|---|---|---|---|
| < 12 | any | any | **Not recommended**. Stage 1 has very low power; Stage 2 evidence is unreliable. Use population-level CGPDT instead. |
| 12-18 | 2-4h | 24-48h | **Marginal**. Set `alpha_detect=0.50`, use `n_bootstrap=999`. Expect high ambiguous rate (30-50%). |
| 18-24 | 2h | 36-48h | **Adequate**. Default parameters work. Sensitivity ~50-70% for strong signals. |
| 24-48 | 1-2h | 48-96h | **Good**. Full power. Default parameters optimal. |
| > 48 | 1-2h | > 48h | **Excellent**. Can tighten thresholds for higher precision. |

### 4.2 Recommended Configurations by Sample Size

#### N < 20 (e.g., 12 timepoints at 4h over 48h)

```python
config = CHORDConfig(
    alpha_detect=0.50,           # very liberal gate
    confidence_thresholds=(0.5, 0.2, -0.5, -0.2),  # relaxed
    confidence_divisor=5.0,      # slightly more decisive
    n_bootstrap=999,             # more stable p-values
    bootstrap_trigger_small_n=8.0,  # trigger bootstrap more often
    bf_small_n_weight=0.3,       # heavily downweight BF
    vmd_amp_gate_default=0.10,   # trust VMD more
    period_dev_n_gate=20,        # don't penalize period deviation
)
```

**Caveats at N < 20**:
- Free-period optimizer has ~1h resolution, so period deviation evidence is unreliable
- BIC/AICc strongly biased toward simpler model
- Bootstrap has limited power (consider n_bootstrap=999)
- Expect 30-50% ambiguous classifications
- **Strongly recommend** supplementing with population-level CGPDT test

#### N = 20-30 (e.g., 24 timepoints at 2h over 48h)

```python
config = CHORDConfig(
    alpha_detect=0.20,
    n_bootstrap=499,
    bootstrap_trigger_small_n=6.0,
    bf_small_n_weight=0.5,
)
```

This is the typical configuration for standard circadian experiments (e.g., the ovary aging dataset with 25 ZT points).

#### N = 30-48

```python
config = CHORDConfig(
    alpha_detect=0.10,
    n_bootstrap=199,
)
```

Default parameters are well-calibrated for this range.

#### N > 48

```python
config = CHORDConfig(
    alpha_detect=0.05,
    confidence_thresholds=(0.7, 0.4, -0.7, -0.4),
    confidence_divisor=7.0,
    n_bootstrap=199,
)
```

With abundant data, you can afford stricter thresholds for higher precision.


---

## 5. Tissue-Type Recommendations

### 5.1 Built-in Presets

Use `CHORDConfig.for_tissue()` for common tissue types:

```python
config = CHORDConfig.for_tissue("ovary", n_samples=24)
```

Available presets:

| Tissue Type | Key Adjustments | Rationale |
|---|---|---|
| `"liver"`, `"scn"` | Default parameters | Strong circadian clock, robust 12h rhythms |
| `"ovary"`, `"uterus"`, `"oviduct"` | Relaxed confidence (0.5/0.25), lower VMD gate (0.12) | 12h rhythms biologically real but weaker; ER stress/UPR-driven |
| `"heart"`, `"muscle"` | Slightly conservative confidence (0.6/0.35) | Moderate 12h rhythms, potential mechanical artifacts |
| `"fibroblast"`, `"cell_line"` | Very liberal (0.5/0.2), low VMD gate (0.10), low BF weight (0.3) | Noisy, weak rhythms, often short time series |
| `"weak_rhythm"`, `"peripheral"`, `"adipose"`, `"kidney"` | Relaxed confidence (0.5/0.25), lower bispectral discount (0.3) | Weak ultradian signals |

### 5.2 Detailed Tissue Guidance

#### Liver / SCN (Strong Circadian Tissues)

These tissues have robust circadian clocks and well-characterized 12h rhythms (e.g., XBP1s-driven ER stress cycle in liver). Default parameters are optimal.

```python
config = CHORDConfig()  # defaults are calibrated on liver-like data
```

**Expected performance**: Sensitivity > 70%, Specificity > 80% for N >= 24.

#### Ovary / Reproductive Tissues

Ovarian tissue presents unique challenges for 12h rhythm analysis:

1. **Weaker ultradian signals**: 12h rhythms in ovary are driven by ER stress and metabolic pathways, not the core clock. Signal-to-noise ratio is lower than liver.
2. **Biological remodeling with aging**: In the ovary aging context, 12h rhythms undergo systematic reprogramming (3,068 genes gain rhythm, 1,681 lose rhythm between Young and Old mice), so the classifier must be sensitive enough to detect both strong and weak 12h signals.
3. **Non-sinusoidal waveforms**: Many ovarian 12h genes show pulsatile expression patterns (e.g., UPR genes), which parametric F-tests may miss.

```python
config = CHORDConfig.for_tissue("ovary", n_samples=24)
# Equivalent to:
config = CHORDConfig(
    confidence_thresholds=(0.5, 0.25, -0.6, -0.3),
    vmd_amp_gate_default=0.12,
)
```

**Key considerations**:
- The relaxed `indep_low=0.25` threshold captures more weak independent signals
- The strict `harm_high=-0.6` threshold maintains specificity for harmonic rejection
- If your ovary data has < 20 timepoints, additionally set `alpha_detect=0.50` and `bootstrap_trigger_small_n=8.0`

#### Heart / Muscle

Heart tissue has moderate 12h rhythms but can exhibit mechanical/contractile artifacts that mimic ultradian oscillations.

```python
config = CHORDConfig.for_tissue("heart", n_samples=24)
```

**Key considerations**:
- Slightly more conservative `indep_low=0.35` to avoid calling mechanical artifacts as independent
- If analyzing cardiac-specific genes known to have 12h rhythms (e.g., metabolic genes), you may relax to default thresholds

#### Cell Lines / In Vitro

Cell line data is typically noisier with weaker rhythms and often shorter time series.

```python
config = CHORDConfig.for_tissue("fibroblast", n_samples=12)
```

**Key considerations**:
- Very liberal detection and classification thresholds
- BF evidence heavily downweighted (`bf_small_n_weight=0.3`) because BIC is unreliable at small N
- **Strongly recommend** using `n_bootstrap=999` for more stable results
- Consider supplementing with population-level CGPDT test across all genes

### 5.3 Custom Tissue Configuration

For tissues not covered by presets, start from the closest preset and adjust:

```python
# Example: brain tissue (moderate rhythms, good data quality)
config = CHORDConfig.for_tissue("liver")  # start from strong-rhythm preset
config.confidence_thresholds = (0.55, 0.28, -0.6, -0.3)  # slightly relax
config.vmd_amp_gate_default = 0.13
```

---

## 6. Interpreting Results

### 6.1 Key Output Fields

```python
result = classify_gene(t, y, config=config)
```

| Field | Type | Description |
|---|---|---|
| `classification` | str | Final classification (see Section 1) |
| `confidence` | float | Confidence score in [-1, 1]. Positive = independent, negative = harmonic |
| `evidence_score` | float | Raw evidence score before tanh mapping |
| `stage1_passed` | bool | Whether gene passed Stage 1 detection |
| `stage1_p_detect` | float | CCT-combined p-value for 12h detection |
| `stage1_best_detector` | str | Which method gave smallest p-value ("ftest", "jtk", "rain", "harmreg") |
| `stage1_waveform_hint` | str | "sinusoidal", "non_sinusoidal", or "unknown" |
| `evidence_details` | dict | Per-evidence-line scores (9 lines) |
| `bayes_factor` | float | M1-free vs M0 Bayes Factor |
| `amp_ratio` | float | 12h/24h amplitude ratio from M1 fit |

### 6.2 Evidence Details

The `evidence_details` dict contains scores for each of the 9 evidence lines:

| Evidence Line | Score Range | Positive = | Negative = |
|---|---|---|---|
| `bayes_factor` | [-2, +2] | M1-free fits better (independent) | M0 fits better (harmonic) |
| `period_deviation` | [-1, +3] | Fitted period deviates from 12.0h | Period locked to exactly 12.0h |
| `amplitude_ratio` | [-2, +3] | Strong 12h relative to 24h | Weak 12h relative to 24h |
| `residual_improvement` | varies | M1 residuals much better than M0 | No improvement |
| `vmd_hilbert` | [-3, +2] | VMD IF shows independent coupling | VMD IF shows harmonic coupling |
| `bispectral` | [-2, +1] | Low bicoherence (no QPC) | High bicoherence (QPC present) |
| `phase_coupling` | varies | Phases uncoupled | Phases locked |
| `bootstrap_lrt` | [-1, +2] | Bootstrap rejects M0 | Bootstrap fails to reject M0 |
| `waveform_asymmetry` | varies | Symmetric waveform | Asymmetric (harmonic-like) waveform |

### 6.3 Diagnostic Checklist

If a gene is classified as `ambiguous`, check:

1. **Stage 1 detection strength** (`stage1_detection_strength`): If < 1.5, the 12h signal may be too weak for reliable disentanglement.
2. **Bayes Factor** (`bayes_factor`): If close to 1.0, the models are indistinguishable. Consider increasing `n_bootstrap`.
3. **Amplitude ratio** (`amp_ratio`): If 0.3-0.5, the 12h component is in the "gray zone" where both independent and harmonic explanations are plausible.
4. **VMD convergence** (`vmd_converged`): If False, VMD failed to decompose the signal. This evidence line contributes 0.
5. **Evidence balance**: Check `evidence_details` for conflicting evidence (e.g., BF says harmonic but period deviation says independent).

---

## 7. Performance Considerations

### 7.1 Runtime Estimates

Per-gene runtime on a single CPU core (Intel Xeon, 2.4 GHz):

| Component | Time (N=24) | Time (N=48) |
|---|---|---|
| Stage 1 (4 detectors + CCT) | ~5 ms | ~10 ms |
| Stage 2 (model fitting) | ~15 ms | ~30 ms |
| Stage 2 (VMD-Hilbert) | ~20 ms | ~40 ms |
| Stage 2 (bispectral) | ~10 ms | ~20 ms |
| Stage 2 (bootstrap, n=199) | ~200 ms | ~400 ms |
| **Total (no bootstrap)** | **~50 ms** | **~100 ms** |
| **Total (with bootstrap)** | **~250 ms** | **~500 ms** |

For 18,000 genes: ~15 min (no bootstrap) to ~75 min (all bootstrap) on a single core.

### 7.2 Reducing Runtime

1. **Increase `alpha_detect`**: Fewer genes enter Stage 2 (but may miss weak signals)
2. **Decrease bootstrap triggers**: `bootstrap_trigger_small_n=3.0`, `bootstrap_trigger_large_n=1.5` — fewer genes trigger bootstrap
3. **Reduce `n_bootstrap`**: 99 instead of 199 (less stable p-values)
4. **Batch processing**: `batch_classify()` processes genes sequentially but can be parallelized externally via multiprocessing

---

## 8. FAQ

### Q: My tissue has very weak 12h rhythms. What should I do?

Use the `"weak_rhythm"` preset and supplement with population-level analysis:

```python
config = CHORDConfig.for_tissue("weak_rhythm", n_samples=24)
results = batch_classify(t, Y_matrix, config=config)

# Also run population-level CGPDT
from chord.bhdt import population_phase_analysis
pop_result = population_phase_analysis(t, Y_matrix)
```

The CGPDT test works even when per-gene evidence is weak, because it aggregates phase information across all genes.

### Q: I have replicates at each timepoint. How should I handle them?

CHORD expects a single expression value per timepoint. Options:

1. **Average replicates** (recommended): Take the mean across replicates at each timepoint. This reduces noise and is the standard approach.
2. **Concatenate as extended time series**: If you have 4 replicates at 24 timepoints, treat as 96 timepoints with repeated time values. This preserves variance information but may violate independence assumptions.

```python
# Option 1: Average replicates
y_mean = expression_matrix.groupby("timepoint").mean()

# Option 2: Concatenate (use with caution)
t_concat = np.tile(t_unique, n_replicates)
y_concat = expression_matrix.values.flatten()
```

### Q: Can I use CHORD for non-12h ultradian rhythms (e.g., 8h, 6h)?

Yes, but you need to adjust `T_base`:

```python
# For 8h rhythm disentanglement (is it independent or 24h/3 harmonic?)
config = CHORDConfig(T_base=24.0, K_harmonics=4)
# The classifier will test whether the 8h component (T_base/3) is independent

# For 16h rhythm (is it independent or 48h/3 harmonic?)
config = CHORDConfig(T_base=48.0)
```

### Q: How do I interpret a high ambiguous rate?

A high ambiguous rate (> 30%) typically indicates:

1. **Insufficient temporal resolution**: Need more timepoints or shorter sampling interval
2. **Low signal-to-noise ratio**: Consider averaging replicates or filtering low-expression genes
3. **Thresholds too strict**: Try relaxing `confidence_thresholds`
4. **Genuine biological ambiguity**: Some genes may truly have mixed 12h signals (partially independent, partially harmonic)

### Q: How does the current architecture differ from earlier versions?

| Feature | Legacy (V4) | Current |
|---|---|---|
| Architecture | Single-stage with post-hoc rescue | Two-stage detect-then-disentangle |
| Detection | F-test primary, JTK/RAIN rescue | CCT fusion of all 4 methods |
| F-test gate (w_12) | Compresses evidence when F-test weak | Removed (Stage 1 handles detection) |
| 24h dominance penalty | Penalizes when 24h >> 12h | Removed (irrelevant to disentanglement) |
| both_weak cap | Caps score when both models fit poorly | Removed (Stage 1 filters weak genes) |
| VMD amplitude gate | 0.3 (strict) | 0.10-0.15 (relaxed, configurable) |
| Non-sinusoidal handling | Post-hoc rescue only | Waveform-aware bispectral discount |
| Configuration | Hardcoded parameters | CHORDConfig with tissue presets |
| Output | Score-based classification | Continuous confidence + classification |

---

## 9. Appendix: Complete CHORDConfig Reference

```python
class CHORDConfig:
    """All parameters with defaults."""

    # Global
    T_base = 24.0                    # Base circadian period (hours)
    alpha_detect = None              # Stage 1 gate (None = auto)

    # Classification
    confidence_thresholds = (0.6, 0.3, -0.6, -0.3)
    confidence_divisor = 6.0

    # Bootstrap
    n_bootstrap = 199
    bootstrap_trigger_small_n = 6.0
    bootstrap_trigger_large_n = 3.0

    # VMD-Hilbert
    vmd_amp_gate_strong = 0.10
    vmd_amp_gate_default = 0.15

    # Bayes Factor
    bf_small_n_threshold = 40
    bf_small_n_weight = 0.5
    small_sample_aicc_threshold = 50

    # Model
    K_harmonics = 3
    period_dev_n_gate = 36

    # Bispectral & Waveform
    bispectral_nonsin_discount = 0.5
    amp_ratio_asymmetry_gate = 0.5
    m0_r2_discount_threshold = 0.5
```

### Tissue Preset Summary

| Preset | alpha_detect | confidence_thresholds | vmd_amp_gate_default | bf_small_n_weight | bispectral_nonsin_discount |
|---|---|---|---|---|---|
| default / liver / scn | auto | (0.6, 0.3, -0.6, -0.3) | 0.15 | 0.5 | 0.5 |
| ovary / reproductive | auto | (0.5, 0.25, -0.6, -0.3) | 0.12 | 0.5 | 0.5 |
| heart / muscle | auto | (0.6, 0.35, -0.55, -0.25) | 0.15 | 0.5 | 0.5 |
| fibroblast / cell_line | auto | (0.5, 0.2, -0.5, -0.2) | 0.10 | 0.3 | 0.5 |
| weak_rhythm / peripheral | auto | (0.5, 0.25, -0.6, -0.3) | 0.12 | 0.5 | 0.3 |
