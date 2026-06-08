# CHORD

**Circadian Harmonic Oscillation Resolution and Disentanglement**

[![PyPI version](https://badge.fury.io/py/chord-rhythm.svg)](https://pypi.org/project/chord-rhythm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A two-stage statistical framework for detecting 12-hour ultradian rhythms in transcriptomic **time-series** and disentangling their generative origin — autonomous oscillators, circadian harmonics, or the intersection of two anti-phase 24h processes.

> **Paper:** Chen P. et al. (2026) *CHORD: Detecting and Classifying Independent 12-Hour Rhythms from Circadian Harmonics.* (submitted).

## The Problem

Gene expression time-series often show 12-hour periodicity. A purely spectral "12h peak" conflates **three** mechanistically distinct origins:

- **Class A — circadian harmonic:** the 2nd harmonic of a non-sinusoidal 24h waveform; phase-locked to the clock, no dedicated 12h machinery.
- **Class B — autonomous oscillator:** a dedicated 12h pathway with its own phase (e.g., the IRE1α–XBP1s ER-stress cycle); survives clock ablation.
- **Class C — intersection artifact:** an apparent 12h emerging when two anti-phase 24h processes are superimposed at the level of synthesis-minus-degradation (Hughes 2009 mechanism); no genuine 12h latent.

Standard spectral methods (Fourier, Lomb-Scargle, JTK_CYCLE) detect the peak but cannot tell A from B from C. CHORD resolves this **ternary** question.

## How It Works

**Stage 1 — Detection:** Fuses four complementary methods (parametric F-test, JTK_CYCLE, RAIN, harmonic regression) via the Cauchy Combination Test (CCT), providing robust detection under arbitrary dependence.

**Stage 2 — Ternary disentanglement (A / B / C):** A calibrated discriminator built on four *orthogonal* statistics, each targeting an identifiable axis of the generative taxonomy:

| Statistic | Separates |
|-----------|-----------|
| Phase-freedom F-test (is the 12h phase locked to 2·φ₂₄?) | autonomous **B** vs driven **{A, C}** |
| Twin-peak symmetry (are the two daily peaks equal?) | **B** vs **C** |
| Amplitude structure (A₁₂/A₂₄ ratio + harmonic-decay residual) | **A** vs **C** |
| 24h-dominance decision prior | guards strong-circadian genes against over-calling B |

The statistics feed a class-weighted multinomial logistic model. This replaces CHORD's earlier additive 12-evidence score, which an internal audit found to be net-negative (most evidence sat on non-identifying axes); the rebuilt discriminator improves apples-to-apples disentanglement AUC from **0.773 → 0.826** on the same hard benchmark.

## Performance

**Stage 1 — Detection** (synthetic benchmark: 15 scenarios × 50 replicates = 750 genes, 48 timepoints):

| Sensitivity | Specificity | Precision | F1 |
|-------------|-------------|-----------|------|
| 92.5% | 96.7% | 99.1% | **0.957** |

**Stage 2 — Ternary disentanglement** (held-out A/B/C benchmark with 12h-SNR-controlled sampling):

| Axis | Metric |
|------|--------|
| Autonomous **B** vs driven **{A, C}** | apples-to-apples AUC **0.826** (legacy 0.773) |
| **B** vs **C** (twin-peak asymmetry) | AUC **0.87–0.95** across 24 Class-C parametrizations |
| **A** vs **C** (amplitude structure) | AUC ≈ 1.0 |

**Interventional gold standard (the key biological test).** On the dense XBP1-LKO time-series (GSE130890, 2h sampling × 48h), genes CHORD calls autonomous-**B** have their 12h rhythm **collapse** when the XBP1/IRE1 12h clock is knocked out (12h KO/WT median 0.27, 89% die), while driven-**A** (24h harmonic) 12h is **preserved** (0.63); the circadian 24h positive control is intervention-specific (KO/WT 0.91). B-vs-driven separation **p = 1.8 × 10⁻¹⁹**, robust to amplitude matching. This confirms CHORD's autonomous/driven distinction reflects real biology, not a synthetic-benchmark artifact.

> **Scope.** CHORD is a **time-series** method (dense sampling, e.g. ≤2–4h). The binding constraint on identifiability is 12h-SNR, not sampling density — see the identifiability budget in the paper. Cross-sectional / unordered-snapshot inference is **not** supported: it is defeated by post-mortem stress confounding and per-gene SNR below the classification floor.

## Installation

```bash
pip install chord-rhythm
```

From source:

```bash
git clone https://github.com/chenpg2/CHORD-Circadian-Harmonic-Oscillation-Resolution-and-Disentanglement.git
cd CHORD-Circadian-Harmonic-Oscillation-Resolution-and-Disentanglement
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install chord-rhythm[bayes]   # Bayesian inference (NumPyro/JAX)
pip install chord-rhythm[deep]    # Neural ODE (PyTorch)
pip install chord-rhythm[viz]     # Visualization (matplotlib)
pip install chord-rhythm[full]    # Everything
```

## Quick Start

### Ternary A/B/C disentanglement (current method)

```python
import numpy as np
from chord.bhdt.stage2_orthogonal import disentangle_ternary

t = np.arange(0, 48, 2.0)  # 2h sampling over 48h
y = expression_data          # 1D array

result = disentangle_ternary(t, y, T_base=24.0)
print(result["class"])           # 'A_harmonic' | 'B_independent' | 'C_intersection' | 'ambiguous' | 'none'
print(result["autonomy_score"])  # P(B): autonomous-vs-driven decision score
print(result["proba"])           # calibrated {'A','B','C'} probabilities
```

### Single gene (binary classifier)

```python
from chord.bhdt.classifier import classify_gene

result = classify_gene(t, y)
print(result["classification"])  # 'independent', 'harmonic', 'ambiguous', ...
print(result["confidence"])      # continuous score in [-1, 1]
```

### Batch analysis

```python
from chord.bhdt.classifier import batch_classify

results = batch_classify(t, Y_matrix, gene_names=gene_list)
# Returns DataFrame with classification for each gene
```

### Command line

```bash
chord detect expression.csv -t 0,2,4,...,46 -o results.csv
```

## Key Results

- **Detection:** F1 = 0.957 on synthetic data; 60.5% known 12h gene recovery on real data (Hughes 2009, 11 datasets)
- **Ternary disentanglement:** B-vs-driven apples-to-apples AUC 0.826 (legacy 0.773); B-vs-C AUC 0.87–0.95; A-vs-C AUC ≈ 1.0
- **Interventional validation:** in XBP1-LKO (GSE130890), autonomous-B 12h collapses (KO/WT 0.27) while harmonic-A 12h persists (0.63), p = 1.8 × 10⁻¹⁹; circadian 24h preserved
- **Robustness:** binding constraint is 12h-SNR (≈ A₁₂/σ·√M), not sampling density — autonomy AUC ≥ 0.95 at SNR ≳ 1.1
- **Speed:** 9.7 ms/gene median; ~3 min for 20,000 genes on single CPU

## Project Structure

```
chord/
├── src/chord/
│   ├── bhdt/                     # Core algorithm
│   │   ├── classifier.py         # Two-stage binary classifier
│   │   ├── stage2_orthogonal.py  # Ternary A/B/C discriminator (current Stage 2)
│   │   ├── matrix_pencil.py      # LS-ESPRIT spectral front-end (high-N regimes)
│   │   ├── detection/            # Stage 1: CCT-fused detection
│   │   ├── inference.py          # Evidence computation
│   │   ├── models.py             # M0/M1 model fitting
│   │   ├── bispectral.py         # Bispectral bicoherence
│   │   ├── hilbert_if.py         # VMD-Hilbert IF analysis
│   │   └── bootstrap.py          # Parametric bootstrap LRT
│   ├── simulation/               # Synthetic data generation
│   │   ├── generator.py          # Scenario generator (incl. Class-C intersection ODE)
│   │   └── ternary_benchmark.py  # Honest A/B/C benchmark (12h-SNR controlled)
│   ├── data/                     # Dataset loaders (GEO)
│   └── benchmarks/               # Method comparison framework
├── scripts/                      # Benchmark and analysis scripts
├── tests/                        # Test suite
├── docs/                         # Documentation
└── results/                      # Benchmark results (v9)
```

## Citation

```bibtex
@article{chen2026chord,
  title={CHORD: Detecting and Classifying Independent 12-Hour Rhythms
         from Circadian Harmonics},
  author={Chen, Peigen},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
