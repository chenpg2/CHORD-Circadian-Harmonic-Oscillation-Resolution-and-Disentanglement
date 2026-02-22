# CHORD

**Circadian Harmonic Oscillation Resolution and Disentanglement**

[![PyPI version](https://badge.fury.io/py/chord-rhythm.svg)](https://pypi.org/project/chord-rhythm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A two-stage statistical framework for detecting 12-hour ultradian rhythms in transcriptomic data and disentangling independent oscillations from circadian harmonic artifacts.

> **Paper:** Chen P. et al. (2026) *CHORD: Detecting and Classifying Independent 12-Hour Rhythms from Circadian Harmonics.* (submitted).

## The Problem

Gene expression time-series often show 12-hour periodicity. This can arise from two fundamentally different mechanisms:

- **Independent 12h oscillators** — driven by dedicated molecular pathways (e.g., IRE1α-XBP1s ER stress cycle)
- **Circadian harmonics** — mathematical artifacts of non-sinusoidal 24h waveforms

Standard spectral methods (Fourier, Lomb-Scargle, JTK_CYCLE) cannot distinguish between these two cases. CHORD solves this.

## How It Works

**Stage 1 — Detection:** Fuses four complementary methods (parametric F-test, JTK_CYCLE, RAIN, harmonic regression) via the Cauchy Combination Test (CCT), providing robust detection under arbitrary dependence.

**Stage 2 — Disentanglement:** Evaluates 12 independent lines of evidence — BIC-based Bayes factors, phase freedom F-tests, amplitude ratios, VMD-Hilbert instantaneous frequency, bispectral bicoherence, and more — to classify each detected rhythm as independent or harmonic.

## Performance

| Benchmark | Sensitivity | Specificity | Precision | F1 |
|-----------|-------------|-------------|-----------|------|
| Tier-1: Detection | 92.5% | 96.7% | 99.1% | **0.957** |
| Tier-2: Disentanglement | 88.6% | 100% | 100% | **0.939** |

*Synthetic benchmark: 15 scenarios × 50 replicates = 750 genes, 48 timepoints.*

On real data (Hughes 2009, 11 datasets), CHORD recovers 60.5% of known 12h genes (F1 = 0.590), outperforming all six comparison methods. BMAL1-KO validation shows 72% reduction in harmonic classification, confirming biological validity.

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

### Single gene

```python
import numpy as np
from chord.bhdt.classifier import classify_gene

t = np.arange(0, 48, 2.0)  # 2h sampling over 48h
y = expression_data          # 1D array

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

- **Detection:** F1 = 0.957 on synthetic data; 60.5% known 12h gene recovery on real data
- **Disentanglement:** F1 = 0.939 with perfect specificity (zero false harmonic calls)
- **Biological validation:** BMAL1-KO harmonic ratio drops 32.1% → 8.9% (72% reduction)
- **Cross-species:** 100% recovery of 9 conserved 12h genes across 11 datasets
- **Robustness:** 50% detection at SNR = 0.5; maintains advantage across all sampling resolutions
- **Speed:** 9.7 ms/gene median; ~3 min for 20,000 genes on single CPU

## Project Structure

```
chord/
├── src/chord/
│   ├── bhdt/                  # Core algorithm
│   │   ├── classifier.py      # Two-stage classifier (main entry point)
│   │   ├── detection/         # Stage 1: CCT-fused detection
│   │   ├── inference.py       # Evidence computation
│   │   ├── models.py          # M0/M1 model fitting
│   │   ├── bispectral.py      # Bispectral bicoherence
│   │   ├── hilbert_if.py      # VMD-Hilbert IF analysis
│   │   └── bootstrap.py       # Parametric bootstrap LRT
│   ├── simulation/            # Synthetic data generation
│   ├── data/                  # Dataset loaders (GEO)
│   └── benchmarks/            # Method comparison framework
├── scripts/                   # Benchmark and analysis scripts
├── tests/                     # Test suite
├── docs/                      # Documentation
└── results/                   # Benchmark results (v9)
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
