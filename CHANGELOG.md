# Changelog

## [1.0.0] - 2026-02-22

First public release accompanying the iMeta manuscript submission.

### Algorithm
- Two-stage detect-then-disentangle architecture
- Stage 1: Cauchy Combination Test (CCT) fusion of parametric F-test, JTK_CYCLE, RAIN, and harmonic regression
- Stage 2: 12-evidence Bayesian scoring with BIC Bayes factors, phase freedom F-test, amplitude ratios, VMD-Hilbert IF, bispectral bicoherence, harmonic coherence
- Adaptive CCT weighting (zero weight for p ≥ 0.5)
- Three-gate architecture (Gate A: CCT threshold, Gate B: residual periodicity, Gate C: multi-method agreement)
- Confidence mapping via tanh with configurable divisor

### Benchmarks
- Synthetic: 15 scenarios × 50 replicates, Tier-1 F1 = 0.957, Tier-2 F1 = 0.939
- Real data: 11 datasets, 7 methods compared
- BMAL1-KO validation: harmonic ratio 32.1% → 8.9%
- Cross-species: 100% recovery of 9 conserved 12h genes
- Robustness: graceful degradation across sampling resolutions; 50% detection at SNR = 0.5

### Version History (internal)
- v6: Initial multi-evidence framework (Tier-2 F1 = 0.767)
- v7: Restructured evidence scoring (Tier-2 F1 = 0.928)
- v8: Amplitude ratio calibration (Tier-2 F1 = 0.940)
- v9: Adaptive CCT + m0_r2_k4 discriminator (Tier-2 F1 = 0.983 synthetic, 0.939 benchmark)
