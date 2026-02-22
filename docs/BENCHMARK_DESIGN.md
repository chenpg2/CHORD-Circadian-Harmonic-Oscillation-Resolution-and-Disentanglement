# CHORD Benchmark: Tiered Evaluation Design

## The Problem

The original benchmark evaluated all methods on a 4-class task:
`independent_ultradian`, `harmonic`, `circadian_only`, `non_rhythmic`.

Classical methods (Lomb-Scargle, Cosinor, Harmonic Regression) **cannot**
distinguish a true independent 12h oscillator from a 12h harmonic artifact
of a non-sinusoidal 24h waveform. They detect "12h power" but have no
mechanism to determine its origin.

Two of 12 scenarios (`sawtooth_harmonic`, `peaked_harmonic`) have
`has_harmonic_12h=True` and `has_independent_12h=False`. Including these
in classical methods' binary 12h accuracy penalizes them for a distinction
they were never designed to make, giving CHORD/BHDT an unfair advantage.

## Tiered Reporting

### Tier-1: Binary 12h Detection (fair cross-method comparison)

- Question: "Is there a 12h component in this signal?"
- For **classical methods**: excludes `sawtooth_harmonic` and
  `peaked_harmonic` scenarios (where the correct binary answer is
  ambiguous — there IS 12h power, but it's not independent).
- For **BHDT**: includes all 12 scenarios.
- Metrics: accuracy, sensitivity, specificity.

### Tier-2: Harmonic Disentanglement (CHORD-unique capability)

- Question: "Is the 12h component independent or a harmonic artifact?"
- Only evaluated for methods that can make this distinction (BHDT).
- Uses full 4-class evaluation across all 12 scenarios.
- Metrics: accuracy, macro-F1, confusion matrix.

## Implementation

`summarize_benchmark()` returns the original `per_method`, `per_scenario`,
and `overall` keys unchanged, plus two new keys:

- `tier1_binary_12h`: dict of `{method: {accuracy, sensitivity, specificity, n_samples}}`
- `tier2_disentangle`: dict of `{method: {accuracy, macro_f1, confusion_matrix, labels, n_samples}}`
