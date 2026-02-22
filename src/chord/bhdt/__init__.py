"""BHDT — Bayesian Harmonic Disentanglement of Transcriptomic rhythms.

Public API
----------
bhdt_analytic              : Single-gene harmonic disentanglement analysis.
classify_gene              : Two-stage detect-then-disentangle classifier (recommended).
batch_classify             : Batch classification.
CHORDConfig                : Classifier configuration with tissue presets.
bispectral_coupling_test   : Bispectral harmonic coupling test (BHCT).
cross_gene_phase_test      : Population-level phase distribution test (CGPDT).
savage_dickey_bf           : Exact Bayes Factor via Savage-Dickey density ratio.
population_phase_analysis  : Dataset-level phase coherence analysis.
"""

from chord.bhdt.inference import bhdt_analytic, population_phase_analysis
from chord.bhdt.classifier import classify_gene, batch_classify, CHORDConfig
from chord.bhdt.bispectral import bispectral_coupling_test, bhct_evidence
from chord.bhdt.cross_gene_phase import cross_gene_phase_test, batch_extract_phases
from chord.bhdt.savage_dickey import savage_dickey_bf, savage_dickey_evidence

__all__ = [
    "bhdt_analytic",
    "classify_gene",
    "batch_classify",
    "CHORDConfig",
    "bispectral_coupling_test",
    "bhct_evidence",
    "cross_gene_phase_test",
    "batch_extract_phases",
    "savage_dickey_bf",
    "savage_dickey_evidence",
    "population_phase_analysis",
]
