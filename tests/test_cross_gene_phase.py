"""Tests for Cross-Gene Phase Distribution Test (CGPDT)."""

import numpy as np
import pytest

from chord.bhdt.cross_gene_phase import (
    batch_extract_phases,
    cross_gene_phase_evidence,
    cross_gene_phase_test,
    extract_phases,
    rayleigh_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_time(n_points=48, duration=48.0):
    """Evenly spaced time points over `duration` hours."""
    return np.linspace(0, duration, n_points, endpoint=False)


# ---------------------------------------------------------------------------
# test_extract_phases_pure_cosine
# ---------------------------------------------------------------------------

class TestExtractPhases:
    def test_pure_cosine_24h(self):
        """A pure 24h cosine should give phase ~0 and correct amplitude."""
        t = _make_time(48, 48.0)
        y = 3.0 * np.cos(2 * np.pi * t / 24.0)
        result = extract_phases(t, y, periods=[24.0, 12.0])

        assert result["amplitudes"][24.0] == pytest.approx(3.0, abs=0.1)
        # Phase should be ~0 (or ~2*pi, which is equivalent)
        phi_24 = result["phases"][24.0]
        assert phi_24 == pytest.approx(0.0, abs=0.1) or phi_24 == pytest.approx(
            2 * np.pi, abs=0.1
        )
        # 12h amplitude should be ~0
        assert result["amplitudes"][12.0] == pytest.approx(0.0, abs=0.1)

    def test_pure_cosine_12h(self):
        """A pure 12h cosine should give correct phase for the 12h component."""
        t = _make_time(48, 48.0)
        phi_true = 1.5
        y = 2.0 * np.cos(2 * np.pi * t / 12.0 - phi_true)
        result = extract_phases(t, y, periods=[24.0, 12.0])

        assert result["amplitudes"][12.0] == pytest.approx(2.0, abs=0.1)
        assert result["phases"][12.0] == pytest.approx(phi_true, abs=0.1)

    def test_mixed_signal(self):
        """Signal with both 24h and 12h components."""
        t = _make_time(48, 48.0)
        y = (
            5.0 * np.cos(2 * np.pi * t / 24.0 - 0.5)
            + 2.0 * np.cos(2 * np.pi * t / 12.0 - 1.0)
        )
        result = extract_phases(t, y, periods=[24.0, 12.0])

        assert result["amplitudes"][24.0] == pytest.approx(5.0, abs=0.2)
        assert result["amplitudes"][12.0] == pytest.approx(2.0, abs=0.2)
        assert result["phases"][24.0] == pytest.approx(0.5, abs=0.15)
        assert result["phases"][12.0] == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------------------
# test_rayleigh_concentrated / test_rayleigh_uniform
# ---------------------------------------------------------------------------

class TestRayleighTest:
    def test_concentrated_angles(self):
        """Angles concentrated around a single direction should reject uniformity."""
        rng = np.random.default_rng(42)
        # Concentrated around pi/4 with small spread
        angles = rng.vonmises(mu=np.pi / 4, kappa=10.0, size=50)
        angles = angles % (2 * np.pi)

        result = rayleigh_test(angles)
        assert result["p_value"] < 0.01
        assert result["R"] > 0.5
        assert result["n"] == 50

    def test_uniform_angles(self):
        """Uniformly distributed angles should NOT reject uniformity."""
        rng = np.random.default_rng(123)
        angles = rng.uniform(0, 2 * np.pi, size=50)

        result = rayleigh_test(angles)
        assert result["p_value"] > 0.05
        assert result["R"] < 0.3

    def test_empty_input(self):
        """Empty array should return NaN values."""
        result = rayleigh_test(np.array([]))
        assert np.isnan(result["R"])
        assert result["n"] == 0

    def test_single_angle(self):
        """Single angle: R should be 1.0."""
        result = rayleigh_test(np.array([1.0]))
        assert result["R"] == pytest.approx(1.0)

    def test_small_n(self):
        """Small n (< 10) uses simpler approximation."""
        rng = np.random.default_rng(7)
        angles = rng.vonmises(mu=0.0, kappa=5.0, size=8)
        result = rayleigh_test(angles)
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["n"] == 8


# ---------------------------------------------------------------------------
# test_harmonic_population_detected
# ---------------------------------------------------------------------------

class TestCrossGenePhaseTest:
    def test_harmonic_population_detected(self):
        """50 genes with phi_12 = 2*phi_24 + const should be detected as harmonic."""
        rng = np.random.default_rng(42)
        n_genes = 50
        delta = 0.8  # constant phase offset

        phi_24 = rng.uniform(0, 2 * np.pi, size=n_genes)
        # Harmonic relationship: phi_12 = 2*phi_24 + delta + small noise
        phi_12 = (2.0 * phi_24 + delta + rng.normal(0, 0.15, size=n_genes)) % (
            2 * np.pi
        )

        result = cross_gene_phase_test(phi_24, phi_12)

        assert result["classification"] == "harmonic"
        assert result["evidence_score"] < 0
        assert result["rayleigh"]["p_value"] < 0.05
        assert result["n_genes"] == n_genes

    def test_independent_population_detected(self):
        """50 genes with random phi_12 should be detected as independent."""
        rng = np.random.default_rng(99)
        n_genes = 50

        phi_24 = rng.uniform(0, 2 * np.pi, size=n_genes)
        phi_12 = rng.uniform(0, 2 * np.pi, size=n_genes)  # no coupling

        result = cross_gene_phase_test(phi_24, phi_12)

        assert result["classification"] == "independent"
        assert result["evidence_score"] > 0
        assert result["rayleigh"]["p_value"] > 0.2

    def test_mixed_population(self):
        """Mix of harmonic and independent genes — result depends on proportions."""
        rng = np.random.default_rng(77)
        n_harmonic = 30
        n_independent = 20
        delta = 1.0

        # Harmonic genes
        phi_24_h = rng.uniform(0, 2 * np.pi, size=n_harmonic)
        phi_12_h = (2.0 * phi_24_h + delta + rng.normal(0, 0.1, size=n_harmonic)) % (
            2 * np.pi
        )

        # Independent genes
        phi_24_i = rng.uniform(0, 2 * np.pi, size=n_independent)
        phi_12_i = rng.uniform(0, 2 * np.pi, size=n_independent)

        phi_24 = np.concatenate([phi_24_h, phi_24_i])
        phi_12 = np.concatenate([phi_12_h, phi_12_i])

        result = cross_gene_phase_test(phi_24, phi_12)

        # With 60% harmonic genes, the residuals should still show concentration
        # but the signal is diluted. The test should still detect it or be inconclusive.
        assert result["classification"] in ("harmonic", "inconclusive")
        assert result["n_genes"] == 50

    def test_small_population(self):
        """Test with only 10 genes (minimum practical case)."""
        rng = np.random.default_rng(55)
        n_genes = 10
        delta = 0.5

        phi_24 = rng.uniform(0, 2 * np.pi, size=n_genes)
        phi_12 = (2.0 * phi_24 + delta + rng.normal(0, 0.1, size=n_genes)) % (
            2 * np.pi
        )

        result = cross_gene_phase_test(phi_24, phi_12)

        # With tight coupling and 10 genes, should still detect harmonic
        assert result["classification"] == "harmonic"
        assert result["evidence_score"] < 0
        assert result["n_genes"] == 10

    def test_too_few_genes(self):
        """Fewer than 5 genes should return inconclusive."""
        phi_24 = np.array([0.1, 0.2, 0.3])
        phi_12 = np.array([0.5, 0.6, 0.7])

        result = cross_gene_phase_test(phi_24, phi_12)

        assert result["classification"] == "inconclusive"
        assert result["evidence_score"] == 0
        assert result["n_genes"] == 3

    def test_mismatched_lengths(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            cross_gene_phase_test(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# test_cross_gene_phase_evidence
# ---------------------------------------------------------------------------

class TestCrossGenePhaseEvidence:
    def test_strong_harmonic(self):
        """Very concentrated residuals → strong harmonic evidence."""
        rng = np.random.default_rng(42)
        n = 100
        phi_24 = rng.uniform(0, 2 * np.pi, size=n)
        phi_12 = (2.0 * phi_24 + 1.0 + rng.normal(0, 0.05, size=n)) % (2 * np.pi)

        ev = cross_gene_phase_evidence(phi_24, phi_12)
        assert ev["evidence_score"] <= -2
        assert ev["classification"] == "harmonic"
        assert ev["p_value"] < 0.01

    def test_strong_independent(self):
        """Uniform residuals → strong independent evidence."""
        rng = np.random.default_rng(99)
        n = 100
        phi_24 = rng.uniform(0, 2 * np.pi, size=n)
        phi_12 = rng.uniform(0, 2 * np.pi, size=n)

        ev = cross_gene_phase_evidence(phi_24, phi_12)
        assert ev["evidence_score"] >= 1
        assert ev["classification"] == "independent"
        assert ev["p_value"] > 0.2


# ---------------------------------------------------------------------------
# test_batch_extract_phases
# ---------------------------------------------------------------------------

class TestBatchExtractPhases:
    def test_matches_single_gene(self):
        """Batch extraction should match single-gene extraction."""
        rng = np.random.default_rng(42)
        t = _make_time(48, 48.0)
        n_genes = 5

        # Generate random signals
        Y = np.zeros((n_genes, len(t)))
        for i in range(n_genes):
            amp_24 = rng.uniform(1, 5)
            amp_12 = rng.uniform(0.5, 2)
            phi_24 = rng.uniform(0, 2 * np.pi)
            phi_12 = rng.uniform(0, 2 * np.pi)
            Y[i] = (
                amp_24 * np.cos(2 * np.pi * t / 24.0 - phi_24)
                + amp_12 * np.cos(2 * np.pi * t / 12.0 - phi_12)
                + rng.normal(0, 0.1, size=len(t))
            )

        batch = batch_extract_phases(t, Y)

        # Compare with single-gene extraction
        for i in range(n_genes):
            single = extract_phases(t, Y[i])
            assert batch["phi_24"][i] == pytest.approx(
                single["phases"][24.0], abs=1e-10
            )
            assert batch["phi_12"][i] == pytest.approx(
                single["phases"][12.0], abs=1e-10
            )
            assert batch["amp_24"][i] == pytest.approx(
                single["amplitudes"][24.0], abs=1e-10
            )
            assert batch["amp_12"][i] == pytest.approx(
                single["amplitudes"][12.0], abs=1e-10
            )

    def test_single_gene_matrix(self):
        """1D input should be treated as a single gene."""
        t = _make_time(48, 48.0)
        y = 3.0 * np.cos(2 * np.pi * t / 24.0)

        batch = batch_extract_phases(t, y.reshape(1, -1))
        assert batch["phi_24"].shape == (1,)
        assert batch["amp_24"][0] == pytest.approx(3.0, abs=0.1)

    def test_shape_mismatch(self):
        """Mismatched shapes should raise ValueError."""
        t = _make_time(48, 48.0)
        Y = np.zeros((3, 24))  # wrong number of timepoints

        with pytest.raises(ValueError, match="columns"):
            batch_extract_phases(t, Y)

    def test_end_to_end_with_phase_test(self):
        """Full pipeline: batch extract → phase test."""
        rng = np.random.default_rng(42)
        t = _make_time(48, 48.0)
        n_genes = 30
        delta = 0.7

        Y = np.zeros((n_genes, len(t)))
        for i in range(n_genes):
            phi_24 = rng.uniform(0, 2 * np.pi)
            phi_12 = 2.0 * phi_24 + delta + rng.normal(0, 0.1)
            Y[i] = (
                3.0 * np.cos(2 * np.pi * t / 24.0 - phi_24)
                + 1.5 * np.cos(2 * np.pi * t / 12.0 - phi_12)
                + rng.normal(0, 0.2, size=len(t))
            )

        batch = batch_extract_phases(t, Y)
        result = cross_gene_phase_test(batch["phi_24"], batch["phi_12"])

        assert result["classification"] == "harmonic"
        assert result["evidence_score"] < 0
