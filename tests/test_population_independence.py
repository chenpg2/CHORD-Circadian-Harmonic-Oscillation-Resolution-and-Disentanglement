"""
Tests for Chi-square Population Independence Test.

Tests whether 12h rhythm detection is statistically independent of 24h
rhythm detection across a population of genes, following Zhu et al. (2024).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chord.bhdt.population_independence import chi_square_independence_test


# ============================================================================
# Test 1: Harmonic-dependent population (12h subset of 24h)
# ============================================================================

class TestChiSquareHarmonicPopulation:

    def test_chi_square_harmonic_population(self):
        """When 12h genes are a subset of 24h genes, detection should be
        dependent: p < 0.05 and O/E > 1.5."""
        rng = np.random.default_rng(42)
        n = 500
        # 40% of genes are 24h+
        is_24h = np.zeros(n, dtype=bool)
        is_24h[:200] = True
        # 12h genes are a strict subset of 24h genes (first 150 of 200)
        is_12h = np.zeros(n, dtype=bool)
        is_12h[:150] = True

        result = chi_square_independence_test(is_24h, is_12h)

        assert result["p_value"] < 0.05, (
            "p={:.4f} should be < 0.05 for harmonic (dependent) population".format(
                result["p_value"]
            )
        )
        assert result["observed_expected_ratio"] > 1.5, (
            "O/E={:.2f} should be > 1.5 for harmonic population".format(
                result["observed_expected_ratio"]
            )
        )
        assert result["interpretation"] == "harmonic_dependent"


# ============================================================================
# Test 2: Independent population (12h and 24h uncorrelated)
# ============================================================================

class TestChiSquareIndependentPopulation:

    def test_chi_square_independent_population(self):
        """When 12h and 24h are assigned independently, expect p > 0.05
        or O/E close to 1."""
        rng = np.random.default_rng(123)
        n = 500
        # Independently assign 30% as 24h+ and 20% as 12h+
        is_24h = rng.random(n) < 0.30
        is_12h = rng.random(n) < 0.20

        result = chi_square_independence_test(is_24h, is_12h)

        # For truly independent draws, O/E should be near 1
        assert 0.5 < result["observed_expected_ratio"] < 2.0, (
            "O/E={:.2f} should be near 1.0 for independent population".format(
                result["observed_expected_ratio"]
            )
        )
        # Interpretation should not be harmonic_dependent
        assert result["interpretation"] != "harmonic_dependent"


# ============================================================================
# Test 3: Return structure — contingency table shape and all keys
# ============================================================================

class TestChiSquareReturnsContingencyTable:

    def test_chi_square_returns_contingency_table(self):
        """Result dict should contain all required keys and a (2,2) table."""
        n = 100
        is_24h = np.array([True] * 30 + [False] * 70)
        is_12h = np.array([True] * 20 + [False] * 80)

        result = chi_square_independence_test(is_24h, is_12h)

        required_keys = [
            "p_value",
            "chi2_statistic",
            "observed_expected_ratio",
            "contingency_table",
            "n_both",
            "n_24h_only",
            "n_12h_only",
            "n_neither",
            "interpretation",
        ]
        for key in required_keys:
            assert key in result, "Missing key: {}".format(key)

        table = result["contingency_table"]
        assert table.shape == (2, 2), (
            "Contingency table shape should be (2,2), got {}".format(table.shape)
        )
        # Table cells should sum to n
        assert table.sum() == n, (
            "Table sum should be {}, got {}".format(n, table.sum())
        )
        # Count fields should sum to n
        counts_sum = (
            result["n_both"]
            + result["n_24h_only"]
            + result["n_12h_only"]
            + result["n_neither"]
        )
        assert counts_sum == n, (
            "Count fields should sum to {}, got {}".format(n, counts_sum)
        )


# ============================================================================
# Test 4: Small sample triggers Fisher exact test
# ============================================================================

class TestChiSquareSmallSampleFisher:

    def test_chi_square_small_sample_fisher(self):
        """With very small sample where expected counts < 5, the function
        should still return a valid p-value (using Fisher exact test)."""
        # 10 genes total, very sparse
        is_24h = np.array([True, True, True, False, False,
                           False, False, False, False, False])
        is_12h = np.array([True, True, False, False, False,
                           False, False, False, False, False])

        result = chi_square_independence_test(is_24h, is_12h)

        # Should not crash and should return a valid p-value
        assert 0.0 <= result["p_value"] <= 1.0, (
            "p_value={} should be in [0, 1]".format(result["p_value"])
        )
        assert result["chi2_statistic"] >= 0.0
        assert result["contingency_table"].shape == (2, 2)


# ============================================================================
# Test 5: Edge case — all genes are both 24h+ and 12h+
# ============================================================================

class TestChiSquareAllSame:

    def test_chi_square_all_same(self):
        """When all genes are 24h+ and 12h+, the function should not crash.
        The contingency table will have zero-count cells."""
        n = 50
        is_24h = np.ones(n, dtype=bool)
        is_12h = np.ones(n, dtype=bool)

        result = chi_square_independence_test(is_24h, is_12h)

        # Should not crash
        assert result["contingency_table"].shape == (2, 2)
        assert result["n_both"] == n
        assert result["n_24h_only"] == 0
        assert result["n_12h_only"] == 0
        assert result["n_neither"] == 0
        # p_value should still be valid
        assert 0.0 <= result["p_value"] <= 1.0
