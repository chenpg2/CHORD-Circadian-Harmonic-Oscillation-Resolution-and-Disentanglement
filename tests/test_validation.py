"""Tests for CHORD validation on public datasets."""
import pytest


def test_validate_hughes2009_runs():
    """Validation pipeline should run without errors on synthetic data."""
    from chord.validation.validate_hughes2009 import validate_hughes2009
    results = validate_hughes2009(cache_dir="/tmp/chord_validation_test", method="analytic", n_jobs=1)
    assert "detection_rate_12h" in results
    assert "false_positive_rate" in results
    assert "disentangle_accuracy" in results
    assert 0 <= results["detection_rate_12h"] <= 1
    assert 0 <= results["false_positive_rate"] <= 1
    assert 0 <= results["disentangle_accuracy"] <= 1


def test_validate_hughes2009_reasonable_detection():
    """CHORD should detect 12h in at least some known 12h genes."""
    from chord.validation.validate_hughes2009 import validate_hughes2009
    results = validate_hughes2009(cache_dir="/tmp/chord_validation_test", method="analytic", n_jobs=1)
    # With synthetic data (low noise), detection should be reasonable
    assert results["detection_rate_12h"] > 0.3, (
        f"Detection rate too low: {results['detection_rate_12h']:.1%}"
    )


def test_validate_hughes2009_low_false_positive():
    """CHORD should not detect 12h in noise/circadian-only genes too often."""
    from chord.validation.validate_hughes2009 import validate_hughes2009
    results = validate_hughes2009(cache_dir="/tmp/chord_validation_test", method="analytic", n_jobs=1)
    assert results["false_positive_rate"] < 0.5, (
        f"False positive rate too high: {results['false_positive_rate']:.1%}"
    )


def test_validate_zhu2023_runs():
    """Validation pipeline should run on both WT and KO."""
    from chord.validation.validate_zhu2023 import validate_zhu2023
    results = validate_zhu2023(cache_dir="/tmp/chord_validation_zhu_test", method="analytic", n_jobs=1)
    assert "ko_independent_rate" in results
    assert "wt_harmonic_rate" in results


def test_validate_zhu2023_ko_independent():
    """In BMAL1 KO, 12h independent genes should persist and be classified as independent."""
    from chord.validation.validate_zhu2023 import validate_zhu2023
    results = validate_zhu2023(cache_dir="/tmp/chord_validation_zhu_test", method="analytic", n_jobs=1)
    # KO independent genes should still be detected
    assert results["ko_independent_detection"] > 0.3, (
        f"KO independent detection too low: {results['ko_independent_detection']:.1%}"
    )


def test_validate_zhu2023_ko_circadian_abolished():
    """In BMAL1 KO, circadian genes should lose their rhythm."""
    from chord.validation.validate_zhu2023 import validate_zhu2023
    results = validate_zhu2023(cache_dir="/tmp/chord_validation_zhu_test", method="analytic", n_jobs=1)
    assert results["ko_circadian_abolished"] > 0.5, (
        f"Circadian not abolished enough in KO: {results['ko_circadian_abolished']:.1%}"
    )


# ---------------------------------------------------------------------------
# Mure 2018 baboon liver — cross-species validation (real GEO data)
# ---------------------------------------------------------------------------

# Use a shared cache dir that contains the pre-downloaded FPKM file.
# The FPKM file (~108 MB) is expected at:
#   <cache_dir>/GSE98965_baboon_tissue_expression_FPKM.csv.gz
_MURE_CACHE = "/home/data2/fangcong2/ovary_aging/scripts/chord/.chord_cache"


def test_validate_mure2018_runs():
    """Mure 2018 validation pipeline should run on real baboon liver data."""
    from chord.validation.validate_mure2018 import validate_mure2018
    results = validate_mure2018(
        cache_dir=_MURE_CACHE, tissue="LIV",
        method="analytic", n_jobs=1,
    )
    assert results["source"] == "GEO_FPKM"
    assert results["organism"] == "Papio anubis"
    assert results["n_genes_analyzed"] > 1000
    assert "known_12h_detection_rate" in results
    assert "core_circadian_correct_rate" in results
    assert "housekeeping_false_positive_rate" in results
    assert "cross_species_12h_overlap" in results


def test_validate_mure2018_known_genes_found():
    """Known gene lists should have reasonable overlap with baboon data."""
    from chord.validation.validate_mure2018 import validate_mure2018
    results = validate_mure2018(
        cache_dir=_MURE_CACHE, tissue="LIV",
        method="analytic", n_jobs=1,
    )
    # At least some known 12h genes should be present in the dataset
    assert results["known_12h_found"] >= 10, (
        f"Only {results['known_12h_found']} known 12h genes found in baboon data"
    )
    # At least some core circadian genes should be present
    assert results["core_circadian_found"] >= 5, (
        f"Only {results['core_circadian_found']} core circadian genes found"
    )


def test_validate_mure2018_12h_detection():
    """CHORD should detect 12h rhythms in at least some known 12h genes."""
    from chord.validation.validate_mure2018 import validate_mure2018
    results = validate_mure2018(
        cache_dir=_MURE_CACHE, tissue="LIV",
        method="analytic", n_jobs=1,
    )
    # Detection rate: at least some known 12h genes should be detected
    # (lower bar than mouse since baboon data has only 12 timepoints over 24h)
    assert results["known_12h_detection_rate"] > 0.0, (
        "No known 12h genes detected at all in baboon liver"
    )


def test_validate_mure2018_circadian_classification():
    """Core circadian genes should be classified as circadian_only."""
    from chord.validation.validate_mure2018 import validate_mure2018
    results = validate_mure2018(
        cache_dir=_MURE_CACHE, tissue="LIV",
        method="analytic", n_jobs=1,
    )
    # At least some circadian genes should be correctly classified
    assert results["core_circadian_correct_rate"] > 0.0, (
        "No core circadian genes classified as circadian_only"
    )
