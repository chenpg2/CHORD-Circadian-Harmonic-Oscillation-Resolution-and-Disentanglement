"""Tests for CHORD public dataset loaders and known gene lists."""

import shutil
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Hughes 2009
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hughes_cache():
    d = tempfile.mkdtemp(prefix="chord_test_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_hughes2009_structure(hughes_cache):
    from chord.data.geo_loader import load_hughes2009

    data = load_hughes2009(cache_dir=hughes_cache)
    assert "expr" in data
    assert "timepoints" in data
    assert "gene_names" in data
    assert "metadata" in data
    assert data["expr"].ndim == 2
    assert data["expr"].shape[1] == len(data["timepoints"])
    assert len(data["gene_names"]) == data["expr"].shape[0]


def test_load_hughes2009_timepoints(hughes_cache):
    from chord.data.geo_loader import load_hughes2009

    data = load_hughes2009(cache_dir=hughes_cache)
    tp = data["timepoints"]
    assert tp[0] == 18.0
    assert tp[-1] == 65.0
    assert len(tp) == 48


def test_load_hughes2009_downsampled(hughes_cache):
    from chord.data.geo_loader import load_hughes2009

    full = load_hughes2009(cache_dir=hughes_cache, downsample_2h=False)
    down = load_hughes2009(cache_dir=hughes_cache, downsample_2h=True)
    assert down["expr"].shape[1] == full["expr"].shape[1] // 2
    assert len(down["timepoints"]) == len(full["timepoints"]) // 2
    # Downsampled timepoints should be every other full timepoint
    np.testing.assert_array_equal(down["timepoints"], full["timepoints"][::2])


def test_load_hughes2009_gene_count(hughes_cache):
    from chord.data.geo_loader import load_hughes2009

    data = load_hughes2009(cache_dir=hughes_cache)
    # Synthetic fallback produces 500 genes
    assert data["expr"].shape[0] >= 100


def test_load_hughes2009_metadata(hughes_cache):
    from chord.data.geo_loader import load_hughes2009

    data = load_hughes2009(cache_dir=hughes_cache)
    meta = data["metadata"]
    assert meta["geo_accession"] == "GSE11923"
    assert meta["organism"] == "Mus musculus"
    assert meta["tissue"] == "liver"


def test_load_hughes2009_cache_reuse(hughes_cache):
    """Second call should load from cache (npz file)."""
    from chord.data.geo_loader import load_hughes2009

    d1 = load_hughes2009(cache_dir=hughes_cache)
    d2 = load_hughes2009(cache_dir=hughes_cache)
    np.testing.assert_array_equal(d1["expr"], d2["expr"])


# ---------------------------------------------------------------------------
# Zhu 2023 BMAL1-KO
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zhu_cache():
    d = tempfile.mkdtemp(prefix="chord_test_cache_zhu_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_zhu2023_structure(zhu_cache):
    from chord.data.geo_loader import load_zhu2023_bmal1ko

    data = load_zhu2023_bmal1ko(cache_dir=zhu_cache)
    assert "ko" in data and "wt" in data
    for condition in ["ko", "wt"]:
        d = data[condition]
        assert "expr" in d
        assert "timepoints" in d
        assert "gene_names" in d
        assert "metadata" in d
        assert d["expr"].ndim == 2
        assert d["expr"].shape[1] == len(d["timepoints"])
        assert len(d["gene_names"]) == d["expr"].shape[0]


def test_load_zhu2023_timepoints(zhu_cache):
    from chord.data.geo_loader import load_zhu2023_bmal1ko

    data = load_zhu2023_bmal1ko(cache_dir=zhu_cache)
    tp = data["wt"]["timepoints"]
    assert tp[0] == 0.0
    assert tp[-1] == 44.0
    assert len(tp) == 12
    # 4h resolution
    np.testing.assert_array_equal(np.diff(tp), np.full(11, 4.0))


def test_load_zhu2023_ko_vs_wt_shape(zhu_cache):
    from chord.data.geo_loader import load_zhu2023_bmal1ko

    data = load_zhu2023_bmal1ko(cache_dir=zhu_cache)
    assert data["wt"]["expr"].shape == data["ko"]["expr"].shape
    assert data["wt"]["gene_names"] == data["ko"]["gene_names"]


def test_load_zhu2023_metadata(zhu_cache):
    from chord.data.geo_loader import load_zhu2023_bmal1ko

    data = load_zhu2023_bmal1ko(cache_dir=zhu_cache)
    assert data["wt"]["metadata"]["condition"] == "wildtype"
    assert data["ko"]["metadata"]["condition"] == "BMAL1_KO"
    assert data["ko"]["metadata"]["geo_accession"] == "GSE171975"


# ---------------------------------------------------------------------------
# Known gene lists
# ---------------------------------------------------------------------------

def test_known_gene_lists():
    from chord.data.known_genes import (
        CORE_CIRCADIAN_GENES,
        KNOWN_12H_GENES_ZHU2017,
        NON_RHYTHMIC_HOUSEKEEPING,
    )

    assert len(CORE_CIRCADIAN_GENES) >= 10
    assert len(KNOWN_12H_GENES_ZHU2017) >= 30
    assert len(NON_RHYTHMIC_HOUSEKEEPING) >= 5
    assert "Xbp1" in KNOWN_12H_GENES_ZHU2017
    assert "Clock" in CORE_CIRCADIAN_GENES
    assert "Actb" in NON_RHYTHMIC_HOUSEKEEPING


def test_known_gene_lists_no_overlap():
    from chord.data.known_genes import (
        CORE_CIRCADIAN_GENES,
        KNOWN_12H_GENES_ZHU2017,
        NON_RHYTHMIC_HOUSEKEEPING,
    )

    circ = set(CORE_CIRCADIAN_GENES)
    ultr = set(KNOWN_12H_GENES_ZHU2017)
    house = set(NON_RHYTHMIC_HOUSEKEEPING)
    assert circ.isdisjoint(ultr), f"Overlap: {circ & ultr}"
    assert circ.isdisjoint(house), f"Overlap: {circ & house}"
    assert ultr.isdisjoint(house), f"Overlap: {ultr & house}"
