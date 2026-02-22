"""Tests for Zhu 2024 loader and human gene lists."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


def test_zhu2024_loader_imports():
    from chord.data.zhu2024_loader import load_zhu2024
    assert callable(load_zhu2024)


def test_known_genes_human_lists():
    from chord.data.known_genes import (
        KNOWN_12H_GENES_ZHU2024,
        CORE_CIRCADIAN_GENES_HUMAN,
        NON_RHYTHMIC_HOUSEKEEPING_HUMAN,
        CONSERVED_12H_GENES_CROSS_SPECIES,
    )
    assert len(KNOWN_12H_GENES_ZHU2024) >= 20
    assert len(CORE_CIRCADIAN_GENES_HUMAN) >= 10
    assert "XBP1" in KNOWN_12H_GENES_ZHU2024
    assert "CLOCK" in CORE_CIRCADIAN_GENES_HUMAN
    # All conserved genes should be in the full list
    for g in CONSERVED_12H_GENES_CROSS_SPECIES:
        assert g in KNOWN_12H_GENES_ZHU2024


def test_no_overlap_12h_circadian():
    from chord.data.known_genes import (
        KNOWN_12H_GENES_ZHU2024,
        CORE_CIRCADIAN_GENES_HUMAN,
    )
    overlap = set(KNOWN_12H_GENES_ZHU2024) & set(CORE_CIRCADIAN_GENES_HUMAN)
    assert len(overlap) == 0, "12h and circadian lists should not overlap: {}".format(overlap)
