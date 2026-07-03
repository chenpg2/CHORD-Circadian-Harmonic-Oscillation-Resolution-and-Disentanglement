"""Known circadian and ultradian gene lists for validation."""

# ---------------------------------------------------------------------------
# Mouse gene lists (for Hughes 2009, Zhu 2023)
# ---------------------------------------------------------------------------

# Core circadian clock genes (should be classified as circadian_only)
CORE_CIRCADIAN_GENES = [
    "Clock", "Arntl",  # Arntl = Bmal1
    "Per1", "Per2", "Per3",
    "Cry1", "Cry2",
    "Nr1d1", "Nr1d2",  # Rev-erb alpha/beta
    "Rora", "Rorb", "Rorc",
    "Dbp", "Tef", "Hlf",
    "Npas2",
]

# Curated 38-gene 12h reference set (used to score detector RECOVERY, not as an
# exhaustive ground truth). Two functional programs:
#   * Program 1 (30 genes): UPR / ER-stress / proteostasis, anchored to IRE1a-XBP1s.
#   * Program 2 (8 genes):  lipid / sterol metabolism (coupled to the hepatic 12h clock).
# In-code attribution: Zhu et al. 2017 (Cell Metab 25:1305-1319), "Table S2 / most robust
# 12h genes by eigenvalue-pencil". VERIFY BEFORE FINALIZING (see manuscript supplement
# supp_38gene_provenance.md and references/reference_audit.md): confirm the exact source
# table for Program 1, and that Program 2's lipid genes are from Zhu 2017 rather than a
# different source (e.g. Meng 2020 XBP1->NAFLD, Dion 2022). Do not cite meng2020/dion2022/
# pan2020 here until those (currently mis-cited/fabricated) references are fixed.
KNOWN_12H_GENES_ZHU2017 = [
    "Xbp1", "Atf4", "Atf6", "Atf6b",
    "Ddit3",  # CHOP
    "Hspa5",  # BiP/GRP78
    "Hsp90b1",  # GRP94
    "Pdia4", "Pdia6",
    "Dnajb9", "Dnajb11",
    "Edem1",
    "Sec61a1", "Sec61b",
    "Srp54a",
    "Stt3a", "Stt3b",
    "Uggt1",
    "Manf",
    "Hyou1",
    "Calr",  # Calreticulin
    "Canx",  # Calnexin
    "P4hb",
    "Ero1l",  # Ero1-like
    "Derl1", "Derl2",
    "Herpud1",
    "Sel1l",
    "Os9",
    "Erlec1",
    # Lipid metabolism 12h genes
    "Fasn", "Acaca", "Scd1",
    "Hmgcr", "Hmgcs1",
    "Sqle", "Fdft1",
    "Srebf1",
]

# Genes that should NOT have 12h rhythms (negative controls)
NON_RHYTHMIC_HOUSEKEEPING = [
    "Actb", "Gapdh", "Hprt", "Tbp", "Rpl13a",
    "B2m", "Sdha", "Hmbs", "Ywhaz", "Ubc",
]


# ---------------------------------------------------------------------------
# Primate / human-style gene lists (for Mure 2018 baboon data)
# Baboon (Papio anubis) uses human-style uppercase gene symbols.
# ---------------------------------------------------------------------------

# Core circadian clock genes — human/primate orthologs
CORE_CIRCADIAN_GENES_PRIMATE = [
    "CLOCK", "ARNTL",  # BMAL1
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",  # REV-ERBα/β
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF",
    "NPAS2",
]

# Known 12h genes — human/primate orthologs of Zhu 2017 mouse 12h genes
# ER stress / UPR pathway (conserved across mammals)
KNOWN_12H_GENES_PRIMATE = [
    "XBP1", "ATF4", "ATF6", "ATF6B",
    "DDIT3",   # CHOP
    "HSPA5",   # BiP/GRP78
    "HSP90B1", # GRP94
    "PDIA4", "PDIA6",
    "DNAJB9", "DNAJB11",
    "EDEM1",
    "SEC61A1", "SEC61B",
    "SRP54",   # human ortholog of mouse Srp54a
    "STT3A", "STT3B",
    "UGGT1",
    "MANF",
    "HYOU1",
    "CALR",    # Calreticulin
    "CANX",    # Calnexin
    "P4HB",
    "ERO1A",   # human ortholog of mouse Ero1l
    "DERL1", "DERL2",
    "HERPUD1",
    "SEL1L",
    "OS9",
    "ERLEC1",
    # Lipid metabolism 12h genes
    "FASN", "ACACA", "SCD",  # human SCD (mouse Scd1)
    "HMGCR", "HMGCS1",
    "SQLE", "FDFT1",
    "SREBF1",
]

# Housekeeping genes — primate (negative controls, should NOT be rhythmic)
NON_RHYTHMIC_HOUSEKEEPING_PRIMATE = [
    "ACTB", "GAPDH", "HPRT1", "TBP", "RPL13A",
    "B2M", "SDHA", "HMBS", "YWHAZ", "UBC",
]


# ---------------------------------------------------------------------------
# Cross-species mapping utilities
# ---------------------------------------------------------------------------

def mouse_to_primate_symbol(mouse_symbol):
    """Convert mouse gene symbol to primate/human convention.

    Mouse: mixed case (e.g. Arntl, Xbp1, Hspa5)
    Human/primate: uppercase (e.g. ARNTL, XBP1, HSPA5)

    Some genes have different names across species; this handles
    known exceptions.
    """
    _MOUSE_TO_HUMAN = {
        "Arntl": "ARNTL",
        "Ero1l": "ERO1A",
        "Srp54a": "SRP54",
        "Scd1": "SCD",
        "Hprt": "HPRT1",
    }
    if mouse_symbol in _MOUSE_TO_HUMAN:
        return _MOUSE_TO_HUMAN[mouse_symbol]
    return mouse_symbol.upper()


# ---------------------------------------------------------------------------
# Human gene lists (for Zhu 2024 peripheral blood)
# ---------------------------------------------------------------------------

# Known 12h genes from Zhu 2024 (UPR/protein metabolism, human symbols)
KNOWN_12H_GENES_ZHU2024 = [
    "XBP1", "HSPA5", "DNAJB9", "HERPUD1", "ATF6B", "UGGT1",
    "SEC61A1", "PDIA4", "CALR", "CANX", "EDEM1", "ERO1A",
    "DERL1", "SEL1L", "SYVN1", "OS9", "ERLEC1", "DNAJC3",
    "HYOU1", "SDF2L1", "ATF4", "ATF6", "DDIT3", "HSP90B1",
    "PDIA6", "DNAJB11", "SEC61B", "STT3A", "STT3B", "MANF",
    "P4HB",
]

# Core circadian genes (human symbols, for Zhu 2024)
CORE_CIRCADIAN_GENES_HUMAN = [
    "CLOCK", "ARNTL", "PER1", "PER2", "PER3",
    "CRY1", "CRY2", "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC", "DBP", "TEF", "HLF", "NPAS2",
]

# Housekeeping genes (human symbols)
NON_RHYTHMIC_HOUSEKEEPING_HUMAN = [
    "ACTB", "GAPDH", "HPRT1", "TBP", "RPL13A",
    "B2M", "SDHA", "HMBS", "YWHAZ", "UBC",
]

# Cross-species conserved 12h genes (human-mouse-sea anemone)
# From Zhu 2024 supplementary
CONSERVED_12H_GENES_CROSS_SPECIES = [
    "XBP1", "HSPA5", "DNAJB9", "CALR", "CANX",
    "PDIA4", "SEC61A1", "EDEM1", "HYOU1",
]
