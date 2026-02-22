"""GEO dataset loaders for CHORD validation.

All loaders use real public GEO data.  Loading order:
  Tier 1 — NPZ cache (fast, local)
  Tier 2 — GEOparse / direct download (requires network)
  Fail   — RuntimeError (no synthetic fallback)
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_hughes2009(cache_dir="~/.chord_cache", downsample_2h=False):
    """Load Hughes 2009 mouse liver microarray (GSE11923).

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    downsample_2h : bool
        If True, keep every other timepoint (24 points at 2h resolution).

    Returns
    -------
    dict with keys:
        expr        : ndarray (genes x timepoints)
        timepoints  : ndarray of CT hours
        gene_names  : list[str]
        metadata    : dict with provenance info

    Raises
    ------
    RuntimeError
        If data cannot be loaded from cache or downloaded via GEOparse.
    """
    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / "hughes2009.npz"

    expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: try loading from cache ---
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        expr = data["expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: try GEOparse ---
    if expr is None:
        try:
            import GEOparse
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "Hughes 2009 (GSE11923) data not cached and GEOparse is not "
                "installed.  Install GEOparse (`pip install GEOparse`) and "
                "re-run, or manually place the NPZ cache at: " + str(npz_file)
            )

        try:
            gse = GEOparse.get_GEO("GSE11923", destdir=str(cache_path))
            pivot = gse.pivot_samples("VALUE")
            sample_meta = gse.phenotype_data

            # Extract CT hours from sample titles
            ct_hours = []
            for sid in pivot.columns:
                title = sample_meta.loc[sid, "title"]
                m = re.search(r'CT(\d+)', title)
                if m:
                    ct = float(m.group(1))
                else:
                    m = re.search(r'[Cc]ircadian\s+time\s+(\d+)', title)
                    if m:
                        ct = float(m.group(1))
                    else:
                        m = re.search(r'(\d+)\s*$', title)
                        ct = float(m.group(1)) if m else None
                if ct is None:
                    raise ValueError(
                        f"Cannot parse CT hour from title: {title!r}"
                    )
                ct_hours.append(ct)

            sort_idx = np.argsort(ct_hours)
            ct_hours = np.array(ct_hours)[sort_idx]
            pivot = pivot.iloc[:, sort_idx]
            raw_expr = pivot.values.astype(np.float64)

            # Map probe IDs to gene symbols
            gpl = list(gse.gpls.values())[0]
            if "Gene Symbol" in gpl.table.columns:
                probe2gene = dict(
                    zip(gpl.table["ID"], gpl.table["Gene Symbol"])
                )
                probe_ids = pivot.index.tolist()
                gene_symbols = [
                    str(probe2gene.get(pid, "")).split("///")[0].strip()
                    for pid in probe_ids
                ]
                has_symbol = [
                    bool(g) and g != "nan" and g != ""
                    for g in gene_symbols
                ]
                raw_expr = raw_expr[has_symbol]
                gene_symbols = [
                    g for g, h in zip(gene_symbols, has_symbol) if h
                ]
                # Deduplicate: keep probe with highest mean expression
                df_dedup = pd.DataFrame({
                    "gene": gene_symbols,
                    "mean_expr": raw_expr.mean(axis=1),
                    "idx": range(len(gene_symbols)),
                })
                best_idx = df_dedup.groupby("gene")["mean_expr"].idxmax().values
                raw_expr = raw_expr[best_idx]
                gene_names_arr = [gene_symbols[i] for i in best_idx]
            else:
                gene_names_arr = pivot.index.tolist()

            # Filter bottom 10% by mean expression
            means = raw_expr.mean(axis=1)
            threshold = np.percentile(means, 10)
            keep = means >= threshold
            raw_expr = raw_expr[keep]
            gene_names_arr = [
                g for g, k in zip(gene_names_arr, keep) if k
            ]

            expr = raw_expr
            timepoints = ct_hours
            gene_names = gene_names_arr
            source = "GEOparse"

            # Cache
            np.savez(
                npz_file,
                expr=expr,
                timepoints=timepoints,
                gene_names=np.array(gene_names, dtype=object),
                source=np.array("GEOparse"),
            )
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to download/parse Hughes 2009 (GSE11923): {e}"
            ) from e

    # --- Downsample ---
    if downsample_2h:
        idx = np.arange(0, len(timepoints), 2)
        expr = expr[:, idx]
        timepoints = timepoints[idx]

    metadata = {
        "dataset": "Hughes 2009",
        "geo_accession": "GSE11923",
        "organism": "Mus musculus",
        "tissue": "liver",
        "platform": "Affymetrix Mouse430_2",
        "source": source,
        "n_genes": expr.shape[0],
        "n_timepoints": expr.shape[1],
    }

    return {
        "expr": expr,
        "timepoints": timepoints,
        "gene_names": gene_names,
        "metadata": metadata,
    }


def load_zhu2023_bmal1ko(cache_dir="~/.chord_cache"):
    """Load Zhu 2023 BMAL1 KO temporal RNA-seq (GSE171975).

    Data is loaded from supplementary xlsx files (WT normalized reads,
    KO raw reads), with replicates averaged per timepoint.

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.

    Returns
    -------
    dict with keys 'ko' and 'wt', each containing:
        expr        : ndarray (genes x timepoints)
        timepoints  : ndarray of CT hours
        gene_names  : list[str]
        metadata    : dict with provenance info

    Raises
    ------
    RuntimeError
        If data cannot be loaded from cache or downloaded.
    """
    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / "zhu2023_bmal1ko.npz"

    wt_expr = ko_expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: cache ---
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        wt_expr = data["wt_expr"]
        ko_expr = data["ko_expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: download supplementary xlsx from GEO ---
    if wt_expr is None:
        try:
            import pandas as pd
            import urllib.request
        except ImportError:
            raise RuntimeError(
                "Zhu 2023 (GSE171975) data not cached and pandas/urllib "
                "not available."
            )

        base_url = (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/"
            "GSE171975/suppl/"
        )
        wt_xlsx = cache_path / "GSE171975_R370_Liver_WT_reads.xlsx"
        ko_xlsx = cache_path / "GSE171975_Liver_BMALKO_reads.xlsx"

        try:
            for local, fname in [
                (wt_xlsx, "GSE171975_R370_Liver_WT_reads.xlsx"),
                (ko_xlsx, "GSE171975_Liver_BMALKO_reads.xlsx"),
            ]:
                if not local.exists():
                    url = base_url + fname
                    warnings.warn(f"Downloading {fname} from GEO ...")
                    urllib.request.urlretrieve(url, str(local))

            # --- Parse WT (use normalized columns) ---
            wt_df = pd.read_excel(str(wt_xlsx))
            norm_cols = [c for c in wt_df.columns if ".normalized" in c]
            if not norm_cols:
                raise ValueError(
                    "No '.normalized' columns found in WT xlsx"
                )

            wt_meta = []
            for c in norm_cols:
                m = re.search(r'CT(\d+)_([A-Z])\.normalized', c)
                if m:
                    wt_meta.append({
                        "col": c,
                        "ct": int(m.group(1)),
                        "rep": m.group(2),
                    })
            wt_cts = sorted(set(d["ct"] for d in wt_meta))
            gene_names_wt = wt_df.iloc[:, 0].tolist()

            wt_avg = np.column_stack([
                wt_df[[d["col"] for d in wt_meta if d["ct"] == ct]]
                .mean(axis=1).values
                for ct in wt_cts
            ])

            # --- Parse KO (raw columns only) ---
            ko_df = pd.read_excel(str(ko_xlsx))
            ko_raw_cols = [c for c in ko_df.columns if ".raw" in c]
            ko_meta = []
            for c in ko_raw_cols:
                m = re.search(r'CT(\d+)_([A-Z])\.raw', c)
                if m:
                    ko_meta.append({
                        "col": c,
                        "ct": int(m.group(1)),
                        "rep": m.group(2),
                    })
            ko_cts = sorted(set(d["ct"] for d in ko_meta))
            gene_names_ko = ko_df.iloc[:, 0].tolist()

            ko_avg = np.column_stack([
                ko_df[[d["col"] for d in ko_meta if d["ct"] == ct]]
                .mean(axis=1).values
                for ct in ko_cts
            ])

            # Intersect gene sets
            common_genes = sorted(
                set(gene_names_wt) & set(gene_names_ko)
            )
            wt_idx = {g: i for i, g in enumerate(gene_names_wt)}
            ko_idx = {g: i for i, g in enumerate(gene_names_ko)}

            wt_expr_all = np.array(
                [wt_avg[wt_idx[g]] for g in common_genes]
            )
            ko_expr_all = np.array(
                [ko_avg[ko_idx[g]] for g in common_genes]
            )

            # Filter bottom 10%
            means = (
                wt_expr_all.mean(axis=1) + ko_expr_all.mean(axis=1)
            ) / 2
            threshold = np.percentile(means, 10)
            keep = means >= threshold
            wt_expr = wt_expr_all[keep].astype(np.float64)
            ko_expr = ko_expr_all[keep].astype(np.float64)
            gene_names = [
                g for g, k in zip(common_genes, keep) if k
            ]
            timepoints = np.array(wt_cts, dtype=np.float64)
            source = "GEO_xlsx"

            np.savez(
                npz_file,
                wt_expr=wt_expr,
                ko_expr=ko_expr,
                timepoints=timepoints,
                gene_names=np.array(gene_names, dtype=object),
                source=np.array(source),
            )
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to download/parse Zhu 2023 (GSE171975): {e}"
            ) from e

    base_meta = {
        "dataset": "Zhu 2023 BMAL1-KO",
        "geo_accession": "GSE171975",
        "organism": "Mus musculus",
        "tissue": "liver",
        "platform": "RNA-seq",
        "source": source,
    }

    return {
        "wt": {
            "expr": wt_expr,
            "timepoints": timepoints,
            "gene_names": gene_names,
            "metadata": {**base_meta, "condition": "wildtype",
                         "n_genes": wt_expr.shape[0],
                         "n_timepoints": wt_expr.shape[1]},
        },
        "ko": {
            "expr": ko_expr,
            "timepoints": timepoints,
            "gene_names": gene_names,
            "metadata": {**base_meta, "condition": "BMAL1_KO",
                         "n_genes": ko_expr.shape[0],
                         "n_timepoints": ko_expr.shape[1]},
        },
    }


def load_mure2018(cache_dir="~/.chord_cache", tissue="LIV", min_fpkm=1.0):
    """Load Mure 2018 baboon tissue RNA-seq (GSE98965).

    Mure et al. (Science 2018) profiled 64 baboon tissues every 2h over 24h.
    Data is downloaded from the series supplementary FPKM matrix.

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    tissue : str
        Tissue abbreviation (e.g. 'LIV' for liver, 'KID' for kidney).
        Default 'LIV' for cross-species comparison with mouse liver.
    min_fpkm : float
        Minimum mean FPKM to keep a gene. Default 1.0.

    Returns
    -------
    dict with keys:
        expr        : ndarray (genes x timepoints)
        timepoints  : ndarray of ZT hours
        gene_names  : list[str]
        metadata    : dict with provenance info

    Raises
    ------
    RuntimeError
        If the FPKM supplementary file cannot be downloaded.
    """
    import pandas as pd

    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / f"mure2018_{tissue}.npz"

    expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: try loading from NPZ cache ---
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        expr = data["expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: parse from downloaded FPKM CSV ---
    if expr is None:
        fpkm_gz = cache_path / "GSE98965_baboon_tissue_expression_FPKM.csv.gz"

        # Download if not cached
        if not fpkm_gz.exists():
            url = (
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE98nnn/"
                "GSE98965/suppl/GSE98965_baboon_tissue_expression_FPKM.csv.gz"
            )
            try:
                import urllib.request
                warnings.warn(
                    f"Downloading Mure 2018 FPKM matrix (~108 MB) from "
                    f"{url} ..."
                )
                urllib.request.urlretrieve(url, str(fpkm_gz))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Mure 2018 FPKM data: {e}\n"
                    f"You can manually download from:\n  {url}\n"
                    f"and place it at:\n  {fpkm_gz}"
                ) from e

        # Verify file integrity (should be > 100 MB)
        file_size = fpkm_gz.stat().st_size
        if file_size < 100_000_000:
            raise RuntimeError(
                f"FPKM file appears incomplete ({file_size / 1e6:.1f} MB, "
                f"expected ~108 MB). Delete {fpkm_gz} and retry."
            )

        # Build column list for the requested tissue
        tissue_upper = tissue.upper()
        zt_hours = list(range(0, 24, 2))
        tissue_cols = [f"{tissue_upper}.ZT{h:02d}" for h in zt_hours]
        usecols = ["Symbol"] + tissue_cols

        try:
            df = pd.read_csv(fpkm_gz, compression="gzip", usecols=usecols)
        except ValueError as e:
            header = pd.read_csv(fpkm_gz, compression="gzip", nrows=0)
            all_cols = header.columns.tolist()
            tissues = sorted({
                c.rsplit(".", 1)[0] for c in all_cols
                if "." in c and c.rsplit(".", 1)[1].startswith("ZT")
            })
            raise ValueError(
                f"Tissue '{tissue}' not found in Mure 2018 data. "
                f"Available tissues: {tissues}"
            ) from e

        # Drop genes without a symbol or with Ensembl-only IDs
        df = df[df["Symbol"].notna()].copy()
        df = df[~df["Symbol"].str.startswith("ENSPANG")].copy()

        # Deduplicate: keep highest mean expression per gene
        expr_vals = df[tissue_cols].astype(np.float64)
        df["_mean_expr"] = expr_vals.mean(axis=1)
        df = df.sort_values("_mean_expr", ascending=False)
        df = df.drop_duplicates(subset="Symbol", keep="first")

        # Filter by minimum expression
        df = df[df["_mean_expr"] >= min_fpkm].copy()
        df = df.sort_values("Symbol").reset_index(drop=True)

        gene_names = df["Symbol"].tolist()
        expr = df[tissue_cols].values.astype(np.float64)
        timepoints = np.array(zt_hours, dtype=np.float64)
        source = "GEO_FPKM"

        # Cache to NPZ
        np.savez(
            npz_file,
            expr=expr,
            timepoints=timepoints,
            gene_names=np.array(gene_names, dtype=object),
            source=np.array(source),
        )

    metadata = {
        "dataset": "Mure 2018",
        "geo_accession": "GSE98965",
        "organism": "Papio anubis",
        "tissue": tissue,
        "platform": "RNA-seq (Illumina HiSeq 2500)",
        "source": source,
        "n_genes": expr.shape[0],
        "n_timepoints": expr.shape[1],
        "reference": "Mure et al., Science 359:eaao0318 (2018)",
    }

    return {
        "expr": expr,
        "timepoints": timepoints,
        "gene_names": gene_names,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Zhang 2014 — 12-tissue circadian atlas (GSE54650 / GSE54652)
# ---------------------------------------------------------------------------

# Tissue code -> full name mapping for Zhang 2014
ZHANG2014_TISSUES = {
    "Adr": "Adrenal gland",
    "Aor": "Aorta",
    "BFat": "Brown adipose (BAT)",
    "Bstm": "Brainstem",
    "Cer": "Cerebellum",
    "Hrt": "Heart",
    "Hyp": "Hypothalamus",
    "Kid": "Kidney",
    "Liv": "Liver",
    "Lun": "Lung",
    "Mus": "Skeletal muscle",
    "WFat": "White adipose (WAT)",
}


def load_zhang2014(cache_dir="~/.chord_cache", tissue="Liv", min_expr=50.0):
    """Load Zhang 2014 mouse multi-tissue circadian atlas (GSE54650).

    Zhang et al. (PNAS 2014) profiled 12 mouse tissues every 2h for 48h
    using Affymetrix MoGene-1_0-st arrays (no replicates).

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    tissue : str
        Tissue abbreviation. One of: Adr, Aor, BFat, Bstm, Cer, Hrt,
        Hyp, Kid, Liv, Lun, Mus, WFat.
    min_expr : float
        Minimum mean expression to keep a gene. Default 50.0
        (microarray intensity units).

    Returns
    -------
    dict with keys:
        expr        : ndarray (genes x timepoints)
        timepoints  : ndarray of CT hours
        gene_names  : list[str]
        metadata    : dict with provenance info

    Raises
    ------
    RuntimeError
        If data cannot be downloaded or parsed.
    ValueError
        If tissue code is not recognized.
    """
    import pandas as pd

    if tissue not in ZHANG2014_TISSUES:
        raise ValueError(
            "Unknown tissue '{}'. Available: {}".format(
                tissue, sorted(ZHANG2014_TISSUES.keys()))
        )

    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / "zhang2014_{}.npz".format(tissue)

    expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: NPZ cache ---
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        expr = data["expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: Parse from series matrix + GPL annotation ---
    if expr is None:
        matrix_gz = cache_path / "GSE54650_series_matrix.txt.gz"
        annot_gz = cache_path / "GPL6246.annot.gz"

        # Download series matrix if needed
        if not matrix_gz.exists():
            url = (
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/"
                "GSE54650/matrix/GSE54650_series_matrix.txt.gz"
            )
            try:
                import urllib.request
                warnings.warn(
                    "Downloading Zhang 2014 series matrix (~20 MB) "
                    "from {} ...".format(url)
                )
                urllib.request.urlretrieve(url, str(matrix_gz))
            except Exception as e:
                raise RuntimeError(
                    "Failed to download Zhang 2014 series matrix: "
                    "{}".format(e)
                ) from e

        # Download GPL6246 annotation if needed
        if not annot_gz.exists():
            url = (
                "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/"
                "GPL6246/annot/GPL6246.annot.gz"
            )
            try:
                import urllib.request
                warnings.warn(
                    "Downloading GPL6246 annotation from {} ...".format(url)
                )
                urllib.request.urlretrieve(url, str(annot_gz))
            except Exception as e:
                raise RuntimeError(
                    "Failed to download GPL6246 annotation: {}".format(e)
                ) from e

        # --- Parse sample titles to get tissue/timepoint mapping ---
        import gzip as _gzip
        sample_titles = []
        sample_ids = []
        data_start = None
        with _gzip.open(str(matrix_gz), "rt") as f:
            for i, line in enumerate(f):
                if line.startswith("!Sample_title"):
                    sample_titles = [
                        t.strip('"') for t in line.strip().split("\t")[1:]
                    ]
                elif line.startswith("!Sample_geo_accession"):
                    sample_ids = [
                        t.strip('"') for t in line.strip().split("\t")[1:]
                    ]
                elif line.startswith("!series_matrix_table_begin"):
                    data_start = i + 1
                    break

        if data_start is None:
            raise RuntimeError("Could not find data table in series matrix")

        # Identify columns for the requested tissue
        tissue_col_idx = []
        tissue_ct_hours = []
        for j, title in enumerate(sample_titles):
            m = re.match(r"(\w+)_CT(\d+)", title)
            if m and m.group(1) == tissue:
                tissue_col_idx.append(j)
                tissue_ct_hours.append(float(m.group(2)))

        if not tissue_col_idx:
            raise ValueError(
                "No samples found for tissue '{}' in series matrix".format(
                    tissue)
            )

        # Sort by CT hour
        sort_order = np.argsort(tissue_ct_hours)
        tissue_col_idx = [tissue_col_idx[i] for i in sort_order]
        tissue_ct_hours = np.array(
            [tissue_ct_hours[i] for i in sort_order], dtype=np.float64
        )

        # Read expression matrix (all samples, then subset)
        expr_df = pd.read_csv(
            str(matrix_gz), compression="gzip", sep="\t",
            skiprows=data_start, index_col=0, comment="!",
        )
        # Remove trailing marker row
        expr_df = expr_df[
            ~expr_df.index.astype(str).str.startswith("!")
        ]

        # Subset to tissue columns
        tissue_sample_ids = [sample_ids[i] for i in tissue_col_idx]
        tissue_expr = expr_df[tissue_sample_ids].values.astype(np.float64)
        probe_ids = expr_df.index.values

        # --- Map probes to gene symbols via GPL6246 ---
        # Skip 27 header lines (comments + column descriptions)
        annot_df = pd.read_csv(
            str(annot_gz), compression="gzip", sep="\t",
            skiprows=27, low_memory=False,
        )
        probe2gene = {}
        for _, row in annot_df.iterrows():
            pid = row["ID"]
            sym = row.get("Gene symbol", "")
            if pd.notna(sym) and sym != "" and sym != "---":
                # Convert probe ID to int for matching with expr matrix
                try:
                    pid_int = int(pid)
                except (ValueError, TypeError):
                    continue
                # Take first symbol if multiple (separated by ///)
                probe2gene[pid_int] = str(sym).split("///")[0].strip()

        # Map and filter
        gene_symbols = [probe2gene.get(pid, "") for pid in probe_ids]
        has_symbol = [
            bool(g) and g != "nan" and g != "" for g in gene_symbols
        ]
        tissue_expr = tissue_expr[has_symbol]
        gene_symbols = [g for g, h in zip(gene_symbols, has_symbol) if h]

        # Deduplicate: keep probe with highest mean expression per gene
        means = tissue_expr.mean(axis=1)
        df_dedup = pd.DataFrame({
            "gene": gene_symbols,
            "mean_expr": means,
        })
        best_idx = (
            df_dedup.groupby("gene")["mean_expr"]
            .idxmax().values.astype(int)
        )
        tissue_expr = tissue_expr[best_idx]
        gene_names_arr = [gene_symbols[i] for i in best_idx]

        # Filter by minimum expression
        means = tissue_expr.mean(axis=1)
        keep = means >= min_expr
        tissue_expr = tissue_expr[keep]
        gene_names_arr = [
            g for g, k in zip(gene_names_arr, keep) if k
        ]

        expr = tissue_expr
        timepoints = tissue_ct_hours
        gene_names = gene_names_arr
        source = "GEO_series_matrix"

        # Cache to NPZ
        np.savez(
            str(npz_file),
            expr=expr,
            timepoints=timepoints,
            gene_names=np.array(gene_names, dtype=object),
            source=np.array(source),
        )

    metadata = {
        "dataset": "Zhang 2014",
        "geo_accession": "GSE54650",
        "organism": "Mus musculus",
        "tissue": tissue,
        "tissue_name": ZHANG2014_TISSUES[tissue],
        "platform": "Affymetrix MoGene-1_0-st",
        "source": source,
        "n_genes": expr.shape[0],
        "n_timepoints": expr.shape[1],
        "reference": "Zhang et al., PNAS 111:16219-16224 (2014)",
    }

    return {
        "expr": expr,
        "timepoints": timepoints,
        "gene_names": gene_names,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Scp2 KO 2015 — GSE67426 (mouse liver, Scp2 KO vs WT)
# ---------------------------------------------------------------------------

def load_scp2ko_2015(cache_dir="~/.chord_cache", genotype="wt", min_expr=1.0):
    """Load Scp2 KO 2015 mouse liver RNA-seq (GSE67426).

    Jouffe et al. profiled Scp2 KO and WT mouse livers every 2h over
    3 consecutive days (72 samples: 2 genotypes x 12 timepoints x 3 reps).

    Parameters
    ----------
    cache_dir : str
        Directory for caching downloaded data.
    genotype : str
        Which genotype to return: 'wt' or 'ko'. Default 'wt'.
    min_expr : float
        Minimum mean expression to keep a gene. Default 1.0.

    Returns
    -------
    dict with keys:
        expr        : ndarray (genes x timepoints)
        timepoints  : ndarray of ZT hours
        gene_names  : list[str]
        metadata    : dict with provenance info

    Raises
    ------
    RuntimeError
        If data cannot be downloaded or parsed.
    ValueError
        If genotype is not 'wt' or 'ko'.
    """
    import pandas as pd

    genotype = genotype.lower()
    if genotype not in ("wt", "ko"):
        raise ValueError(
            "genotype must be 'wt' or 'ko', got '{}'".format(genotype)
        )

    cache_path = Path(os.path.expanduser(cache_dir))
    cache_path.mkdir(parents=True, exist_ok=True)
    npz_file = cache_path / "scp2ko_2015_{}.npz".format(genotype)

    expr = timepoints = gene_names = None
    source = None

    # --- Tier 1: NPZ cache ---
    if npz_file.exists():
        data = np.load(npz_file, allow_pickle=True)
        expr = data["expr"]
        timepoints = data["timepoints"]
        gene_names = list(data["gene_names"])
        source = str(data["source"]) if "source" in data else "cache"

    # --- Tier 2: Parse from series matrix ---
    if expr is None:
        matrix_gz = cache_path / "GSE67426_series_matrix.txt.gz"

        # Download series matrix if needed
        if not matrix_gz.exists():
            url = (
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/"
                "GSE67426/matrix/GSE67426_series_matrix.txt.gz"
            )
            try:
                import urllib.request
                warnings.warn(
                    "Downloading GSE67426 series matrix from {} ...".format(
                        url)
                )
                urllib.request.urlretrieve(url, str(matrix_gz))
            except Exception as e:
                raise RuntimeError(
                    "Failed to download GSE67426 series matrix: {}".format(e)
                ) from e

        # --- Parse sample metadata from header ---
        import gzip as _gzip
        sample_titles = []
        sample_ids = []
        platform_id = None
        data_start = None
        with _gzip.open(str(matrix_gz), "rt") as f:
            for i, line in enumerate(f):
                if line.startswith("!Sample_title"):
                    sample_titles = [
                        t.strip('"') for t in line.strip().split("\t")[1:]
                    ]
                elif line.startswith("!Sample_geo_accession"):
                    sample_ids = [
                        t.strip('"') for t in line.strip().split("\t")[1:]
                    ]
                elif line.startswith("!Sample_platform_id"):
                    vals = [
                        t.strip('"') for t in line.strip().split("\t")[1:]
                    ]
                    if vals:
                        platform_id = vals[0]
                elif line.startswith("!series_matrix_table_begin"):
                    data_start = i + 1
                    break

        if data_start is None:
            raise RuntimeError(
                "Could not find data table in GSE67426 series matrix"
            )

        # --- Identify columns for the requested genotype ---
        # Sample titles like "SCP2_KO_t0_rep1", "WT_t0_rep1"
        geno_col_idx = []
        geno_timepoints = []
        geno_reps = []
        for j, title in enumerate(sample_titles):
            # Try pattern: SCP2_KO_t<N>_rep<M> or WT_t<N>_rep<M>
            m_ko = re.match(
                r"SCP2_KO_t(\d+)_rep(\d+)", title, re.IGNORECASE
            )
            m_wt = re.match(
                r"WT_t(\d+)_rep(\d+)", title, re.IGNORECASE
            )
            if genotype == "ko" and m_ko:
                geno_col_idx.append(j)
                geno_timepoints.append(float(m_ko.group(1)))
                geno_reps.append(int(m_ko.group(2)))
            elif genotype == "wt" and m_wt:
                geno_col_idx.append(j)
                geno_timepoints.append(float(m_wt.group(1)))
                geno_reps.append(int(m_wt.group(2)))

        if not geno_col_idx:
            raise RuntimeError(
                "No samples found for genotype '{}' in GSE67426. "
                "Sample titles: {}".format(genotype, sample_titles[:5])
            )

        # Read expression matrix
        expr_df = pd.read_csv(
            str(matrix_gz), compression="gzip", sep="\t",
            skiprows=data_start, index_col=0, comment="!",
        )
        expr_df = expr_df[
            ~expr_df.index.astype(str).str.startswith("!")
        ]

        # Subset to genotype columns
        geno_sample_ids = [sample_ids[i] for i in geno_col_idx]
        raw_expr = expr_df[geno_sample_ids].values.astype(np.float64)
        row_ids = expr_df.index.values

        # --- Average replicates per timepoint ---
        unique_tp = sorted(set(geno_timepoints))
        avg_expr = np.column_stack([
            raw_expr[
                :,
                [k for k, tp in enumerate(geno_timepoints) if tp == t],
            ].mean(axis=1)
            for t in unique_tp
        ])
        tp_array = np.array(unique_tp, dtype=np.float64)

        # --- Map row IDs to gene symbols ---
        # For RNA-seq, row IDs may already be gene symbols.
        # Check if they look like numeric probe IDs (need platform annot)
        # or gene symbols / Ensembl IDs.
        row_id_strs = [str(rid) for rid in row_ids]
        numeric_ids = all(
            re.match(r"^\d+$", rid) for rid in row_id_strs[:100]
        )

        if numeric_ids and platform_id:
            # Need platform annotation to map probe IDs to gene symbols
            annot_gz = cache_path / "{}.annot.gz".format(platform_id)
            if not annot_gz.exists():
                # Derive FTP path from platform ID
                prefix = platform_id[:len(platform_id) - 3] + "nnn"
                url = (
                    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/{}/{}"
                    "/annot/{}.annot.gz".format(prefix, platform_id,
                                                platform_id)
                )
                try:
                    import urllib.request
                    warnings.warn(
                        "Downloading {} annotation from {} ...".format(
                            platform_id, url)
                    )
                    urllib.request.urlretrieve(url, str(annot_gz))
                except Exception:
                    annot_gz = None

            if annot_gz and annot_gz.exists():
                # Skip comment lines (start with #)
                skip = 0
                with _gzip.open(str(annot_gz), "rt") as f:
                    for line in f:
                        if line.startswith("#"):
                            skip += 1
                        else:
                            break
                annot_df = pd.read_csv(
                    str(annot_gz), compression="gzip", sep="\t",
                    skiprows=skip, low_memory=False,
                )
                probe2gene = {}
                sym_col = None
                for col in ["Gene symbol", "Gene Symbol",
                            "GENE_SYMBOL", "Symbol"]:
                    if col in annot_df.columns:
                        sym_col = col
                        break
                if sym_col:
                    for _, row in annot_df.iterrows():
                        pid = row["ID"]
                        sym = row.get(sym_col, "")
                        if pd.notna(sym) and sym != "" and sym != "---":
                            try:
                                pid_int = int(pid)
                            except (ValueError, TypeError):
                                continue
                            probe2gene[pid_int] = (
                                str(sym).split("///")[0].strip()
                            )

                gene_symbols = [
                    probe2gene.get(rid, "") for rid in row_ids
                ]
            else:
                # Fallback: use row IDs as-is
                gene_symbols = row_id_strs
        else:
            # Row IDs are likely gene symbols or Ensembl IDs
            gene_symbols = row_id_strs

        # Filter out rows without a valid gene symbol
        has_symbol = [
            bool(g) and g != "nan" and g != "" for g in gene_symbols
        ]
        avg_expr = avg_expr[has_symbol]
        gene_symbols = [
            g for g, h in zip(gene_symbols, has_symbol) if h
        ]

        # Deduplicate: keep gene with highest mean expression
        means = avg_expr.mean(axis=1)
        df_dedup = pd.DataFrame({
            "gene": gene_symbols,
            "mean_expr": means,
        })
        best_idx = (
            df_dedup.groupby("gene")["mean_expr"]
            .idxmax().values.astype(int)
        )
        avg_expr = avg_expr[best_idx]
        gene_names_arr = [gene_symbols[i] for i in best_idx]

        # Filter by minimum expression
        means = avg_expr.mean(axis=1)
        keep = means >= min_expr
        avg_expr = avg_expr[keep]
        gene_names_arr = [
            g for g, k in zip(gene_names_arr, keep) if k
        ]

        expr = avg_expr
        timepoints = tp_array
        gene_names = gene_names_arr
        source = "GEO_series_matrix"

        # Cache to NPZ
        np.savez(
            str(npz_file),
            expr=expr,
            timepoints=timepoints,
            gene_names=np.array(gene_names, dtype=object),
            source=np.array(source),
        )

    condition = "Scp2_KO" if genotype == "ko" else "wildtype"
    metadata = {
        "dataset": "Scp2 KO 2015",
        "geo_accession": "GSE67426",
        "organism": "Mus musculus",
        "tissue": "liver",
        "condition": condition,
        "platform": "RNA-seq",
        "source": source,
        "n_genes": expr.shape[0],
        "n_timepoints": expr.shape[1],
        "reference": (
            "Jouffe et al., PLoS Genet 12:e1005865 (2016)"
        ),
    }

    return {
        "expr": expr,
        "timepoints": timepoints,
        "gene_names": gene_names,
        "metadata": metadata,
    }
