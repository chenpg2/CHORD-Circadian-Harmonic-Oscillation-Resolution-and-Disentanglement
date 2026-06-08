"""Process GSE130890 (Pan 2020) WIG coverage tracks into a gene x sample matrix.

WT + XBP1 liver-specific KO mouse liver, 2h/48h (24 timepoints x 2 reps x 2
genotypes = 96 samples). GEO deposits only per-sample WIG coverage (mm10), so we
quantify gene-level expression by summing read coverage (value x span) over each
gene's mm10 RefSeq exons. Downloads one WIG at a time, quantifies, deletes it
(keeps disk < 200 MB), and is resumable (per-sample .npy cache).

Output: ~/.chord_cache/gse130890_xbp1ko.npz  {wt_expr, ko_expr, timepoints, gene_names}
"""
import gzip
import os
import re
import subprocess

import numpy as np

CACHE = os.path.expanduser("~/.chord_cache")
WORK = os.path.join(CACHE, "GSE130890")
PARTS = os.path.join(WORK, "parts")
os.makedirs(PARTS, exist_ok=True)
REFGENE = os.path.join(CACHE, "mm10_refGene.txt.gz")
FILELIST_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130890/suppl/filelist.txt"
BINSIZE = 200
STD_CHROMS = {f"chr{c}" for c in list(range(1, 20)) + ["X", "Y"]}


def build_gene_model():
    """Return (gene_names, binmap) where binmap[(chrom,bin)] = list of gene indices,
    built from the union of RefSeq exons per gene symbol (mm10)."""
    sym_to_idx, gene_names = {}, []
    exons_by_gene = {}  # idx -> list of (chrom, start, end)
    with gzip.open(REFGENE, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            chrom, sym = c[2], c[12]
            if chrom not in STD_CHROMS:
                continue
            starts = [int(x) for x in c[9].split(",") if x]
            ends = [int(x) for x in c[10].split(",") if x]
            if sym not in sym_to_idx:
                sym_to_idx[sym] = len(gene_names)
                gene_names.append(sym)
                exons_by_gene[sym_to_idx[sym]] = []
            gi = sym_to_idx[sym]
            for s, e in zip(starts, ends):
                exons_by_gene[gi].append((chrom, s, e))
    binmap = {}
    for gi, exons in exons_by_gene.items():
        for chrom, s, e in exons:
            for b in range(s // BINSIZE, e // BINSIZE + 1):
                binmap.setdefault((chrom, b), set()).add(gi)
    binmap = {k: tuple(v) for k, v in binmap.items()}
    print(f"gene model: {len(gene_names)} genes, {len(binmap)} exon bins", flush=True)
    return gene_names, binmap


def quantify_wig(path, n_genes, binmap):
    cov = np.zeros(n_genes, dtype=np.float64)
    chrom, span = None, 1
    with gzip.open(path, "rt") as f:
        for line in f:
            if line[0] == "v":  # variableStep chrom=.. span=..
                p = line.split()
                chrom = p[1].split("=")[1]
                span = int(p[2].split("=")[1])
                continue
            i = line.find(" ")
            if i < 0:
                continue
            pos = int(line[:i]); val = float(line[i + 1:])
            genes = binmap.get((chrom, pos // BINSIZE))
            if genes:
                w = val * span
                for gi in genes:
                    cov[gi] += w
    return cov


def sample_table():
    out = subprocess.run(["curl", "-s", "--max-time", "120", FILELIST_URL],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        c = ln.split("\t")
        if len(c) < 5 or c[4] != "WIG":
            continue
        fn = c[1]
        m = re.match(r"(GSM\d+)_(WT|LKO)_CT(\d+)([AB])_", fn)
        if not m:
            continue
        gsm, geno, ct, rep = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{fn}"
        rows.append({"gsm": gsm, "geno": geno, "ct": ct, "rep": rep, "fn": fn, "url": url})
    return rows


def main():
    gene_names, binmap = build_gene_model()
    n = len(gene_names)
    np.save(os.path.join(PARTS, "gene_names.npy"), np.array(gene_names, dtype=object))
    rows = sample_table()
    print(f"{len(rows)} samples to process", flush=True)
    for k, r in enumerate(rows):
        out_npy = os.path.join(PARTS, f"{r['gsm']}.npy")
        if os.path.exists(out_npy):
            continue
        wig = os.path.join(WORK, r["fn"])
        ok = False
        for _ in (1, 2):
            rc = subprocess.run(["curl", "-s", "--max-time", "600", "-o", wig, r["url"]]).returncode
            if rc == 0 and os.path.exists(wig) and os.path.getsize(wig) > 1_000_000:
                ok = True
                break
        if not ok:
            print(f"FAIL download {r['gsm']} ({r['fn']}) — left unprocessed", flush=True)
            continue
        cov = quantify_wig(wig, n, binmap)
        np.save(out_npy, cov)
        os.remove(wig)
        print(f"[{k+1}/{len(rows)}] {r['gsm']} {r['geno']} CT{r['ct']}{r['rep']} "
              f"done (Alb-rank check sum={cov.sum():.2e})", flush=True)
    assemble(gene_names, rows)


def assemble(gene_names, rows):
    cts = sorted({r["ct"] for r in rows})
    n = len(gene_names)
    mats = {"WT": np.full((n, len(cts)), np.nan), "LKO": np.full((n, len(cts)), np.nan)}
    for geno in ("WT", "LKO"):
        for j, ct in enumerate(cts):
            reps = []
            for r in rows:
                if r["geno"] == geno and r["ct"] == ct:
                    p = os.path.join(PARTS, f"{r['gsm']}.npy")
                    if os.path.exists(p):
                        reps.append(np.load(p))
            if reps:
                mats[geno][:, j] = np.mean(reps, axis=0)
    out = os.path.join(CACHE, "gse130890_xbp1ko.npz")
    np.savez(out, wt_expr=mats["WT"], ko_expr=mats["LKO"],
             timepoints=np.array(cts, dtype=float),
             gene_names=np.array(gene_names, dtype=object), source="GSE130890_WIG_mm10")
    done = sum(1 for r in rows if os.path.exists(os.path.join(PARTS, f"{r['gsm']}.npy")))
    print(f"SAVED {out}: {n} genes x {len(cts)} timepoints, {done}/{len(rows)} samples used", flush=True)


if __name__ == "__main__":
    main()
