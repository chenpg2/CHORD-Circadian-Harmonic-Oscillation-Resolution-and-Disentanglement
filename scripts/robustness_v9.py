#!/usr/bin/env python3
"""
CHORD v9 Robustness Test — CHORD-only downsampling + noise.
Competing method results can be taken from v6 benchmark.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from chord.bhdt.classifier import classify_gene, CHORDConfig
from chord.data.geo_loader import load_hughes2009
from chord.simulation.generator import independent_superposition

def run_chord_on_data(expr, tp, gene_names, label, verbose=True):
    """Run CHORD on a dataset."""
    results = []
    n = len(gene_names)
    t0 = time.time()
    for i, (y, g) in enumerate(zip(expr, gene_names)):
        if verbose and (i+1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            print(f"    CHORD on {label}: {i+1}/{n} genes ({rate:.0f} genes/s)", flush=True)
        try:
            r = classify_gene(tp, y)
            results.append({
                "gene": g, "method": "CHORD", "dataset": label,
                "p_value": r.get("stage1_p_detect", 1.0),
                "has_12h": r.get("stage1_passed", False),
                "classification": r.get("classification", "unknown"),
                "confidence": r.get("confidence", 0.0),
            })
        except Exception as e:
            results.append({
                "gene": g, "method": "CHORD", "dataset": label,
                "p_value": 1.0, "has_12h": False,
                "classification": "error", "confidence": 0.0,
            })
    return pd.DataFrame(results)

def main():
    output_dir = "results/benchmark_v9"
    
    print("=" * 60)
    print("CHORD v9 Robustness — Downsampling Experiment")
    print("=" * 60)
    
    # Load full-resolution Hughes2009
    d = load_hughes2009(downsample_2h=False)
    expr_full, tp_full = d["expr"], d["timepoints"]
    gene_names = d["gene_names"]
    print(f"  Loaded Hughes2009: {len(gene_names)} genes x {len(tp_full)} timepoints")
    
    configs = [
        (1, "48tp_1h", 48),
        (2, "24tp_2h", 24),
        (3, "16tp_3h", 16),
        (4, "12tp_4h", 12),
        (6, "8tp_6h",  8),
    ]
    
    all_ds = []
    for step, label, expected_n in configs:
        idx = np.arange(0, len(tp_full), step)
        expr_ds = expr_full[:, idx]
        tp_ds = tp_full[idx]
        print(f"\n  Downsampling: {label} ({len(tp_ds)} timepoints)")
        
        df = run_chord_on_data(expr_ds, tp_ds, gene_names, f"downsample_{label}")
        df["downsample_config"] = label
        df["n_timepoints"] = len(tp_ds)
        all_ds.append(df)
    
    ds_results = pd.concat(all_ds, ignore_index=True)
    ds_results.to_csv(f"{output_dir}/downsampling_chord_v9.csv", index=False)
    print(f"\n  Saved downsampling results: {len(ds_results)} rows")
    
    # Noise experiment
    print("\n" + "=" * 60)
    print("CHORD v9 Robustness — Noise Experiment")
    print("=" * 60)
    
    snr_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    t = np.arange(0, 48, 1.0)
    rng = np.random.RandomState(42)
    n_reps = 20
    
    all_noise = []
    for snr in snr_levels:
        print(f"\n  SNR = {snr}")
        for rep in range(n_reps):
            # SNR = A_12 / noise_sd; with A_12=1.5, noise_sd = A_12/snr
            noise_sd = 1.5 / snr
            result = independent_superposition(seed=rng.randint(0, 2**31), noise_sd=noise_sd)
            y = result["y"]
            r = classify_gene(t, y)
            all_noise.append({
                "snr": snr, "replicate": rep, "method": "CHORD",
                "has_12h": r.get("stage1_passed", False),
                "classification": r.get("classification", "unknown"),
                "confidence": r.get("confidence", 0.0),
            })
    
    noise_results = pd.DataFrame(all_noise)
    noise_results.to_csv(f"{output_dir}/noise_chord_v9.csv", index=False)
    print(f"\n  Saved noise results: {len(noise_results)} rows")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\nDownsampling (CHORD detection rate on known 12h genes):")
    from chord.data.known_genes import KNOWN_12H_GENES_ZHU2017
    known = set(KNOWN_12H_GENES_ZHU2017)
    for cfg in ["48tp_1h", "24tp_2h", "16tp_3h", "12tp_4h", "8tp_6h"]:
        sub = ds_results[ds_results["downsample_config"] == cfg]
        known_in = known & set(sub["gene"])
        if known_in:
            det = sub[sub["gene"].isin(known_in) & sub["has_12h"].astype(bool)]
            print(f"  {cfg}: {len(det)}/{len(known_in)} = {len(det)/len(known_in):.1%}")
    
    print("\nNoise (CHORD detection rate):")
    for snr in snr_levels:
        sub = noise_results[noise_results["snr"] == snr]
        rate = sub["has_12h"].astype(bool).mean()
        print(f"  SNR={snr}: {rate:.1%}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
