#!/bin/bash
# Full CHORD benchmark for publication
# Estimated runtime: ~12-14 hours
# Started: $(date)

set -e
export PYTHONPATH="/home/data2/fangcong2/ovary_aging/scripts/chord/src:$PYTHONPATH"

SCRIPT="/home/data2/fangcong2/ovary_aging/scripts/chord/scripts/benchmark_final.py"
OUTDIR="/home/data2/fangcong2/ovary_aging/scripts/chord/results/benchmark"
LOGFILE="${OUTDIR}/full_benchmark.log"

echo "========================================" | tee -a "$LOGFILE"
echo "CHORD Full Benchmark Started: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Phase 1: Synthetic (50 reps) — ~2 min
echo "[$(date '+%H:%M:%S')] Phase 1: Synthetic benchmark..." | tee -a "$LOGFILE"
python "$SCRIPT" --mode synthetic --reps 50 --output "$OUTDIR" 2>&1 | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S')] Phase 1 DONE" | tee -a "$LOGFILE"

# Phase 2: Real data (gt_only, 5 datasets) — ~2 min
echo "[$(date '+%H:%M:%S')] Phase 2: Real data benchmark..." | tee -a "$LOGFILE"
python "$SCRIPT" --mode real --gt-only \
  --datasets hughes2009_2h,zhu2023_wt,zhu2023_ko,mure2018_liver,zhang2014_liver \
  --output "$OUTDIR" 2>&1 | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S')] Phase 2 DONE" | tee -a "$LOGFILE"

# Phase 3: Robustness (downsampling + noise) — ~12 hours
echo "[$(date '+%H:%M:%S')] Phase 3: Robustness experiments..." | tee -a "$LOGFILE"
python "$SCRIPT" --mode robustness --reps 20 --output "$OUTDIR" 2>&1 | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S')] Phase 3 DONE" | tee -a "$LOGFILE"

# Phase 4: Generate figures
echo "[$(date '+%H:%M:%S')] Phase 4: Generating figures..." | tee -a "$LOGFILE"
FIGSCRIPT="/home/data2/fangcong2/ovary_aging/scripts/chord/scripts/benchmark_figures.py"
if [ -f "$FIGSCRIPT" ]; then
    python "$FIGSCRIPT" --input "$OUTDIR" --output "${OUTDIR}/figures/" 2>&1 | tee -a "$LOGFILE"
fi
echo "[$(date '+%H:%M:%S')] Phase 4 DONE" | tee -a "$LOGFILE"

echo "========================================" | tee -a "$LOGFILE"
echo "ALL PHASES COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
