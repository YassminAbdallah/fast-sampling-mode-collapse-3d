#!/usr/bin/env bash
# DDIM-50 baseline for Tables 2 & 3 — runs BraTS, then IXI, sequentially.
# Uses the existing trained DDPM checkpoints; no retraining required.
# Wall-clock on Apple M-series: ~1 to 2 hours total (sampling is the cost).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/ddim_baseline_5seed"
LOG="$OUTDIR/ddim_baseline_stdout.log"
mkdir -p "$OUTDIR"

echo "[$(date)] Launching DDIM-50 baseline. Log: $LOG"
echo ""
echo "==========================================="
echo "  Dataset: BraTS 2023"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item_ddim_baseline/eval_ddim_50.py \
    --dataset brats \
    --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  Dataset: IXI"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item_ddim_baseline/eval_ddim_50.py \
    --dataset ixi \
    --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] DDIM-50 baseline finished."
echo "Outputs:"
echo "  $OUTDIR/brats_ddim50.json"
echo "  $OUTDIR/ixi_ddim50.json"
