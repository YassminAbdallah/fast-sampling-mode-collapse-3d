#!/usr/bin/env bash
# Phase 1 — Conditional 128³ VQ-GAN training (~24 hours on M-series)
#
# Outputs:
#   results/paper4/brats_128cubed_conditional/phase1_shared.pt
#   results/paper4/brats_128cubed_conditional/phase1_log.json
#
# Logs to: results/paper4/brats_128cubed_conditional/phase1_stdout.log
set -euo pipefail
cd "$(dirname "$0")/../.."   # → repo root

OUTDIR="results/paper4/brats_128cubed_conditional"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/phase1_stdout.log"

echo "[$(date)] Launching Phase 1 conditional VQ-GAN. Logs: $LOG"
# caffeinate -i prevents idle sleep so multi-day training survives overnight.
# PYTHONUNBUFFERED=1 + python -u keep stdout flushing line-by-line for tee.
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \
    --data-path data/brats_conditional_128.pt \
    --output-dir "$OUTDIR" \
    --phase vqgan \
    --batch-size 2 \
    --epochs-vqgan 80 \
    --seed 42 2>&1 | tee "$LOG"

echo "[$(date)] Phase 1 finished. Verify $OUTDIR/phase1_shared.pt exists."
