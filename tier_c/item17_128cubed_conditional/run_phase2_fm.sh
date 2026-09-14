#!/usr/bin/env bash
# Phase 2 — Conditional 128³ FM teacher training (~24 hours on M-series)
# Requires Phase 1 to be complete (phase1_shared.pt present).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional"
LOG="$OUTDIR/phase2_stdout.log"

if [ ! -f "$OUTDIR/phase1_shared.pt" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $OUTDIR/phase1_shared.pt"
    echo "Run ./run_phase1_vqgan.sh first."
    exit 1
fi

echo "[$(date)] Launching Phase 2 conditional FM teacher. Logs: $LOG"
# caffeinate -i prevents idle sleep so multi-day training survives overnight.
# PYTHONUNBUFFERED=1 + python -u keep stdout flushing line-by-line for tee.
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \
    --data-path data/brats_conditional_128.pt \
    --output-dir "$OUTDIR" \
    --phase fm \
    --batch-size 2 \
    --epochs-fm 80 \
    --seed 42 2>&1 | tee "$LOG"

echo "[$(date)] Phase 2 finished. Verify $OUTDIR/fm/final.pt exists."
