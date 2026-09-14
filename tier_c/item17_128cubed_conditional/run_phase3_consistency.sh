#!/usr/bin/env bash
# Phase 3 — Conditional 128³ Consistency Distillation (~12 hours on M-series)
# Requires Phases 1 and 2 to be complete.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional"
LOG="$OUTDIR/phase3_stdout.log"

if [ ! -f "$OUTDIR/phase1_shared.pt" ] || [ ! -f "$OUTDIR/fm/final.pt" ]; then
    echo "ERROR: Phase 1 or Phase 2 checkpoint missing in $OUTDIR/"
    exit 1
fi

echo "[$(date)] Launching Phase 3 conditional Consistency Distillation. Logs: $LOG"
# caffeinate -i prevents idle sleep so multi-day training survives overnight.
# PYTHONUNBUFFERED=1 + python -u keep stdout flushing line-by-line for tee.
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \
    --data-path data/brats_conditional_128.pt \
    --output-dir "$OUTDIR" \
    --phase consistency \
    --batch-size 2 \
    --epochs-consistency 60 \
    --seed 42 2>&1 | tee "$LOG"

echo "[$(date)] Phase 3 finished. Verify $OUTDIR/consistency/final.pt exists."
