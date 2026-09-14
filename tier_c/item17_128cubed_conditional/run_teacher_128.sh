#!/usr/bin/env bash
# Train the 128³ teacher segmenter (~6-12 h on M-series).
# Outputs: results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional/e7_teacher_128"
LOG="$OUTDIR/teacher_stdout.log"
mkdir -p "$OUTDIR"

if [ ! -f "data/brats_conditional_128.pt" ]; then
    echo "ERROR: data/brats_conditional_128.pt missing."
    echo "Run prepare_conditional_128.py first."
    exit 1
fi

echo "[$(date)] Launching 128³ teacher training. Logs: $LOG"
PYTHONUNBUFFERED=1 caffeinate -i python -u \
    tier_c/item17_128cubed_conditional/train_teacher_128.py \
    --data-path data/brats_conditional_128.pt \
    --output-dir "$OUTDIR" \
    --max-iters 5000 \
    --n-seeds 5 2>&1 | tee "$LOG"

echo "[$(date)] Teacher training finished. Verify $OUTDIR/teacher_best.pt exists."
