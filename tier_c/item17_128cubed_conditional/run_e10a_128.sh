#!/usr/bin/env bash
# 128³ E10a dose-response — 4 unique-count conditions × 3 seeds + real-only × 3 seeds.
# Wall-clock ~4-6 days continuous on M-series (~22 h per training run × 15 runs).
# Reduce by setting MAXITERS=1500 below.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional/e10a_128"
LOG="$OUTDIR/e10a_stdout.log"
mkdir -p "$OUTDIR"

TEACHER="results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt"
if [ ! -f "$TEACHER" ]; then
    echo "ERROR: 128³ teacher missing at $TEACHER"
    echo "Run run_teacher_128.sh first (or use the Seed 0 checkpoint already saved)."
    exit 1
fi

# Recipe v2: 5 seeds × 2000 iters per condition. Matches 64³ protocol on iters,
# strengthens to 5 seeds.
MAXITERS="${MAXITERS:-2000}"
NSEEDS="${NSEEDS:-5}"

echo "[$(date)] Launching 128³ E10a (v2 recipe). Logs: $LOG"
echo "    max_iters = $MAXITERS  n_seeds = $NSEEDS"
# caffeinate -s: prevents system sleep even when lid is closed (AC power required).
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item17_128cubed_conditional/e10a_128.py \
    --data-path data/brats_conditional_128.pt \
    --ckpt-dir results/paper4/brats_128cubed_conditional \
    --teacher-path "$TEACHER" \
    --output-dir "$OUTDIR" \
    --max-iters "$MAXITERS" \
    --n-seeds "$NSEEDS" 2>&1 | tee "$LOG"

echo "[$(date)] 128³ E10a finished. Verify $OUTDIR/e10a_results_128.json exists."
