#!/usr/bin/env bash
# T1 pre-emptive: task-independent P/R/D/C for Consistency@50 and Shortcut@50
# on IXI + BraTS 64-cubed, using the tumour-segmentation teacher's
# bottleneck features as the (task-independent) feature space.
# Wall-clock estimate on Apple M1: ~60 min total (both datasets).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/t1_pr_task_independent"
LOG="$OUTDIR/t1_pr_stdout.log"
mkdir -p "$OUTDIR"

echo "[$(date)] Launching T1 task-independent P/R/D/C run."
echo "  Log: $LOG"
echo ""

# caffeinate -s: no sleep even with lid closed on AC.
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/t1_task_independent_pr/t1_pr_teacher_features.py \
    --dataset both 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] Finished."
echo "  Verify: $OUTDIR/t1_pr_results.json"
