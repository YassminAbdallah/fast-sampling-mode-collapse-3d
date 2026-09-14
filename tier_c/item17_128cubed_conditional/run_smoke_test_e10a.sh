#!/usr/bin/env bash
# Smoke test the v2 NaN-robust recipe before committing to the 7-day full run.
#
# Trains ONE seed × ONE condition (unique_500, seed=42) for 500 iters only.
# Wall-clock ~1.5 hours. If this finishes without NaN aborts and reports a
# sensible Dice (anything > 0.3), the recipe is validated and the full
# run can launch with confidence.
#
# Output goes to a SEPARATE directory so it does NOT touch the real E10a
# partial JSON.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional/e10a_128_smoke"
LOG="$OUTDIR/smoke_stdout.log"
mkdir -p "$OUTDIR"

TEACHER="results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt"
if [ ! -f "$TEACHER" ]; then
    echo "ERROR: 128³ teacher missing at $TEACHER"
    exit 1
fi

echo "[$(date)] Smoke test: 1 seed × unique_500 × 500 iters. Logs: $LOG"
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item17_128cubed_conditional/e10a_128.py \
    --data-path data/brats_conditional_128.pt \
    --ckpt-dir results/paper4/brats_128cubed_conditional \
    --teacher-path "$TEACHER" \
    --output-dir "$OUTDIR" \
    --max-iters 500 \
    --n-seeds 1 2>&1 | tee "$LOG"

echo ""
echo "[$(date)] Smoke test finished."
echo ""
echo "Pass criteria:"
echo "  * Seed 42 of unique_500 finishes with status 'ok' (not 'nan_aborted')"
echo "  * Final Dice > 0.3 (any positive number with a 500-iter under-trained model)"
echo "  * skipped_nan counter stays at 0 or small (<10)"
echo ""
echo "If those hold, the v2 recipe is validated. Launch the full run:"
echo "  bash tier_c/item17_128cubed_conditional/run_e10a_128.sh"
