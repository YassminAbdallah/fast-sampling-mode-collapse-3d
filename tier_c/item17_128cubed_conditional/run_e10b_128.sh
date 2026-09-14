#!/usr/bin/env bash
# 128³ E10b — practical augmentation comparison.
# 3 fractions × 4 conditions × 3 seeds = 36 training runs ≈ ~10 days on M-series.
# Reuses the conditional 128³ generators + teacher from §5.8.2 / §5.8.3.
# Recipe v2 numerical hardening applied (matches E10a 128³).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional/e10b_128"
LOG="$OUTDIR/e10b_stdout.log"
mkdir -p "$OUTDIR"

TEACHER="results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt"
if [ ! -f "$TEACHER" ]; then
    echo "ERROR: 128³ teacher missing at $TEACHER"
    exit 1
fi
for ck in phase1_shared.pt shortcut/final.pt consistency/final.pt; do
    if [ ! -f "results/paper4/brats_128cubed_conditional/$ck" ]; then
        echo "ERROR: missing checkpoint results/paper4/brats_128cubed_conditional/$ck"
        exit 1
    fi
done

NSEEDS="${NSEEDS:-3}"
MAXITERS="${MAXITERS:-2000}"

echo "[$(date)] Launching 128³ E10b. Logs: $LOG"
echo "    max_iters = $MAXITERS  n_seeds = $NSEEDS"
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item17_128cubed_conditional/e10b_128.py \
    --data-path data/brats_conditional_128.pt \
    --ckpt-dir results/paper4/brats_128cubed_conditional \
    --teacher-path "$TEACHER" \
    --output-dir "$OUTDIR" \
    --max-iters "$MAXITERS" \
    --n-seeds "$NSEEDS" 2>&1 | tee "$LOG"

echo "[$(date)] 128³ E10b finished. Verify $OUTDIR/e10b_results_128.json exists."
