#!/usr/bin/env bash
# Conditional 128³ evaluation — reproduce §5.4 Table 6 at higher resolution.
# Wall-clock ~30-60 min on M-series. Outputs:
#   results/paper4/brats_128cubed_conditional/paper_eval_5seed_conditional/eval_results_128_conditional_5seed.json
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/brats_128cubed_conditional"
LOG="$OUTDIR/eval_stdout.log"

for ck in phase1_shared.pt consistency/final.pt shortcut/final.pt; do
    if [ ! -f "$OUTDIR/$ck" ]; then
        echo "ERROR: missing checkpoint $OUTDIR/$ck"
        echo "Run Phases 1-4 first."
        exit 1
    fi
done

echo "[$(date)] Launching conditional 128³ evaluation. Logs: $LOG"
PYTHONUNBUFFERED=1 caffeinate -i python -u \
    tier_c/item17_128cubed_conditional/eval_128_conditional.py \
    --data-path data/brats_conditional_128.pt \
    --ckpt-dir "$OUTDIR" \
    --output-dir "$OUTDIR/paper_eval_5seed_conditional" \
    --steps 50 \
    --n-per-class 32 2>&1 | tee "$LOG"

echo "[$(date)] Eval finished."
