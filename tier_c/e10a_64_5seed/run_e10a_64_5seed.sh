#!/usr/bin/env bash
# 64-cubed E10a 5-seed replication.
# Wall-clock estimate on Apple M1: ~20-25 h continuous.
# Per-seed checkpointing: if interrupted, restart the same command.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/e10a_64_5seed"
LOG="$OUTDIR/e10a_64_5seed_stdout.log"
mkdir -p "$OUTDIR"

MAX_ITERS="${MAX_ITERS:-2000}"

echo "[$(date)] Launching 64³ E10a 5-seed replication."
echo "  max_iters = $MAX_ITERS"
echo "  Log: $LOG"
echo ""

# caffeinate -s prevents sleep on the AC-powered Mac even with lid closed.
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/e10a_64_5seed/e10a_64_5seed.py \
    --max-iters "$MAX_ITERS" 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] Finished."
echo "  Verify: $OUTDIR/e10a_64_5seed_results.json"
