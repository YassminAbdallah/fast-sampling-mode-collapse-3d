#!/usr/bin/env bash
# Run the 2 missing cells of the 2x2 {EMA, no-EMA} x {L2, Pseudo-Huber} factorial.
# Cell C (EMA + Pseudo-Huber) and Cell D (no-EMA + L2) are trained sequentially.
# Total wall-clock on Apple M-series: ~18-24 hours (1 day).
#
# The other two cells are already trained:
#   ema_l2          -> standard Consistency Distillation (results/.../consistency/final.pt)
#   noema_pseudohuber -> Improved CD (results/.../improved_cd_brats/improved_cd/final.pt)
set -euo pipefail
cd "$(dirname "$0")/../.."

GEN_DIR="results/paper4/brats_benchmark_20260324_154926"
DATA="data/brats_conditional_64.pt"
OUTDIR="results/paper4/cd_2x2"
LOG="$OUTDIR/run_2x2_stdout.log"
mkdir -p "$OUTDIR"

# Sanity check inputs
for path in "$GEN_DIR" "$DATA"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $path"
        exit 1
    fi
done

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "[$(date)] Launching 2x2 factorial. Logs: $LOG"
echo "    epochs = $EPOCHS  batch = $BATCH_SIZE"
echo ""
echo "==========================================="
echo "  Cell C: EMA + Pseudo-Huber"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item15b_2x2_factorial/train_2x2_factorial.py \
    --variant ema_pseudohuber \
    --gen-dir "$GEN_DIR" \
    --data-path "$DATA" \
    --output-dir "$OUTDIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  Cell D: no-EMA + L2"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/item15b_2x2_factorial/train_2x2_factorial.py \
    --variant noema_l2 \
    --gen-dir "$GEN_DIR" \
    --data-path "$DATA" \
    --output-dir "$OUTDIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] 2x2 factorial finished."
echo "Verify:"
echo "  $OUTDIR/ema_pseudohuber/final.pt"
echo "  $OUTDIR/noema_l2/final.pt"
