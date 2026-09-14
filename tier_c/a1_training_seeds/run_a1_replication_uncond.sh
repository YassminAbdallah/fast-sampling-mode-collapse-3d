#!/usr/bin/env bash
# A1 UNCONDITIONAL — 2 CD + 2 Shortcut retrainings + eval on the Table 3 setup.
# Speaks to whether the "CD collapses to ~2%" unconditional headline is
# training-seed-stable or artifact.
# Wall-clock on M1: ~1.5 hours total (same as conditional).
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/train_seed_replication_uncond"
LOG="$OUTDIR/a1_uncond_stdout.log"
mkdir -p "$OUTDIR"

echo "[$(date)] A1 UNCONDITIONAL replication started."
echo "  Log: $LOG"
echo ""

CD_EPOCHS="${CD_EPOCHS:-80}"
SC_EPOCHS="${SC_EPOCHS:-150}"

echo "  CD epochs = $CD_EPOCHS, Shortcut epochs = $SC_EPOCHS"

# ---- CD, training seed 100 ----
echo ""
echo "==========================================="
echo "  [1/4] Consistency Distillation UNCONDITIONAL, train seed 100"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed_uncond.py \
    --method consistency --train-seed 100 --epochs "$CD_EPOCHS" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  [2/4] Consistency Distillation UNCONDITIONAL, train seed 200"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed_uncond.py \
    --method consistency --train-seed 200 --epochs "$CD_EPOCHS" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  [3/4] Shortcut FM UNCONDITIONAL, train seed 100"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed_uncond.py \
    --method shortcut --train-seed 100 --epochs "$SC_EPOCHS" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  [4/4] Shortcut FM UNCONDITIONAL, train seed 200"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed_uncond.py \
    --method shortcut --train-seed 200 --epochs "$SC_EPOCHS" 2>&1 | tee -a "$LOG"

echo ""
echo "==========================================="
echo "  Evaluating all six unconditional models"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/eval_replicated_seeds_uncond.py 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] A1 UNCONDITIONAL finished."
echo "Verify: $OUTDIR/a1_eval_summary_uncond.json"
