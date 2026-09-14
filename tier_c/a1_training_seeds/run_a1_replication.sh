#!/usr/bin/env bash
# A1 — replicate CD and Shortcut with 2 additional training seeds each at 64³.
# Total: 4 trainings (2 CD + 2 Shortcut) + 1 evaluation pass.
# Wall-clock on Apple M1: ~5-8 hours.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTDIR="results/paper4/train_seed_replication"
LOG="$OUTDIR/a1_stdout.log"
mkdir -p "$OUTDIR"

echo "[$(date)] A1 training-seed replication started. Log: $LOG"
echo ""

# Reuse variables from the environment if set (useful for shorter test runs)
CD_EPOCHS="${CD_EPOCHS:-80}"        # matches original CD
SC_EPOCHS="${SC_EPOCHS:-150}"       # matches original Shortcut FM

echo "  CD epochs = $CD_EPOCHS, Shortcut epochs = $SC_EPOCHS"
echo ""

# ---- CD, training seed 100 ----
echo "==========================================="
echo "  [1/4] Consistency Distillation, train seed 100"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed.py \
    --method consistency --train-seed 100 --epochs "$CD_EPOCHS" 2>&1 | tee -a "$LOG"

# ---- CD, training seed 200 ----
echo ""
echo "==========================================="
echo "  [2/4] Consistency Distillation, train seed 200"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed.py \
    --method consistency --train-seed 200 --epochs "$CD_EPOCHS" 2>&1 | tee -a "$LOG"

# ---- Shortcut, training seed 100 ----
echo ""
echo "==========================================="
echo "  [3/4] Shortcut FM, train seed 100"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed.py \
    --method shortcut --train-seed 100 --epochs "$SC_EPOCHS" 2>&1 | tee -a "$LOG"

# ---- Shortcut, training seed 200 ----
echo ""
echo "==========================================="
echo "  [4/4] Shortcut FM, train seed 200"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/train_replicate_seed.py \
    --method shortcut --train-seed 200 --epochs "$SC_EPOCHS" 2>&1 | tee -a "$LOG"

# ---- Evaluate all six models (2 original + 4 new) ----
echo ""
echo "==========================================="
echo "  Evaluating all six trained models"
echo "==========================================="
PYTHONUNBUFFERED=1 caffeinate -s python -u \
    tier_c/a1_training_seeds/eval_replicated_seeds.py 2>&1 | tee -a "$LOG"

echo ""
echo "[$(date)] A1 replication finished."
echo "Verify: $OUTDIR/a1_eval_summary.json"
