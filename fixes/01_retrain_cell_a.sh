#!/usr/bin/env bash
# =============================================================================
# Train Cell A (EMA + L2) of the {EMA, no-EMA} x {L2, Pseudo-Huber} factorial
# under the same schedule as the other three cells (80 epochs, batch size 4),
# so that every contrast within the factorial is schedule-matched. The
# main-benchmark Consistency Distillation checkpoint (150 epochs, batch 2)
# is left untouched.
#
# RUNTIME  ~10-20 min on Apple M-series.
#
# Usage:  bash fixes/01_retrain_cell_a.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

TRAINER="tier_c/item15b_2x2_factorial/train_2x2_factorial.py"
GEN_DIR="results/paper4/brats_benchmark_20260324_154926"
DATA="data/brats_conditional_64.pt"
OUTDIR="results/paper4/cd_2x2"
EPOCHS=80
BATCH=4

# ---------------------------------------------------------------------------
# Step 1: register the ema_l2 variant so argparse accepts it.
# ---------------------------------------------------------------------------
python3 - "$TRAINER" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]); src = p.read_text()
if '"ema_l2"' in src.split("VARIANT_DESCRIPTIONS")[1].split("}")[0]:
    print("  ema_l2 already registered - no patch needed"); sys.exit(0)
src = src.replace(
    '"noema_l2":        "No-EMA target + L2 loss          (Cell D: \'no-EMA + L2\')",',
    '"noema_l2":        "No-EMA target + L2 loss          (Cell D: \'no-EMA + L2\')",\n'
    '    "ema_l2":          "EMA target + L2 loss             (Cell A: schedule-matched retrain)",',
    1)
p.write_text(src)
print("  patched: ema_l2 added to VARIANT_DESCRIPTIONS")
PY

# ---------------------------------------------------------------------------
# Step 2: train Cell A under the B/C/D recipe.
#         Writes to cd_2x2/ema_l2/ -- the ORIGINAL 150-epoch checkpoint in
#         the benchmark dir is left untouched, so Tables 2/3/6 do not move.
# ---------------------------------------------------------------------------
echo ""
echo "=== Training Cell A (EMA + L2) @ ${EPOCHS} epochs, batch ${BATCH} ==="
python3 "$TRAINER" \
    --variant ema_l2 \
    --gen-dir "$GEN_DIR" \
    --data-path "$DATA" \
    --output-dir "$OUTDIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --num-classes 2 \
    2>&1 | tee "$OUTDIR/cell_a_retrain_stdout.log"

# ---------------------------------------------------------------------------
# Step 3: rebuild the factorial table. --no-reuse-existing forces a fresh
#         eval of every cell so all four are scored under one protocol.
# ---------------------------------------------------------------------------
echo ""
echo "=== Re-evaluating all four cells ==="
python3 tier_c/item15b_2x2_factorial/eval_2x2_factorial.py \
    --n-samples 64 \
    --no-reuse-existing \
    2>&1 | tee "$OUTDIR/cell_a_reeval_stdout.log"

cat <<'EOF'

=============================================================================
DONE. Now update the manuscript:

  1. §5.2 -- replace Cell A's 32.7%-of-real diversity with the NEW schedule-
     matched number from results/paper4/cd_2x2/factorial_2x2_table.json.
     THE CONCLUSION MAY CHANGE. The old claim was "given EMA, Pseudo-Huber
     improves diversity over L2 (49.9% vs 32.7%)". If the schedule-matched
     Cell A now lands near 49.9%, that claim dissolves and the honest
     statement becomes "the loss function is not a meaningful lever once
     training is schedule-matched." Report whatever you get.

  2. Appendix A.3 -- currently says "trained for 150 epochs at batch size 4".
     The benchmark config says batch_size = 2. Rewrite to state both recipes:
        - main benchmark generators: 150 epochs, batch 2
        - 2x2 factorial cells:        80 epochs, batch 4
=============================================================================
EOF
