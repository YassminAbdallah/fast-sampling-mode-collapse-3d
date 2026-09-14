#!/bin/bash
# ==============================================================
# Paper 3: Shortcut FM Ablation Study
# ==============================================================
#
# Tests 4 ablation variants against the default shortcut config:
#
#   1. sc0.00         — FM-only + d-conditioning (NO SC loss at all)
#   2. sc0.25_nocurr  — SC loss but NO d_max curriculum (d_max=1.0 from start)
#   3. sc0.10         — Low SC ratio (10% vs default 25%)
#   4. sc0.50         — High SC ratio (50% vs default 25%)
#
# Default (already trained): sc_ratio=0.25, d_max curriculum ON
#
# Usage:
#   bash run_ablation_shortcut.sh ixi     # IXI dataset
#   bash run_ablation_shortcut.sh brats   # BraTS dataset
#
# Prerequisites:
#   - Existing benchmark results with trained VQ-GAN
#   - Set RESUME_DIR below to point to your benchmark results
# ==============================================================

DATASET=${1:-ixi}

# Set resume directory based on dataset
if [ "$DATASET" == "ixi" ]; then
    RESUME_DIR="results/ixi_benchmark_20260221_045741"
elif [ "$DATASET" == "brats" ]; then
    RESUME_DIR="results/brats_benchmark_20260221_193306"
else
    echo "Usage: bash run_ablation_shortcut.sh [ixi|brats]"
    exit 1
fi

echo "============================================================"
echo "  ABLATION STUDY: Shortcut FM Components"
echo "  Dataset: $DATASET"
echo "  Resume from: $RESUME_DIR"
echo "============================================================"
echo ""

# Ablation 1: FM-only + d-conditioning, NO SC loss
echo ">>> Ablation 1/4: sc_ratio=0.00 (FM-only, no SC loss)"
python flow_matching_3d.py \
    --resume-dir $RESUME_DIR \
    --method shortcut \
    --dataset $DATASET \
    --sc-ratio 0.0 \
    --epochs-p2 150

echo ""

# Ablation 2: SC loss but NO curriculum (d_max=1.0 from start)
echo ">>> Ablation 2/4: sc_ratio=0.25, NO d_max curriculum"
python flow_matching_3d.py \
    --resume-dir $RESUME_DIR \
    --method shortcut \
    --dataset $DATASET \
    --sc-ratio 0.25 \
    --no-curriculum \
    --epochs-p2 150

echo ""

# Ablation 3: Low SC ratio (10%)
echo ">>> Ablation 3/4: sc_ratio=0.10"
python flow_matching_3d.py \
    --resume-dir $RESUME_DIR \
    --method shortcut \
    --dataset $DATASET \
    --sc-ratio 0.10 \
    --epochs-p2 150

echo ""

# Ablation 4: High SC ratio (50%)
echo ">>> Ablation 4/4: sc_ratio=0.50"
python flow_matching_3d.py \
    --resume-dir $RESUME_DIR \
    --method shortcut \
    --dataset $DATASET \
    --sc-ratio 0.50 \
    --epochs-p2 150

echo ""
echo "============================================================"
echo "  ABLATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  $RESUME_DIR/*/shortcut_sc0.00/"
echo "  $RESUME_DIR/*/shortcut_sc0.25_nocurr/"
echo "  $RESUME_DIR/*/shortcut_sc0.10/"
echo "  $RESUME_DIR/*/shortcut_sc0.50/"
echo ""
echo "Compare against default: $RESUME_DIR/*/shortcut/"
echo ""
echo "To get rigorous eval numbers, run evaluate_ablation.py next."
