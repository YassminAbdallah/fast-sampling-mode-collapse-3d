#!/bin/bash
# ============================================================
# Paper 3: Complete Experiment Execution Script
# ============================================================
# Run experiments in order. Each section is independent.
# Total estimated time: ~6-8 hours
# ============================================================

echo "============================================================"
echo "  Paper 3: Experiment Execution Plan"
echo "============================================================"
echo ""
echo "  EXP 1: Codebook utilization (IXI + BraTS)     ~15 min"
echo "  EXP 2: Feature diversity (IXI + BraTS)         ~30 min"
echo "  EXP 3: BraTS ablation (4 training runs)        ~1.5 hrs"
echo "  EXP 4: Pixel-space trajectory (IXI)            ~3-4 hrs"
echo ""
echo "  IMPORTANT: models_shared.py must be in the same directory!"
echo "============================================================"

# ============================================================
# EXP 1: CODEBOOK UTILIZATION (~15 min, no training)
# ============================================================

echo "--- EXP 1: Codebook Utilization ---"

python codebook_utilization.py \
    --run-dir results/ixi_benchmark_20260221_045741 \
    --data-path data/ixi_preprocessed_64.pt \
    --n-samples 64

python codebook_utilization.py \
    --run-dir results/brats_benchmark_20260221_193306 \
    --data-path data/brats_preprocessed_64.pt \
    --n-samples 64


# ============================================================
# EXP 2: FEATURE DIVERSITY (~30 min, no training)
# ============================================================

echo "--- EXP 2: Feature Diversity ---"

python feature_diversity.py \
    --run-dir results/ixi_benchmark_20260221_045741 \
    --data-path data/ixi_preprocessed_64.pt \
    --n-samples 64

python feature_diversity.py \
    --run-dir results/brats_benchmark_20260221_193306 \
    --data-path data/brats_preprocessed_64.pt \
    --n-samples 64


# ============================================================
# EXP 3: BRATS ABLATION (~1.5 hrs, 4 training runs)
# ============================================================

echo "--- EXP 3: BraTS Ablation ---"

# Step 1: Patch flow_matching_3d.py (run once)
python patch_ablation_args.py

# Step 2: Run 4 ablation variants (each ~21 min)

# 3a. No SC loss (SC ratio = 0.0)
python flow_matching_3d.py \
    --dataset brats \
    --resume-dir results/brats_benchmark_20260221_193306 \
    --method shortcut \
    --sc-ratio 0.00 \
    --ablation-name sc0.00 \
    --epochs-p2 150

# 3b. SC ratio = 0.10
python flow_matching_3d.py \
    --dataset brats \
    --resume-dir results/brats_benchmark_20260221_193306 \
    --method shortcut \
    --sc-ratio 0.10 \
    --ablation-name sc0.10 \
    --epochs-p2 150

# 3c. SC ratio = 0.50
python flow_matching_3d.py \
    --dataset brats \
    --resume-dir results/brats_benchmark_20260221_193306 \
    --method shortcut \
    --sc-ratio 0.50 \
    --ablation-name sc0.50 \
    --epochs-p2 150

# 3d. No curriculum (SC = 0.25 but d_max = 1.0 from start)
python flow_matching_3d.py \
    --dataset brats \
    --resume-dir results/brats_benchmark_20260221_193306 \
    --method shortcut \
    --no-curriculum \
    --ablation-name no_curriculum \
    --epochs-p2 150

# Step 3: Evaluate all ablation variants
python evaluate_ablation.py \
    --run-dir results/brats_benchmark_20260221_193306 \
    --data-path data/brats_preprocessed_64.pt \
    --n-samples 64


# ============================================================
# EXP 4: PIXEL-SPACE TRAJECTORY (~3-4 hrs)
# ============================================================

echo "--- EXP 4: Pixel-Space Trajectory ---"

python pixel_trajectory_comparison.py \
    --data-path data/ixi_preprocessed_64.pt \
    --latent-run-dir results/ixi_benchmark_20260221_045741 \
    --epochs 150 \
    --n-trajectories 16


# ============================================================
echo ""
echo "  All experiments complete!"
echo "  Check results/ for outputs."
echo "============================================================"
