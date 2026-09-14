#!/usr/bin/env bash
# Re-run Kynkäänniemi P/R + Naeem D/C in PCA-128 feature space.
#
# Closes peer-review punch-list item #4 / Task #104. The original 64³
# benchmark used raw 4,096-D VQ-GAN encoder features for k-NN manifold
# estimation, which is pathological in high dimensions (Kynkäänniemi
# et al. 2019 recommend Inception-2048 → 128 reduction). This re-runs
# the same evaluation in 128-D PCA space.
#
# Wall-clock estimate (M-series, 64³ models): ~30-60 min PER DATASET.
# Total ~1-2 hours for both BraTS + IXI.
#
# Outputs go to a NEW directory (precision_recall_pca128/) to preserve
# the original results for traceability.
set -euo pipefail
cd "$(dirname "$0")/../.."

BRATS_RUN="results/paper3/brats_benchmark_20260221_193306"
IXI_RUN="results/paper3/ixi_benchmark_20260221_045741"
BRATS_DATA="data/brats_preprocessed_64.pt"
IXI_DATA="data/ixi_preprocessed_64.pt"

for path in "$BRATS_RUN" "$IXI_RUN" "$BRATS_DATA" "$IXI_DATA"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $path"
        exit 1
    fi
done

echo "[$(date)] Launching P/R PCA-128 re-runs (BraTS + IXI)"
echo "  Output: results/paper3/<benchmark>/<data_subdir>/precision_recall_pca128/"

echo ""
echo "==============================================="
echo "  1/2  BraTS — re-running P/R + D/C @ PCA-128"
echo "==============================================="
PYTHONUNBUFFERED=1 caffeinate -i python -u src/paper3/precision_recall_metrics.py \
    --run-dir "$BRATS_RUN" \
    --data-path "$BRATS_DATA" \
    --n-samples 64 \
    --seeds 42 123 456 789 1337 \
    --pca-dim 128 2>&1 | tee "$BRATS_RUN/pr_pca128_stdout.log"

echo ""
echo "==============================================="
echo "  2/2  IXI — re-running P/R + D/C @ PCA-128"
echo "==============================================="
PYTHONUNBUFFERED=1 caffeinate -i python -u src/paper3/precision_recall_metrics.py \
    --run-dir "$IXI_RUN" \
    --data-path "$IXI_DATA" \
    --n-samples 64 \
    --seeds 42 123 456 789 1337 \
    --pca-dim 128 2>&1 | tee "$IXI_RUN/pr_pca128_stdout.log"

echo ""
echo "[$(date)] Both P/R PCA-128 re-runs finished."
echo "Verify:"
echo "  $BRATS_RUN/100pct_300vol/precision_recall_pca128/precision_recall_results.json"
echo "  $IXI_RUN/100pct_200vol/precision_recall_pca128/precision_recall_results.json"
