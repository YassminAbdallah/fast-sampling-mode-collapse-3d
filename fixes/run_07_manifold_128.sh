#!/usr/bin/env bash
# 128-cubed manifold metrics, distributional metrics and diversity.
# Inference only, no training. ~35-50 min on Apple M1 MPS.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/paper4/manifold_metrics_128

echo "=== SMOKE TEST (2 min): 1 seed, 16 samples ==="
python3 fixes/07_manifold_metrics_128.py --seeds 42 --n-samples 16 \
  --out results/paper4/manifold_metrics_128/smoke.json

echo
read -r -p "Smoke test looks right? Run the full 5-seed pass (~45 min)? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "stopped."; exit 0; }

caffeinate -s python3 fixes/07_manifold_metrics_128.py \
  2>&1 | tee results/paper4/manifold_metrics_128/stdout.log
