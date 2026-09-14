#!/usr/bin/env python3
"""
K-Sensitivity Analysis for Precision/Recall
=============================================

Tests whether the zero-Recall finding for Consistency Distillation
is robust across different k values, or is an artifact of k=3.

Runs Kynkäänniemi Precision/Recall with k = 1, 3, 5, 7, 10
for the key methods: FM@50, Shortcut@4, Consistency@4, Consistency@50.

Usage:
    python k_sensitivity.py --run-dir results/ixi_benchmark_20260221_045741 \
        --data-path data/ixi_preprocessed_64.pt

    python k_sensitivity.py --run-dir results/brats_benchmark_20260221_193306 \
        --data-path data/brats_preprocessed_64.pt

Output:
    {data_subdir}/precision_recall/k_sensitivity.json
    {data_subdir}/precision_recall/fig_k_sensitivity.pdf/png

Requires: models_shared.py in the same directory.
"""

import os, json, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

from models_shared import (
    Encoder3D, Decoder3D, VectorQuantizer,
    load_unet, load_vqgan, sample_latent
)

K_VALUES = [1, 3, 5, 7, 10]
SEEDS = [42, 123, 456]


@torch.no_grad()
def extract_features(volumes, enc, dev, batch_size=8):
    features = []
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size].to(dev)
        z = enc(batch)
        features.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(features, axis=0)


def kynkaanniemi_precision_recall(real_features, gen_features, k=3):
    """Kynkäänniemi et al. (2019) Precision and Recall."""
    nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(real_features)
    real_dists, _ = nn_real.kneighbors(real_features)
    real_radii = real_dists[:, k]

    nn_gen = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(gen_features)
    gen_dists, _ = nn_gen.kneighbors(gen_features)
    gen_radii = gen_dists[:, k]

    # Precision
    dists_gen_to_real, idx_gen_to_real = nn_real.kneighbors(gen_features, n_neighbors=1)
    precision = float(np.mean(dists_gen_to_real[:, 0] <= real_radii[idx_gen_to_real[:, 0]]))

    # Recall
    dists_real_to_gen, idx_real_to_gen = nn_gen.kneighbors(real_features, n_neighbors=1)
    recall = float(np.mean(dists_real_to_gen[:, 0] <= gen_radii[idx_real_to_gen[:, 0]]))

    return precision, recall


METHODS_TO_TEST = [
    ("fm", 50, "FM@50"),
    ("shortcut", 4, "Shortcut@4"),
    ("shortcut", 128, "Shortcut@128"),
    ("consistency", 1, "Consistency@1"),
    ("consistency", 4, "Consistency@4"),
    ("consistency", 50, "Consistency@50"),
]


def main():
    parser = argparse.ArgumentParser(description="K-sensitivity for Precision/Recall")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else run_dir

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Results dir: {data_dir}")
    print(f"K values: {K_VALUES}")

    # Load VQ-GAN
    enc, dec, vq, _ = load_vqgan(data_dir / "phase1_shared.pt", dev)

    # Load real data
    vols = torch.load(args.data_path, weights_only=True)
    if vols.dim() == 4: vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} real volumes")

    print("Extracting real features...")
    real_features = extract_features(vols, enc, dev)
    print(f"  Shape: {real_features.shape}")

    out_dir = data_dir / "precision_recall"
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    for method, steps, label in METHODS_TO_TEST:
        ckpt = data_dir / method / "final.pt"
        if not ckpt.exists():
            print(f"Skipping {label}"); continue

        unet = load_unet(ckpt, dev)
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

        all_results[label] = {}

        for k in K_VALUES:
            seed_p, seed_r = [], []
            for seed in SEEDS:
                torch.manual_seed(seed)
                if dev.type == "mps": torch.mps.manual_seed(seed)
                np.random.seed(seed)

                # Generate
                all_gen = []
                batch = min(8, args.n_samples)
                with torch.no_grad():
                    for start in range(0, args.n_samples, batch):
                        bs = min(batch, args.n_samples - start)
                        z = sample_latent(unet, method, bs, dev, steps)
                        gen = dec(z).clamp(0, 1)
                        all_gen.append(gen.cpu())
                gen_vols = torch.cat(all_gen, 0)
                gen_features = extract_features(gen_vols, enc, dev)

                p, r = kynkaanniemi_precision_recall(real_features, gen_features, k=k)
                seed_p.append(p)
                seed_r.append(r)

            all_results[label][k] = {
                "precision_mean": float(np.mean(seed_p)),
                "precision_std": float(np.std(seed_p)),
                "recall_mean": float(np.mean(seed_r)),
                "recall_std": float(np.std(seed_r)),
                "precision_seeds": seed_p,
                "recall_seeds": seed_r,
            }
            print(f"  k={k:>2}: P={np.mean(seed_p):.3f}±{np.std(seed_p):.3f} "
                  f"R={np.mean(seed_r):.3f}±{np.std(seed_r):.3f}")

        del unet
        if dev.type == "mps": torch.mps.empty_cache()

    # Save
    with open(out_dir / "k_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Figure: Recall vs k for each method
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "FM@50": "#3498db",
        "Shortcut@4": "#9b59b6",
        "Shortcut@128": "#8e44ad",
        "Consistency@1": "#e74c3c",
        "Consistency@4": "#c0392b",
        "Consistency@50": "#a93226",
    }
    markers = {
        "FM@50": "o", "Shortcut@4": "D", "Shortcut@128": "d",
        "Consistency@1": "^", "Consistency@4": "v", "Consistency@50": "s",
    }

    # Panel A: Recall vs k
    ax = axes[0]
    for label in all_results:
        ks = sorted(all_results[label].keys())
        recalls = [all_results[label][k]["recall_mean"] for k in ks]
        stds = [all_results[label][k]["recall_std"] for k in ks]
        ax.errorbar(ks, recalls, yerr=stds, label=label,
                    color=colors.get(label, "#999"),
                    marker=markers.get(label, "o"),
                    linewidth=2, capsize=4, markersize=7)
    ax.set_xlabel("k (nearest neighbors)", fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title("Recall vs k — Sensitivity Analysis", fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)

    # Panel B: Precision vs k
    ax = axes[1]
    for label in all_results:
        ks = sorted(all_results[label].keys())
        precs = [all_results[label][k]["precision_mean"] for k in ks]
        stds = [all_results[label][k]["precision_std"] for k in ks]
        ax.errorbar(ks, precs, yerr=stds, label=label,
                    color=colors.get(label, "#999"),
                    marker=markers.get(label, "o"),
                    linewidth=2, capsize=4, markersize=7)
    ax.set_xlabel("k (nearest neighbors)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs k — Sensitivity Analysis", fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_k_sensitivity.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / "fig_k_sensitivity.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  K-SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<20} | " + " | ".join([f"k={k}" for k in K_VALUES]))
    print(f"  {'-'*20}-+-" + "-+-".join([f"-----" for _ in K_VALUES]))
    for label in all_results:
        recalls = [f"{all_results[label][k]['recall_mean']:.3f}" for k in K_VALUES]
        print(f"  {label:<20} | " + " | ".join(recalls))

    print(f"\nSaved to {out_dir}/k_sensitivity.json and fig_k_sensitivity.pdf")


if __name__ == "__main__":
    main()
