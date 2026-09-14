#!/usr/bin/env python3
"""
Feature Diversity Analysis
===========================

Proves that sample diversity translates to meaningful feature-space coverage.
Uses the trained VQ-GAN encoder as a feature extractor.

For each method: generate N samples → decode → encode to get features →
measure coverage of real feature space.

Usage:
    python feature_diversity.py --run-dir results/ixi_benchmark_20260221_045741 --data-path data/ixi_preprocessed_64.pt
    python feature_diversity.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt

Output:
    {run-dir}/feature_diversity/
        feature_diversity.json
        fig_feature_coverage.pdf
        fig_feature_pca.pdf
"""

import os, json, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

from models_shared import load_unet, load_vqgan, sample_latent, Decoder3D

METHODS_TO_TEST = [
    ("fm", 50, "FM@50"),
    ("shortcut", 1, "Shortcut@1"),
    ("shortcut", 4, "Shortcut@4"),
    ("shortcut", 16, "Shortcut@16"),
    ("shortcut", 128, "Shortcut@128"),
    ("consistency", 1, "Consistency@1"),
    ("consistency", 4, "Consistency@4"),
    ("consistency", 50, "Consistency@50"),
    ("rectified", 10, "Rectified@10"),
]


def extract_features(volumes, enc, dev, batch_size=4):
    """Encode volumes through VQ-GAN encoder, flatten to feature vectors."""
    features = []
    with torch.no_grad():
        for i in range(0, len(volumes), batch_size):
            z = enc(volumes[i:i+batch_size].to(dev))
            features.append(z.view(z.shape[0], -1).cpu())
    return torch.cat(features).numpy()


def compute_coverage(real_features, syn_features, k=5):
    nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(real_features)
    real_dists, _ = nn_real.kneighbors(real_features)
    r = np.median(real_dists[:, k])
    nn_syn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(syn_features)
    syn_dists, _ = nn_syn.kneighbors(real_features)
    return float((syn_dists[:, 0] <= r).mean()), float(r)


def compute_density(real_features, syn_features, k=5):
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    real_dists, _ = nn_real.kneighbors(real_features)
    r = np.median(real_dists[:, k-1])
    nn_real2 = NearestNeighbors(radius=r, metric='euclidean').fit(real_features)
    counts = nn_real2.radius_neighbors(syn_features, return_distance=False)
    return float(np.mean([len(c) for c in counts]) / k)


def pca_overlap(real_features, syn_features, n_components=10):
    pca_r = PCA(n_components=n_components).fit(real_features)
    pca_s = PCA(n_components=n_components).fit(syn_features)
    overlaps = [float(abs(np.dot(pca_r.components_[i], pca_s.components_[i]))) for i in range(n_components)]
    return {"component_overlaps": overlaps, "mean_overlap": float(np.mean(overlaps)),
            "real_explained_var": pca_r.explained_variance_ratio_.tolist(),
            "syn_explained_var": pca_s.explained_variance_ratio_.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else run_dir

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}, Results dir: {data_dir}")

    enc, dec, vq, p1_state = load_vqgan(data_dir / "phase1_shared.pt", dev)

    vols = torch.load(args.data_path, weights_only=True)
    if vols.dim() == 4: vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} real volumes")

    print("Extracting real features...")
    real_feats = extract_features(vols, enc, dev)
    print(f"  Real feature shape: {real_feats.shape}")

    out_dir = data_dir / "feature_diversity"; out_dir.mkdir(exist_ok=True)
    results = {}

    for method, steps, label in METHODS_TO_TEST:
        ckpt_path = data_dir / method / "final.pt"
        if not ckpt_path.exists():
            print(f"Skipping {label}"); continue

        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        unet = load_unet(ckpt_path, dev)

        print(f"  Generating {args.n_samples} samples...")
        all_vols = []
        batch = min(8, args.n_samples)
        with torch.no_grad():
            for start in range(0, args.n_samples, batch):
                bs = min(batch, args.n_samples - start)
                z = sample_latent(unet, method, bs, dev, steps)
                gen = dec(z).clamp(0, 1)
                all_vols.append(gen.cpu())
        syn_vols = torch.cat(all_vols)

        print(f"  Extracting synthetic features...")
        syn_feats = extract_features(syn_vols, enc, dev)

        print(f"  Computing metrics...")
        coverage, radius = compute_coverage(real_feats, syn_feats, k=5)
        density = compute_density(real_feats, syn_feats, k=5)
        pca_info = pca_overlap(real_feats, syn_feats, n_components=10)

        results[label] = {
            "coverage": coverage, "coverage_radius": radius, "density": density,
            "pca_mean_overlap": pca_info["mean_overlap"],
            "pca_component_overlaps": pca_info["component_overlaps"],
            "pca_syn_explained_var": pca_info["syn_explained_var"],
        }
        print(f"  Coverage: {coverage:.1%}  Density: {density:.2f}  PCA overlap: {pca_info['mean_overlap']:.3f}")
        del unet

    with open(out_dir / "feature_diversity.json", "w") as f:
        json.dump(results, f, indent=2)

    # ================================================================
    # Figure 1: Coverage + Density
    # ================================================================
    labels = list(results.keys())
    coverages = [results[l]["coverage"]*100 for l in labels]
    densities = [results[l]["density"] for l in labels]
    cmap = {}
    for l in labels:
        if l.startswith("FM"): cmap[l] = "#2196F3"
        elif l.startswith("Shortcut"): cmap[l] = "#FF9800"
        elif l.startswith("Consistency"): cmap[l] = "#F44336"
        elif l.startswith("Rectified"): cmap[l] = "#4CAF50"
        else: cmap[l] = "#999"
    colors = [cmap[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    bars1 = ax1.barh(range(len(labels)), coverages, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(labels))); ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Feature Coverage (%)"); ax1.set_title("Coverage of Real Feature Space", fontweight='bold')
    for b,v in zip(bars1, coverages): ax1.text(v+1, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
    ax1.invert_yaxis()

    bars2 = ax2.barh(range(len(labels)), densities, color=colors, edgecolor='white')
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Feature Density"); ax2.set_title("Density in Real Feature Space", fontweight='bold')
    for b,v in zip(bars2, densities): ax2.text(v+0.02, b.get_y()+b.get_height()/2, f'{v:.2f}', va='center', fontsize=9)
    ax2.invert_yaxis()
    plt.tight_layout()
    for ext in ['pdf','png']: plt.savefig(out_dir/f'fig_feature_coverage.{ext}', dpi=200, bbox_inches='tight')
    plt.close()

    # ================================================================
    # Figure 2: PCA 2D projection
    # ================================================================
    pca = PCA(n_components=2).fit(real_feats)
    real_2d = pca.transform(real_feats)
    key_methods = ["FM@50", "Shortcut@4", "Consistency@4"]
    fig, axes = plt.subplots(1, len(key_methods)+1, figsize=(5*(len(key_methods)+1), 5))

    axes[0].scatter(real_2d[:,0], real_2d[:,1], c='gray', alpha=0.5, s=20)
    axes[0].set_title("Real Data", fontweight='bold'); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    for idx, ml in enumerate(key_methods):
        ax = axes[idx+1]
        ax.scatter(real_2d[:,0], real_2d[:,1], c='gray', alpha=0.2, s=15)
        method = ml.split("@")[0].lower(); steps = int(ml.split("@")[1])
        ckpt_path = data_dir / method / "final.pt"
        if ckpt_path.exists() and ml in results:
            unet = load_unet(ckpt_path, dev)
            with torch.no_grad():
                z = sample_latent(unet, method, min(args.n_samples, 64), dev, steps)
                syn_v = dec(z).clamp(0, 1)
                syn_f = extract_features(syn_v, enc, dev)
                syn_2d = pca.transform(syn_f)
            del unet
            ax.scatter(syn_2d[:,0], syn_2d[:,1], c=cmap.get(ml,'blue'), alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
            cov = results[ml]["coverage"]*100
            ax.text(0.05, 0.95, f"Coverage: {cov:.0f}%", transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(ml, fontweight='bold'); ax.set_xlabel("PC1")

    plt.tight_layout()
    for ext in ['pdf','png']: plt.savefig(out_dir/f'fig_feature_pca.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
