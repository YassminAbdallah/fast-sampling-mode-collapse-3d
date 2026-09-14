#!/usr/bin/env python3
"""
Standard Precision/Recall/Density/Coverage Metrics
====================================================

Implements the established manifold metrics:
  - Precision & Recall (Kynkäänniemi et al., NeurIPS 2019)
  - Density & Coverage (Naeem et al., ICML 2020)

Both computed in VQ-GAN encoder feature space (flattened 8³×8 = 4096-dim).

These complement the existing custom coverage from feature_diversity.py
with metrics that have published definitions and are directly comparable
to other papers.

Usage:
    python precision_recall_metrics.py \
        --run-dir results/ixi_benchmark_20260221_045741 \
        --data-path data/ixi_preprocessed_64.pt \
        --n-samples 64 --seeds 42 123 456 789 1337

    python precision_recall_metrics.py \
        --run-dir results/brats_benchmark_20260221_193306 \
        --data-path data/brats_preprocessed_64.pt \
        --n-samples 64 --seeds 42 123 456 789 1337

Output:
    {run-dir}/{data_subdir}/precision_recall/
        precision_recall_results.json
        fig_precision_recall.pdf
        table_precision_recall.tex

Requires: models_shared.py in the same directory.

References:
    Kynkäänniemi et al. (2019). "Improved Precision and Recall Metric
        for Assessing Generative Models." NeurIPS 2019.
    Naeem et al. (2020). "Reliable Fidelity and Diversity Metrics
        for Generative Models." ICML 2020.
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


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(volumes, enc, dev, batch_size=8):
    """Encode volumes through VQ-GAN encoder → flatten to feature vectors.
    
    Input:  (N, 1, 64, 64, 64)
    Output: (N, 4096)  — flattened latent 8×8×8×8
    """
    features = []
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size].to(dev)
        z = enc(batch)  # (B, 8, 8, 8, 8)
        features.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(features, axis=0)


# ============================================================
# Kynkäänniemi et al. (2019): Precision & Recall
# ============================================================
# 
# Precision: fraction of generated samples falling within the
#   support of the real distribution (estimated via k-NN manifolds)
# Recall: fraction of real samples falling within the support of
#   the generated distribution
#
# Algorithm:
#   1. For each real sample, find its k-th nearest real neighbor → radius r_i
#   2. Precision = fraction of generated samples that fall within
#      at least one real sample's k-NN ball
#   3. Similarly for recall, with roles swapped

def kynkaanniemi_precision_recall(real_features, gen_features, k=3):
    """
    Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019).
    
    Args:
        real_features: (N_r, D) numpy array
        gen_features:  (N_g, D) numpy array
        k: number of nearest neighbors for manifold estimation
    
    Returns:
        precision: float in [0, 1]
        recall:    float in [0, 1]
    """
    # Real manifold: k-NN distances among real samples
    nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(real_features)
    real_dists, _ = nn_real.kneighbors(real_features)
    # k-th neighbor distance (index k because index 0 is self)
    real_radii = real_dists[:, k]

    # Gen manifold: k-NN distances among generated samples
    nn_gen = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(gen_features)
    gen_dists, _ = nn_gen.kneighbors(gen_features)
    gen_radii = gen_dists[:, k]

    # Precision: for each gen sample, check if it falls inside any real ball
    # Find distance from each gen sample to nearest real sample
    dists_gen_to_real, idx_gen_to_real = nn_real.kneighbors(gen_features, n_neighbors=1)
    dists_gen_to_real = dists_gen_to_real[:, 0]
    nearest_real_idx = idx_gen_to_real[:, 0]
    precision = float(np.mean(dists_gen_to_real <= real_radii[nearest_real_idx]))

    # Recall: for each real sample, check if it falls inside any gen ball
    dists_real_to_gen, idx_real_to_gen = nn_gen.kneighbors(real_features, n_neighbors=1)
    dists_real_to_gen = dists_real_to_gen[:, 0]
    nearest_gen_idx = idx_real_to_gen[:, 0]
    recall = float(np.mean(dists_real_to_gen <= gen_radii[nearest_gen_idx]))

    return precision, recall


# ============================================================
# Naeem et al. (2020): Density & Coverage
# ============================================================
#
# Coverage: fraction of real samples whose k-NN ball contains
#   at least one generated sample. Similar to recall but uses
#   real-sample radii only.
# Density: average number of generated samples within each real
#   sample's k-NN ball, normalized by k. Measures how well
#   generated samples concentrate around real data.

def naeem_density_coverage(real_features, gen_features, k=5):
    """
    Density and Coverage (Naeem et al., ICML 2020).
    
    Args:
        real_features: (N_r, D) numpy array
        gen_features:  (N_g, D) numpy array
        k: number of nearest neighbors for radius estimation
    
    Returns:
        density:  float ≥ 0
        coverage: float in [0, 1]
    """
    n_real = len(real_features)
    n_gen = len(gen_features)

    # Compute k-NN radii from real samples
    nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(real_features)
    real_dists, _ = nn_real.kneighbors(real_features)
    real_radii = real_dists[:, k]  # k-th neighbor distance (excluding self)

    # Coverage: fraction of real samples with ≥1 gen neighbor within radius
    # Density: average count of gen samples within each real ball / k
    nn_real_for_gen = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(real_features)

    # For each real sample, count how many gen samples fall within its ball
    coverage_count = 0
    density_sum = 0.0

    # Efficient: compute distances from all gen to all real
    # For large N this should use batching, but 64-200 samples is fine
    for i in range(n_real):
        dists = np.linalg.norm(gen_features - real_features[i:i+1], axis=1)
        count_in_ball = int(np.sum(dists <= real_radii[i]))
        if count_in_ball > 0:
            coverage_count += 1
        density_sum += count_in_ball

    coverage = coverage_count / n_real
    density = density_sum / (n_real * k)

    return float(density), float(coverage)


# ============================================================
# Methods to evaluate
# ============================================================

METHODS_TO_TEST = [
    ("fm", [10, 25, 50]),
    ("rectified", [5, 10, 25]),
    ("consistency", [1, 4, 16, 50]),
    ("shortcut", [1, 4, 16, 50, 128]),
    ("ddpm", [1000]),
]


def main():
    parser = argparse.ArgumentParser(description="Standard Precision/Recall/Density/Coverage metrics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1337])
    parser.add_argument("--k-pr", type=int, default=3, help="k for Precision/Recall (Kynkäänniemi)")
    parser.add_argument("--k-dc", type=int, default=5, help="k for Density/Coverage (Naeem)")
    parser.add_argument("--conditional", action="store_true",
                        help="Conditional mode: data-path is a conditional dataset")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of conditioning classes (default 2)")
    parser.add_argument("--class-label", type=int, default=None,
                        help="Class label for conditional generation (default: None = mixed)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else run_dir

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Results dir: {data_dir}")
    print(f"Seeds: {args.seeds}")

    # Load VQ-GAN
    enc, dec, vq, _ = load_vqgan(data_dir / "phase1_shared.pt", dev)

    # Load real data
    num_classes = 0
    class_label = args.class_label

    if args.conditional:
        num_classes = args.num_classes
        cond_data = torch.load(args.data_path, weights_only=False)
        vols = cond_data["volumes"]
        labels = cond_data["labels"]
        train_idx = cond_data["split_info"]["train_idx"]
        vols = vols[train_idx]
        labels = labels[train_idx]
        print(f"Conditional mode: {len(vols)} training volumes, {num_classes} classes")
        if class_label is not None:
            print(f"  Generating class: {class_label}")
    else:
        vols = torch.load(args.data_path, weights_only=True)

    if vols.dim() == 4: vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} real volumes")

    print("Extracting real features...")
    real_features = extract_features(vols, enc, dev)
    print(f"  Real feature shape: {real_features.shape}")

    out_dir = data_dir / ("precision_recall_cond" if args.conditional else "precision_recall")
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    for method, step_counts in METHODS_TO_TEST:
        ckpt_path = data_dir / method / "final.pt"
        if not ckpt_path.exists():
            print(f"Skipping {method} (no checkpoint)")
            continue

        unet = load_unet(ckpt_path, dev, num_classes=num_classes)

        for steps in step_counts:
            label = f"{method}@{steps}"
            print(f"\n{'='*50}")
            print(f"  {label}")
            print(f"{'='*50}")

            seed_results = {
                "precision": [], "recall": [],
                "density": [], "coverage": [],
            }

            for seed in args.seeds:
                torch.manual_seed(seed)
                if dev.type == "mps": torch.mps.manual_seed(seed)
                elif dev.type == "cuda": torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                # Generate samples
                all_gen = []
                batch = min(8, args.n_samples)
                with torch.no_grad():
                    for start in range(0, args.n_samples, batch):
                        bs = min(batch, args.n_samples - start)
                        z = sample_latent(unet, method, bs, dev, steps,
                                          class_label=class_label)
                        gen = dec(z).clamp(0, 1)
                        all_gen.append(gen.cpu())
                gen_vols = torch.cat(all_gen, 0)

                # Extract generated features
                gen_features = extract_features(gen_vols, enc, dev)

                # Kynkäänniemi Precision/Recall
                prec, rec = kynkaanniemi_precision_recall(
                    real_features, gen_features, k=args.k_pr)
                seed_results["precision"].append(prec)
                seed_results["recall"].append(rec)

                # Naeem Density/Coverage
                dens, cov = naeem_density_coverage(
                    real_features, gen_features, k=args.k_dc)
                seed_results["density"].append(dens)
                seed_results["coverage"].append(cov)

                print(f"    Seed {seed}: P={prec:.3f} R={rec:.3f} D={dens:.3f} C={cov:.3f}")

            # Aggregate
            result = {}
            for metric in seed_results:
                vals = seed_results[metric]
                result[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "seeds": vals,
                }
            all_results[label] = result

            print(f"  → P={result['precision']['mean']:.3f}±{result['precision']['std']:.3f} "
                  f"R={result['recall']['mean']:.3f}±{result['recall']['std']:.3f} "
                  f"D={result['density']['mean']:.3f}±{result['density']['std']:.3f} "
                  f"C={result['coverage']['mean']:.3f}±{result['coverage']['std']:.3f}")

        del unet
        if dev.type == "mps": torch.mps.empty_cache()
        elif dev.type == "cuda": torch.cuda.empty_cache()

    # Save results
    save_data = {
        "metadata": {
            "seeds": args.seeds,
            "n_samples": args.n_samples,
            "k_precision_recall": args.k_pr,
            "k_density_coverage": args.k_dc,
            "feature_dim": real_features.shape[1],
            "n_real": len(real_features),
        },
        "results": all_results,
    }
    with open(out_dir / "precision_recall_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # ============================================================
    # Figures
    # ============================================================

    # Figure: Precision vs Recall scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Color and marker maps
    method_styles = {
        "fm": ("#3498db", "o", "FM"),
        "rectified": ("#2ecc71", "s", "Rectified"),
        "consistency": ("#e74c3c", "^", "Consistency"),
        "shortcut": ("#9b59b6", "D", "Shortcut FM"),
        "ddpm": ("#f39c12", "v", "DDPM"),
    }

    # Panel A: Precision vs Recall
    ax = axes[0]
    for label, result in all_results.items():
        method = label.split("@")[0]
        steps = label.split("@")[1]
        color, marker, mname = method_styles.get(method, ("#999", "o", method))
        ax.scatter(result["recall"]["mean"], result["precision"]["mean"],
                   c=color, marker=marker, s=80, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(steps, (result["recall"]["mean"], result["precision"]["mean"]),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')

    # Legend (one entry per method)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                              markersize=8, label=n)
                       for _, (c, m, n) in method_styles.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower left')
    ax.set_xlabel("Recall (Kynkäänniemi et al., 2019)", fontsize=10)
    ax.set_ylabel("Precision (Kynkäänniemi et al., 2019)", fontsize=10)
    ax.set_title("Precision vs Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Density vs Coverage
    ax = axes[1]
    for label, result in all_results.items():
        method = label.split("@")[0]
        steps = label.split("@")[1]
        color, marker, mname = method_styles.get(method, ("#999", "o", method))
        ax.scatter(result["coverage"]["mean"], result["density"]["mean"],
                   c=color, marker=marker, s=80, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(steps, (result["coverage"]["mean"], result["density"]["mean"]),
                    fontsize=7, ha='left', va='bottom', xytext=(3, 3),
                    textcoords='offset points')

    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    ax.set_xlabel("Coverage (Naeem et al., 2020)", fontsize=10)
    ax.set_ylabel("Density (Naeem et al., 2020)", fontsize=10)
    ax.set_title("Density vs Coverage")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_precision_recall.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / "fig_precision_recall.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

    # ============================================================
    # LaTeX table
    # ============================================================

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Standard generative quality and diversity metrics. "
        r"Precision/Recall: Kynk\"a\"anniemi et al.\ (2019), $k{=}" + str(args.k_pr) + r"$. "
        r"Density/Coverage: Naeem et al.\ (2020), $k{=}" + str(args.k_dc) + r"$. "
        r"Mean over " + str(len(args.seeds)) + r" seeds.}",
        r"\label{tab:precision_recall}",
        r"\footnotesize",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Precision $\uparrow$ & Recall $\uparrow$ & Density $\uparrow$ & Coverage $\uparrow$ \\",
        r"\midrule",
    ]

    for label in sorted(all_results.keys(), key=lambda x: (x.split("@")[0], int(x.split("@")[1]))):
        r = all_results[label]
        lines.append(
            f"  {label} & "
            f"{r['precision']['mean']:.3f}$\\pm${r['precision']['std']:.3f} & "
            f"{r['recall']['mean']:.3f}$\\pm${r['recall']['std']:.3f} & "
            f"{r['density']['mean']:.3f}$\\pm${r['density']['std']:.3f} & "
            f"{r['coverage']['mean']:.3f}$\\pm${r['coverage']['std']:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    (out_dir / "table_precision_recall.tex").write_text("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"  All results saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
