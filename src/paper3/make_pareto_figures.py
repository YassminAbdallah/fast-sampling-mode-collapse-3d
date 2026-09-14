#!/usr/bin/env python3
"""
Regenerate Pareto Frontier Figures
====================================

Generates three versions of the quality-diversity Pareto plot:
  1. Fig. 1 (main): SSIM vs Diversity (original, kept for familiarity)
  2. Fig. S_frd: FRD vs Diversity (supplementary)
  3. Fig. S_lfid: Latent FID vs Diversity (supplementary, alternative)

Reads from:
  - eval_results.json (or eval_results_5seed.json): SSIM, diversity
  - distributional_results.json: FRD, Latent FID

Usage:
    # IXI
    python make_pareto_figures.py \
        --eval-json results/ixi_benchmark_20260221_045741/100pct_200vol/paper_eval_5seed/eval_results_5seed.json \
        --dist-json results/ixi_benchmark_20260221_045741/100pct_200vol/distributional_metrics/distributional_results.json \
        --real-div 0.0838 \
        --dataset-name "IXI (200 healthy brain volumes)" \
        --out-dir results/ixi_benchmark_20260221_045741/100pct_200vol/paper_figures

    # BraTS
    python make_pareto_figures.py \
        --eval-json results/brats_benchmark_20260221_193306/100pct_300vol/paper_eval_5seed/eval_results_5seed.json \
        --dist-json results/brats_benchmark_20260221_193306/100pct_300vol/distributional_metrics/distributional_results.json \
        --real-div 0.0500 \
        --dataset-name "BraTS 2023 (300 glioma volumes)" \
        --out-dir results/brats_benchmark_20260221_193306/100pct_300vol/paper_figures
"""

import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# Style config
# ============================================================

METHOD_STYLES = {
    "ddpm":        {"color": "#f39c12", "marker": "v",  "label": "DDPM",           "zorder": 3},
    "fm":          {"color": "#3498db", "marker": "o",  "label": "Flow Matching",  "zorder": 4},
    "rectified":   {"color": "#2ecc71", "marker": "s",  "label": "Rectified Flow", "zorder": 4},
    "consistency": {"color": "#e74c3c", "marker": "^",  "label": "Consistency",    "zorder": 5},
    "shortcut":    {"color": "#9b59b6", "marker": "D",  "label": "Shortcut FM",    "zorder": 6},
}


def load_eval_data(eval_path):
    """Load eval results. Handles both 3-seed and 5-seed formats."""
    with open(eval_path) as f:
        data = json.load(f)
    # 5-seed format has "results" key
    if "results" in data:
        return data["results"]
    return data


def load_dist_data(dist_path):
    """Load distributional results."""
    with open(dist_path) as f:
        return json.load(f)


def get_diversity_pct(div_val, real_div):
    """Convert raw diversity to % of real."""
    return (div_val / real_div) * 100 if real_div > 0 else 0


def make_ssim_pareto(eval_data, real_div, dataset_name, out_dir):
    """Fig. 1: SSIM vs Diversity (% of real) — main paper figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: SSIM vs Diversity scatter
    ax = axes[0]

    # Add collapse zone
    ax.axvspan(0, 50, alpha=0.08, color='red', label='Collapse zone (<50%)')
    ax.axvline(x=50, color='red', alpha=0.3, linestyle='--', linewidth=0.8)

    for method, style in METHOD_STYLES.items():
        if method not in eval_data:
            continue
        steps_data = eval_data[method]
        for steps_str, metrics in sorted(steps_data.items(), key=lambda x: int(x[0])):
            steps = int(steps_str)

            ssim = metrics["ssim"]["mean"] if isinstance(metrics["ssim"], dict) else metrics["ssim"]
            div_raw = metrics["diversity"]["mean"] if isinstance(metrics["diversity"], dict) else metrics["diversity"]
            div_pct = get_diversity_pct(div_raw, real_div)

            # Skip DDPM@50 and @200 (produce noise, not real samples)
            if method == "ddpm" and steps < 1000:
                continue

            ax.scatter(div_pct, ssim, c=style["color"], marker=style["marker"],
                       s=90, zorder=style["zorder"], edgecolors='white', linewidths=0.5)
            ax.annotate(str(steps), (div_pct, ssim), fontsize=7,
                        ha='left', va='bottom', xytext=(3, 3),
                        textcoords='offset points')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=s["marker"], color='w',
                              markerfacecolor=s["color"], markersize=8, label=s["label"])
                       for _, s in METHOD_STYLES.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower left')
    ax.set_xlabel("Sample Diversity (% of real data)", fontsize=11)
    ax.set_ylabel("SSIM ↑", fontsize=11)
    # Widen the SSIM axis so a small (~0.03) inter-method SSIM difference is not
    # visually exaggerated by an auto-scaled truncated axis.
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 105.0)
    ax.set_title(f"Quality–Diversity Frontier", fontsize=12)
    ax.grid(True, alpha=0.2)

    # Panel B: Diversity vs NFE
    ax = axes[1]
    for method, style in METHOD_STYLES.items():
        if method not in eval_data:
            continue
        steps_data = eval_data[method]
        nfes, divs = [], []
        for steps_str, metrics in sorted(steps_data.items(), key=lambda x: int(x[0])):
            steps = int(steps_str)
            if method == "ddpm" and steps < 1000:
                continue
            div_raw = metrics["diversity"]["mean"] if isinstance(metrics["diversity"], dict) else metrics["diversity"]
            nfes.append(steps)
            divs.append(get_diversity_pct(div_raw, real_div))

        if nfes:
            ax.plot(nfes, divs, color=style["color"], marker=style["marker"],
                    markersize=7, linewidth=2, label=style["label"],
                    zorder=style["zorder"], markeredgecolor='white', markeredgewidth=0.5)

    ax.axhline(y=50, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=11)
    ax.set_ylabel("Diversity (% of real data)", fontsize=11)
    ax.set_title("Diversity vs Inference Steps", fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)

    plt.suptitle(dataset_name, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_ssim_pareto.pdf", bbox_inches='tight', dpi=200)
    fig.savefig(out_dir / "fig1_ssim_pareto.png", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved fig1_ssim_pareto.pdf/png")


def make_frd_pareto(eval_data, dist_data, real_div, dataset_name, out_dir):
    """Supplementary: FRD vs Diversity — distributional quality on y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: FRD vs Diversity scatter
    ax = axes[0]
    ax.axvspan(0, 50, alpha=0.08, color='red')
    ax.axvline(x=50, color='red', alpha=0.3, linestyle='--', linewidth=0.8)

    for method, style in METHOD_STYLES.items():
        if method not in dist_data:
            continue
        for steps_str, d_metrics in sorted(dist_data[method].items(), key=lambda x: int(x[0])):
            steps = int(steps_str)
            frd = d_metrics["frd"]

            # Get matching diversity from eval_data
            if method in eval_data and steps_str in eval_data[method]:
                e_metrics = eval_data[method][steps_str]
                div_raw = e_metrics["diversity"]["mean"] if isinstance(e_metrics["diversity"], dict) else e_metrics["diversity"]
                div_pct = get_diversity_pct(div_raw, real_div)
            else:
                continue

            ax.scatter(div_pct, frd, c=style["color"], marker=style["marker"],
                       s=90, zorder=style["zorder"], edgecolors='white', linewidths=0.5)
            ax.annotate(str(steps), (div_pct, frd), fontsize=7,
                        ha='left', va='bottom', xytext=(3, 3),
                        textcoords='offset points')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=s["marker"], color='w',
                              markerfacecolor=s["color"], markersize=8, label=s["label"])
                       for _, s in METHOD_STYLES.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    ax.set_xlabel("Sample Diversity (% of real data)", fontsize=11)
    ax.set_ylabel("FRD ↓ (lower is better)", fontsize=11)
    ax.set_title("Distributional Quality–Diversity Frontier", fontsize=12)
    ax.grid(True, alpha=0.2)

    # Panel B: Latent FID vs Diversity
    ax = axes[1]
    ax.axvspan(0, 50, alpha=0.08, color='red')
    ax.axvline(x=50, color='red', alpha=0.3, linestyle='--', linewidth=0.8)

    for method, style in METHOD_STYLES.items():
        if method not in dist_data:
            continue
        for steps_str, d_metrics in sorted(dist_data[method].items(), key=lambda x: int(x[0])):
            steps = int(steps_str)
            lfid = d_metrics["latent_fid"]

            if method in eval_data and steps_str in eval_data[method]:
                e_metrics = eval_data[method][steps_str]
                div_raw = e_metrics["diversity"]["mean"] if isinstance(e_metrics["diversity"], dict) else e_metrics["diversity"]
                div_pct = get_diversity_pct(div_raw, real_div)
            else:
                continue

            ax.scatter(div_pct, lfid, c=style["color"], marker=style["marker"],
                       s=90, zorder=style["zorder"], edgecolors='white', linewidths=0.5)
            ax.annotate(str(steps), (div_pct, lfid), fontsize=7,
                        ha='left', va='bottom', xytext=(3, 3),
                        textcoords='offset points')

    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
    ax.set_xlabel("Sample Diversity (% of real data)", fontsize=11)
    ax.set_ylabel("Latent FID ↓ (lower is better)", fontsize=11)
    ax.set_title("Latent FID–Diversity Frontier", fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.suptitle(dataset_name, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig_supp_frd_pareto.pdf", bbox_inches='tight', dpi=200)
    fig.savefig(out_dir / "fig_supp_frd_pareto.png", bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  Saved fig_supp_frd_pareto.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Pareto frontier figures")
    parser.add_argument("--eval-json", required=True, help="Path to eval_results.json or eval_results_5seed.json")
    parser.add_argument("--dist-json", required=True, help="Path to distributional_results.json")
    parser.add_argument("--real-div", type=float, required=True, help="Real data diversity value")
    parser.add_argument("--dataset-name", required=True, help="Dataset label for figure title")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval data from {args.eval_json}")
    eval_data = load_eval_data(args.eval_json)

    print(f"Loading distributional data from {args.dist_json}")
    dist_data = load_dist_data(args.dist_json)

    print(f"Real diversity: {args.real_div}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {out_dir}")

    # Generate main SSIM-based Pareto (Fig. 1)
    print("\nGenerating SSIM-based Pareto (Fig. 1)...")
    make_ssim_pareto(eval_data, args.real_div, args.dataset_name, out_dir)

    # Generate FRD-based Pareto (Supplementary)
    print("\nGenerating FRD/Latent FID Pareto (Supplementary)...")
    make_frd_pareto(eval_data, dist_data, args.real_div, args.dataset_name, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
