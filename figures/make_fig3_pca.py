"""
Figure 3: Feature-space PCA projections.
Shows how FM@50 and Shortcut@4 scatter across the real distribution while
Consistency@4 collapses into a tight cluster.

Usage:
    python make_fig3_pca.py \
        --run-dir results/ixi_benchmark_20260221_045741/100pct_200vol \
        --output-dir figures/

Expects: radiomic feature arrays saved during the coverage analysis, one .npy
file per method at {run_dir}/coverage_analysis/{method}_radiomic_features.npy
and real_radiomic_features.npy for the reference set.

If the .npy files don't exist, set --extract-from-volumes to regenerate them
from the saved sample volumes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


METHODS_TO_PLOT = [
    ("fm_50", "FM @ 50 steps", "#2E75B6"),
    ("shortcut_4", "Shortcut FM @ 4 steps", "#7030A0"),
    ("consistency_4", "Consistency @ 4 steps", "#C00000"),
]


def load_features(coverage_dir: Path, tag: str):
    path = coverage_dir / f"{tag}_radiomic_features.npy"
    if not path.exists():
        print(f"[warn] {path} not found — skipping {tag}")
        return None
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir = args.run_dir / "coverage_analysis"

    real_features = load_features(coverage_dir, "real")
    if real_features is None:
        raise SystemExit(
            f"Cannot find real_radiomic_features.npy in {coverage_dir}.\n"
            "Generate radiomic features first (see precision_recall_metrics.py)."
        )

    # Fit PCA on real features and project everything into the same 2D space.
    pca = PCA(n_components=2)
    real_2d = pca.fit_transform(real_features)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)

    for ax, (tag, label, color) in zip(axes, METHODS_TO_PLOT):
        synth = load_features(coverage_dir, tag)
        if synth is None:
            ax.set_title(f"{label}\n(features not found)", fontsize=10)
            ax.set_axis_off()
            continue

        synth_2d = pca.transform(synth)

        # Real points in grey background
        ax.scatter(
            real_2d[:, 0], real_2d[:, 1],
            c="lightgrey",
            s=22,
            alpha=0.7,
            label="Real",
            edgecolors="none",
        )
        # Synthetic points colored
        ax.scatter(
            synth_2d[:, 0], synth_2d[:, 1],
            c=color,
            s=30,
            alpha=0.85,
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )

        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "Feature-space PCA: synthetic samples projected into the real data's principal-component space",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    out_pdf = args.output_dir / "fig3_feature_pca.pdf"
    out_png = args.output_dir / "fig3_feature_pca.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved {out_pdf} and {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
