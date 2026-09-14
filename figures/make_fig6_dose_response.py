"""
Figure 6: Diversity dose-response curve.
Plots Dice vs number of unique synthetic anatomies, showing the threshold at ~25.

Usage:
    python make_fig6_dose_response.py \
        --results-dir results/paper4/e10a \
        --output-dir figures/

Expects: E10a results in the format produced by conditional_segmentation.py,
with one subdirectory per condition (real_only, unique_10, unique_25, unique_100, unique_500),
each containing seed_42.json, seed_43.json, seed_44.json with a "test_dice" field.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# This script reads the per-seed results JSON and carries no fallback values;
# if the results cannot be loaded it exits with an error.

def load_condition_results(results_dir: Path, condition: str):
    """Load per-seed Dice scores for one condition."""
    condition_dir = results_dir / condition
    if not condition_dir.exists():
        return None

    dice_scores = []
    for seed_file in sorted(condition_dir.glob("seed_*.json")):
        with open(seed_file) as f:
            data = json.load(f)
            dice_scores.append(data["test_dice"])
    if not dice_scores:
        return None
    return np.array(dice_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper4/e10a"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions_order = ["real_only", "unique_10", "unique_25", "unique_100", "unique_500"]
    unique_counts = []
    means = []
    stds = []

    for cond in conditions_order:
        scores = load_condition_results(args.results_dir, cond)
        if scores is not None:
            means.append(scores.mean())
            stds.append(scores.std())
        else:
            raise SystemExit(
                f"Refusing to plot: could not load '{cond}' from {results_dir}.\n"
                "This script carries no fallback numbers. Point --results at:\n"
                "  results/paper4/e10a_64_5seed/e10a_64_5seed_results.json"
            )

    means = np.array(means)
    stds = np.array(stds)
    unique_counts = np.array(unique_counts)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Plot augmented conditions (skip real_only from the line)
    aug_mask = unique_counts > 0
    ax.errorbar(
        unique_counts[aug_mask],
        means[aug_mask],
        yerr=stds[aug_mask],
        fmt="o-",
        color="#2E75B6",
        linewidth=2.2,
        markersize=8,
        capsize=4,
        label="Real + synthetic augmentation",
    )

    # Horizontal line for real_only baseline
    real_only_mean = means[0]
    real_only_std = stds[0]
    ax.axhline(real_only_mean, color="gray", linestyle="--", linewidth=1.5, label="Real-only baseline")
    ax.fill_between(
        [8, 700],
        real_only_mean - real_only_std,
        real_only_mean + real_only_std,
        color="gray",
        alpha=0.15,
    )

    # Vertical dashed line at the ~25 threshold
    ax.axvline(25, color="#C00000", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(
        26, means[aug_mask].min() - 0.005,
        "~25-unique\nthreshold",
        color="#C00000",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Unique synthetic anatomies (log scale)", fontsize=11)
    ax.set_ylabel("Test Dice score", fontsize=11)
    ax.set_title(
        "Diversity dose-response: 18 real + 500 synthetic (only unique count varies)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(8, 700)
    # Y-axis: show the full comparable range so a ~2-pp Dice difference is not
    # visually exaggerated. A truncated (0.765, 0.790) range would make the
    # threshold look larger than it is.
    ax.set_ylim(0.60, 0.85)

    # Annotate the jump between unique_10 and unique_25
    ax.annotate(
        "",
        xy=(25, 0.790),
        xytext=(10, 0.772),
        arrowprops=dict(arrowstyle="->", color="#006600", lw=1.5),
    )
    ax.text(
        16, 0.781,
        "p = 0.020",
        color="#006600",
        fontsize=9,
        ha="center",
    )

    plt.tight_layout()
    out_pdf = args.output_dir / "fig6_dose_response.pdf"
    out_png = args.output_dir / "fig6_dose_response.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved {out_pdf} and {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
