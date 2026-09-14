#!/usr/bin/env python3
"""
Visual verification of candidate near-copies (left-tail check).

For the Consistency Distillation synthetic samples whose nearest-neighbor
distance to the real training set falls below the real-to-real LOO minimum
(the "candidate near-copy" set surfaced by memorization_nn.py), this script
renders each candidate synthetic volume next to its nearest real volume in
three orthogonal mid-slices (axial, coronal, sagittal).

The decision after visual inspection:

  * If candidates LOOK like the same anatomy as their NN real volumes
    (lesion in same place, ventricle shape matches, sulci align) →
    write "memorization" concretely in §6.2.

  * If candidates look like *similar* brains that happen to be close in
    feature space (generic similarity, not anatomy-specific) →
    soften wording to "candidate near-duplicates in VQ-GAN feature space"
    and keep the §6.2 framing as "concentration onto a small number of
    specific training patients" (mode-collapse geometry, not patient
    copy reproduction).

Reads: results/analyses/memorization_nn.csv (already produced)
       results/paper4/brats_128cubed_conditional/e10b_128/syn_pool_consistency_4.pt
       data/brats_conditional_128.pt

Writes:
    results/analyses/memorization_visual_check.png
    results/analyses/memorization_visual_check.csv  (sub-table of candidates)

Usage:
    cd fast-sampling-mode-collapse-3d/
    python tier_c/analyses/memorization_visual_check.py

    # Defaults to the consistency_4 method and the real-to-real-min threshold.
    # Override:
    python tier_c/analyses/memorization_visual_check.py \\
        --method consistency_4 --threshold 13.328  # use p5 instead of min
    python tier_c/analyses/memorization_visual_check.py --max-rows 20
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"


def mid_slices(vol):
    """Return (axial, coronal, sagittal) mid-slices from a volume of shape
    (1, D, H, W) or (D, H, W)."""
    if vol.ndim == 4:
        vol = vol[0]
    D, H, W = vol.shape
    return {
        "axial":    vol[D // 2, :, :],
        "coronal":  vol[:, H // 2, :],
        "sagittal": vol[:, :, W // 2],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="consistency_4",
                   choices=["consistency_4", "shortcut_50"])
    p.add_argument("--csv", default="results/analyses/memorization_nn.csv")
    p.add_argument("--summary-json",
                   default="results/analyses/memorization_nn.json")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold; default = real-to-real LOO minimum "
                        "from the summary JSON.")
    p.add_argument("--use-p5", action="store_true",
                   help="Use real-to-real p5 instead of min as threshold "
                        "(broader, ~15%% of synthetic pool).")
    p.add_argument("--syn-pool",
                   default="results/paper4/brats_128cubed_conditional"
                           "/e10b_128/syn_pool_consistency_4.pt")
    p.add_argument("--data",
                   default="data/brats_conditional_128.pt")
    p.add_argument("--output-dir", default="results/analyses")
    p.add_argument("--max-rows", type=int, default=20,
                   help="Max number of candidate pairs to render. The full "
                        "list still gets dumped to CSV.")
    args = p.parse_args()

    csv_path = REPO / args.csv
    json_path = REPO / args.summary_json
    if not csv_path.exists():
        sys.exit(f"ERROR: missing {csv_path}\nRun memorization_nn.py first.")
    if not json_path.exists():
        sys.exit(f"ERROR: missing {json_path}\nRun memorization_nn.py first.")

    # ----- Decide threshold -----
    with open(json_path) as f:
        summary = json.load(f)
    base = summary.get("_real_to_real_baseline", {})
    if not base:
        sys.exit("ERROR: summary JSON missing _real_to_real_baseline. Re-run memorization_nn.py.")
    if args.threshold is not None:
        thresh = args.threshold
        thresh_label = f"explicit threshold = {thresh:.3f}"
    elif args.use_p5:
        thresh = base["p5"]
        thresh_label = f"real-to-real p5 = {thresh:.3f}"
    else:
        thresh = base["min"]
        thresh_label = f"real-to-real min = {thresh:.3f}"
    print(f"Threshold: {thresh_label}")

    # ----- Read per-volume NN data -----
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr if r["method"] == args.method]
    print(f"Total {args.method} synthetic samples: {len(rows)}")

    candidates = []
    for r in rows:
        d = float(r["nn_distance_to_real_train"])
        if d < thresh:
            candidates.append({
                "synthetic_idx": int(r["synthetic_idx"]),
                "real_idx": int(r["nn_real_train_index"]),
                "nn_distance": d,
            })
    candidates.sort(key=lambda c: c["nn_distance"])
    print(f"Candidates below threshold: {len(candidates)} / {len(rows)}")

    if not candidates:
        sys.exit("No candidates below threshold. Nothing to render.")

    # Save the candidate sub-table
    out_dir = Path(REPO / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = out_dir / "memorization_visual_check.csv"
    with open(sub_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["synthetic_idx", "real_idx", "nn_distance"])
        w.writeheader()
        w.writerows(candidates)
    print(f"Saved candidate sub-table: {sub_csv}")

    # ----- Load synthetic and real volumes -----
    print(f"Loading synthetic pool: {args.syn_pool}")
    syn = torch.load(REPO / args.syn_pool, weights_only=False, map_location="cpu")
    if syn.dim() == 4:
        syn = syn.unsqueeze(1)
    syn_np = syn.float().numpy()
    print(f"  Synthetic shape: {syn_np.shape}")

    print(f"Loading real conditional dataset: {args.data}")
    raw = torch.load(REPO / args.data, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict) or "volumes" not in raw:
        sys.exit("ERROR: data is not a conditional dict.")
    vols = raw["volumes"]
    splits = raw.get("split_info") or {}
    train_idx = splits.get("train_idx") or list(range(len(vols)))
    train_vols = vols[train_idx]
    if train_vols.dim() == 4:
        train_vols = train_vols.unsqueeze(1)
    real_np = train_vols.float().numpy()
    print(f"  Real training shape: {real_np.shape}")

    # ----- Render -----
    n_to_render = min(args.max_rows, len(candidates))
    print(f"\nRendering {n_to_render} candidate pairs...")

    cols_per_pair = 3  # axial, coronal, sagittal
    fig, axes = plt.subplots(n_to_render, 2 * cols_per_pair,
                              figsize=(2 * cols_per_pair * 2.2, n_to_render * 2.2))
    if n_to_render == 1:
        axes = axes.reshape(1, -1)

    view_names = ["axial", "coronal", "sagittal"]
    for row, cand in enumerate(candidates[:n_to_render]):
        si, ri, d = cand["synthetic_idx"], cand["real_idx"], cand["nn_distance"]
        s_vol = syn_np[si]   # shape (1, D, H, W)
        r_vol = real_np[ri]
        s_slices = mid_slices(s_vol)
        r_slices = mid_slices(r_vol)

        for c, view in enumerate(view_names):
            # Real on the left
            ax = axes[row, c]
            ax.imshow(r_slices[view].T, cmap="gray", origin="lower",
                      vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"REAL  {view}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"real#{ri}\n"
                              f"syn#{si}\n"
                              f"d={d:.3f}",
                              fontsize=7, rotation=0, labelpad=28,
                              va="center", ha="right")

            # Synthetic on the right
            ax = axes[row, c + cols_per_pair]
            ax.imshow(s_slices[view].T, cmap="gray", origin="lower",
                      vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"SYNTHETIC  {view}", fontsize=8)

    fig.suptitle(
        f"Candidate near-copies (left tail below {thresh_label})\n"
        f"{args.method}: each row is one candidate synthetic sample (right)\n"
        f"alongside its nearest real training volume (left).\n"
        f"Anatomy match = memorization; generic similarity = candidate only.",
        y=1.005, fontsize=10)
    fig.tight_layout()
    png = out_dir / "memorization_visual_check.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"\nOpen {png} and decide:")
    print(f"  - Same anatomy (lesion location, ventricle shape, sulci match) → memorization")
    print(f"  - Generic similarity only → candidate near-duplicates (feature-space)")


if __name__ == "__main__":
    main()
