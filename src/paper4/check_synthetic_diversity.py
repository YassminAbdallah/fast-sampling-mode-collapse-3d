#!/usr/bin/env python3
"""
Quick diagnostic: Does Shortcut@50 produce meaningfully diverse tumors?
=======================================================================

Run BEFORE committing to the 21-hour dose-response experiment.
Takes ~5 minutes. Answers two questions:

1. Do 500 Shortcut@50 volumes have diverse tumors (size, location, shape)?
2. Does subsampling to 10 unique volumes actually reduce tumor diversity?

If both answers are YES → dose-response experiment is worth running.
If answer 1 is NO → model lacks diversity, experiment will be flat.

Usage:
    python check_synthetic_diversity.py \
        --gen-dir results/brats_benchmark_20260324_154926 \
        --teacher-path results/paper4/e7_teacher/teacher_best.pt \
        --data-path data/brats_conditional_64.pt \
        --num-classes 2
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models_shared import load_vqgan, load_unet, sample_latent

try:
    from monai.networks.nets import BasicUNet
except ImportError:
    print("ERROR: MONAI not installed. Run: pip install monai")
    sys.exit(1)


def get_teacher_masks(volumes, teacher, device, threshold=0.5, batch_size=8):
    """Get binary tumor masks from teacher segmenter."""
    teacher.eval()
    all_masks = []
    with torch.no_grad():
        for i in range(0, len(volumes), batch_size):
            batch = volumes[i:i+batch_size].to(device)
            logits = teacher(batch)
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # tumor channel
            masks = (probs > threshold).float().cpu()
            all_masks.append(masks)
    return torch.cat(all_masks, dim=0)


def compute_tumor_stats(masks):
    """Compute per-volume tumor statistics from binary masks.
    
    Returns dict with arrays of per-volume stats:
      - volume: number of tumor voxels
      - centroid_x/y/z: tumor centroid coordinates
      - has_tumor: whether any tumor was detected
    """
    N = len(masks)
    stats = {
        "volume": np.zeros(N),
        "centroid_x": np.zeros(N),
        "centroid_y": np.zeros(N),
        "centroid_z": np.zeros(N),
        "has_tumor": np.zeros(N, dtype=bool),
    }
    
    for i in range(N):
        m = masks[i, 0]  # (D, H, W)
        tvox = m.sum().item()
        stats["volume"][i] = tvox
        stats["has_tumor"][i] = tvox > 0
        
        if tvox > 0:
            coords = torch.nonzero(m, as_tuple=False).float()  # (K, 3)
            centroid = coords.mean(dim=0)
            stats["centroid_x"][i] = centroid[0].item()
            stats["centroid_y"][i] = centroid[1].item()
            stats["centroid_z"][i] = centroid[2].item()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check synthetic tumor diversity")
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--data-path", required=True, help="Conditional dataset for real comparison")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--n-gen", type=int, default=200, help="Volumes to generate (200 is enough)")
    parser.add_argument("--output-dir", type=str, default="results/paper4/diagnostics")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    gen_dir = Path(args.gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir

    _, dec, _, _ = load_vqgan(data_dir / "phase1_shared.pt", device)
    unet = load_unet(data_dir / "shortcut" / "final.pt", device, num_classes=args.num_classes)

    teacher = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                        features=(32, 64, 128, 256, 32, 32)).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    # Load real data for comparison
    cond_data = torch.load(args.data_path, weights_only=False)
    real_vols = cond_data["volumes"]
    real_segs = cond_data["segs"]
    train_idx = cond_data["split_info"]["train_idx"]
    real_vols = real_vols[train_idx]
    real_segs = real_segs[train_idx]
    # Binarize real segs
    if real_segs.dim() == 4:
        real_segs = real_segs.unsqueeze(1)
    real_segs_binary = (real_segs > 0).float()

    # Generate synthetic volumes
    n_gen = args.n_gen
    print(f"\nGenerating {n_gen} volumes from Shortcut@50 (class=1)...")
    all_vols = []
    batch_size = 8
    for start in range(0, n_gen, batch_size):
        bs = min(batch_size, n_gen - start)
        z = sample_latent(unet, "shortcut", bs, device, 50, class_label=1)
        with torch.no_grad():
            vols = dec(z).clamp(0, 1)
        all_vols.append(vols.cpu())
        print(f"  {min(start+bs, n_gen)}/{n_gen}", end='\r')
    print()
    syn_vols = torch.cat(all_vols, dim=0)

    # Get teacher masks for synthetic
    print("Getting teacher masks for synthetic volumes...")
    syn_masks = get_teacher_masks(syn_vols, teacher, device)

    # Compute stats
    print("\nComputing tumor statistics...")
    syn_stats = compute_tumor_stats(syn_masks)
    real_stats = compute_tumor_stats(real_segs_binary[:n_gen])

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Report
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  TUMOR DIVERSITY DIAGNOSTIC")
    print(f"{'='*60}")

    # Detection rate
    syn_det = syn_stats["has_tumor"].mean()
    real_det = real_stats["has_tumor"].mean()
    print(f"\n  Tumor detection rate:")
    print(f"    Synthetic: {syn_det:.1%} ({syn_stats['has_tumor'].sum():.0f}/{n_gen})")
    print(f"    Real:      {real_det:.1%} ({real_stats['has_tumor'].sum():.0f}/{len(real_stats['has_tumor'])})")

    # Volume distribution
    syn_vox = syn_stats["volume"][syn_stats["has_tumor"]]
    real_vox = real_stats["volume"][real_stats["has_tumor"]]
    print(f"\n  Tumor volume (voxels) — synthetic with tumor:")
    print(f"    Mean: {syn_vox.mean():.0f}  Std: {syn_vox.std():.0f}")
    print(f"    Min:  {syn_vox.min():.0f}  Max: {syn_vox.max():.0f}")
    print(f"    IQR:  [{np.percentile(syn_vox, 25):.0f}, {np.percentile(syn_vox, 75):.0f}]")
    print(f"  Tumor volume (voxels) — real:")
    print(f"    Mean: {real_vox.mean():.0f}  Std: {real_vox.std():.0f}")
    print(f"    Min:  {real_vox.min():.0f}  Max: {real_vox.max():.0f}")
    print(f"    IQR:  [{np.percentile(real_vox, 25):.0f}, {np.percentile(real_vox, 75):.0f}]")

    # Centroid diversity
    syn_has = syn_stats["has_tumor"]
    syn_cx = syn_stats["centroid_x"][syn_has]
    syn_cy = syn_stats["centroid_y"][syn_has]
    syn_cz = syn_stats["centroid_z"][syn_has]
    real_has = real_stats["has_tumor"]
    real_cx = real_stats["centroid_x"][real_has]
    real_cy = real_stats["centroid_y"][real_has]
    real_cz = real_stats["centroid_z"][real_has]

    print(f"\n  Centroid spread (std of x,y,z):")
    print(f"    Synthetic: x={syn_cx.std():.1f}  y={syn_cy.std():.1f}  z={syn_cz.std():.1f}")
    print(f"    Real:      x={real_cx.std():.1f}  y={real_cy.std():.1f}  z={real_cz.std():.1f}")

    # Number of distinct centroid locations (binned to 4x4x4 grid)
    def count_distinct_bins(cx, cy, cz, bins=4, vol_size=64):
        bx = np.clip((cx / vol_size * bins).astype(int), 0, bins-1)
        by = np.clip((cy / vol_size * bins).astype(int), 0, bins-1)
        bz = np.clip((cz / vol_size * bins).astype(int), 0, bins-1)
        unique_bins = set(zip(bx, by, bz))
        return len(unique_bins)

    syn_bins = count_distinct_bins(syn_cx, syn_cy, syn_cz)
    real_bins = count_distinct_bins(real_cx, real_cy, real_cz)
    print(f"\n  Distinct centroid bins (4×4×4 grid = 64 possible):")
    print(f"    Synthetic: {syn_bins}/64")
    print(f"    Real:      {real_bins}/64")

    # KEY QUESTION: Does subsampling reduce tumor diversity?
    print(f"\n{'='*60}")
    print(f"  SUBSAMPLING TEST (does 10 unique → less tumor diversity?)")
    print(f"{'='*60}")

    rng = np.random.RandomState(42)
    for n_unique in [10, 25, 100, n_gen]:
        if n_unique < n_gen:
            idx = rng.choice(n_gen, n_unique, replace=False)
        else:
            idx = np.arange(n_gen)
        sub_masks = syn_masks[idx]
        sub_stats = compute_tumor_stats(sub_masks)
        sub_has = sub_stats["has_tumor"]
        sub_vox = sub_stats["volume"][sub_has]
        sub_cx = sub_stats["centroid_x"][sub_has]
        sub_cy = sub_stats["centroid_y"][sub_has]
        sub_cz = sub_stats["centroid_z"][sub_has]

        n_bins = count_distinct_bins(sub_cx, sub_cy, sub_cz) if sub_has.sum() > 0 else 0
        vol_std = sub_vox.std() if len(sub_vox) > 1 else 0
        det_rate = sub_has.mean()

        print(f"  {n_unique:>4} unique: det={det_rate:.0%}, "
              f"vol_std={vol_std:.0f}, "
              f"centroid_bins={n_bins}/64, "
              f"vol_range=[{sub_vox.min():.0f}-{sub_vox.max():.0f}]" if len(sub_vox) > 0 else "no tumors")

    # ============================================================
    # Visual grid: 16 random synthetic samples
    # ============================================================
    print(f"\n  Generating visual grid...")
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    # 16 samples: show axial slice + tumor overlay
    sample_idx = rng.choice(min(n_gen, syn_stats["has_tumor"].sum()), 16, replace=False)
    # Only pick volumes with tumors
    tumor_idx = np.where(syn_stats["has_tumor"])[0]
    if len(tumor_idx) >= 16:
        show_idx = rng.choice(tumor_idx, 16, replace=False)
    else:
        show_idx = tumor_idx[:16]

    for i, idx in enumerate(show_idx):
        row = i // 4
        col_vol = (i % 4) * 2
        col_mask = col_vol + 1

        vol = syn_vols[idx, 0, :, :, 32].numpy()  # axial mid-slice
        mask = syn_masks[idx, 0, :, :, 32].numpy()

        axes[row, col_vol].imshow(vol, cmap='gray', vmin=0, vmax=1)
        axes[row, col_vol].set_title(f"#{idx} vol={syn_stats['volume'][idx]:.0f}", fontsize=8)
        axes[row, col_vol].axis('off')

        # Overlay
        axes[row, col_mask].imshow(vol, cmap='gray', vmin=0, vmax=1)
        if mask.max() > 0:
            axes[row, col_mask].imshow(mask, cmap='Reds', alpha=0.4)
        cx, cy = syn_stats["centroid_y"][idx], syn_stats["centroid_x"][idx]
        if syn_stats["has_tumor"][idx]:
            axes[row, col_mask].plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
        axes[row, col_mask].set_title(f"tumor overlay", fontsize=8)
        axes[row, col_mask].axis('off')

    plt.suptitle("Shortcut@50 class=1: 16 random samples with teacher tumor masks", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / "synthetic_tumor_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved grid to {outdir / 'synthetic_tumor_grid.png'}")

    # Centroid scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (c1, c2, label) in zip(axes, [
        (syn_cx, syn_cy, "X vs Y"), (syn_cx, syn_cz, "X vs Z"), (syn_cy, syn_cz, "Y vs Z")
    ]):
        ax.scatter(c1, c2, alpha=0.3, s=10, c='purple', label=f'Synthetic (n={len(c1)})')
        r1 = real_cx if 'X' in label.split(' vs ')[0] else real_cy
        r2 = real_cy if 'Y' in label.split(' vs ')[1] else real_cz
        if 'X' in label.split(' vs ')[0]: r1 = real_cx
        elif 'Y' in label.split(' vs ')[0]: r1 = real_cy
        if 'Y' in label.split(' vs ')[1]: r2 = real_cy
        elif 'Z' in label.split(' vs ')[1]: r2 = real_cz
        ax.scatter(r1, r2, alpha=0.3, s=10, c='green', label=f'Real (n={len(r1)})')
        ax.set_title(f"Centroid {label}")
        ax.set_xlim(0, 64); ax.set_ylim(0, 64)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Tumor centroid locations: synthetic vs real", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "centroid_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved centroid scatter to {outdir / 'centroid_scatter.png'}")

    # Verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    diverse_enough = syn_bins >= real_bins * 0.5 and syn_vox.std() >= real_vox.std() * 0.3
    if diverse_enough:
        print(f"  ✓ Synthetic tumors show meaningful variation.")
        print(f"    Dose-response experiment should show a signal.")
    else:
        print(f"  ⚠ Synthetic tumor diversity is limited.")
        print(f"    Dose-response may be flat — consider this a finding in itself.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
