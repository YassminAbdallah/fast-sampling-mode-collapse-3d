#!/usr/bin/env python3
"""
E-voxel: Compute tumor voxel counts at 64³ from BraTS segmentation masks.
=========================================================================

Purpose: Decide how many PADP components are viable at 64³ resolution.

Decision rule (from Paper4_Handoff_v5):
  - If median tumor > 200 voxels: keep all 4 PADP components (TVD, TLE, BSV, SDS)
  - If median tumor 50-200 voxels: keep TVD, TLE, SDS. Drop BSV.
  - If median tumor < 50 voxels: keep TVD and TLE only.

Usage:
    python e_voxel_check.py
    python e_voxel_check.py --seg-path data/brats_seg_preprocessed_64.pt

If the preprocessed segmentation file doesn't exist, this script will
create it from the raw BraTS NIfTI files (same preprocessing as volumes).
"""

import os, sys, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_brats_segs(brats_dir="data/brats", target_size=64, max_volumes=1000, save_path=None):
    """Load and preprocess BraTS segmentation masks to 64³."""
    import nibabel as nib

    # Find segmentation files
    files = sorted(glob.glob(f"{brats_dir}/**/*-seg.nii.gz", recursive=True))
    if not files:
        files = sorted(glob.glob(f"{brats_dir}/**/*_seg.nii.gz", recursive=True))
    if not files:
        files = sorted(glob.glob(f"{brats_dir}/**/seg*.nii.gz", recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No segmentation files found in {brats_dir}. "
            "Expected pattern: *-seg.nii.gz or *_seg.nii.gz"
        )

    print(f"Found {len(files)} segmentation files")
    segs = []
    for i, f in enumerate(files[:max_volumes]):
        d = nib.load(f).get_fdata().astype(np.float32)
        # BraTS seg labels: 0=background, 1=necrotic core, 2=edema, 3/4=enhancing
        # Convert to binary: any tumor > 0
        d = (d > 0).astype(np.float32)
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
        # Nearest-neighbor interpolation for segmentation masks
        d = F.interpolate(d, size=(target_size, target_size, target_size),
                         mode='nearest')
        segs.append(d.squeeze(0))
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{min(len(files), max_volumes)}")

    segs = torch.stack(segs)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(segs, save_path)
        print(f"Saved preprocessed segmentations to {save_path}")
    return segs


def analyze_tumor_voxels(seg):
    """Analyze tumor voxel counts and print PADP decision."""
    # seg shape: (N, 1, D, H, W) or (N, D, H, W) — binary masks
    if seg.dim() == 4:
        # No channel dimension: (N, D, H, W)
        tumor_voxels = (seg > 0).sum(dim=(1, 2, 3))
    else:
        # With channel dimension: (N, 1, D, H, W)
        tumor_voxels = (seg > 0).sum(dim=(1, 2, 3, 4))

    print(f"\n{'='*60}")
    print(f"  E-VOXEL: Tumor Voxel Analysis at 64³")
    print(f"{'='*60}")
    print(f"  Total volumes:          {len(tumor_voxels)}")
    print(f"  Mean tumor voxels:      {tumor_voxels.float().mean():.0f}")
    print(f"  Median tumor voxels:    {tumor_voxels.float().median():.0f}")
    print(f"  Min:                    {tumor_voxels.min().item()}")
    print(f"  Max:                    {tumor_voxels.max().item()}")
    print(f"  Std:                    {tumor_voxels.float().std():.0f}")
    print(f"  Fraction < 50 voxels:   {(tumor_voxels < 50).float().mean():.2%}")
    print(f"  Fraction < 200 voxels:  {(tumor_voxels < 200).float().mean():.2%}")
    print(f"  Fraction == 0 voxels:   {(tumor_voxels == 0).float().mean():.2%}")

    # Percentiles
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = torch.quantile(tumor_voxels.float(), p / 100.0)
        print(f"  {p}th percentile:        {val:.0f}")

    median = tumor_voxels.float().median().item()
    print(f"\n{'='*60}")
    print(f"  PADP DECISION:")
    if median > 200:
        print(f"  Median = {median:.0f} > 200  →  ALL 4 components: TVD, TLE, BSV, SDS")
    elif median >= 50:
        print(f"  Median = {median:.0f} ∈ [50, 200]  →  3 components: TVD, TLE, SDS (drop BSV)")
    else:
        print(f"  Median = {median:.0f} < 50  →  2 components: TVD, TLE only")
    print(f"{'='*60}")

    return tumor_voxels


def main():
    parser = argparse.ArgumentParser(description="E-voxel: Tumor voxel count analysis")
    parser.add_argument('--seg-path', type=str, default='data/brats_seg_preprocessed_64.pt',
                        help='Path to preprocessed segmentation tensor')
    parser.add_argument('--brats-dir', type=str, default='data/brats',
                        help='Raw BraTS directory (used if seg-path does not exist)')
    parser.add_argument('--max-volumes', type=int, default=1000)
    args = parser.parse_args()

    if os.path.exists(args.seg_path):
        print(f"Loading cached segmentations: {args.seg_path}")
        seg = torch.load(args.seg_path, weights_only=True)
    else:
        print(f"No cached file at {args.seg_path}, preprocessing from {args.brats_dir}...")
        seg = preprocess_brats_segs(
            brats_dir=args.brats_dir,
            save_path=args.seg_path,
            max_volumes=args.max_volumes
        )

    print(f"Segmentation shape: {seg.shape}")
    analyze_tumor_voxels(seg)


if __name__ == "__main__":
    main()
