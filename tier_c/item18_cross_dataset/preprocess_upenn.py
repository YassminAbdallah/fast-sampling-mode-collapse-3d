#!/usr/bin/env python3
"""
Item 18 — UPenn-GBM preprocessing
==================================

Converts UPenn-GBM raw NIfTI volumes into a single .pt tensor with the same
shape and format as data/brats_conditional_64.pt and data/brats_seg_preprocessed_64.pt.

What it does:
  - Scans a directory of UPenn-GBM subjects
  - For each subject, locates the T2-FLAIR volume and segmentation mask
  - Skull-strip is assumed to have been done by the dataset (UPenn-GBM is preprocessed)
  - Intensity-normalises T2-FLAIR (min-max to [0,1] using non-zero voxels)
  - Resamples both volume and mask to 64x64x64 via trilinear (volume) and nearest-neighbor (mask)
  - Binarises the segmentation mask (any non-zero label -> 1) for compatibility with the existing pipeline
  - Saves two tensors:
      data/upenn_volumes_64.pt   (N, 1, 64, 64, 64) float32 in [0, 1]
      data/upenn_seg_64.pt       (N, 64, 64, 64) int64 in {0, 1}

UPenn-GBM file conventions (from TCIA):
  Each subject is a folder containing files like:
    <subject_id>_FLAIR.nii.gz
    <subject_id>_segm.nii.gz
  Filenames vary slightly between subjects. The script accepts several patterns.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item18_cross_dataset/preprocess_upenn.py \\
        --input-dir /path/to/UPenn-GBM/raw/ \\
        --output-dir data/ \\
        --max-subjects 100

Expected wall time: ~20 minutes for ~60 subjects on M-series.

If your UPenn-GBM directory structure differs from the assumptions below, the
script prints a clear error showing the first few files it found, so you can
either point it at the right subdirectory or adjust the file-pattern arguments.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def find_flair(subj_dir: Path):
    """Search for the FLAIR volume inside a subject directory."""
    patterns = [
        "*FLAIR*.nii.gz", "*flair*.nii.gz",
        "*FLAIR*.nii", "*flair*.nii",
        "*_T2_FLAIR*.nii.gz", "*_T2_FLAIR*.nii",
    ]
    for p in patterns:
        hits = sorted(subj_dir.glob(p))
        if hits:
            return hits[0]
    return None


def find_seg_for_subject(subj_name: str, structural_dir: Path, seg_dir: Path = None):
    """Search for the segmentation file matching a SPECIFIC subject name.

    CRITICAL: only matches files whose name starts with subj_name. Never returns
    a generic *segm*.nii.gz match because that would silently pair the wrong
    subject's tumor mask with the FLAIR (data corruption).

    Supports both layouts:
      (a) Nested: structural_dir/<subj_name>/<subj_name>_segm.nii.gz
      (b) Flat:   seg_dir/<subj_name>_segm.nii.gz
    """
    # Subject-specific patterns ONLY — these all start with subj_name
    subj_patterns = [
        f"{subj_name}_segm.nii.gz",
        f"{subj_name}_automated_approx_segm.nii.gz",
        f"{subj_name}*segm*.nii.gz",
        f"{subj_name}*seg*.nii.gz",
    ]
    # First, look in seg_dir (flat layout) if provided
    if seg_dir is not None and seg_dir.exists():
        for p in subj_patterns:
            hits = sorted(seg_dir.glob(p))
            if hits:
                return hits[0]
    # Fall back to nested layout: the subject's own structural folder.
    # Here a generic *segm* match IS safe because we are already inside the
    # subject's own folder, so any segm file there belongs to that subject.
    subj_dir = structural_dir / subj_name
    if subj_dir.exists():
        for p in [
            f"{subj_name}*segm*.nii.gz",
            "*segm*.nii.gz", "*segm*.nii",
            "*automated_approx_segm*.nii.gz",
            "*tumor*.nii.gz",
        ]:
            hits = sorted(subj_dir.glob(p))
            if hits:
                return hits[0]
    return None


def load_nifti(path: Path):
    """Load a NIfTI file. Returns numpy array (D, H, W)."""
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel not installed. Run: pip install nibabel")
        sys.exit(1)
    img = nib.load(str(path))
    arr = img.get_fdata()
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def resample_to_64(vol: np.ndarray, mode: str = "trilinear") -> np.ndarray:
    """Resample volume to (64, 64, 64) using PyTorch interpolation.

    mode = 'trilinear' for intensity volumes, 'nearest' for masks.
    """
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
    if mode == "trilinear":
        out = F.interpolate(t, size=(64, 64, 64), mode="trilinear", align_corners=False)
    else:
        out = F.interpolate(t, size=(64, 64, 64), mode="nearest")
    return out.squeeze(0).squeeze(0).numpy()


def normalize_intensity(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] using non-zero voxels.

    Robust against outlier-bright voxels (e.g., skull tissue in unstripped FLAIR):
    uses the 1st and 99th percentiles of non-zero voxels rather than full min/max,
    then clips. This preserves the dynamic range of brain tissue when the volume
    has not been skull-stripped.
    """
    mask = vol > 0
    if mask.sum() < 100:
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[mask]
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.0))
    out = (vol - vmin) / max(vmax - vmin, 1e-6)
    out[~mask] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Path to images_structural_unstripped/ (or images_structural/): "
                             "the folder containing one subdirectory per subject with FLAIR inside")
    parser.add_argument("--seg-dir", default=None,
                        help="Optional. Path to images_segm/ (flat layout — all *_segm.nii.gz files "
                             "directly inside). If not given, segmentations are assumed to live in "
                             "the same subject subdirectory as the FLAIR file.")
    parser.add_argument("--output-dir", default="data/",
                        help="Where to save upenn_volumes_64.pt and upenn_seg_64.pt")
    parser.add_argument("--max-subjects", type=int, default=100,
                        help="Maximum number of subjects to include")
    parser.add_argument("--subj-pattern", default="*",
                        help="Glob pattern for subject directories (default: every subdirectory)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    seg_dir = Path(args.seg_dir).expanduser().resolve() if args.seg_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)
    if seg_dir is not None and not seg_dir.exists():
        print(f"ERROR: --seg-dir given but not found: {seg_dir}")
        sys.exit(1)

    subj_dirs = sorted([d for d in input_dir.glob(args.subj_pattern) if d.is_dir()])
    print(f"Found {len(subj_dirs)} candidate subject directories in {input_dir}")
    if seg_dir:
        seg_files = sorted(seg_dir.glob("*segm*.nii.gz"))
        print(f"Found {len(seg_files)} segmentation files in {seg_dir}")
    if len(subj_dirs) == 0:
        first_few = list(input_dir.iterdir())[:5]
        print(f"  First few entries of {input_dir}: {[p.name for p in first_few]}")
        print(f"  If subjects are one level deeper, point --input-dir at that subdirectory.")
        sys.exit(1)

    volumes = []
    masks = []
    skipped = []

    for i, subj in enumerate(subj_dirs[: args.max_subjects]):
        flair = find_flair(subj)
        seg = find_seg_for_subject(subj.name, input_dir, seg_dir)
        if flair is None:
            skipped.append((subj.name, "missing FLAIR"))
            continue
        if seg is None:
            skipped.append((subj.name, "missing segmentation"))
            continue
        try:
            v_arr = load_nifti(flair)
            s_arr = load_nifti(seg)
        except Exception as e:
            skipped.append((subj.name, f"load error: {e}"))
            continue

        # Resample to 64^3
        v64 = resample_to_64(v_arr, mode="trilinear")
        s64 = resample_to_64(s_arr, mode="nearest")
        # Normalize intensity
        v64 = normalize_intensity(v64)
        # Binarize segmentation (any positive label -> 1) for compatibility with
        # the existing binary segmentation pipeline (see data/brats_seg_preprocessed_64.pt format)
        s64 = (s64 > 0).astype(np.int64)
        volumes.append(v64)
        masks.append(s64)
        if (i + 1) % 10 == 0 or i < 5:
            tumor_voxels = int(s64.sum())
            print(f"  [{i+1:3d}/{min(len(subj_dirs), args.max_subjects)}] {subj.name}: "
                  f"tumor={tumor_voxels} voxels, "
                  f"intensity range [{v64.min():.3f}, {v64.max():.3f}]")

    if not volumes:
        print(f"ERROR: no subjects successfully processed. Skipped: {skipped[:5]}")
        sys.exit(1)

    V = torch.from_numpy(np.stack(volumes, axis=0)).unsqueeze(1)  # (N, 1, 64, 64, 64)
    S = torch.from_numpy(np.stack(masks, axis=0))                 # (N, 64, 64, 64)

    vol_path = output_dir / "upenn_volumes_64.pt"
    seg_path = output_dir / "upenn_seg_64.pt"
    torch.save(V.float(), vol_path)
    torch.save(S.long(), seg_path)

    print()
    print(f"Saved {V.shape[0]} volumes to {vol_path}  shape={tuple(V.shape)}")
    print(f"Saved {S.shape[0]} masks   to {seg_path}  shape={tuple(S.shape)}  "
          f"unique={sorted(torch.unique(S).tolist())}")

    # Per-subject tumor voxel counts: detects collisions where multiple subjects got the same mask
    tumor_counts = (S > 0).long().sum(dim=(1, 2, 3)).tolist()
    unique_counts = set(tumor_counts)
    print(f"Distinct tumor-voxel counts across {len(tumor_counts)} subjects: {len(unique_counts)}")
    if len(unique_counts) < max(2, len(tumor_counts) // 4):
        print("WARNING: many subjects share identical tumor counts. "
              "This usually indicates the same mask was assigned to multiple subjects "
              "(data alignment bug). Inspect the script logic.")
    if skipped:
        n_missing_seg = sum(1 for _, r in skipped if "missing segmentation" in r)
        n_missing_flair = sum(1 for _, r in skipped if "missing FLAIR" in r)
        n_other = len(skipped) - n_missing_seg - n_missing_flair
        print(f"\nSkipped {len(skipped)} subjects:")
        print(f"  - missing segmentation: {n_missing_seg}")
        print(f"  - missing FLAIR:        {n_missing_flair}")
        print(f"  - other errors:         {n_other}")
        print(f"  First 5 skipped:")
        for name, reason in skipped[:5]:
            print(f"    - {name}: {reason}")


if __name__ == "__main__":
    main()
