#!/usr/bin/env python3
"""
E-grade + E0: Prepare BraTS Conditional Dataset
================================================

Two-step process:
1. E-grade: Extract grade labels from BraTS metadata, or compute
   tumor-size proxy (median split on segmentation mask volume).
2. E0: Save conditional dataset with volumes + labels for training.

Grade determination:
  - FIRST tries to read WHO grade from BraTS 2023 GLI metadata (CSV/JSON).
  - If unavailable, computes tumor volume from seg masks and splits at median:
    Class 0 = "Small tumor" (below median)
    Class 1 = "Large tumor" (above median)

Output: data/brats_conditional_64.pt containing:
  {
    "volumes": Tensor (N, 1, 64, 64, 64),
    "labels":  Tensor (N,) of int64 (0 or 1),
    "segs":    Tensor (N, 1, 64, 64, 64) — segmentation masks,
    "label_type": "grade" or "size_proxy",
    "label_names": ["LGG", "HGG"] or ["Small tumor", "Large tumor"],
    "split_info": {train_idx, val_idx, test_idx},
    "tumor_volumes": Tensor (N,) — voxel counts per volume,
  }

Usage:
    python prepare_conditional_brats.py
    python prepare_conditional_brats.py --brats-dir data/brats --force-size-proxy
"""

import os, sys, json, csv, re, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def find_brats_metadata(brats_dir):
    """Try to find BraTS 2023 GLI metadata with WHO grade info."""
    candidates = [
        "mapping.csv", "name_mapping.csv", "BraTS2023_GLI_mapping.csv",
        "metadata.csv", "participants.tsv", "metadata.json",
    ]
    for c in candidates:
        p = Path(brats_dir) / c
        if p.exists():
            return p
    # Search recursively
    for ext in ["*.csv", "*.tsv", "*.json"]:
        found = glob.glob(f"{brats_dir}/{ext}")
        for f in found:
            # Quick check: does it contain "grade" or "Grade"?
            try:
                with open(f, 'r') as fh:
                    header = fh.readline().lower()
                    if 'grade' in header:
                        return Path(f)
            except:
                pass
    return None


def parse_grade_from_metadata(metadata_path, subject_ids):
    """
    Parse WHO grade from BraTS metadata file.
    Returns dict: {subject_id: grade_class} where grade_class is 0 (LGG) or 1 (HGG).

    BraTS 2023 GLI uses WHO 2021 grading:
      Grade 2, 3 → LGG (class 0)
      Grade 4    → HGG (class 1)
    """
    ext = metadata_path.suffix.lower()
    grade_map = {}

    if ext == '.csv':
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find the grade column (various naming conventions)
                grade_val = None
                sid = None
                for k, v in row.items():
                    kl = k.lower().strip()
                    if kl in ('grade', 'who_grade', 'tumor_grade', 'who grade'):
                        grade_val = v.strip()
                    if kl in ('brats21id', 'brats_id', 'subject_id', 'id', 'name',
                              'brats2023id', 'subjectid'):
                        sid = v.strip()

                if sid and grade_val:
                    # Parse grade: "2", "3", "4", "II", "III", "IV", "LGG", "HGG"
                    gv = grade_val.upper().strip()
                    if gv in ('2', 'II', 'LGG', 'LOW', 'LOW GRADE', 'LOW-GRADE'):
                        grade_map[sid] = 0
                    elif gv in ('3', 'III'):
                        grade_map[sid] = 0  # Grade 3 = LGG in WHO 2021
                    elif gv in ('4', 'IV', 'HGG', 'HIGH', 'HIGH GRADE', 'HIGH-GRADE'):
                        grade_map[sid] = 1
                    else:
                        print(f"  Warning: unknown grade '{grade_val}' for {sid}")

    elif ext == '.tsv':
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Same logic as CSV
                grade_val = None; sid = None
                for k, v in row.items():
                    kl = k.lower().strip()
                    if 'grade' in kl: grade_val = v.strip()
                    if kl in ('participant_id', 'subject_id', 'id'): sid = v.strip()
                if sid and grade_val:
                    gv = grade_val.upper().strip()
                    if gv in ('2', 'II', 'LGG', '3', 'III'):
                        grade_map[sid] = 0
                    elif gv in ('4', 'IV', 'HGG'):
                        grade_map[sid] = 1

    elif ext == '.json':
        with open(metadata_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                sid = entry.get('id') or entry.get('subject_id') or entry.get('name', '')
                grade = entry.get('grade') or entry.get('who_grade', '')
                if sid and grade:
                    gv = str(grade).upper().strip()
                    if gv in ('2', 'II', 'LGG', '3', 'III'):
                        grade_map[sid] = 0
                    elif gv in ('4', 'IV', 'HGG'):
                        grade_map[sid] = 1

    return grade_map


def extract_subject_id(filepath):
    """Extract BraTS subject ID from filepath.
    e.g. .../BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz -> BraTS-GLI-00000-000
    """
    name = Path(filepath).stem
    # Remove modality suffixes
    for suffix in ['-t2f', '-t1c', '-t1n', '-t2w', '-seg',
                   '.nii', '_t2f', '_t1c', '_t1n', '_t2w', '_seg']:
        name = name.replace(suffix, '')
    return name


def compute_size_proxy_labels(tumor_volumes):
    """Split at median tumor volume: 0=small, 1=large."""
    median_vol = tumor_volumes.float().median().item()
    labels = (tumor_volumes.float() > median_vol).long()
    print(f"  Size proxy: median tumor volume = {median_vol:.0f} voxels")
    print(f"  Class 0 (Small tumor, ≤{median_vol:.0f}): {(labels == 0).sum()} volumes")
    print(f"  Class 1 (Large tumor, >{median_vol:.0f}): {(labels == 1).sum()} volumes")
    return labels


def create_splits(n, train_frac=0.60, val_frac=0.20, seed=42):
    """Create train/val/test split indices.
    Default: 60/20/20 → ~750/250/251 for 1251 BraTS volumes.
    Adjust for smaller datasets.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train_idx": indices[:n_train].tolist(),
        "val_idx": indices[n_train:n_train + n_val].tolist(),
        "test_idx": indices[n_train + n_val:].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare BraTS conditional dataset")
    parser.add_argument('--brats-dir', type=str, default='data/brats',
                        help='Raw BraTS directory with NIfTI files')
    parser.add_argument('--vol-path', type=str, default='data/brats_preprocessed_64.pt',
                        help='Path to preprocessed volume tensor (from Paper 2)')
    parser.add_argument('--seg-path', type=str, default='data/brats_seg_preprocessed_64.pt',
                        help='Path to preprocessed segmentation tensor')
    parser.add_argument('--output', type=str, default='data/brats_conditional_64.pt',
                        help='Output path for conditional dataset')
    parser.add_argument('--force-size-proxy', action='store_true',
                        help='Skip metadata search, use tumor-size proxy directly')
    parser.add_argument('--max-volumes', type=int, default=1300)
    parser.add_argument('--target-size', type=int, default=64)
    args = parser.parse_args()

    # ---- Load or create volumes ----
    if os.path.exists(args.vol_path):
        print(f"Loading cached volumes: {args.vol_path}")
        volumes = torch.load(args.vol_path, weights_only=True)
    else:
        print(f"ERROR: Volume file not found at {args.vol_path}")
        print("Run flow_matching_3d.py --dataset brats first to create it,")
        print("or preprocess BraTS volumes manually.")
        sys.exit(1)

    # ---- Load or create segmentations ----
    if os.path.exists(args.seg_path):
        print(f"Loading cached segmentations: {args.seg_path}")
        segs = torch.load(args.seg_path, weights_only=True)
    else:
        print(f"Segmentation file not found at {args.seg_path}")
        print("Run e_voxel_check.py first to create it.")
        sys.exit(1)

    N = min(len(volumes), len(segs))
    volumes = volumes[:N]
    segs = segs[:N]
    print(f"\nDataset: {N} volumes, shape {volumes.shape}")

    # ---- Compute tumor volumes ----
    if segs.dim() == 4:
        tumor_volumes = (segs > 0).sum(dim=(1, 2, 3))
    else:
        tumor_volumes = (segs > 0).sum(dim=(1, 2, 3, 4))
    print(f"Tumor voxels — mean: {tumor_volumes.float().mean():.0f}, "
          f"median: {tumor_volumes.float().median():.0f}")

    # ---- E-grade: Get labels ----
    label_type = "size_proxy"
    label_names = ["Small tumor", "Large tumor"]
    labels = None

    if not args.force_size_proxy:
        print(f"\n--- E-grade: Searching for grade metadata in {args.brats_dir} ---")
        meta_path = find_brats_metadata(args.brats_dir)
        if meta_path:
            print(f"  Found metadata: {meta_path}")

            # Get subject IDs from volume file ordering
            # We need to match the ordering of volumes to the metadata
            modality = "t2f"
            vol_files = sorted(glob.glob(f"{args.brats_dir}/**/*-{modality}.nii.gz", recursive=True))
            if not vol_files:
                vol_files = sorted(glob.glob(f"{args.brats_dir}/**/*.nii.gz", recursive=True))
                vol_files = [f for f in vol_files if "-seg" not in f and "_seg" not in f]

            subject_ids = [extract_subject_id(f) for f in vol_files[:N]]
            grade_map = parse_grade_from_metadata(meta_path, subject_ids)

            if len(grade_map) > 0:
                # Match subject IDs to grades
                matched = 0
                grade_labels = torch.zeros(N, dtype=torch.long)
                for i, sid in enumerate(subject_ids):
                    if sid in grade_map:
                        grade_labels[i] = grade_map[sid]
                        matched += 1
                    else:
                        # Try partial match
                        for key in grade_map:
                            if key in sid or sid in key:
                                grade_labels[i] = grade_map[key]
                                matched += 1
                                break

                coverage = matched / N
                print(f"  Grade labels matched: {matched}/{N} ({coverage:.1%})")

                if coverage >= 0.80:
                    labels = grade_labels
                    label_type = "grade"
                    label_names = ["LGG", "HGG"]
                    print(f"  Using WHO grade labels!")
                    print(f"  Class 0 (LGG): {(labels == 0).sum()} volumes")
                    print(f"  Class 1 (HGG): {(labels == 1).sum()} volumes")
                else:
                    print(f"  Coverage too low ({coverage:.1%}), falling back to size proxy")
            else:
                print(f"  No grade information found in metadata, using size proxy")
        else:
            print(f"  No metadata file found, using size proxy")

    if labels is None:
        print(f"\n--- E-grade: Computing tumor-size proxy labels ---")
        labels = compute_size_proxy_labels(tumor_volumes)
        label_type = "size_proxy"
        label_names = ["Small tumor", "Large tumor"]

    # ---- Create splits ----
    splits = create_splits(N)
    print(f"\n--- Dataset splits ---")
    print(f"  Train: {len(splits['train_idx'])} volumes")
    print(f"  Val:   {len(splits['val_idx'])} volumes")
    print(f"  Test:  {len(splits['test_idx'])} volumes")

    # Verify class balance in train split
    train_labels = labels[splits['train_idx']]
    print(f"  Train class 0: {(train_labels == 0).sum()} | class 1: {(train_labels == 1).sum()}")

    # ---- Save conditional dataset ----
    dataset = {
        "volumes": volumes,
        "labels": labels,
        "segs": segs,
        "label_type": label_type,
        "label_names": label_names,
        "split_info": splits,
        "tumor_volumes": tumor_volumes,
        "num_classes": 2,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(dataset, args.output)
    print(f"\n--- Saved conditional dataset to {args.output} ---")
    print(f"  Volumes: {volumes.shape}")
    print(f"  Labels: {labels.shape} ({label_type}: {label_names})")
    print(f"  Segs: {segs.shape}")
    print(f"  Splits: train={len(splits['train_idx'])}, "
          f"val={len(splits['val_idx'])}, test={len(splits['test_idx'])}")

    # ---- Summary for next steps ----
    print(f"\n{'='*60}")
    print(f"  NEXT STEPS:")
    print(f"  1. Train conditional VQ-GAN + generative models:")
    print(f"     python flow_matching_3d.py --dataset brats-cond \\")
    print(f"       --conditional --num-classes 2 \\")
    print(f"       --methods fm shortcut consistency rectified")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
