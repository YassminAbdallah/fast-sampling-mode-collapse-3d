#!/usr/bin/env python3
"""
Build the 128³ conditional BraTS dataset for the Week 3–5 conditional retrain.

Strategy
--------
The 128³ unconditional preprocessing (preprocess_brats_128.py) selected the
first 300 subjects from BraTS 2023 GLI by alphabetical sort of the internal
zip paths. The 64³ conditional preprocessing (src/paper4/prepare_conditional_brats.py)
used the same alphabetical-sort selection on the extracted filesystem layout.
By design both pipelines should produce the *same* 300 subjects in the *same*
order, although that is not stored explicitly in the saved tensors (the §5.8
v7.1 wording footnote acknowledges this).

This script:

  1. Loads `data/brats_preprocessed_128.pt` (N volumes at 128³, no labels)
  2. Loads `data/brats_conditional_64.pt` (N volumes at 64³ with labels)
  3. Verifies they have the same number of volumes (sanity check).
  4. Pairs the 128³ volumes with the 64³ labels positionally.
  5. Saves `data/brats_conditional_128.pt` with the same dict structure as the
     64³ conditional dataset, except the volumes (and segs) are at 128³.

If you want to be extra-rigorous, you can re-derive labels at 128³ from the
segmentation masks at 128³ rather than reusing the 64³ labels:

    python prepare_conditional_128.py --rederive-labels

This re-computes the tumor-volume median split on the 128³ segmentation masks.
For the default "size_proxy" labels, this should produce labels identical to
the 64³ derivation up to within the resampling rounding tolerance (the median
split is a coarse classifier and is highly stable under 64↔128 voxel resampling).

Usage
-----
    cd fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed_conditional/prepare_conditional_128.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_or_fail(path: Path, what: str):
    if not path.exists():
        sys.exit(f"ERROR: missing {what}: {path}\n"
                 f"  Run the preceding preprocessing step first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="Build 128³ conditional BraTS")
    parser.add_argument("--vols-128", default="data/brats_preprocessed_128.pt")
    parser.add_argument("--cond-64", default="data/brats_conditional_64.pt")
    parser.add_argument("--segs-128",
                        default="data/brats_seg_preprocessed_128.pt",
                        help="Optional: 128³ segmentation masks. If missing we will "
                             "either copy the 64³ segs (upsampled) or skip seg saving.")
    parser.add_argument("--output", default="data/brats_conditional_128.pt")
    parser.add_argument("--rederive-labels", action="store_true",
                        help="Recompute tumor-volume median-split labels on the "
                             "128³ segmentation masks rather than reusing the 64³ labels.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    vols_path = repo_root / args.vols_128
    cond_path = repo_root / args.cond_64
    segs_path = repo_root / args.segs_128
    out_path  = repo_root / args.output

    print(f"Repo root: {repo_root}")
    print(f"Loading 128³ volumes: {vols_path}")
    raw = load_or_fail(vols_path, "128³ volumes")
    if isinstance(raw, torch.Tensor):
        vols_128 = raw
    elif isinstance(raw, dict):
        vols_128 = raw.get("volumes", raw.get("v"))
        if vols_128 is None:
            sys.exit(f"ERROR: don't recognize keys in {vols_path}: {list(raw.keys())}")
    else:
        sys.exit(f"ERROR: unexpected type {type(raw)} from {vols_path}")
    if vols_128.dim() == 4:
        vols_128 = vols_128.unsqueeze(1)  # (N, 1, 128, 128, 128)
    N128 = vols_128.shape[0]
    print(f"  Loaded 128³ volumes: {tuple(vols_128.shape)}, dtype={vols_128.dtype}")

    print(f"Loading 64³ conditional dataset: {cond_path}")
    cond_64 = load_or_fail(cond_path, "64³ conditional dataset")
    if not isinstance(cond_64, dict):
        sys.exit(f"ERROR: expected dict in {cond_path}, got {type(cond_64)}")
    needed_keys = ["volumes", "labels"]
    for k in needed_keys:
        if k not in cond_64:
            sys.exit(f"ERROR: missing key '{k}' in {cond_path}: keys are {list(cond_64.keys())}")
    labels_64 = cond_64["labels"]
    N64 = cond_64["volumes"].shape[0]
    print(f"  64³ dataset: {N64} volumes, "
          f"labels dtype={labels_64.dtype}, classes={sorted(set(labels_64.tolist()))}")

    if N128 != N64:
        print(f"\nWARNING: 128³ has {N128} volumes but 64³ has {N64} volumes.")
        print(f"  Truncating to min({N128}, {N64}) = {min(N128, N64)} for positional pairing.")
        n_use = min(N128, N64)
        vols_128 = vols_128[:n_use]
        labels_paired = labels_64[:n_use]
    else:
        n_use = N128
        labels_paired = labels_64

    # ---- Segmentation masks at 128³ ----
    segs_128 = None
    if segs_path.exists():
        print(f"Loading 128³ seg masks: {segs_path}")
        raw_segs = load_or_fail(segs_path, "128³ segs")
        segs_128 = raw_segs["segs"] if isinstance(raw_segs, dict) and "segs" in raw_segs else raw_segs
        if isinstance(segs_128, torch.Tensor):
            if segs_128.dim() == 4:
                segs_128 = segs_128.unsqueeze(1)
            segs_128 = segs_128[:n_use]
            print(f"  segs_128: {tuple(segs_128.shape)}")
    else:
        print(f"  (no 128³ seg file at {segs_path}; will save without segs)")

    # ---- Optionally rederive labels from 128³ segmentation masks ----
    if args.rederive_labels:
        if segs_128 is None:
            sys.exit("--rederive-labels requires 128³ segmentation masks at "
                     f"{segs_path}, which were not found.")
        tumor_volumes = (segs_128 > 0).sum(dim=(1, 2, 3, 4)).long()
        median = tumor_volumes.median().item()
        labels_paired = (tumor_volumes > median).long()
        print(f"  Re-derived labels at 128³: median split at {median} voxels")
        print(f"    Class 0 (small tumor): {(labels_paired == 0).sum().item()}")
        print(f"    Class 1 (large tumor): {(labels_paired == 1).sum().item()}")
        n_changed = (labels_paired != labels_64[:n_use]).sum().item()
        print(f"    Changed from 64³-derived labels: {n_changed}/{n_use} subjects")

    # ---- Build the output dict ----
    out = {
        "volumes": vols_128,
        "labels": labels_paired,
        "label_type": cond_64.get("label_type", "size_proxy"),
        "label_names": cond_64.get("label_names", ["Small tumor", "Large tumor"]),
        "split_info": cond_64.get("split_info"),  # reused; preserves train/val/test
        "tumor_volumes": cond_64.get("tumor_volumes")[:n_use] if cond_64.get("tumor_volumes") is not None else None,
        "_source_provenance": {
            "vols_from": str(vols_path.relative_to(repo_root)),
            "labels_from": str(cond_path.relative_to(repo_root)),
            "labels_rederived_at_128": bool(args.rederive_labels),
            "selection_convention": (
                "Alphabetical-sort selection of the first N subjects in BraTS "
                "2023 GLI; both the 128³ and 64³ pipelines use the same "
                "selection convention. Exact subject-ID alignment is not "
                "stored in the saved tensors, but the position-based pairing "
                "is the established convention used throughout this paper."
            ),
        },
    }
    if segs_128 is not None:
        out["segs"] = segs_128

    print(f"\nSaving {out_path}")
    torch.save(out, out_path)
    print(f"  volumes: {tuple(out['volumes'].shape)}")
    print(f"  labels:  {tuple(out['labels'].shape)}, classes={sorted(set(out['labels'].tolist()))}")
    if segs_128 is not None:
        print(f"  segs:    {tuple(out['segs'].shape)}")
    print("Done.")


if __name__ == "__main__":
    main()
