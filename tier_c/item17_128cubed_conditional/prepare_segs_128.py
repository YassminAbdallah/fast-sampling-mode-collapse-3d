#!/usr/bin/env python3
"""
Build 128³ segmentation masks for E10a downstream training.

We don't have 128³ segs from the BraTS preprocessing pipeline (only 64³ segs
were saved). The cleanest way to get 128³ masks is to upsample the existing
64³ binary tumor masks with nearest-neighbor interpolation. Binary masks have
very little high-frequency content, so this introduces no meaningful artifacts
for downstream segmentation. The teacher trained on these upsampled masks is
fully self-consistent: it predicts on the same upsampled grid that its
training labels live on.

Outputs:
  data/brats_seg_preprocessed_128.pt   # tensor (N, 1, 128, 128, 128) uint8
  data/brats_conditional_128.pt        # augmented in place to add 'segs' key

Usage:
    cd fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed_conditional/prepare_segs_128.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    seg64_path = REPO_ROOT / "data" / "brats_seg_preprocessed_64.pt"
    cond128_path = REPO_ROOT / "data" / "brats_conditional_128.pt"
    seg128_path = REPO_ROOT / "data" / "brats_seg_preprocessed_128.pt"

    if not seg64_path.exists():
        sys.exit(f"ERROR: missing 64³ segs at {seg64_path}")

    print(f"Loading 64³ segs: {seg64_path}")
    segs64 = torch.load(seg64_path, weights_only=False, map_location="cpu")
    if isinstance(segs64, dict):
        segs64 = segs64.get("segs", segs64.get("v", segs64.get("volumes")))
    if segs64.dim() == 4:
        segs64 = segs64.unsqueeze(1)  # (N, 1, 64, 64, 64)
    print(f"  Loaded shape {tuple(segs64.shape)}, dtype {segs64.dtype}")

    # Binarize defensively (some preprocessors store multi-class)
    segs64_b = (segs64 > 0).float()

    print("Upsampling 64³ → 128³ with nearest-neighbor (preserves binary mask boundaries).")
    # interpolate expects float, returns float; we cast back to uint8 for storage
    segs128 = F.interpolate(segs64_b, scale_factor=2.0, mode="nearest").to(torch.uint8)
    print(f"  Output shape: {tuple(segs128.shape)}")
    print(f"  Tumor-voxel counts per subject: median={segs128.sum(dim=(1,2,3,4)).median().item()}, "
          f"min={segs128.sum(dim=(1,2,3,4)).min().item()}, "
          f"max={segs128.sum(dim=(1,2,3,4)).max().item()}")

    print(f"Saving {seg128_path}")
    torch.save(segs128, seg128_path)

    # Augment the conditional dataset with segs (for downstream training).
    if cond128_path.exists():
        print(f"Augmenting {cond128_path} with 'segs' key …")
        cond = torch.load(cond128_path, weights_only=False, map_location="cpu")
        if not isinstance(cond, dict):
            sys.exit("ERROR: brats_conditional_128.pt is not a dict — re-run "
                     "prepare_conditional_128.py first.")
        n_cond = cond["volumes"].shape[0]
        if n_cond != segs128.shape[0]:
            print(f"  WARNING: cond has {n_cond} volumes, segs has {segs128.shape[0]}. "
                  f"Truncating to {min(n_cond, segs128.shape[0])}.")
            n = min(n_cond, segs128.shape[0])
            cond["segs"] = segs128[:n]
        else:
            cond["segs"] = segs128
        torch.save(cond, cond128_path)
        print(f"  Updated {cond128_path}: keys = {list(cond.keys())}")
    else:
        print(f"  (No {cond128_path} found; only saved standalone segs.)")

    print("Done.")


if __name__ == "__main__":
    main()
