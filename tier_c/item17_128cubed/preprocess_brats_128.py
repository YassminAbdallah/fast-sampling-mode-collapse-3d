#!/usr/bin/env python3
"""
Item 17 — Re-preprocess BraTS 2023 GLI at 128³ resolution
==========================================================

Reads the BraTS 2023 GLI training zip and produces 128³ FLAIR volumes
for the 128³ resolution-validation experiment.

What it does:
  - Streams subjects directly from the zip (no full unzip needed)
  - For each subject, loads the T2-FLAIR NIfTI file
  - Trilinear resample to 128×128×128
  - Robust intensity normalization (1st/99th percentile clip → [0, 1])
  - Saves data/brats_preprocessed_128.pt as a (N, 1, 128, 128, 128) tensor

Why not just upsample the existing 64³?
  Upsampling 64³ would create artificial blur, not actual high-resolution data.
  The whole point of this experiment is to test whether findings hold on
  data with genuine 128³ resolution information.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed/preprocess_brats_128.py \\
        --zip-path datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \\
        --output data/brats_preprocessed_128.pt \\
        --max-subjects 300

Estimated wall time: ~40-60 minutes on M-series for 300 subjects.

Output: data/brats_preprocessed_128.pt — shape (N, 1, 128, 128, 128), float32, [0, 1].
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_nifti_from_zip(zf: zipfile.ZipFile, internal_path: str):
    """Load a .nii.gz file from inside a zip via a short-lived temp file.

    BytesIO doesn't work here because nibabel relies on the .gz file extension /
    path detection to know the stream is gzipped. Writing to a temp file with the
    correct suffix lets nibabel auto-detect and decompress.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel not installed. Run: pip install nibabel")
        sys.exit(1)

    import tempfile
    suffix = ".nii.gz" if internal_path.endswith(".nii.gz") else ".nii"
    with zf.open(internal_path) as src, \
         tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(src.read())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        arr = img.get_fdata()
        if arr.ndim == 4:
            arr = arr[..., 0]
        return arr.astype(np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def resample_to_target(vol: np.ndarray, target: int = 128) -> np.ndarray:
    """Trilinear resample to (target, target, target)."""
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
    out = F.interpolate(t, size=(target, target, target),
                        mode="trilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).numpy()


def normalize_intensity(vol: np.ndarray) -> np.ndarray:
    """Robust min-max normalization using 1st-99th percentile of non-zero voxels."""
    mask = vol > 0
    if mask.sum() < 100:
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[mask]
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.0))
    out = (vol - vmin) / max(vmax - vmin, 1e-6)
    out[~mask] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def find_flair_paths_in_zip(zf: zipfile.ZipFile):
    """Return a list of (subject_id, internal_flair_path) tuples for every subject in the zip.

    BraTS 2023 GLI naming convention: files are named like
        <subject>/<subject>-t2f.nii.gz
    where the modality token is one of:
        -t1n   T1 native
        -t1c   T1 contrast-enhanced
        -t2w   T2 weighted
        -t2f   T2-FLAIR  ← what we want
        -seg   segmentation

    Older BraTS releases used <subject>_flair.nii.gz / <subject>-flair.nii.gz. We accept both.
    """
    subject_files = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if not (name.endswith(".nii.gz") or name.endswith(".nii")):
            continue
        lower = name.lower()
        is_flair = (
            "-t2f.nii" in lower or "_t2f.nii" in lower      # BraTS 2023 GLI
            or "-flair.nii" in lower or "_flair.nii" in lower  # older releases
        )
        if not is_flair:
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        subj_id = parts[-2]
        # Prefer the first match per subject (alphabetical zip order should give a stable choice)
        if subj_id not in subject_files:
            subject_files[subj_id] = name
    return [(sid, subject_files[sid]) for sid in sorted(subject_files.keys())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", required=True,
                        help="Path to BraTS 2023 GLI training zip "
                             "(datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip)")
    parser.add_argument("--output", default="data/brats_preprocessed_128.pt",
                        help="Where to save the (N, 1, 128, 128, 128) tensor")
    parser.add_argument("--max-subjects", type=int, default=300,
                        help="Maximum number of subjects to include (default 300 to match BraTS 64³)")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Target cubic resolution (default 128)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        sys.exit(1)

    print(f"Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        flair_paths = find_flair_paths_in_zip(zf)
        print(f"Found {len(flair_paths)} subjects with FLAIR files")
        if len(flair_paths) == 0:
            print("ERROR: no FLAIR files found. Sample entries in zip:")
            for n in zf.namelist()[:10]:
                print(f"  {n}")
            sys.exit(1)

        volumes = []
        subj_ids = []
        skipped = []

        for i, (subj_id, internal_path) in enumerate(flair_paths[: args.max_subjects]):
            try:
                arr = load_nifti_from_zip(zf, internal_path)
                v = resample_to_target(arr, target=args.target_size)
                v = normalize_intensity(v)
                volumes.append(v)
                subj_ids.append(subj_id)
                if (i + 1) % 10 == 0 or i < 3:
                    print(f"  [{i+1:3d}/{min(len(flair_paths), args.max_subjects)}] "
                          f"{subj_id}: shape {v.shape}, range [{v.min():.3f}, {v.max():.3f}]")
            except Exception as e:
                skipped.append((subj_id, str(e)[:60]))

    if not volumes:
        print(f"ERROR: no subjects successfully processed. Skipped: {skipped[:5]}")
        sys.exit(1)

    V = torch.from_numpy(np.stack(volumes, axis=0)).unsqueeze(1).float()
    torch.save(V, output_path)

    print()
    print(f"Saved {V.shape[0]} volumes to {output_path}  shape={tuple(V.shape)}  dtype={V.dtype}")
    print(f"Approximate disk size: {V.numel() * 4 / 1e9:.2f} GB")
    if skipped:
        print(f"\nSkipped {len(skipped)} subjects. First 3:")
        for sid, reason in skipped[:3]:
            print(f"  - {sid}: {reason}")


if __name__ == "__main__":
    main()
