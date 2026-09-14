#!/usr/bin/env python3
"""
Item 18 — Cross-dataset generalization: BraTS Shortcut@50 augmentation tested on UPenn-GBM
============================================================================================

Tests whether augmentation using the existing BraTS-trained Shortcut@50 model
transfers to a different glioma dataset (UPenn-GBM). No new generative training
is performed — we re-use the existing trained checkpoint at:
   results/paper4/brats_benchmark_20260324_154926/100pct_180vol/shortcut/final.pt

Experiment design (mirrors E10b from the existing paper, restricted to one fraction):

  Conditions evaluated at 10% UPenn-GBM real data (~6 volumes):
    (a) Real only            - baseline
    (b) Real + classical aug - random flips, 90° rotations
    (c) Real + Shortcut@50   - 500 BraTS-generated synthetic + UPenn-trained-teacher pseudo-labels

  Each condition: 3 seeds, 2000 iterations, binary tumor segmentation.

Phases:
  1. Train a UPenn-GBM teacher segmenter on the train portion (~30 volumes), to
     produce pseudo-labels for the synthetic volumes (~2 hours).
  2. Generate 500 synthetic volumes from the existing BraTS-trained Shortcut@50
     model (or load from cache if previously generated).
  3. Pseudo-label the synthetic volumes with the UPenn-GBM teacher.
  4. Run the three augmentation conditions × 3 seeds = 9 segmenter runs.

Total: ~10-12 hours on Apple M-series.

Output: results/paper4/cross_dataset_upenn/upenn_results.json with per-condition Dice + seed-level data.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item18_cross_dataset/cross_dataset_upenn.py \\
        --upenn-vols data/upenn_volumes_64.pt \\
        --upenn-segs data/upenn_seg_64.pt \\
        --gen-dir results/paper4/brats_benchmark_20260324_154926 \\
        --output-dir results/paper4/cross_dataset_upenn \\
        --num-classes-gen 2 \\
        --n-seeds 3
"""

import argparse
import json
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from monai.networks.nets import BasicUNet
    from monai.losses import DiceCELoss
except ImportError:
    print("ERROR: MONAI not installed. Run: pip install monai")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_vqgan, load_unet, sample_latent

warnings.filterwarnings("ignore")


# ============================================================
# DATASETS
# ============================================================

class BinarySegDataset(Dataset):
    """Volumes + binary tumor mask, with optional flip-based augmentation."""

    def __init__(self, volumes, segs, augment=False):
        self.v = volumes if volumes.dim() == 5 else volumes.unsqueeze(1)
        self.s = segs if segs.dim() == 4 else segs.squeeze(1) if segs.dim() == 5 else segs
        if self.s.dim() == 3:  # (N, 64, 64, 64) — add channel dim later for BasicUNet
            pass
        # Force binary
        self.s = (self.s > 0).long()
        self.augment = augment

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol = self.v[i].float()
        seg = self.s[i].long()
        if seg.dim() == 3:
            seg = seg.unsqueeze(0)  # (1, 64, 64, 64) for DiceCELoss
        if self.augment:
            for d in [1, 2, 3]:
                if random.random() < 0.5:
                    vol = torch.flip(vol, dims=[d])
                    seg = torch.flip(seg, dims=[d])
        return vol, seg


class ClassicalAugDataset(Dataset):
    """Adds flips + 90° rotations on the fly."""

    def __init__(self, volumes, segs):
        self.base = BinarySegDataset(volumes, segs, augment=False)

    def __len__(self):
        return len(self.base) * 4  # over-sample to give more augmentation

    def __getitem__(self, idx):
        vol, seg = self.base[idx % len(self.base)]
        # random flips
        for d in [1, 2, 3]:
            if random.random() < 0.5:
                vol = torch.flip(vol, dims=[d])
                seg = torch.flip(seg, dims=[d])
        # one random 90° rotation in one random plane
        plane = random.choice([(1, 2), (1, 3), (2, 3)])
        k = random.randint(0, 3)
        if k > 0:
            vol = torch.rot90(vol, k, dims=plane)
            seg = torch.rot90(seg, k, dims=plane)
        # intensity jitter
        vol = (vol * (0.9 + 0.2 * random.random())).clamp(0, 1)
        return vol, seg


# ============================================================
# Training loop
# ============================================================

def create_seg_model(device):
    return BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        features=[32, 64, 128, 256, 32, 32],
    ).to(device)


def train_seg(model, train_ds, n_iters, device, batch_size=4, lr=1e-3, seed=42, log_every=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    iter_loader = iter(loader)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
    model.train()
    losses = []
    for it in range(n_iters):
        try:
            vols, segs = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            vols, segs = next(iter_loader)
        vols = vols.to(device).float()
        segs = segs.to(device).long()
        logits = model(vols)
        loss = criterion(logits, segs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (it + 1) % log_every == 0:
            print(f"    iter {it+1:5d}/{n_iters}  loss={np.mean(losses[-log_every:]):.4f}")
    return float(np.mean(losses[-200:]))


@torch.no_grad()
def evaluate_seg(model, test_ds, device, batch_size=8):
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    dices = []
    for vols, segs in loader:
        vols = vols.to(device).float()
        segs = segs.to(device).long()
        pred = model(vols).argmax(dim=1, keepdim=True)
        target = segs
        # binary Dice
        pred_b = (pred > 0).float()
        true_b = (target > 0).float()
        inter = (pred_b * true_b).sum(dim=(1, 2, 3, 4))
        denom = pred_b.sum(dim=(1, 2, 3, 4)) + true_b.sum(dim=(1, 2, 3, 4))
        dice = ((2.0 * inter + 1e-6) / (denom + 1e-6))
        dices.extend(dice.cpu().tolist())
    return float(np.mean(dices))


# ============================================================
# Generation from existing trained Shortcut FM
# ============================================================

@torch.no_grad()
def generate_shortcut_volumes(gen_dir, n_volumes, device, num_classes=2, class_label=1, steps=50):
    gen_dir = Path(gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir
    p1_path = data_dir / "phase1_shared.pt"
    ckpt_path = data_dir / "shortcut" / "final.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN not found: {p1_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Shortcut FM not found: {ckpt_path}")
    _, dec, _, _ = load_vqgan(p1_path, device)
    dec.eval()
    unet = load_unet(ckpt_path, device, num_classes=num_classes)
    unet.eval()
    out = []
    rem = n_volumes
    while rem > 0:
        bs = min(8, rem)
        z = sample_latent(unet, "shortcut", bs, device, steps, class_label=class_label)
        v = dec(z).clamp(0, 1).cpu()
        out.append(v)
        rem -= bs
        print(f"  generated {n_volumes - rem}/{n_volumes}", end="\r")
    print()
    return torch.cat(out, dim=0)[:n_volumes]


# ============================================================
# Pseudo-labelling
# ============================================================

@torch.no_grad()
def pseudo_label(teacher, volumes, device, batch_size=4):
    teacher.eval()
    out = []
    for i in range(0, len(volumes), batch_size):
        v = volumes[i:i + batch_size].to(device).float()
        if v.dim() == 4:
            v = v.unsqueeze(1)
        pred = teacher(v).argmax(dim=1)
        out.append(pred.cpu())
    return torch.cat(out, dim=0)  # (N, 64, 64, 64) binary


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upenn-vols", default="data/upenn_volumes_64.pt",
                        help="Preprocessed UPenn-GBM volumes (output of preprocess_upenn.py)")
    parser.add_argument("--upenn-segs", default="data/upenn_seg_64.pt",
                        help="Preprocessed UPenn-GBM binary masks")
    parser.add_argument("--gen-dir", default="results/paper4/brats_benchmark_20260324_154926",
                        help="Existing BraTS benchmark dir with trained Shortcut@50 and VQ-GAN")
    parser.add_argument("--output-dir", default="results/paper4/cross_dataset_upenn",
                        help="Where to save results")
    parser.add_argument("--num-classes-gen", type=int, default=2,
                        help="Generator's conditioning classes (2 = small/large tumor)")
    parser.add_argument("--n-synth", type=int, default=500,
                        help="Synthetic volumes to generate")
    parser.add_argument("--real-frac", type=float, default=0.10,
                        help="UPenn-GBM real data fraction used in each condition")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of segmenter seeds per condition")
    parser.add_argument("--teacher-iters", type=int, default=3000,
                        help="Iterations for the UPenn-GBM teacher segmenter")
    parser.add_argument("--seg-iters", type=int, default=2000,
                        help="Iterations per segmenter run")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load UPenn-GBM data ----
    vols = torch.load(args.upenn_vols, weights_only=False, map_location="cpu")
    segs = torch.load(args.upenn_segs, weights_only=False, map_location="cpu")
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    if segs.dim() == 5:
        segs = segs.squeeze(1)
    n_total = len(vols)
    print(f"UPenn-GBM: {n_total} volumes, vols shape {tuple(vols.shape)}, segs shape {tuple(segs.shape)}")
    print(f"Seg unique values: {sorted(torch.unique(segs).tolist())}")

    # Split: 60% train / 20% val / 20% test  (typical for a small cross-dataset test)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_vols = vols[train_idx]; train_segs = segs[train_idx]
    val_vols = vols[val_idx]; val_segs = segs[val_idx]
    test_vols = vols[test_idx]; test_segs = segs[test_idx]
    print(f"Split: train={len(train_vols)} val={len(val_vols)} test={len(test_vols)}")

    test_ds = BinarySegDataset(test_vols, test_segs, augment=False)

    # ---- Phase 1: UPenn-GBM teacher segmenter ----
    teacher_ckpt = outdir / "upenn_teacher.pt"
    if teacher_ckpt.exists():
        print(f"\n========== PHASE 1: Loading cached UPenn teacher ==========")
        teacher = create_seg_model(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f"\n========== PHASE 1: Training UPenn-GBM teacher segmenter ({args.teacher_iters} iters) ==========")
        teacher_train_vols = torch.cat([train_vols, val_vols], dim=0)
        teacher_train_segs = torch.cat([train_segs, val_segs], dim=0)
        teacher_ds = BinarySegDataset(teacher_train_vols, teacher_train_segs, augment=True)
        teacher = create_seg_model(device)
        train_seg(teacher, teacher_ds, args.teacher_iters, device, batch_size=4, seed=42, log_every=500)
        torch.save(teacher.state_dict(), teacher_ckpt)
    t_dice = evaluate_seg(teacher, test_ds, device)
    print(f"\nUPenn teacher test Dice: {t_dice:.4f}")
    with open(outdir / "teacher_metrics.json", "w") as f:
        json.dump({"test_dice": t_dice}, f, indent=2)

    # ---- Phase 2: Generate synthetic volumes from existing BraTS Shortcut@50 ----
    synth_cache = outdir / f"shortcut50_{args.n_synth}vols.pt"
    if synth_cache.exists():
        print(f"\n========== PHASE 2: Loading cached synthetic volumes ==========")
        synth_vols = torch.load(synth_cache, weights_only=False)
    else:
        print(f"\n========== PHASE 2: Generating {args.n_synth} synthetic volumes from BraTS Shortcut@50 ==========")
        synth_vols = generate_shortcut_volumes(
            args.gen_dir, args.n_synth, device,
            num_classes=args.num_classes_gen, class_label=1, steps=50,
        )
        torch.save(synth_vols, synth_cache)
    print(f"Synthetic volumes shape: {tuple(synth_vols.shape)}")

    # ---- Phase 3: Pseudo-label synthetic volumes with UPenn teacher ----
    print(f"\n========== PHASE 3: Pseudo-labeling synthetic volumes with UPenn teacher ==========")
    pseudo_segs = pseudo_label(teacher, synth_vols, device, batch_size=4)
    tumor_voxels = int(pseudo_segs.sum())
    print(f"Pseudo-label tumor voxels: {tumor_voxels:,} (avg {tumor_voxels/len(pseudo_segs):.0f} per volume)")

    # ---- Phase 4: Cross-dataset experiment ----
    print(f"\n========== PHASE 4: Cross-dataset segmentation conditions ==========")
    n_real = max(1, int(args.real_frac * len(train_vols)))
    g = torch.Generator().manual_seed(42)
    real_idx = torch.randperm(len(train_vols), generator=g)[:n_real]
    real_v = train_vols[real_idx]; real_s = train_segs[real_idx]
    print(f"Using {n_real} real UPenn-GBM volumes ({args.real_frac:.0%} of training)")

    results = {
        "design": vars(args),
        "n_upenn_total": n_total,
        "n_upenn_train": len(train_vols),
        "n_upenn_val": len(val_vols),
        "n_upenn_test": len(test_vols),
        "n_real_used": n_real,
        "teacher_test_dice": t_dice,
        "conditions": {},
    }

    def run_condition(name, build_ds_fn):
        dices = []
        for seed in range(42, 42 + args.n_seeds):
            print(f"\n  Condition {name}, seed={seed}")
            ds = build_ds_fn()
            model = create_seg_model(device)
            train_seg(model, ds, args.seg_iters, device, batch_size=4, seed=seed, log_every=500)
            d = evaluate_seg(model, test_ds, device)
            print(f"    test Dice = {d:.4f}")
            dices.append(d)
        results["conditions"][name] = {
            "dices": dices,
            "mean": float(np.mean(dices)),
            "std": float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
        }

    # (a) Real only
    print("\n----- (a) Real only -----")
    run_condition("real_only", lambda: BinarySegDataset(real_v, real_s, augment=True))

    # (b) Real + classical augmentation
    print("\n----- (b) Real + Classical Augmentation -----")
    run_condition("real_classical_aug",
                  lambda: ClassicalAugDataset(real_v, real_s))

    # (c) Real + Shortcut@50 with pseudo-labels
    print("\n----- (c) Real + Shortcut@50 (BraTS-trained, UPenn-pseudo-labeled) -----")
    def _build_synth():
        real_ds = BinarySegDataset(real_v, real_s, augment=True)
        synth_ds = BinarySegDataset(synth_vols, pseudo_segs.unsqueeze(1) if pseudo_segs.dim() == 4 else pseudo_segs, augment=True)
        return ConcatDataset([real_ds, synth_ds])
    run_condition("real_shortcut50", _build_synth)

    # ---- Save ----
    results["timestamp"] = datetime.utcnow().isoformat() + "Z"
    out_path = outdir / "upenn_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"  UPenn-GBM cross-dataset summary ({args.real_frac:.0%} real = {n_real} vol)")
    print("=" * 60)
    for name, r in results["conditions"].items():
        print(f"  {name:>22}  Dice = {r['mean']:.4f} ± {r['std']:.4f}")


if __name__ == "__main__":
    main()
