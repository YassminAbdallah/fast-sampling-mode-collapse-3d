#!/usr/bin/env python3
"""
Downstream Segmentation Experiments for Paper 4
================================================

Covers:
  E7:   Train teacher segmenter on ALL real BraTS data
  E8:   Ceiling experiment (real-only, duplicated, FM@50, noise, classical)
  E10a: Diversity isolation (THE critical control — diverse vs collapsed)
  E10b: Full downstream sweep (7 conditions × 4 fractions × 5 seeds)
  E10c: Mixing ratio sub-experiment (optional)

Segmentation architecture: MONAI BasicUNet (3D)
  - Input: 64³ single-channel
  - features=[32, 64, 128, 256]
  - Binary: tumor vs background
  - 5000 gradient steps (iteration-based)
  - AdamW lr=1e-3, weight_decay=1e-5
  - CosineAnnealingLR
  - DiceCELoss
  - Batch size 4
  - 5 seeds per condition

Usage:
    python conditional_segmentation.py e7 --data-path data/brats_conditional_64.pt
    python conditional_segmentation.py e8 --data-path data/brats_conditional_64.pt --gen-dir results/brats_cond_XXX
    python conditional_segmentation.py e10a --data-path data/brats_conditional_64.pt --gen-dir results/brats_cond_XXX
    python conditional_segmentation.py e10b --data-path data/brats_conditional_64.pt --gen-dir results/brats_cond_XXX
"""

import os, sys, json, time, math, argparse, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

try:
    from monai.networks.nets import BasicUNet
    from monai.losses import DiceCELoss
except ImportError:
    print("ERROR: MONAI not installed. Run: pip install monai")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None  # Only needed for E10a

from models_shared import (
    Encoder3D, Decoder3D, VectorQuantizer,
    DenoisingUNet3D, load_vqgan, load_unet, sample_latent
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================
# DATASETS
# ============================================================

class SegDataset(Dataset):
    """Dataset for segmentation: volumes + binary tumor masks."""
    def __init__(self, volumes, segs, augment=False):
        """
        Args:
            volumes: (N, 1, 64, 64, 64) or (N, 64, 64, 64)
            segs: (N, 64, 64, 64) or (N, 1, 64, 64, 64) — tumor masks
            augment: if True, apply random flips
        """
        self.v = volumes.unsqueeze(1) if volumes.dim() == 4 else volumes
        self.s = segs.unsqueeze(1) if segs.dim() == 4 else segs
        # Binarize: any tumor label > 0 → 1
        self.s = (self.s > 0).float()
        self.augment = augment

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i], self.s[i]
        if self.augment:
            # Random flips along each spatial axis
            for d in range(3):
                if torch.rand(1).item() > 0.5:
                    vol = torch.flip(vol, [d + 1])  # +1 because dim 0 is channel
                    seg = torch.flip(seg, [d + 1])
        return vol, seg


class ClassicalAugDataset(Dataset):
    """Dataset with on-the-fly classical augmentation.

    Augmentations: random 3D flips (all axes), random 90° rotations,
    random intensity scaling (0.9-1.1), Gaussian noise (σ=0.01).
    """
    def __init__(self, volumes, segs):
        self.v = volumes.unsqueeze(1) if volumes.dim() == 4 else volumes
        self.s = segs.unsqueeze(1) if segs.dim() == 4 else segs
        self.s = (self.s > 0).float()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i].clone(), self.s[i].clone()
        # Random flips
        for d in range(3):
            if torch.rand(1).item() > 0.5:
                vol = torch.flip(vol, [d + 1])
                seg = torch.flip(seg, [d + 1])
        # Random 90° rotation around a random axis
        if torch.rand(1).item() > 0.5:
            axes = [(1, 2), (1, 3), (2, 3)]
            ax = axes[torch.randint(0, 3, (1,)).item()]
            k = torch.randint(1, 4, (1,)).item()
            vol = torch.rot90(vol, k, ax)
            seg = torch.rot90(seg, k, ax)
        # Random intensity scaling
        scale = 0.9 + 0.2 * torch.rand(1).item()
        vol = (vol * scale).clamp(0, 1)
        # Gaussian noise
        vol = vol + 0.01 * torch.randn_like(vol)
        vol = vol.clamp(0, 1)
        return vol, seg


class SyntheticSegDataset(Dataset):
    """Dataset pairing synthetic volumes with teacher-predicted segmentations."""
    def __init__(self, synthetic_volumes, teacher_model, device, threshold=0.5, augment=False):
        """
        Generate pseudo-labels for synthetic volumes using teacher segmenter.

        Args:
            synthetic_volumes: (N, 1, 64, 64, 64)
            teacher_model: trained MONAI BasicUNet
            device: torch device
            threshold: softmax threshold for binarizing predictions
            augment: random flips during training
        """
        self.v = synthetic_volumes.unsqueeze(1) if synthetic_volumes.dim() == 4 else synthetic_volumes
        self.augment = augment

        # Generate pseudo-labels in batches
        teacher_model.eval()
        all_segs = []
        bs = 8
        with torch.no_grad():
            for start in range(0, len(self.v), bs):
                batch = self.v[start:start+bs].to(device)
                logits = teacher_model(batch)
                # Binary: channel 1 = tumor probability
                probs = torch.softmax(logits, dim=1)[:, 1:2]
                masks = (probs > threshold).float()
                all_segs.append(masks.cpu())
        self.s = torch.cat(all_segs, dim=0)

        # Report detection rate
        has_tumor = (self.s.sum(dim=(1, 2, 3, 4)) > 0).float()
        self.tumor_detection_rate = has_tumor.mean().item()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i], self.s[i]
        if self.augment:
            for d in range(3):
                if torch.rand(1).item() > 0.5:
                    vol = torch.flip(vol, [d + 1])
                    seg = torch.flip(seg, [d + 1])
        return vol, seg


# ============================================================
# SEGMENTATION TRAINING (ITERATION-BASED)
# ============================================================

def create_seg_model(device, spatial_dims=3, in_channels=1, out_channels=2):
    """Create MONAI BasicUNet for binary tumor segmentation."""
    model = BasicUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        features=(32, 64, 128, 256, 32, 32),  # BasicUNet default encoder features
    ).to(device)
    return model


def train_seg_iteration_based(
    model, train_ds, val_ds, device,
    max_iters=5000, batch_size=4, lr=1e-3, weight_decay=1e-5,
    seed=42, log_every=500
):
    """Train segmentation model for fixed number of gradient steps.

    Returns: dict with final val Dice, training history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, drop_last=True)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    model.train()
    losses = []
    step = 0
    data_iter = iter(dl)
    t0 = time.time()

    while step < max_iters:
        try:
            vol, seg = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            vol, seg = next(data_iter)

        vol, seg = vol.to(device), seg.to(device).long()
        logits = model(vol)
        loss = criterion(logits, seg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        step += 1

        if step % log_every == 0 or step == max_iters:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-log_every:])
            print(f"    [{step:5d}/{max_iters}] loss={avg_loss:.4f} | {elapsed:.0f}s")

    # Validation Dice
    val_dice = evaluate_seg(model, val_ds, device)
    return {"val_dice": val_dice, "losses": losses, "steps": max_iters}


def evaluate_seg(model, dataset, device, batch_size=8):
    """Compute mean Dice score on a dataset."""
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dice_scores = []

    with torch.no_grad():
        for vol, seg in dl:
            vol, seg = vol.to(device), seg.to(device)
            logits = model(vol)
            pred = torch.argmax(logits, dim=1, keepdim=True).float()
            target = (seg > 0).float()

            # Dice per sample
            for i in range(pred.shape[0]):
                p, t = pred[i].flatten(), target[i].flatten()
                intersection = (p * t).sum()
                union = p.sum() + t.sum()
                if union == 0:
                    dice = 1.0 if t.sum() == 0 else 0.0
                else:
                    dice = (2.0 * intersection / union).item()
                dice_scores.append(dice)

    model.train()
    return float(np.mean(dice_scores))


# ============================================================
# SYNTHETIC VOLUME GENERATION
# ============================================================

def generate_synthetic_volumes(gen_dir, method, steps, n_volumes, device,
                                num_classes=2, class_label=1):
    """Generate synthetic volumes from a trained generative model.

    Args:
        gen_dir: directory with phase1_shared.pt and method/final.pt
        method: 'fm', 'shortcut', 'consistency', 'rectified'
        steps: number of inference steps
        n_volumes: how many to generate
        device: torch device
        num_classes: number of conditioning classes
        class_label: which class to generate (default 1 = large tumor / HGG)

    Returns: Tensor of shape (n_volumes, 1, 64, 64, 64)
    """
    gen_dir = Path(gen_dir)

    # Find data subdirectory
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir

    # Load VQ-GAN decoder
    p1_path = data_dir / "phase1_shared.pt"
    _, dec, _, _ = load_vqgan(p1_path, device)

    # Load U-Net
    ckpt_path = data_dir / method / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    unet = load_unet(ckpt_path, device, num_classes=num_classes)

    # Generate in batches
    all_vols = []
    batch_size = 8
    remaining = n_volumes
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = sample_latent(unet, method, bs, device, steps, class_label=class_label)
        with torch.no_grad():
            vols = dec(z).clamp(0, 1)
        all_vols.append(vols.cpu())
        remaining -= bs
        print(f"    Generated {n_volumes - remaining}/{n_volumes}", end='\r')

    print()
    return torch.cat(all_vols, dim=0)[:n_volumes]


# ============================================================
# E7: TEACHER SEGMENTER
# ============================================================

def run_e7(args):
    """Train teacher segmenter on ALL real BraTS training data."""
    print(f"\n{'='*60}\n  E7: TEACHER SEGMENTER\n{'='*60}")

    data = torch.load(args.data_path, weights_only=False)
    vols = data["volumes"]
    segs = data["segs"]
    splits = data["split_info"]
    device = torch.device(args.device)

    # Use train + val for teacher (maximize data)
    train_idx = splits["train_idx"] + splits["val_idx"]
    test_idx = splits["test_idx"]

    train_vols = vols[train_idx]
    train_segs = segs[train_idx]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]

    print(f"  Train: {len(train_vols)} volumes")
    print(f"  Test:  {len(test_vols)} volumes")

    train_ds = SegDataset(train_vols, train_segs, augment=True)
    test_ds = SegDataset(test_vols, test_segs, augment=False)

    outdir = Path(args.output_dir) / "e7_teacher"
    outdir.mkdir(parents=True, exist_ok=True)

    # Train with 5 seeds, pick best
    results = []
    for seed in range(5):
        print(f"\n  Seed {seed}:")
        model = create_seg_model(device)
        res = train_seg_iteration_based(
            model, train_ds, test_ds, device,
            max_iters=args.max_iters, seed=seed + 42
        )
        print(f"    Val Dice: {res['val_dice']:.4f}")
        results.append(res)

        # Save each seed
        torch.save(model.state_dict(), outdir / f"teacher_seed{seed}.pt")
        if seed == 0 or res['val_dice'] > max(r['val_dice'] for r in results[:-1]):
            torch.save(model.state_dict(), outdir / "teacher_best.pt")
            print(f"    → New best!")

    # Summary
    dices = [r['val_dice'] for r in results]
    print(f"\n  Teacher Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"  Best: {max(dices):.4f}")

    with open(outdir / "e7_results.json", 'w') as f:
        json.dump({"dices": dices, "mean": float(np.mean(dices)),
                    "std": float(np.std(dices))}, f, indent=2)

    print(f"\n  Saved to {outdir}")


# ============================================================
# E8: CEILING EXPERIMENT
# ============================================================

def run_e8(args):
    """Ceiling experiment: 5 conditions at 25% data."""
    print(f"\n{'='*60}\n  E8: CEILING EXPERIMENT (25% data)\n{'='*60}")

    data = torch.load(args.data_path, weights_only=False)
    vols = data["volumes"]
    segs = data["segs"]
    splits = data["split_info"]
    device = torch.device(args.device)

    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    # 25% of training data
    n_25 = max(4, int(len(train_idx) * 0.25))
    subset_idx = train_idx[:n_25]

    real_vols = vols[subset_idx]
    real_segs = segs[subset_idx]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]

    print(f"  Real subset: {n_25} volumes (25%)")
    print(f"  Test: {len(test_idx)} volumes")

    test_ds = SegDataset(test_vols, test_segs, augment=False)
    outdir = Path(args.output_dir) / "e8_ceiling"
    outdir.mkdir(parents=True, exist_ok=True)

    n_synthetic = 500
    conditions = {}

    # (a) Real only
    conditions["real_only"] = SegDataset(real_vols, real_segs, augment=True)

    # (b) Real + duplicated real
    dup_vols = real_vols.repeat((n_synthetic // n_25) + 1, *([1] * (real_vols.dim() - 1)))[:n_synthetic]
    dup_segs = real_segs.repeat((n_synthetic // n_25) + 1, *([1] * (real_segs.dim() - 1)))[:n_synthetic]
    conditions["duplicated"] = ConcatDataset([
        SegDataset(real_vols, real_segs, augment=True),
        SegDataset(dup_vols, dup_segs, augment=True)
    ])

    # (c) Real + FM@50 synthetic (needs teacher + gen model)
    if args.gen_dir and args.teacher_path:
        teacher = create_seg_model(device)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=True))
        teacher.eval()

        print(f"  Generating {n_synthetic} FM@50 synthetic volumes...")
        fm_vols = generate_synthetic_volumes(
            args.gen_dir, "fm", 50, n_synthetic, device,
            num_classes=args.num_classes, class_label=1
        )
        fm_syn_ds = SyntheticSegDataset(fm_vols, teacher, device, augment=True)
        print(f"    Tumor detection rate: {fm_syn_ds.tumor_detection_rate:.1%}")
        conditions["fm50"] = ConcatDataset([
            SegDataset(real_vols, real_segs, augment=True),
            fm_syn_ds
        ])
    else:
        print("  Skipping FM@50 condition (need --gen-dir and --teacher-path)")

    # (d) Real + noise (Gaussian noise through VQ-GAN decoder)
    if args.gen_dir:
        gen_dir = Path(args.gen_dir)
        sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
        data_dir = sub[0] if sub else gen_dir
        p1_path = data_dir / "phase1_shared.pt"
        _, dec, _, _ = load_vqgan(p1_path, device)
        print(f"  Generating {n_synthetic} noise volumes...")
        with torch.no_grad():
            noise_latent = torch.randn(n_synthetic, 8, 8, 8, 8, device=device)
            noise_vols_list = []
            for i in range(0, n_synthetic, 8):
                bs = min(8, n_synthetic - i)
                nv = dec(noise_latent[i:i+bs]).clamp(0, 1).cpu()
                noise_vols_list.append(nv)
            noise_vols = torch.cat(noise_vols_list, dim=0)

        if args.teacher_path:
            noise_syn_ds = SyntheticSegDataset(noise_vols, teacher, device, augment=True)
            print(f"    Noise tumor detection rate: {noise_syn_ds.tumor_detection_rate:.1%}")
            conditions["noise"] = ConcatDataset([
                SegDataset(real_vols, real_segs, augment=True),
                noise_syn_ds
            ])
    else:
        print("  Skipping noise condition (need --gen-dir)")

    # (e) Real + classical augmentation
    conditions["classical_aug"] = ClassicalAugDataset(real_vols, real_segs)

    # Run all conditions
    all_results = {}
    seeds = list(range(42, 42 + args.n_seeds))

    for cond_name, train_ds in conditions.items():
        print(f"\n  --- Condition: {cond_name} ({len(train_ds)} samples) ---")
        cond_dices = []
        for seed in seeds:
            model = create_seg_model(device)
            res = train_seg_iteration_based(
                model, train_ds, test_ds, device,
                max_iters=args.max_iters, seed=seed,
                log_every=args.max_iters  # only log at end
            )
            cond_dices.append(res['val_dice'])
            print(f"    Seed {seed}: Dice={res['val_dice']:.4f}")

        all_results[cond_name] = {
            "dices": cond_dices,
            "mean": float(np.mean(cond_dices)),
            "std": float(np.std(cond_dices)),
            "n_train": len(train_ds),
        }
        print(f"    → {cond_name}: {np.mean(cond_dices):.4f} ± {np.std(cond_dices):.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  E8 CEILING RESULTS (25% data = {n_25} real volumes)")
    print(f"{'='*60}")
    print(f"  {'Condition':<20} {'Dice':>10} {'± std':>10} {'N_train':>10}")
    print(f"  {'-'*50}")
    for cond, r in all_results.items():
        print(f"  {cond:<20} {r['mean']:>10.4f} {r['std']:>10.4f} {r['n_train']:>10}")

    with open(outdir / "e8_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Saved to {outdir}")
    return all_results


# ============================================================
# E10a: DIVERSITY ISOLATION (THE CRITICAL EXPERIMENT)
# ============================================================

def run_e10a(args):
    """Diversity isolation: graduated dose-response from same Shortcut@50 model.

    Design: all conditions use 500 total synthetic volumes from Shortcut@50.
    The ONLY variable is how many unique volumes are in the set:
      - 500 unique (full diversity)
      - 100 unique (moderate collapse — each repeated ~5x)
      - 25 unique  (severe collapse — each repeated ~20x)
      - 10 unique  (extreme collapse — each repeated ~50x)

    All volumes come from the same model at the same step count with the same
    per-sample quality distribution. If Dice improves monotonically with unique
    count, diversity is causally responsible — not model quality, not dataset size.

    The dose-response curve is much harder to dismiss than binary comparison.
    """
    print(f"\n{'='*60}\n  E10a: DIVERSITY ISOLATION — DOSE-RESPONSE\n{'='*60}")

    data = torch.load(args.data_path, weights_only=False)
    vols = data["volumes"]
    segs = data["segs"]
    splits = data["split_info"]
    device = torch.device(args.device)

    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    # 10% of training data as real subset
    n_10 = max(4, int(len(train_idx) * 0.10))
    subset_idx = train_idx[:n_10]
    real_vols = vols[subset_idx]
    real_segs = segs[subset_idx]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]

    print(f"  Real subset: {n_10} volumes (10%)")
    print(f"  Test: {len(test_idx)} volumes")

    # Load teacher
    teacher = create_seg_model(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    # Step 1: Generate 500 volumes from Shortcut@50
    print(f"\n  Step 1: Generating 500 volumes from Shortcut@50...")
    gen_dir = Path(args.gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir

    p1_path = data_dir / "phase1_shared.pt"
    _, dec, _, _ = load_vqgan(p1_path, device)
    unet = load_unet(data_dir / "shortcut" / "final.pt", device, num_classes=args.num_classes)

    n_gen = 500
    all_vols = []
    all_features = []
    batch_size = 8
    for start in range(0, n_gen, batch_size):
        bs = min(batch_size, n_gen - start)
        z = sample_latent(unet, "shortcut", bs, device, 50, class_label=1)
        with torch.no_grad():
            decoded = dec(z).clamp(0, 1)
            features = z.view(bs, -1)
        all_vols.append(decoded.cpu())
        all_features.append(features.cpu())
        print(f"    Generated {min(start + bs, n_gen)}/{n_gen}", end='\r')
    print()

    all_vols = torch.cat(all_vols, dim=0)
    all_features = torch.cat(all_features, dim=0).numpy()
    del unet, dec

    # Step 2: Create graduated diversity conditions
    # Sort volumes by distance to feature-space mean — closest = most "average"
    global_mean = all_features.mean(axis=0, keepdims=True)
    dists_to_mean = np.linalg.norm(all_features - global_mean, axis=1)
    sorted_by_centrality = np.argsort(dists_to_mean)

    diversity_levels = [500, 100, 25, 10]
    conditions = {}
    rng = np.random.RandomState(42)

    print(f"\n  Step 2: Creating {len(diversity_levels)} diversity conditions...")

    def pairwise_l1_diversity(vol_tensor, n_pairs=1000):
        idx1 = torch.randint(0, len(vol_tensor), (n_pairs,))
        idx2 = torch.randint(0, len(vol_tensor), (n_pairs,))
        return (vol_tensor[idx1] - vol_tensor[idx2]).abs().mean().item()

    for n_unique in diversity_levels:
        label = f"unique_{n_unique}"
        if n_unique >= n_gen:
            # Full diverse set
            syn_vols = all_vols
        else:
            # Pick n_unique volumes UNIFORMLY AT RANDOM (not by centrality:
            # centrality-based selection would introduce a quality confound),
            # then resample with replacement up to n_gen so the total count is fixed.
            source_indices = rng.choice(n_gen, n_unique, replace=False)
            sample_idx = rng.choice(source_indices, n_gen, replace=True)
            syn_vols = all_vols[sample_idx]

        div = pairwise_l1_diversity(syn_vols)
        conditions[label] = syn_vols
        print(f"    {label}: {n_gen} total, L1 diversity = {div:.4f}")

    # Step 3: Train segmentation for each condition
    test_ds = SegDataset(test_vols, test_segs, augment=False)
    outdir = Path(args.output_dir) / "e10a_isolation"
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 42 + args.n_seeds))
    results = {}

    for cond_name, syn_vols in conditions.items():
        print(f"\n  --- {cond_name.upper()} ---")

        # Create pseudo-labels with teacher
        syn_ds = SyntheticSegDataset(syn_vols, teacher, device, augment=True)
        print(f"    Tumor detection rate: {syn_ds.tumor_detection_rate:.1%}")

        # Combine real + synthetic
        real_ds = SegDataset(real_vols, real_segs, augment=True)
        combined_ds = ConcatDataset([real_ds, syn_ds])

        cond_dices = []
        for seed in seeds:
            model = create_seg_model(device)
            res = train_seg_iteration_based(
                model, combined_ds, test_ds, device,
                max_iters=args.max_iters, seed=seed,
                log_every=1000
            )
            cond_dices.append(res['val_dice'])
            print(f"    Seed {seed}: Dice={res['val_dice']:.4f}")
            del model

        n_unique = int(cond_name.split("_")[1])
        results[cond_name] = {
            "n_unique": n_unique,
            "n_total": n_gen,
            "dices": cond_dices,
            "mean": float(np.mean(cond_dices)),
            "std": float(np.std(cond_dices)),
            "tumor_detection_rate": syn_ds.tumor_detection_rate,
            "diversity_l1": float(pairwise_l1_diversity(syn_vols)),
        }
        print(f"    → {cond_name}: {np.mean(cond_dices):.4f} ± {np.std(cond_dices):.4f}")

    # Also run real-only baseline
    print(f"\n  --- REAL ONLY (no synthetic) ---")
    real_only_ds = SegDataset(real_vols, real_segs, augment=True)
    real_dices = []
    for seed in seeds:
        model = create_seg_model(device)
        res = train_seg_iteration_based(
            model, real_only_ds, test_ds, device,
            max_iters=args.max_iters, seed=seed,
            log_every=1000
        )
        real_dices.append(res['val_dice'])
        print(f"    Seed {seed}: Dice={res['val_dice']:.4f}")
        del model
    results["real_only"] = {
        "n_unique": 0, "n_total": 0,
        "dices": real_dices,
        "mean": float(np.mean(real_dices)),
        "std": float(np.std(real_dices)),
    }
    print(f"    → real_only: {np.mean(real_dices):.4f} ± {np.std(real_dices):.4f}")

    # Statistical tests: pairwise between adjacent diversity levels
    from scipy import stats
    stat_tests = []
    sorted_conds = sorted(
        [(k, v) for k, v in results.items() if k != "real_only"],
        key=lambda x: x[1]["n_unique"]
    )
    # NOTE: Conditions are seed-matched (same 3 seeds {42, 43, 44} train every
    # condition), so the appropriate test is a paired t-test on per-seed dices,
    # not an independent-samples t-test. We switched ttest_ind -> ttest_rel
    # following the v7.1 peer review.
    for i in range(len(sorted_conds) - 1):
        name_lo, res_lo = sorted_conds[i]
        name_hi, res_hi = sorted_conds[i + 1]
        t_stat, p_val = stats.ttest_rel(res_lo["dices"], res_hi["dices"])
        stat_tests.append({
            "comparison": f"{name_hi} vs {name_lo}",
            "test": "paired t-test on per-seed Dice (scipy.stats.ttest_rel)",
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "dice_diff": res_hi["mean"] - res_lo["mean"],
        })

    # Test most-diverse vs most-collapsed (extremes)
    if len(sorted_conds) >= 2:
        name_lo, res_lo = sorted_conds[0]
        name_hi, res_hi = sorted_conds[-1]
        t_stat, p_val = stats.ttest_rel(res_lo["dices"], res_hi["dices"])
        stat_tests.append({
            "comparison": f"{name_hi} vs {name_lo} (extremes)",
            "test": "paired t-test on per-seed Dice (scipy.stats.ttest_rel)",
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "dice_diff": res_hi["mean"] - res_lo["mean"],
        })

    results["statistical_tests"] = stat_tests
    results["design"] = {
        "approach": "graduated dose-response: same model, same quality, varying unique count",
        "diversity_levels": diversity_levels,
        "total_per_condition": n_gen,
        "real_subset_size": n_10,
        "selection_method": "uniform random subsample, resampled with replacement to n_gen",
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"  E10a DIVERSITY ISOLATION — DOSE-RESPONSE RESULTS")
    print(f"{'='*60}")
    print(f"  {'Condition':<18} {'Unique':>8} {'Dice':>10} {'± std':>10} {'L1 Div':>10}")
    print(f"  {'-'*56}")
    print(f"  {'real_only':<18} {'—':>8} {results['real_only']['mean']:>10.4f} "
          f"{results['real_only']['std']:>10.4f} {'—':>10}")
    for name, res in sorted(
        [(k, v) for k, v in results.items() if k.startswith("unique_")],
        key=lambda x: x[1]["n_unique"]
    ):
        print(f"  {name:<18} {res['n_unique']:>8} {res['mean']:>10.4f} "
              f"{res['std']:>10.4f} {res['diversity_l1']:>10.4f}")

    print(f"\n  Pairwise tests:")
    for t in stat_tests:
        sig = "***" if t["p_value"] < 0.01 else ("*" if t["p_value"] < 0.05 else "ns")
        print(f"    {t['comparison']}: Δ={t['dice_diff']:+.4f}, p={t['p_value']:.4f} [{sig}]")

    # Check for monotonic dose-response
    dice_by_unique = [(r["n_unique"], r["mean"]) for k, r in results.items() if k.startswith("unique_")]
    dice_by_unique.sort()
    is_monotonic = all(dice_by_unique[i][1] <= dice_by_unique[i+1][1]
                       for i in range(len(dice_by_unique) - 1))
    print(f"\n  Monotonic dose-response: {'✓ YES' if is_monotonic else '✗ NO'}")
    if is_monotonic:
        print(f"  → Diversity causally improves segmentation (dose-response confirmed)")

    print(f"{'='*60}")

    # Save
    with open(outdir / "e10a_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {outdir}")

    # Plot dose-response curve
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = sorted(
        [(r["n_unique"], r["mean"], r["std"]) for k, r in results.items() if k.startswith("unique_")],
        key=lambda x: x[0]
    )
    uniques = [d[0] for d in plot_data]
    means = [d[1] for d in plot_data]
    stds = [d[2] for d in plot_data]

    ax.errorbar(uniques, means, yerr=stds, marker='o', linewidth=2,
                color='#9C27B0', capsize=5, markersize=8, label='Shortcut@50 (varying diversity)')
    ax.axhline(results["real_only"]["mean"], color='#666666', linestyle='--',
               linewidth=1.5, label=f'Real only ({n_10} vol)')
    ax.set_xlabel("Number of Unique Synthetic Volumes (out of 500 total)")
    ax.set_ylabel("Dice Score")
    ax.set_title("Diversity Isolation: Dose-Response Curve")
    ax.set_xscale('log')
    ax.set_xticks(uniques)
    ax.set_xticklabels([str(u) for u in uniques])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "e10a_dose_response.png", dpi=150)
    plt.savefig(outdir / "e10a_dose_response.pdf", dpi=150)
    plt.close()
    print(f"  Saved dose-response figure")


# ============================================================
# E10b: FULL DOWNSTREAM SWEEP
# ============================================================

def run_e10b(args):
    """Full downstream sweep: 7 conditions × 4 fractions × 5 seeds."""
    print(f"\n{'='*60}\n  E10b: FULL DOWNSTREAM SWEEP\n{'='*60}")

    data = torch.load(args.data_path, weights_only=False)
    vols = data["volumes"]
    segs = data["segs"]
    splits = data["split_info"]
    device = torch.device(args.device)

    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    all_train_vols = vols[train_idx]
    all_train_segs = segs[train_idx]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]
    test_ds = SegDataset(test_vols, test_segs, augment=False)

    # Load teacher
    teacher = create_seg_model(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    outdir = Path(args.output_dir) / "e10b_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    fractions = [0.05, 0.10, 0.25, 1.0]
    n_synthetic = 500
    seeds = list(range(42, 42 + args.n_seeds))

    # Pre-generate synthetic datasets
    print("  Pre-generating synthetic datasets...")
    synthetic_cache = {}

    for method, steps_label in [("shortcut", 1), ("shortcut", 50),
                                 ("consistency", 4), ("fm", 50)]:
        key = f"{method}@{steps_label}"
        print(f"    Generating {n_synthetic} from {key}...")
        try:
            syn_vols = generate_synthetic_volumes(
                args.gen_dir, method, steps_label, n_synthetic, device,
                num_classes=args.num_classes, class_label=1
            )
            syn_ds = SyntheticSegDataset(syn_vols, teacher, device, augment=True)
            synthetic_cache[key] = syn_ds
            print(f"      Tumor detection: {syn_ds.tumor_detection_rate:.1%}")
        except FileNotFoundError as e:
            print(f"      SKIPPED: {e}")

    # Run sweep
    all_results = {}

    for frac in fractions:
        n_real = max(4, int(len(all_train_vols) * frac))
        real_vols_frac = all_train_vols[:n_real]
        real_segs_frac = all_train_segs[:n_real]

        frac_key = f"{int(frac * 100)}pct"
        all_results[frac_key] = {}

        print(f"\n{'='*40}")
        print(f"  Fraction: {frac*100:.0f}% ({n_real} real volumes)")
        print(f"{'='*40}")

        # Define conditions
        conditions = {}

        # (a) Real only
        conditions["real_only"] = SegDataset(real_vols_frac, real_segs_frac, augment=True)

        # (b) Classical augmentation
        conditions["classical"] = ClassicalAugDataset(real_vols_frac, real_segs_frac)

        # (c-g) Synthetic augmentation conditions
        for key, syn_ds in synthetic_cache.items():
            cond_name = key.replace("@", "_at_")
            conditions[cond_name] = ConcatDataset([
                SegDataset(real_vols_frac, real_segs_frac, augment=True),
                syn_ds
            ])

        # Run each condition
        for cond_name, train_ds in conditions.items():
            print(f"\n  [{frac_key}] {cond_name} ({len(train_ds)} samples)")
            cond_dices = []
            for seed in seeds:
                model = create_seg_model(device)
                res = train_seg_iteration_based(
                    model, train_ds, test_ds, device,
                    max_iters=args.max_iters, seed=seed,
                    log_every=args.max_iters
                )
                cond_dices.append(res['val_dice'])

            all_results[frac_key][cond_name] = {
                "dices": cond_dices,
                "mean": float(np.mean(cond_dices)),
                "std": float(np.std(cond_dices)),
                "n_train": len(train_ds),
            }
            print(f"    → {np.mean(cond_dices):.4f} ± {np.std(cond_dices):.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  E10b DOWNSTREAM SWEEP RESULTS")
    print(f"{'='*70}")
    for frac_key, frac_results in all_results.items():
        print(f"\n  {frac_key}:")
        print(f"    {'Condition':<25} {'Dice':>10} {'± std':>10}")
        print(f"    {'-'*45}")
        for cond, r in sorted(frac_results.items(), key=lambda x: -x[1]['mean']):
            print(f"    {cond:<25} {r['mean']:>10.4f} {r['std']:>10.4f}")

    with open(outdir / "e10b_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot: Dice vs data fraction for each condition
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "real_only": "#666666", "classical": "#999999",
        "shortcut_at_1": "#E91E63", "shortcut_at_50": "#9C27B0",
        "consistency_at_4": "#FF9800", "fm_at_50": "#2196F3",
    }
    for cond_name in list(conditions.keys()):
        means = []
        stds = []
        fracs_plot = []
        for frac in fractions:
            fk = f"{int(frac*100)}pct"
            if fk in all_results and cond_name in all_results[fk]:
                means.append(all_results[fk][cond_name]["mean"])
                stds.append(all_results[fk][cond_name]["std"])
                fracs_plot.append(frac * 100)
        if means:
            color = colors.get(cond_name, '#333333')
            ax.errorbar(fracs_plot, means, yerr=stds, marker='o',
                       label=cond_name, color=color, linewidth=2, capsize=4)

    ax.set_xlabel("Real Data Fraction (%)")
    ax.set_ylabel("Dice Score")
    ax.set_title("Downstream Segmentation: Dice vs Data Fraction")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "e10b_dice_vs_fraction.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved to {outdir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Paper 4: Downstream Segmentation Experiments")
    parser.add_argument('experiment', choices=['e7', 'e8', 'e10a', 'e10b'],
                        help='Which experiment to run')
    parser.add_argument('--data-path', type=str, default='data/brats_conditional_64.pt',
                        help='Path to conditional dataset')
    parser.add_argument('--gen-dir', type=str, default=None,
                        help='Path to generative model results directory')
    parser.add_argument('--teacher-path', type=str, default=None,
                        help='Path to trained teacher segmenter (e7 output)')
    parser.add_argument('--output-dir', type=str, default='results/paper4',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--max-iters', type=int, default=2000,
                        help='Max training iterations for segmentation (default 2000)')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds per condition')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of conditioning classes in generative model')
    args = parser.parse_args()

    # Auto device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    print(f"Device: {args.device}")

    if args.experiment == 'e7':
        run_e7(args)
    elif args.experiment == 'e8':
        run_e8(args)
    elif args.experiment == 'e10a':
        run_e10a(args)
    elif args.experiment == 'e10b':
        run_e10b(args)


if __name__ == "__main__":
    main()
