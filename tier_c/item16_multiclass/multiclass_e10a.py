#!/usr/bin/env python3
"""
Item 16 — Multi-class BraTS dose-response (Tier C)
====================================================

Re-runs the E10a diversity-isolation experiment with multi-class segmentation
(4 classes: 0 = background, 1 = necrotic, 2 = edema, 3 = enhancing) instead of
binary. Reports the standard BraTS regions:
  - WT (Whole Tumor)     = labels {1, 2, 3}
  - TC (Tumor Core)      = labels {1, 3}
  - ET (Enhancing Tumor) = label {3}

The experiment design is identical to the binary E10a in
src/paper4/conditional_segmentation.py:
  - 4 augmentation pools (10 / 25 / 100 / 500 unique synthetic volumes,
    each replicated to 500 total)
  - Real-only baseline (10% of training data, 18 volumes)
  - 3 segmenter seeds per condition
  - 2000 iterations per run
  - Teacher segmenter trained on all real BraTS, then generates pseudo-labels
    for the 500 Shortcut@50 synthetic volumes

What's different from binary E10a:
  - segmenter has out_channels=4 instead of 2
  - DiceCELoss with to_onehot_y=True, softmax=True (multi-class)
  - evaluation computes per-region Dice (WT, TC, ET) and mean across regions
  - teacher segmenter is also multi-class

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item16_multiclass/multiclass_e10a.py \\
        --data-path data/brats_conditional_64.pt \\
        --seg-path  data/brats_seg_preprocessed_64.pt \\
        --gen-dir   results/paper4/brats_benchmark_20260324_154926 \\
        --output-dir results/paper4/paper4/e10a_isolation_multiclass \\
        --n-seeds 3 \\
        --teacher-iters 5000 \\
        --seg-iters 2000

Estimated total time on Apple M-series:
  - Teacher training (5000 iters): ~2 hours
  - Generating 500 synthetic vols at Shortcut@50: ~5 min (you may already have these cached)
  - Producing multi-class pseudo-labels for 500 vols: ~5 min
  - E10a dose-response: 4 unique-count conditions × 3 seeds × 2000 iters = 12 runs
                        at ~70 min each = ~14 hours
  - Real-only baseline: 3 seeds × 2000 iters = ~3.5 hours
  - Total: ~17-19 hours

Output: results/paper4/paper4/e10a_isolation_multiclass/e10a_multiclass_results.json
        with WT/TC/ET Dice per condition.
"""

import os, sys, json, time, argparse, warnings, random
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

# Add src/paper4 to path so we can import models_shared and load_unet utility
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

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

class MultiClassSegDataset(Dataset):
    """Dataset that preserves multi-class BraTS labels (4 classes)."""
    def __init__(self, volumes, segs, augment=False):
        self.v = volumes.unsqueeze(1) if volumes.dim() == 4 else volumes
        self.s = segs.unsqueeze(1) if segs.dim() == 4 else segs
        # Do NOT binarize — preserve labels {0, 1, 2, 3}
        # Clamp to valid range in case of float storage
        self.s = self.s.long().clamp(0, 3)
        self.augment = augment

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i].float(), self.s[i].long()
        if self.augment:
            for d in [2, 3, 4]:
                if random.random() < 0.5:
                    vol = torch.flip(vol, dims=[d - 1])
                    seg = torch.flip(seg, dims=[d - 1])
        return vol, seg


# ============================================================
# MULTI-CLASS METRICS — BraTS WT/TC/ET
# ============================================================

def region_masks(seg):
    """Given a label tensor with values in {0,1,2,3}, return three binary masks.

    WT (Whole Tumor):    labels 1, 2, 3
    TC (Tumor Core):     labels 1, 3
    ET (Enhancing):      label 3 only
    """
    wt = (seg >= 1).float()
    tc = ((seg == 1) | (seg == 3)).float()
    et = (seg == 3).float()
    return wt, tc, et


def dice_score(pred, target, eps=1e-6):
    """Soft Dice score for binary masks."""
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def per_region_dice(pred_labels, true_labels):
    """Returns (wt, tc, et) Dice given two label tensors (shape (B, 1, D, H, W) or (B, D, H, W))."""
    pred_wt, pred_tc, pred_et = region_masks(pred_labels)
    true_wt, true_tc, true_et = region_masks(true_labels)
    return (
        dice_score(pred_wt, true_wt).item(),
        dice_score(pred_tc, true_tc).item(),
        dice_score(pred_et, true_et).item(),
    )


# ============================================================
# MODEL
# ============================================================

def create_seg_model(device, num_classes=4):
    """4-channel output BasicUNet for multi-class BraTS."""
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=[32, 64, 128, 256, 32, 32],
    ).to(device)
    return model


# ============================================================
# TRAINING LOOP
# ============================================================

def train_seg_iteration_based(
    model, train_dataset, n_iters, device, batch_size=4,
    lr=1e-3, weight_decay=1e-5, log_every=200, seed=42,
):
    """Iteration-based training matching the binary E10a setup."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    iter_loader = iter(loader)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            recent = np.mean(losses[-log_every:])
            print(f"    iter {it+1:5d}/{n_iters}  loss={recent:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return float(np.mean(losses[-200:]))


def evaluate_seg(model, dataset, device, batch_size=8):
    """Returns dict with WT/TC/ET Dice averaged across the test set."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    wts, tcs, ets = [], [], []
    with torch.no_grad():
        for vols, segs in loader:
            vols = vols.to(device).float()
            segs = segs.to(device).long()
            logits = model(vols)
            pred = logits.argmax(dim=1, keepdim=True)
            wt, tc, et = per_region_dice(pred, segs)
            wts.append(wt); tcs.append(tc); ets.append(et)
    return {
        "wt": float(np.mean(wts)),
        "tc": float(np.mean(tcs)),
        "et": float(np.mean(ets)),
        "mean": float(np.mean([np.mean(wts), np.mean(tcs), np.mean(ets)])),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_brats_data(data_path, seg_path):
    """Loads volumes (with class labels) and the original multi-class seg masks.

    Returns (volumes, conditional_labels, multi_class_segs)
    """
    d = torch.load(data_path, weights_only=False, map_location='cpu')
    if isinstance(d, dict):
        volumes = d.get("volumes", d.get("v", None))
        cond_labels = d.get("labels", d.get("y", None))
    else:
        volumes, cond_labels = d, None
    if volumes is None:
        raise ValueError(f"Could not find volumes in {data_path}")

    segs = torch.load(seg_path, weights_only=False, map_location='cpu')
    if isinstance(segs, dict):
        segs = segs.get("segs", segs.get("masks", segs.get("s", None)))
    if segs is None:
        raise ValueError(f"Could not find segs in {seg_path}")

    # Align lengths
    N = min(len(volumes), len(segs))
    volumes = volumes[:N]
    segs = segs[:N]
    if cond_labels is not None:
        cond_labels = cond_labels[:N]

    # Ensure shape (N, 1, 64, 64, 64) for volumes; (N, 64, 64, 64) or (N, 1, 64, 64, 64) for segs
    if volumes.dim() == 4:
        volumes = volumes.unsqueeze(1)
    if segs.dim() == 5:
        segs = segs.squeeze(1)

    print(f"Loaded {N} volumes (shape: {tuple(volumes.shape)}) and segs (shape: {tuple(segs.shape)})")
    print(f"  Seg unique labels: {sorted(torch.unique(segs.long()).tolist())}")
    return volumes, cond_labels, segs


def split_train_val_test(volumes, segs, train_n=180, val_n=60, test_n=60, seed=42):
    """Same fixed split as the binary E10a (seed=42, sizes 180/60/60)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(volumes), generator=g)
    tr = perm[:train_n]
    va = perm[train_n:train_n + val_n]
    te = perm[train_n + val_n:train_n + val_n + test_n]
    return (volumes[tr], segs[tr]), (volumes[va], segs[va]), (volumes[te], segs[te])


# ============================================================
# SYNTHETIC GENERATION (Shortcut@50)
# ============================================================

@torch.no_grad()
def generate_shortcut_volumes(gen_dir, n_volumes, device, num_classes=2, steps=50, class_label=1):
    """Generates n_volumes brain volumes using the trained Shortcut FM model.

    Mirrors the existing generate_synthetic_volumes() in
    src/paper4/conditional_segmentation.py. Locates phase1_shared.pt and
    shortcut/final.pt inside the first *pct_* subfolder of gen_dir.
    """
    gen_dir = Path(gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir
    p1_path = data_dir / "phase1_shared.pt"
    ckpt_path = data_dir / "shortcut" / "final.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint not found: {p1_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Shortcut FM checkpoint not found: {ckpt_path}")

    print(f"  Loading VQ-GAN from {p1_path}")
    _, dec, _, _ = load_vqgan(p1_path, device)
    dec.eval()
    print(f"  Loading Shortcut FM from {ckpt_path}")
    unet = load_unet(ckpt_path, device, num_classes=num_classes)
    unet.eval()

    print(f"  Generating {n_volumes} volumes at Shortcut@{steps} (class_label={class_label}) ...")
    t0 = time.time()
    all_vols = []
    bs = 8
    remaining = n_volumes
    while remaining > 0:
        b = min(bs, remaining)
        z = sample_latent(unet, "shortcut", b, device, steps, class_label=class_label)
        vols = dec(z).clamp(0, 1).cpu()
        all_vols.append(vols)
        remaining -= b
        print(f"    generated {n_volumes - remaining}/{n_volumes}", end="\r")
    print()
    volumes = torch.cat(all_vols, dim=0)[:n_volumes]
    print(f"  Done in {time.time() - t0:.1f}s, shape={tuple(volumes.shape)}")
    return volumes


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/brats_conditional_64.pt",
                        help="Conditional volume + class-label tensor")
    parser.add_argument("--seg-path",  default="data/brats_seg_preprocessed_64.pt",
                        help="Multi-class segmentation masks tensor")
    parser.add_argument("--gen-dir",   default="results/paper4/brats_benchmark_20260324_154926",
                        help="Directory with trained Shortcut FM checkpoint")
    parser.add_argument("--output-dir", default="results/paper4/paper4/e10a_isolation_multiclass",
                        help="Where to save results JSON and figure")
    parser.add_argument("--n-synth", type=int, default=500,
                        help="Total synthetic volumes generated (default 500)")
    parser.add_argument("--unique-counts", nargs="+", type=int, default=[10, 25, 100, 500],
                        help="Unique-volume conditions to test")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of segmenter seeds per condition")
    parser.add_argument("--teacher-iters", type=int, default=5000,
                        help="Iterations to train the multi-class teacher")
    parser.add_argument("--seg-iters", type=int, default=2000,
                        help="Iterations per segmenter training run")
    parser.add_argument("--real-frac", type=float, default=0.10,
                        help="Real-data fraction for augmentation conditions")
    parser.add_argument("--num-classes-gen", type=int, default=2,
                        help="Conditional class count of the generator (2 = small/large tumor)")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | mps (auto if not set)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- Load data -----
    volumes, cond_labels, segs = load_brats_data(args.data_path, args.seg_path)
    (train_v, train_s), (val_v, val_s), (test_v, test_s) = split_train_val_test(
        volumes, segs, train_n=180, val_n=60, test_n=60, seed=42
    )
    print(f"\nSplits: train={len(train_v)} val={len(val_v)} test={len(test_v)}")

    test_ds = MultiClassSegDataset(test_v, test_s, augment=False)

    # ----- Phase 1: Train multi-class teacher (or load cached) -----
    teacher_ckpt = outdir / "teacher_multiclass.pt"
    teacher_metrics_path = outdir / "teacher_metrics.json"
    if teacher_ckpt.exists() and teacher_metrics_path.exists():
        print(f"\n========== PHASE 1: Loading cached teacher from {teacher_ckpt} ==========")
        teacher = create_seg_model(device, num_classes=4)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        with open(teacher_metrics_path) as f:
            teacher_metrics = json.load(f)
        print(f"Cached teacher test Dice — WT={teacher_metrics['wt']:.3f}  "
              f"TC={teacher_metrics['tc']:.3f}  ET={teacher_metrics['et']:.3f}  "
              f"mean={teacher_metrics['mean']:.3f}")
    else:
        print("\n========== PHASE 1: Multi-class teacher segmenter ==========")
        teacher_ds = MultiClassSegDataset(
            torch.cat([train_v, val_v], dim=0),
            torch.cat([train_s, val_s], dim=0),
            augment=True,
        )
        print(f"Teacher training set: {len(teacher_ds)} volumes")
        teacher = create_seg_model(device, num_classes=4)
        train_seg_iteration_based(
            teacher, teacher_ds, args.teacher_iters, device,
            batch_size=4, seed=42, log_every=500,
        )
        teacher_metrics = evaluate_seg(teacher, test_ds, device, batch_size=4)
        print(f"\nTeacher test Dice — WT={teacher_metrics['wt']:.3f}  "
              f"TC={teacher_metrics['tc']:.3f}  ET={teacher_metrics['et']:.3f}  "
              f"mean={teacher_metrics['mean']:.3f}")
        torch.save(teacher.state_dict(), teacher_ckpt)
        with open(teacher_metrics_path, "w") as f:
            json.dump(teacher_metrics, f, indent=2)

    # ----- Phase 2: Generate or load 500 Shortcut@50 synthetic volumes -----
    cache_path = outdir / f"shortcut50_{args.n_synth}vols.pt"
    if cache_path.exists():
        print(f"\n========== PHASE 2: Loading cached synthetic volumes from {cache_path} ==========")
        synth_vols = torch.load(cache_path, weights_only=False)
    else:
        print("\n========== PHASE 2: Generating Shortcut@50 synthetic volumes ==========")
        synth_vols = generate_shortcut_volumes(
            args.gen_dir, args.n_synth, device, num_classes=args.num_classes_gen,
        )
        torch.save(synth_vols, cache_path)

    # ----- Phase 3: Multi-class pseudo-labels for synthetic volumes -----
    print("\n========== PHASE 3: Pseudo-labelling synthetic volumes ==========")
    teacher.eval()
    pseudo_segs = []
    bs = 8
    with torch.no_grad():
        for i in range(0, len(synth_vols), bs):
            v = synth_vols[i:i + bs].to(device).float()
            if v.dim() == 4: v = v.unsqueeze(1)
            logits = teacher(v)
            pseudo = logits.argmax(dim=1, keepdim=True).cpu()
            pseudo_segs.append(pseudo)
    pseudo_segs = torch.cat(pseudo_segs, dim=0).squeeze(1)
    pseudo_dist = torch.bincount(pseudo_segs.flatten(), minlength=4)
    print(f"Pseudo-label distribution: BG={pseudo_dist[0].item()}  "
          f"NCR={pseudo_dist[1].item()}  ED={pseudo_dist[2].item()}  ET={pseudo_dist[3].item()}")

    # ----- Phase 4: Dose-response experiment -----
    print("\n========== PHASE 4: Dose-response (multi-class) ==========")
    results = {"design": vars(args), "teacher_metrics": teacher_metrics, "conditions": {}}

    # Real-only baseline: 10% of training data = 18 volumes
    n_real = max(1, int(args.real_frac * len(train_v)))
    g = torch.Generator().manual_seed(42)
    real_idx = torch.randperm(len(train_v), generator=g)[:n_real]
    real_v = train_v[real_idx]; real_s = train_s[real_idx]
    print(f"\nReal-only baseline: {n_real} volumes, {args.n_seeds} seeds")

    real_only_dices = []
    for seed in range(42, 42 + args.n_seeds):
        print(f"\n  Real-only seed={seed}")
        ds = MultiClassSegDataset(real_v, real_s, augment=True)
        model = create_seg_model(device, num_classes=4)
        train_seg_iteration_based(model, ds, args.seg_iters, device,
                                  batch_size=4, seed=seed, log_every=500)
        metrics = evaluate_seg(model, test_ds, device)
        print(f"    test Dice — WT={metrics['wt']:.3f}  TC={metrics['tc']:.3f}  ET={metrics['et']:.3f}  mean={metrics['mean']:.3f}")
        real_only_dices.append(metrics)
    results["conditions"]["real_only"] = {
        "n_unique": 0, "n_total": n_real,
        "dices": real_only_dices,
        "wt_mean": float(np.mean([d['wt'] for d in real_only_dices])),
        "tc_mean": float(np.mean([d['tc'] for d in real_only_dices])),
        "et_mean": float(np.mean([d['et'] for d in real_only_dices])),
        "mean_mean": float(np.mean([d['mean'] for d in real_only_dices])),
    }

    # Each unique-count condition: random subsample of synth_vols, replicate to n_synth total
    for n_unique in args.unique_counts:
        print(f"\n----- Condition: unique_{n_unique} -----")
        g = torch.Generator().manual_seed(123 + n_unique)
        idx = torch.randperm(len(synth_vols), generator=g)[:n_unique]
        sub_v = synth_vols[idx]
        sub_s = pseudo_segs[idx]
        rep = (args.n_synth + n_unique - 1) // n_unique
        full_v = sub_v.repeat(rep, 1, 1, 1, 1)[:args.n_synth]
        full_s = sub_s.repeat(rep, 1, 1, 1)[:args.n_synth]

        cond_dices = []
        for seed in range(42, 42 + args.n_seeds):
            print(f"\n  unique_{n_unique} seed={seed}")
            real_ds = MultiClassSegDataset(real_v, real_s, augment=True)
            syn_ds = MultiClassSegDataset(full_v, full_s, augment=True)
            combo = ConcatDataset([real_ds, syn_ds])
            model = create_seg_model(device, num_classes=4)
            train_seg_iteration_based(model, combo, args.seg_iters, device,
                                      batch_size=4, seed=seed, log_every=500)
            metrics = evaluate_seg(model, test_ds, device)
            print(f"    test Dice — WT={metrics['wt']:.3f}  TC={metrics['tc']:.3f}  ET={metrics['et']:.3f}  mean={metrics['mean']:.3f}")
            cond_dices.append(metrics)

        key = f"unique_{n_unique}"
        results["conditions"][key] = {
            "n_unique": n_unique,
            "n_total": args.n_synth + n_real,
            "dices": cond_dices,
            "wt_mean": float(np.mean([d['wt'] for d in cond_dices])),
            "tc_mean": float(np.mean([d['tc'] for d in cond_dices])),
            "et_mean": float(np.mean([d['et'] for d in cond_dices])),
            "mean_mean": float(np.mean([d['mean'] for d in cond_dices])),
        }

    # ----- Save and plot -----
    results["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(outdir / "e10a_multiclass_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outdir/'e10a_multiclass_results.json'}")

    # Quick dose-response plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    region_names = ["wt", "tc", "et", "mean"]
    region_titles = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing (ET)", "Mean"]
    real_only = results["conditions"]["real_only"]
    for ax, region, title in zip(axes, region_names, region_titles):
        xs, ys, errs = [], [], []
        for k, v in results["conditions"].items():
            if k == "real_only": continue
            xs.append(v["n_unique"])
            dices = [d[region] for d in v["dices"]]
            ys.append(np.mean(dices))
            errs.append(np.std(dices))
        order = np.argsort(xs)
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; errs = [errs[i] for i in order]
        ax.errorbar(xs, ys, yerr=errs, marker='o', color='purple', label='Shortcut@50')
        ax.axhline(real_only[f"{region}_mean"], color='gray', ls='--', label='Real only (18 vol)')
        ax.set_xscale('log')
        ax.set_xlabel("Unique synthetic volumes")
        ax.set_ylabel("Dice")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "e10a_multiclass_dose_response.png", dpi=150, bbox_inches='tight')
    plt.savefig(outdir / "e10a_multiclass_dose_response.pdf", bbox_inches='tight')
    print(f"Saved: {outdir/'e10a_multiclass_dose_response.png'}")


if __name__ == "__main__":
    main()
