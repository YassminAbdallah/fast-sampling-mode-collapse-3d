#!/usr/bin/env python3
"""
Downstream Segmentation: Direct Augmentation with Pseudo-Labels
================================================================

Tests the paper's core claim: diverse synthetic data provides more
generalization benefit than collapsed synthetic data.

Approach (standard medical imaging augmentation):
  1. Train a TEACHER segmenter on ALL real labeled data
  2. Generate synthetic volumes from each generative method
  3. Apply teacher to create pseudo-labels for synthetic volumes
  4. Train STUDENT segmenters on {real subset + synthetic w/ pseudo-labels}
  5. Compare test Dice across augmentation sources and data fractions

Conditions tested:
  (a) Real only                   -- baseline (no augmentation)
  (b) Real + Consistency@4 synth  -- high quality, low diversity
  (c) Real + Shortcut@4 synth    -- moderate quality, high diversity
  (d) Real + FM@50 synth         -- moderate quality, high diversity

Data fractions: 25%, 50%, 100% of labeled training data
Each with 3 seeds for statistical reliability.

Why this works:
  - Collapsed synthetic data (Consistency) = adding near-identical volumes
    -> same as repeating training examples, minimal benefit
  - Diverse synthetic data (Shortcut, FM) = covering more anatomical
    variation -> genuine augmentation, especially at low data fractions

Usage:
    # If you need to preprocess BraTS with seg masks first:
    python downstream_segmentation.py --mode preprocess \
        --run-dir results/brats_benchmark_20260221_193306 \
        --data-path data/ --brats-raw-dir data/brats/

    # Run full experiment (generate + train teacher + augmented training):
    python downstream_segmentation.py --mode all \
        --run-dir results/brats_benchmark_20260221_193306 \
        --data-path data/brats_preprocessed_64.pt \
        --seg-path data/brats_seg_preprocessed_64.pt \
        --n-synthetic 180

    # If synthetic volumes already generated, just train:
    python downstream_segmentation.py --mode train \
        --run-dir results/brats_benchmark_20260221_193306 \
        --data-path data/brats_preprocessed_64.pt \
        --seg-path data/brats_seg_preprocessed_64.pt

Output:
    {data_subdir}/downstream_seg/
        synthetic_fm50.pt, synthetic_shortcut4.pt, synthetic_consistency4.pt
        pseudo_labels_fm50.pt, pseudo_labels_shortcut4.pt, pseudo_labels_consistency4.pt
        seg_results.json
        fig_downstream_dice.pdf/png
        table_downstream.tex

Requires: models_shared.py in the same directory.
"""

import os, json, argparse, warnings, time, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from models_shared import (
    Decoder3D, load_unet, load_vqgan, sample_latent
)


# ============================================================
# BraTS Preprocessing
# ============================================================

def preprocess_brats_with_seg(brats_dir, output_dir, target_size=64, max_volumes=300):
    """Preprocess BraTS volumes + seg masks to 64 cubed."""
    import nibabel as nib

    brats_dir = Path(brats_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(glob.glob(str(brats_dir / "BraTS-GLI-*")))
    if not subjects:
        subjects = sorted(glob.glob(str(brats_dir / "*" / "BraTS-GLI-*")))
    print(f"Found {len(subjects)} BraTS subjects")

    volumes, seg_masks = [], []
    for i, subj_dir in enumerate(subjects[:max_volumes]):
        subj_dir = Path(subj_dir)
        t2f_files = list(subj_dir.glob("*t2f*nii*"))
        seg_files = list(subj_dir.glob("*seg*nii*"))
        if not t2f_files or not seg_files:
            continue
        try:
            vol = nib.load(str(t2f_files[0])).get_fdata().astype(np.float32)
            seg = nib.load(str(seg_files[0])).get_fdata().astype(np.int64)
            p1, p99 = np.percentile(vol[vol > 0], [1, 99]) if (vol > 0).any() else (0, 1)
            vol = np.clip(vol, p1, p99)
            vol = (vol - p1) / (p99 - p1 + 1e-8)
            seg_binary = (seg > 0).astype(np.int64)
            vol_t = F.interpolate(torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float(),
                                  size=target_size, mode='trilinear', align_corners=False)
            seg_t = F.interpolate(torch.from_numpy(seg_binary).unsqueeze(0).unsqueeze(0).float(),
                                  size=target_size, mode='nearest')
            volumes.append(vol_t.squeeze(0))
            seg_masks.append(seg_t.squeeze(0).squeeze(0).long())
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{min(len(subjects), max_volumes)}")
        except Exception as e:
            print(f"  Error processing {subj_dir.name}: {e}")
            continue

    volumes_tensor = torch.stack(volumes)
    seg_tensor = torch.stack(seg_masks)
    torch.save(volumes_tensor, output_dir / "brats_preprocessed_64.pt")
    torch.save(seg_tensor, output_dir / "brats_seg_preprocessed_64.pt")
    print(f"Saved {len(volumes)} volumes and masks to {output_dir}")
    return volumes_tensor, seg_tensor


# ============================================================
# Segmentation Model (lightweight 3D U-Net)
# ============================================================

class SegResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1))
    def forward(self, x):
        return x + self.net(x)


class SegUNet3D(nn.Module):
    """Lightweight 3D U-Net for binary segmentation."""
    def __init__(self, in_ch=1, n_classes=2, base_ch=32):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv3d(in_ch, base_ch, 3, padding=1), SegResBlock(base_ch))
        self.down1 = nn.Conv3d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.e2 = SegResBlock(base_ch * 2)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.e3 = SegResBlock(base_ch * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv3d(base_ch * 4, base_ch * 2, 3, padding=1))
        self.d2 = SegResBlock(base_ch * 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv3d(base_ch * 2, base_ch, 3, padding=1))
        self.d1 = SegResBlock(base_ch)
        self.out = nn.Conv3d(base_ch, n_classes, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.down1(s1))
        s3 = self.e3(self.down2(s2))
        x = self.up2(s3) + s2
        x = self.d2(x)
        x = self.up1(x) + s1
        x = self.d1(x)
        return self.out(x)


# ============================================================
# Datasets
# ============================================================

class SegDataset(Dataset):
    def __init__(self, volumes, masks, augment=False):
        self.volumes = volumes
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, i):
        vol = self.volumes[i]
        mask = self.masks[i]
        if self.augment:
            for dim in [1, 2, 3]:
                if torch.rand(1).item() > 0.5:
                    vol = vol.flip(dim)
                    mask = mask.flip(dim - 1)
        return vol, mask


# ============================================================
# Metrics
# ============================================================

def dice_score(pred, target, smooth=1e-5):
    pred_flat = pred.reshape(-1).float()
    target_flat = target.reshape(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return float((2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


# ============================================================
# Teacher Training
# ============================================================

def train_teacher(train_vols, train_masks, val_vols, val_masks, dev,
                  epochs=120, lr=1e-3, base_ch=32):
    print(f"\n  Training teacher on {len(train_vols)} real volumes, {epochs} epochs...")
    model = SegUNet3D(1, 2, base_ch).to(dev)
    train_ds = SegDataset(train_vols, train_masks, augment=True)
    val_ds = SegDataset(val_vols, val_masks)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    weights = torch.tensor([1.0, 5.0], device=dev)
    best_dice, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for vols, masks in train_loader:
            vols, masks = vols.to(dev), masks.to(dev)
            logits = model(vols)
            loss = F.cross_entropy(logits, masks, weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for vols, masks in val_loader:
                    vols, masks = vols.to(dev), masks.to(dev)
                    pred = model(vols).argmax(1)
                    for b in range(len(vols)):
                        val_dices.append(dice_score(pred[b], masks[b]))
            mean_dice = float(np.mean(val_dices))
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"val_dice={mean_dice:.4f} (best={best_dice:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Teacher trained: best val Dice = {best_dice:.4f}")
    return model, best_dice


@torch.no_grad()
def generate_pseudo_labels(teacher, volumes, dev, batch_size=8):
    teacher.eval()
    all_masks = []
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size].to(dev)
        logits = teacher(batch)
        pred = logits.argmax(1)
        all_masks.append(pred.cpu())
    return torch.cat(all_masks, 0)


# ============================================================
# Student Training (with augmented data)
# ============================================================

def train_student(real_vols, real_masks, syn_vols, syn_masks,
                  val_vols, val_masks, dev,
                  epochs=100, lr=1e-3, base_ch=32, label=""):
    model = SegUNet3D(1, 2, base_ch).to(dev)

    if syn_vols is not None and len(syn_vols) > 0:
        combined_vols = torch.cat([real_vols, syn_vols], 0)
        combined_masks = torch.cat([real_masks, syn_masks], 0)
        print(f"    Training on {len(real_vols)} real + {len(syn_vols)} synthetic = {len(combined_vols)} total")
    else:
        combined_vols = real_vols
        combined_masks = real_masks
        print(f"    Training on {len(real_vols)} real only")

    train_ds = SegDataset(combined_vols, combined_masks, augment=True)
    val_ds = SegDataset(val_vols, val_masks)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    weights = torch.tensor([1.0, 5.0], device=dev)
    best_dice, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for vols, masks in train_loader:
            vols, masks = vols.to(dev), masks.to(dev)
            logits = model(vols)
            loss = F.cross_entropy(logits, masks, weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for vols, masks in val_loader:
                    vols, masks = vols.to(dev), masks.to(dev)
                    pred = model(vols).argmax(1)
                    for b in range(len(vols)):
                        val_dices.append(dice_score(pred[b], masks[b]))
            mean_dice = float(np.mean(val_dices))
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 40 == 0 or epoch == epochs - 1:
                print(f"      Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                      f"val_dice={mean_dice:.4f} (best={best_dice:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_dice


def evaluate_model(model, test_vols, test_masks, dev):
    model.eval()
    test_ds = SegDataset(test_vols, test_masks)
    test_loader = DataLoader(test_ds, batch_size=4)
    dices = []
    with torch.no_grad():
        for vols, masks in test_loader:
            vols, masks = vols.to(dev), masks.to(dev)
            pred = model(vols).argmax(1)
            for b in range(len(vols)):
                dices.append(dice_score(pred[b], masks[b]))
    return float(np.mean(dices)), float(np.std(dices)), dices


# ============================================================
# Synthetic Data Generation
# ============================================================

@torch.no_grad()
def generate_synthetic_dataset(unet, dec, method, steps, n, dev, batch_size=8):
    all_vols = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = sample_latent(unet, method, bs, dev, steps)
        gen = dec(z).clamp(0, 1)
        all_vols.append(gen.cpu())
    return torch.cat(all_vols, 0)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Downstream segmentation via direct augmentation")
    parser.add_argument("--mode", choices=["generate", "train", "all", "preprocess"], default="all")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seg-path", type=str, default=None)
    parser.add_argument("--brats-raw-dir", type=str, default=None)
    parser.add_argument("--n-synthetic", type=int, default=180)
    parser.add_argument("--teacher-epochs", type=int, default=120)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else run_dir
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # --- Preprocess ---
    if args.mode == "preprocess":
        if not args.brats_raw_dir:
            print("ERROR: --brats-raw-dir required"); return
        out = Path(args.data_path)
        if out.is_dir():
            preprocess_brats_with_seg(args.brats_raw_dir, out)
        else:
            preprocess_brats_with_seg(args.brats_raw_dir, out.parent)
        return

    out_dir = data_dir / "downstream_seg"
    out_dir.mkdir(exist_ok=True)

    # --- Generate ---
    if args.mode in ("generate", "all"):
        print("\n" + "=" * 60)
        print("  PHASE 1: Generating Synthetic Datasets")
        print("=" * 60)

        _, dec, _, _ = load_vqgan(data_dir / "phase1_shared.pt", dev)
        gen_configs = [
            ("fm", 50, "synthetic_fm50.pt"),
            ("shortcut", 4, "synthetic_shortcut4.pt"),
            ("consistency", 4, "synthetic_consistency4.pt"),
        ]
        for method, steps, filename in gen_configs:
            ckpt = data_dir / method / "final.pt"
            if not ckpt.exists():
                print(f"  Skipping {method}"); continue
            if (out_dir / filename).exists():
                print(f"  {filename} exists, skipping"); continue
            print(f"\n  Generating {args.n_synthetic} volumes: {method}@{steps}")
            torch.manual_seed(42)
            unet = load_unet(ckpt, dev)
            syn = generate_synthetic_dataset(unet, dec, method, steps, args.n_synthetic, dev)
            torch.save(syn, out_dir / filename)
            print(f"    Saved {syn.shape}")
            del unet
            if dev.type == "mps": torch.mps.empty_cache()
            elif dev.type == "cuda": torch.cuda.empty_cache()
        del dec

        if args.mode == "generate":
            print("\nDone. Run --mode train next."); return

    # --- Train ---
    if args.mode in ("train", "all"):
        if not args.seg_path:
            print("ERROR: --seg-path required"); return

        print("\n" + "=" * 60)
        print("  PHASE 2: Train Teacher + Augmented Students")
        print("=" * 60)

        vols = torch.load(args.data_path, weights_only=True)
        if vols.dim() == 4: vols = vols.unsqueeze(1)
        masks = torch.load(args.seg_path, weights_only=True)
        print(f"Data: {vols.shape}, Masks: {masks.shape}")
        print(f"Tumor prevalence: {(masks > 0).float().mean():.4f}")

        n = len(vols)
        np.random.seed(42)
        perm = np.random.permutation(n)
        n_train, n_val = int(0.6 * n), int(0.2 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_vols, train_masks = vols[train_idx], masks[train_idx]
        val_vols, val_masks = vols[val_idx], masks[val_idx]
        test_vols, test_masks = vols[test_idx], masks[test_idx]
        print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

        # Step 1: Train teacher
        print("\n" + "=" * 60)
        print("  STEP 1: Training Teacher")
        print("=" * 60)

        teacher, teacher_dice = train_teacher(
            train_vols, train_masks, val_vols, val_masks, dev,
            epochs=args.teacher_epochs)
        teacher_test_dice, _, _ = evaluate_model(teacher, test_vols, test_masks, dev)
        print(f"  Teacher test Dice: {teacher_test_dice:.4f}")

        # Step 2: Pseudo-labels
        print("\n" + "=" * 60)
        print("  STEP 2: Generating Pseudo-Labels")
        print("=" * 60)

        syn_data = {}
        for name, filename in [("fm50", "synthetic_fm50.pt"),
                                ("shortcut4", "synthetic_shortcut4.pt"),
                                ("consistency4", "synthetic_consistency4.pt")]:
            vol_path = out_dir / filename
            label_path = out_dir / f"pseudo_labels_{name}.pt"
            if not vol_path.exists():
                print(f"  {filename} not found, skipping"); continue
            syn_vols = torch.load(vol_path, weights_only=True)
            if label_path.exists():
                syn_masks = torch.load(label_path, weights_only=True)
                print(f"  {name}: loaded cached pseudo-labels")
            else:
                print(f"  {name}: generating pseudo-labels for {len(syn_vols)} volumes...")
                syn_masks = generate_pseudo_labels(teacher, syn_vols, dev)
                torch.save(syn_masks, label_path)
                tumor_frac = (syn_masks > 0).float().mean()
                print(f"    Pseudo-label tumor prevalence: {tumor_frac:.4f}")
            syn_data[name] = (syn_vols, syn_masks)

        del teacher
        if dev.type == "mps": torch.mps.empty_cache()

        # Step 3: Train students
        print("\n" + "=" * 60)
        print("  STEP 3: Training Augmented Students")
        print("=" * 60)

        fractions = [0.25, 0.50, 1.0]
        conditions = [("none", None)]
        for name in ["consistency4", "shortcut4", "fm50"]:
            if name in syn_data:
                conditions.append((name, syn_data[name]))

        all_results = {}

        for seed in args.seeds:
            print(f"\n{'=' * 60}")
            print(f"  SEED {seed}")
            print(f"{'=' * 60}")

            for frac in fractions:
                n_use = max(4, int(len(train_vols) * frac))
                torch.manual_seed(seed)
                np.random.seed(seed)
                frac_perm = np.random.permutation(len(train_vols))[:n_use]
                frac_vols = train_vols[frac_perm]
                frac_masks = train_masks[frac_perm]
                frac_label = f"{int(frac * 100)}%"
                print(f"\n  --- {frac_label} data ({n_use} real volumes) ---")

                for cond_name, cond_data in conditions:
                    key = f"{cond_name}_{frac_label}"
                    if key not in all_results:
                        all_results[key] = {"test_dice": [], "val_dice": [],
                                            "condition": cond_name, "frac": frac}
                    syn_v = cond_data[0] if cond_data else None
                    syn_m = cond_data[1] if cond_data else None
                    print(f"\n    Condition: {cond_name}, data: {frac_label}")
                    torch.manual_seed(seed)

                    model, val_dice = train_student(
                        frac_vols, frac_masks, syn_v, syn_m,
                        val_vols, val_masks, dev,
                        epochs=args.student_epochs, label=cond_name)

                    test_dice, test_std, test_dices = evaluate_model(
                        model, test_vols, test_masks, dev)
                    all_results[key]["test_dice"].append(test_dice)
                    all_results[key]["val_dice"].append(val_dice)
                    print(f"    -> Test Dice: {test_dice:.4f} (val: {val_dice:.4f})")

                    del model
                    if dev.type == "mps": torch.mps.empty_cache()
                    elif dev.type == "cuda": torch.cuda.empty_cache()

        # ============================================================
        # Save and visualize
        # ============================================================

        summary = {}
        for key, data in all_results.items():
            dices = data["test_dice"]
            summary[key] = {
                "condition": data["condition"],
                "frac": data["frac"],
                "test_dice_mean": float(np.mean(dices)),
                "test_dice_std": float(np.std(dices)),
                "test_dice_seeds": dices,
                "n_seeds": len(dices),
            }

        save_data = {
            "metadata": {
                "seeds": args.seeds,
                "teacher_epochs": args.teacher_epochs,
                "student_epochs": args.student_epochs,
                "teacher_test_dice": teacher_test_dice,
                "fractions": fractions,
                "n_train_total": n_train,
                "n_val": n_val,
                "n_test": len(test_idx),
                "n_synthetic_per_method": args.n_synthetic,
                "approach": "direct_augmentation_pseudo_labels",
            },
            "results": summary,
        }
        with open(out_dir / "seg_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved to {out_dir / 'seg_results.json'}")

        # --- Figure ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
        styles = {
            "none":          ("#666666", "--", "o", "Real only (no augmentation)"),
            "consistency4":  ("#e74c3c", "-",  "v", "Real + Consistency@4 synthetic"),
            "shortcut4":     ("#9b59b6", "-",  "^", "Real + Shortcut@4 synthetic"),
            "fm50":          ("#3498db", "-",  "D", "Real + FM@50 synthetic"),
        }
        for cond_name, (color, ls, marker, label) in styles.items():
            xs, ys, errs = [], [], []
            for frac in fractions:
                key = f"{cond_name}_{int(frac * 100)}%"
                if key in summary:
                    xs.append(frac * 100)
                    ys.append(summary[key]["test_dice_mean"])
                    errs.append(summary[key]["test_dice_std"])
            if xs:
                ax.errorbar(xs, ys, yerr=errs, color=color, linestyle=ls,
                            marker=marker, markersize=9, capsize=5, linewidth=2.2,
                            label=label, markeredgecolor='white', markeredgewidth=0.8)
        ax.axhline(y=teacher_test_dice, color='#2ecc71', linestyle=':', linewidth=1.5,
                    alpha=0.7, label=f'Teacher (all real, Dice={teacher_test_dice:.3f})')
        ax.set_xlabel("Labeled Training Data (%)", fontsize=12)
        ax.set_ylabel("Test Dice Score", fontsize=12)
        ax.set_title("Downstream Tumor Segmentation:\nDoes Synthetic Diversity Help?", fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xticks([25, 50, 100])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.4)
        plt.tight_layout()
        fig.savefig(out_dir / "fig_downstream_dice.pdf", bbox_inches='tight', dpi=150)
        fig.savefig(out_dir / "fig_downstream_dice.png", bbox_inches='tight', dpi=150)
        plt.close(fig)

        # --- LaTeX table ---
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Downstream BraTS tumor segmentation. Student trained on real "
            r"labeled data augmented with synthetic volumes (pseudo-labeled by teacher). "
            r"Mean $\pm$ std Dice over " + str(len(args.seeds)) + r" seeds.}",
            r"\label{tab:downstream}",
            r"\footnotesize",
            r"\begin{tabular}{l" + "c" * len(fractions) + "}",
            r"\toprule",
            r"Augmentation Source & " + " & ".join(
                [f"{int(f * 100)}\\% labeled" for f in fractions]) + r" \\",
            r"\midrule",
        ]
        for cond_name, (_, _, _, label) in styles.items():
            row = f"  {label}"
            for frac in fractions:
                key = f"{cond_name}_{int(frac * 100)}%"
                if key in summary:
                    m = summary[key]["test_dice_mean"]
                    s = summary[key]["test_dice_std"]
                    row += f" & {m:.3f}$\\pm${s:.3f}"
                else:
                    row += " & ---"
            row += r" \\"
            lines.append(row)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        (out_dir / "table_downstream.tex").write_text("\n".join(lines))

        # --- Summary ---
        print(f"\n{'=' * 60}")
        print(f"  DOWNSTREAM SEGMENTATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Teacher (all real): {teacher_test_dice:.4f}")
        for frac in fractions:
            print(f"\n  {int(frac * 100)}% labeled data:")
            for cond_name in ["none", "consistency4", "shortcut4", "fm50"]:
                key = f"{cond_name}_{int(frac * 100)}%"
                if key in summary:
                    m = summary[key]["test_dice_mean"]
                    s = summary[key]["test_dice_std"]
                    delta = m - summary[f"none_{int(frac * 100)}%"]["test_dice_mean"]
                    sign = "+" if delta >= 0 else ""
                    print(f"    {cond_name:>15}: {m:.4f} +/- {s:.4f} ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()
