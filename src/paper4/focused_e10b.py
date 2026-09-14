#!/usr/bin/env python3
"""
Focused E10b: Downstream Comparison for Merged Paper
=====================================================

Answers the clinical question: "I have X labeled scans. Which augmentation
strategy should I use?"

Conditions:
  (a) Real only — baseline
  (b) Real + Classical augmentation (flips, rotations, intensity jitter)
  (c) Real + Shortcut@50 synthetic (high diversity)
  (d) Real + Consistency@4 synthetic (collapsed, high SSIM)

Data fractions: 5% (~9 vol), 10% (~18 vol), 25% (~45 vol)
Seeds: 3 per condition (42, 43, 44)
Iterations: 2000 per run

Total: 4 conditions × 3 fractions × 3 seeds = 36 runs ≈ 50 hours
  Or with --n-seeds 2: 24 runs ≈ 34 hours

Usage:
    python focused_e10b.py \
        --data-path data/brats_conditional_64.pt \
        --gen-dir results/brats_benchmark_20260324_154926 \
        --teacher-path results/paper4/e7_teacher/teacher_best.pt \
        --output-dir results/paper4/e10b_focused \
        --num-classes 2 \
        --n-seeds 3
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

try:
    from monai.networks.nets import BasicUNet
    from monai.losses import DiceCELoss
except ImportError:
    print("ERROR: MONAI not installed. Run: pip install monai")
    sys.exit(1)

from models_shared import load_vqgan, load_unet, sample_latent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================
# DATASETS (copied from conditional_segmentation.py, standalone)
# ============================================================

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, volumes, segs, augment=False):
        self.v = volumes.unsqueeze(1) if volumes.dim() == 4 else volumes
        self.s = segs.unsqueeze(1) if segs.dim() == 4 else segs
        self.s = (self.s > 0).float()
        self.augment = augment

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


class ClassicalAugDataset(torch.utils.data.Dataset):
    def __init__(self, volumes, segs):
        self.v = volumes.unsqueeze(1) if volumes.dim() == 4 else volumes
        self.s = segs.unsqueeze(1) if segs.dim() == 4 else segs
        self.s = (self.s > 0).float()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i].clone(), self.s[i].clone()
        for d in range(3):
            if torch.rand(1).item() > 0.5:
                vol = torch.flip(vol, [d + 1])
                seg = torch.flip(seg, [d + 1])
        if torch.rand(1).item() > 0.5:
            axes = [(1, 2), (1, 3), (2, 3)]
            ax = axes[torch.randint(0, 3, (1,)).item()]
            k = torch.randint(1, 4, (1,)).item()
            vol = torch.rot90(vol, k, ax)
            seg = torch.rot90(seg, k, ax)
        scale = 0.9 + 0.2 * torch.rand(1).item()
        vol = (vol * scale).clamp(0, 1)
        vol = vol + 0.01 * torch.randn_like(vol)
        vol = vol.clamp(0, 1)
        return vol, seg


class SyntheticSegDataset(torch.utils.data.Dataset):
    def __init__(self, synthetic_volumes, teacher_model, device, threshold=0.5):
        self.v = synthetic_volumes.unsqueeze(1) if synthetic_volumes.dim() == 4 else synthetic_volumes
        teacher_model.eval()
        all_segs = []
        with torch.no_grad():
            for start in range(0, len(self.v), 8):
                batch = self.v[start:start+8].to(device)
                logits = teacher_model(batch)
                probs = torch.softmax(logits, dim=1)[:, 1:2]
                masks = (probs > threshold).float()
                all_segs.append(masks.cpu())
        self.s = torch.cat(all_segs, dim=0)
        has_tumor = (self.s.sum(dim=(1, 2, 3, 4)) > 0).float()
        self.tumor_detection_rate = has_tumor.mean().item()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        vol, seg = self.v[i], self.s[i]
        # Random flips
        for d in range(3):
            if torch.rand(1).item() > 0.5:
                vol = torch.flip(vol, [d + 1])
                seg = torch.flip(seg, [d + 1])
        return vol, seg


# ============================================================
# TRAINING
# ============================================================

def train_seg(train_ds, test_ds, device, max_iters=2000, batch_size=4,
              lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                      features=(32, 64, 128, 256, 32, 32)).to(device)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, drop_last=True)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    model.train()
    step = 0
    data_iter = iter(dl)

    while step < max_iters:
        try:
            vol, seg = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            vol, seg = next(data_iter)

        vol, seg = vol.to(device), seg.to(device).long()
        loss = criterion(model(vol), seg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

    # Evaluate
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    dice_scores = []
    with torch.no_grad():
        for vol, seg in test_dl:
            vol, seg = vol.to(device), seg.to(device)
            pred = torch.argmax(model(vol), dim=1, keepdim=True).float()
            target = (seg > 0).float()
            for i in range(pred.shape[0]):
                p, t = pred[i].flatten(), target[i].flatten()
                inter = (p * t).sum()
                union = p.sum() + t.sum()
                dice = (2.0 * inter / union).item() if union > 0 else (1.0 if t.sum() == 0 else 0.0)
                dice_scores.append(dice)

    del model
    return float(np.mean(dice_scores))


def generate_synthetic(gen_dir, method, steps, n_volumes, device, num_classes=2):
    gen_dir = Path(gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir

    _, dec, _, _ = load_vqgan(data_dir / "phase1_shared.pt", device)
    unet = load_unet(data_dir / method / "final.pt", device, num_classes=num_classes)

    all_vols = []
    for start in range(0, n_volumes, 8):
        bs = min(8, n_volumes - start)
        z = sample_latent(unet, method, bs, device, steps, class_label=1)
        with torch.no_grad():
            vols = dec(z).clamp(0, 1)
        all_vols.append(vols.cpu())

    del unet, dec
    return torch.cat(all_vols, dim=0)[:n_volumes]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Focused E10b: downstream comparison")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--output-dir", default="results/paper4/e10b_focused")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available(): args.device = "mps"
        elif torch.cuda.is_available(): args.device = "cuda"
        else: args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    data = torch.load(args.data_path, weights_only=False)
    vols = data["volumes"]
    segs = data["segs"]
    splits = data["split_info"]
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    all_train_vols = vols[train_idx]
    all_train_segs = segs[train_idx]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]
    test_ds = SegDataset(test_vols, test_segs, augment=False)

    print(f"Total train: {len(all_train_vols)}, Test: {len(test_vols)}")

    # Load teacher
    teacher = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                        features=(32, 64, 128, 256, 32, 32)).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=True))
    teacher.eval()

    # Pre-generate synthetic datasets
    print(f"\n  Generating {args.n_synthetic} Shortcut@50 volumes...")
    shortcut_vols = generate_synthetic(args.gen_dir, "shortcut", 50, args.n_synthetic,
                                        device, args.num_classes)
    shortcut_syn = SyntheticSegDataset(shortcut_vols, teacher, device)
    print(f"    Tumor detection: {shortcut_syn.tumor_detection_rate:.1%}")

    print(f"  Generating {args.n_synthetic} Consistency@4 volumes...")
    consist_vols = generate_synthetic(args.gen_dir, "consistency", 4, args.n_synthetic,
                                      device, args.num_classes)
    consist_syn = SyntheticSegDataset(consist_vols, teacher, device)
    print(f"    Tumor detection: {consist_syn.tumor_detection_rate:.1%}")

    del teacher  # free memory

    # Run experiment
    fractions = [0.05, 0.10, 0.25]
    seeds = list(range(42, 42 + args.n_seeds))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for frac in fractions:
        n_real = max(4, int(len(all_train_vols) * frac))
        real_v = all_train_vols[:n_real]
        real_s = all_train_segs[:n_real]
        frac_key = f"{int(frac * 100)}pct"
        all_results[frac_key] = {"n_real": n_real}

        print(f"\n{'='*60}")
        print(f"  {frac_key}: {n_real} real volumes")
        print(f"{'='*60}")

        conditions = {
            "real_only": SegDataset(real_v, real_s, augment=True),
            "classical_aug": ClassicalAugDataset(real_v, real_s),
            "shortcut_50": ConcatDataset([SegDataset(real_v, real_s, augment=True), shortcut_syn]),
            "consistency_4": ConcatDataset([SegDataset(real_v, real_s, augment=True), consist_syn]),
        }

        for cond_name, train_ds in conditions.items():
            print(f"\n  [{frac_key}] {cond_name} ({len(train_ds)} samples)")
            dices = []
            for seed in seeds:
                dice = train_seg(train_ds, test_ds, device,
                                 max_iters=args.max_iters, seed=seed)
                dices.append(dice)
                print(f"    Seed {seed}: {dice:.4f}")

            result = {
                "dices": dices,
                "mean": float(np.mean(dices)),
                "std": float(np.std(dices)),
                "n_train": len(train_ds),
            }
            all_results[frac_key][cond_name] = result
            print(f"    → {np.mean(dices):.4f} ± {np.std(dices):.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  FOCUSED E10b: DOWNSTREAM COMPARISON RESULTS")
    print(f"{'='*70}")

    for frac_key in sorted(all_results.keys()):
        fr = all_results[frac_key]
        n = fr.get("n_real", "?")
        print(f"\n  {frac_key} ({n} real volumes):")
        print(f"    {'Condition':<20} {'Dice':>10} {'± std':>10} {'N_train':>10}")
        print(f"    {'-'*50}")
        for cond in ["real_only", "classical_aug", "shortcut_50", "consistency_4"]:
            if cond in fr:
                r = fr[cond]
                print(f"    {cond:<20} {r['mean']:>10.4f} {r['std']:>10.4f} {r['n_train']:>10}")

    # Save
    with open(outdir / "e10b_focused_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ============================================================
    # Figure: Dice vs Data Fraction (hero figure for merged paper)
    # ============================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {
        "real_only": "#777777",
        "classical_aug": "#4CAF50",
        "shortcut_50": "#9C27B0",
        "consistency_4": "#FF9800",
    }
    labels = {
        "real_only": "Real only",
        "classical_aug": "Real + Classical aug",
        "shortcut_50": "Real + Shortcut@50",
        "consistency_4": "Real + Consistency@4",
    }

    for cond in ["real_only", "classical_aug", "shortcut_50", "consistency_4"]:
        fracs_plot = []
        means = []
        stds = []
        for frac in fractions:
            fk = f"{int(frac*100)}pct"
            if fk in all_results and cond in all_results[fk]:
                fracs_plot.append(frac * 100)
                means.append(all_results[fk][cond]["mean"])
                stds.append(all_results[fk][cond]["std"])
        if means:
            ax.errorbar(fracs_plot, means, yerr=stds, marker='o',
                        label=labels[cond], color=colors[cond],
                        linewidth=2.5, capsize=5, markersize=8)

    ax.set_xlabel("Real Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Downstream Tumor Segmentation: Augmentation Strategy Comparison", fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([5, 10, 25])
    plt.tight_layout()
    plt.savefig(outdir / "fig_dice_vs_fraction.png", dpi=150)
    plt.savefig(outdir / "fig_dice_vs_fraction.pdf", dpi=150)
    plt.close()

    # ============================================================
    # LaTeX table
    # ============================================================
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Downstream tumor segmentation Dice across data fractions and augmentation strategies. "
        r"Mean $\pm$ std over " + str(args.n_seeds) + r" seeds. "
        r"Bold: best per fraction.}",
        r"\label{tab:downstream}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(fractions) + "}",
        r"\toprule",
        r"Strategy & " + " & ".join([f"{int(f*100)}\\% ({max(4,int(len(all_train_vols)*f))} vol)" for f in fractions]) + r" \\",
        r"\midrule",
    ]

    for cond in ["real_only", "classical_aug", "shortcut_50", "consistency_4"]:
        vals = []
        for frac in fractions:
            fk = f"{int(frac*100)}pct"
            if fk in all_results and cond in all_results[fk]:
                r = all_results[fk][cond]
                vals.append(f"{r['mean']:.3f}$\\pm${r['std']:.3f}")
            else:
                vals.append("---")
        lines.append(f"  {labels[cond]} & " + " & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (outdir / "table_downstream.tex").write_text("\n".join(lines))

    print(f"\n  Saved results to {outdir}")
    print(f"  Saved figure: fig_dice_vs_fraction.pdf")
    print(f"  Saved table: table_downstream.tex")


if __name__ == "__main__":
    main()
