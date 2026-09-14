#!/usr/bin/env python3
"""
Pixel-Space vs Latent-Space Trajectory Comparison
===================================================

Trains FM directly in pixel space (NO VQ-GAN), compares trajectory
straightness to latent-space FM. If pixel trajectories are more curved,
this proves VQ-GAN flattens trajectories.

Usage:
    python pixel_trajectory_comparison.py --data-path data/ixi_preprocessed_64.pt --latent-run-dir results/ixi_benchmark_20260221_045741

Output:
    results/pixel_trajectory_{dataset}_{timestamp}/
        pixel_fm_checkpoint.pt
        trajectory_comparison.json
        fig_trajectory_comparison.pdf
"""

import os, json, time, math, argparse, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from models_shared import DenoisingUNet3D, SinEmb, load_unet


# ============================================================
# Pixel-space U-Net (NEW model, not loading from checkpoint)
# ============================================================

class PixelUNet3D(nn.Module):
    """Lightweight 3D U-Net for pixel-space FM. Input/output: [B, 1, 64, 64, 64]."""
    def __init__(self, in_ch=1, td=128):
        super().__init__()
        self.te = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        self.e1 = nn.Sequential(nn.Conv3d(in_ch, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                                nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.d1 = nn.Conv3d(16, 16, 4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.d2 = nn.Conv3d(32, 32, 4, stride=2, padding=1)
        self.e3 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.d3 = nn.Conv3d(64, 64, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                 nn.Conv3d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.tp1 = nn.Linear(td, 16); self.tp2 = nn.Linear(td, 32)
        self.tp3 = nn.Linear(td, 64); self.tpm = nn.Linear(td, 64)
        self.u3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.dc3 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv3d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.dc2 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv3d(32, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                                nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU())
        self.dc1 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
                                 nn.Conv3d(16, in_ch, 3, padding=1))

    def forward(self, x, t):
        te = self.te(t)
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        h3 = self.e3(self.d2(h2)) + self.tp3(te)[:, :, None, None, None]
        hm = self.mid(self.d3(h3)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc3(torch.cat([self.u3(hm), h3], 1))
        h = self.dc2(torch.cat([self.u2(h), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


class BrainDS(Dataset):
    def __init__(self, vols):
        self.v = vols if vols.dim() == 5 else vols.unsqueeze(1)
    def __len__(self): return len(self.v)
    def __getitem__(self, i): return {"volume": self.v[i]}


def train_pixel_fm(unet, dl, dev, epochs=150, lr=1e-4):
    opt = torch.optim.AdamW(unet.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for e in range(1, epochs + 1):
        el = 0; n = 0; t0 = time.time(); unet.train()
        for batch in dl:
            x = batch["volume"].to(dev)
            z0 = torch.randn_like(x); t = torch.rand(x.shape[0], device=dev)
            te = t[:, None, None, None, None]
            xt = (1 - te) * z0 + te * x
            l = F.mse_loss(unet(xt, t), x - z0)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
            el += l.item(); n += 1
        sch.step()
        if e % 10 == 0 or e == 1:
            print(f"  [{e:3d}/{epochs}] pixel_fm_loss={el/n:.6f} | {time.time()-t0:.1f}s")


def measure_trajectories(unet, n_traj, dev, space, n_dense=256, ch=8, size=8):
    """Measure trajectory straightness and curvature."""
    unet.eval()
    if space == "latent":
        z0 = torch.randn(n_traj, ch, size, size, size, device=dev)
    else:
        z0 = torch.randn(n_traj, 1, 64, 64, 64, device=dev)

    dt = 1.0 / n_dense
    record_every = max(1, n_dense // 32)
    positions = [z0.cpu()]
    z = z0.clone()

    with torch.no_grad():
        for i in range(n_dense):
            t = torch.full((n_traj,), i * dt, device=dev)
            z = z + unet(z, t) * dt
            if i % record_every == 0 or i == n_dense - 1:
                positions.append(z.cpu())

    straightness_scores = []
    max_curvatures = []
    for traj_idx in range(n_traj):
        pts = [p[traj_idx].flatten().numpy() for p in positions]
        direct = np.linalg.norm(pts[-1] - pts[0])
        path_len = sum(np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1))
        straightness_scores.append(direct / max(path_len, 1e-10))

        start, end = pts[0], pts[-1]
        direction = end - start; dir_norm = np.linalg.norm(direction)
        if dir_norm > 0: direction = direction / dir_norm
        devs = []
        for p in pts[1:-1]:
            proj = np.dot(p - start, direction) * direction + start
            devs.append(np.linalg.norm(p - proj) / max(dir_norm, 1e-10))
        max_curvatures.append(max(devs) if devs else 0)

    return {
        "straightness_mean": float(np.mean(straightness_scores)),
        "straightness_std": float(np.std(straightness_scores)),
        "max_curvature_mean": float(np.mean(max_curvatures)),
        "max_curvature_std": float(np.std(max_curvatures)),
        "n_trajectories": n_traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--latent-run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-trajectories", type=int, default=16)
    args = parser.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    vols = torch.load(args.data_path, weights_only=True)
    if vols.dim() == 4: vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} volumes")

    ds_name = "ixi" if "ixi" in args.data_path.lower() else "brats"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/pixel_trajectory_{ds_name}_{ts}"); out_dir.mkdir(parents=True, exist_ok=True)

    # Train pixel-space FM
    print(f"\n{'='*60}\n  TRAINING PIXEL-SPACE FM ({args.epochs} epochs)\n{'='*60}")
    pixel_unet = PixelUNet3D(in_ch=1).to(dev)
    print(f"  Params: {sum(p.numel() for p in pixel_unet.parameters()):,}")
    ds = BrainDS(vols); dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    train_pixel_fm(pixel_unet, dl, dev, epochs=args.epochs)
    torch.save(pixel_unet.state_dict(), out_dir / "pixel_fm_checkpoint.pt")

    # Measure pixel trajectories
    print(f"\n{'='*60}\n  MEASURING PIXEL-SPACE TRAJECTORIES\n{'='*60}")
    pixel_results = measure_trajectories(pixel_unet, args.n_trajectories, dev, "pixel")
    print(f"  Straightness: {pixel_results['straightness_mean']:.4f} ± {pixel_results['straightness_std']:.4f}")
    print(f"  Max curvature: {pixel_results['max_curvature_mean']:.4f} ± {pixel_results['max_curvature_std']:.4f}")
    del pixel_unet

    # Measure latent trajectories
    print(f"\n{'='*60}\n  MEASURING LATENT-SPACE TRAJECTORIES\n{'='*60}")
    latent_dir = Path(args.latent_run_dir)
    sub = sorted([d for d in latent_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    dd = sub[0] if sub else latent_dir
    fm_path = dd / "fm" / "final.pt"
    latent_results = None
    if fm_path.exists():
        latent_unet = load_unet(fm_path, dev)
        latent_results = measure_trajectories(latent_unet, args.n_trajectories, dev, "latent")
        print(f"  Straightness: {latent_results['straightness_mean']:.4f} ± {latent_results['straightness_std']:.4f}")
        print(f"  Max curvature: {latent_results['max_curvature_mean']:.4f} ± {latent_results['max_curvature_std']:.4f}")
        del latent_unet
    else:
        print(f"  No FM checkpoint at {fm_path}")

    # Save
    comparison = {"pixel_space": pixel_results, "latent_space": latent_results, "dataset": ds_name}
    with open(out_dir / "trajectory_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Pixel Space\n(64³ × 1)"]
    straight = [pixel_results["straightness_mean"]]
    straight_err = [pixel_results["straightness_std"]]
    curv = [pixel_results["max_curvature_mean"]]
    curv_err = [pixel_results["max_curvature_std"]]
    colors = ["#E53935"]
    if latent_results:
        labels.append("Latent Space\n(8³ × 8)")
        straight.append(latent_results["straightness_mean"])
        straight_err.append(latent_results["straightness_std"])
        curv.append(latent_results["max_curvature_mean"])
        curv_err.append(latent_results["max_curvature_std"])
        colors.append("#2196F3")

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, straight, yerr=straight_err, color=colors, capsize=5, width=0.5)
    ax1.set_ylabel("Straightness"); ax1.set_title("Trajectory Straightness\n(1.0 = straight)", fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(0.8, 1.01)
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.5)
    for b, v in zip(bars1, straight): ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{v:.4f}', ha='center', fontweight='bold')

    bars2 = ax2.bar(x, [c*100 for c in curv], yerr=[e*100 for e in curv_err], color=colors, capsize=5, width=0.5)
    ax2.set_ylabel("Max Curvature (%)"); ax2.set_title("Max Trajectory Curvature\n(lower = better for shortcuts)", fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    for b, v in zip(bars2, curv): ax2.text(b.get_x()+b.get_width()/2, b.get_height()*100+0.5, f'{v*100:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf','png']: plt.savefig(out_dir/f'fig_trajectory_comparison.{ext}', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}\n  SUMMARY ({ds_name.upper()})\n{'='*60}")
    print(f"  Pixel:  straightness={pixel_results['straightness_mean']:.4f}  curvature={pixel_results['max_curvature_mean']:.4f}")
    if latent_results:
        print(f"  Latent: straightness={latent_results['straightness_mean']:.4f}  curvature={latent_results['max_curvature_mean']:.4f}")
        print(f"  Latent is {latent_results['straightness_mean']/pixel_results['straightness_mean']:.3f}× straighter")
    print(f"\n  Results: {out_dir}")


if __name__ == "__main__":
    main()
