#!/usr/bin/env python3
"""
Visual Quality Grids for Paper 3 (A8)
======================================

Generates publication-quality visual comparison figures:
1. Grid: All 5 methods at best step count (axial + sagittal), with real reference
2. Grid: Shortcut@1, @4, @16, @128 vs FM@50 showing adaptive-step progression
3. Both IXI and BraTS versions

Usage:
    python visual_grids.py --run-dir results/ixi_benchmark_20260221_045741
    python visual_grids.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt --dataset brats
"""

import os, sys, json, math, argparse, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


# ============================================================
# MODEL CLASSES
# ============================================================

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1))
    def forward(self, x): return x + self.net(x)

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, lat_ch=8, n_res=2):
        super().__init__()
        layers = [nn.Conv3d(in_ch, 32, 3, padding=1), nn.SiLU(),
                  nn.Conv3d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
                  nn.Conv3d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
                  nn.Conv3d(128, lat_ch, 4, stride=2, padding=1), nn.SiLU()]
        for _ in range(n_res): layers.append(ResBlock3D(lat_ch))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Upsample3D(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv3d(ci, co, 3, padding=1)
    def forward(self, x): return self.conv(self.up(x))

class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, lat_ch=8, n_res=2):
        super().__init__()
        layers = []
        for _ in range(n_res): layers.append(ResBlock3D(lat_ch))
        self.res = nn.Sequential(*layers)
        self.up = nn.Sequential(
            Upsample3D(lat_ch, 128), nn.SiLU(),
            Upsample3D(128, 64), nn.SiLU(),
            Upsample3D(64, 32), nn.SiLU(),
            nn.Conv3d(32, out_ch, 3, padding=1), nn.Sigmoid())
    def forward(self, x): return self.up(self.res(x))

class VectorQuantizer(nn.Module):
    def __init__(self, ne=256, d=8, beta=0.25):
        super().__init__()
        self.ne, self.d, self.beta = ne, d, beta
        self.emb = nn.Embedding(ne, d)
        nn.init.uniform_(self.emb.weight, -1/ne, 1/ne)
    def forward(self, z):
        zp = z.permute(0, 2, 3, 4, 1).contiguous()
        d = torch.cdist(zp.view(-1, self.d), self.emb.weight)
        idx = d.argmin(-1)
        zq = self.emb(idx).view(zp.shape).permute(0, 4, 1, 2, 3)
        loss = F.mse_loss(zq, z.detach()) + self.beta * F.mse_loss(zq.detach(), z)
        zq = z + (zq - z).detach()
        return zq, loss, idx

class SinEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.d = d
    def forward(self, t):
        h = self.d // 2; e = math.log(10000) / (h - 1)
        e = torch.exp(torch.arange(h, device=t.device) * -e)
        e = t[:, None] * e[None, :]; return torch.cat([e.sin(), e.cos()], -1)

class DenoisingUNet3D(nn.Module):
    def __init__(self, ch=8, td=128):
        super().__init__()
        self.te = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        self.de = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        self.e1 = nn.Sequential(nn.Conv3d(ch, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.d1 = nn.Conv3d(64, 64, 4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                nn.Conv3d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU())
        self.d2 = nn.Conv3d(128, 128, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.Conv3d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
                                 nn.Conv3d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU())
        self.tp1 = nn.Linear(td, 64); self.tp2 = nn.Linear(td, 128); self.tpm = nn.Linear(td, 128)
        self.u2 = Upsample3D(128, 128)
        self.dc2 = nn.Sequential(nn.Conv3d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                                 nn.Conv3d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.u1 = Upsample3D(64, 64)
        self.dc1 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv3d(64, ch, 3, padding=1))
    def forward(self, x, t, d=None):
        te = self.te(t)
        if d is not None: te = te + self.de(d)
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        hm = self.mid(self.d2(h2)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc2(torch.cat([self.u2(hm), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(unet, dec, method, n, dev, steps, ch=8):
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    if method == "shortcut":
        d_val = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * d_val, device=dev)
            d = torch.full((n,), d_val, device=dev)
            z = z + d_val * unet(z, t, d=d)
    elif method == "ddpm":
        T = 1000; b = torch.linspace(1e-4, 0.02, T); a = 1 - b; ac = torch.cumprod(a, 0)
        ss = max(T // steps, 1); ts = list(range(T - 1, -1, -ss))
        for tv in ts:
            t = torch.full((n,), tv, device=dev, dtype=torch.float32)
            np_ = unet(z, t)
            z = (1/math.sqrt(a[tv])) * (z - (b[tv]/math.sqrt(1-ac[tv])) * np_)
            if tv > 0: z = z + math.sqrt(b[tv]) * torch.randn_like(z)
    elif method == "consistency":
        if steps == 1:
            t = torch.zeros(n, device=dev); z = z + unet(z, t)
        else:
            ts_s = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
            for i, t_val in enumerate(ts_s):
                t = torch.full((n,), t_val.item(), device=dev)
                x_hat = z + (1 - t_val) * unet(z, t)
                if i < steps - 1:
                    t_next = ts_s[i+1].item()
                    z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
                else: z = x_hat
    else:  # fm, rectified
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=dev); z = z + unet(z, t) * dt
    return dec(z).clamp(0, 1)


def load_unet(data_dir, method, dev, ch=8):
    unet = DenoisingUNet3D(ch).to(dev)
    p2 = torch.load(data_dir / method / "final.pt", map_location=dev, weights_only=True)
    unet_state = {k.replace("unet.", ""): v for k, v in p2.items() if k.startswith("unet.")}
    if unet_state: unet.load_state_dict(unet_state)
    else: unet.load_state_dict(p2)
    unet.eval()
    return unet


# ============================================================
# FIGURE 1: 5-METHOD COMPARISON (best steps)
# ============================================================

def make_method_comparison(data_dir, dec, vols, dev, out_dir, dataset="ixi"):
    """All 5 methods at their best step count, 3 samples × (axial + sagittal)."""
    # Best steps per method per dataset
    best_steps = {
        "ixi": {"ddpm": 1000, "fm": 10, "rectified": 5, "consistency": 1, "shortcut": 1},
        "brats": {"ddpm": 1000, "fm": 10, "rectified": 5, "consistency": 2, "shortcut": 1},
    }
    steps_map = best_steps.get(dataset, best_steps["ixi"])

    n_samples = 3
    n_methods = 5
    method_order = ["ddpm", "fm", "rectified", "consistency", "shortcut"]
    method_names = {"ddpm": "DDPM\n@1000", "fm": "FM\n@10", "rectified": "Rect. Flow\n@5",
                    "consistency": "Consistency\n@1", "shortcut": "Shortcut FM\n@1 (Ours)"}

    fig, axes = plt.subplots(n_samples, n_methods + 1, figsize=(3.2 * (n_methods + 1), 3.2 * n_samples))
    # "+1" for the real column

    torch.manual_seed(42)

    # Real volumes
    for row in range(n_samples):
        v = vols[row, 0].numpy()
        mid = v.shape[0] // 2
        axes[row, 0].imshow(v[:, :, mid], cmap='gray', vmin=0, vmax=1)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title('Real', fontsize=13, fontweight='bold')

    # Generated volumes
    for col_idx, method in enumerate(method_order):
        ckpt = data_dir / method / "final.pt"
        if not ckpt.exists():
            for row in range(n_samples):
                axes[row, col_idx + 1].axis('off')
                axes[row, col_idx + 1].text(0.5, 0.5, 'N/A', transform=axes[row, col_idx + 1].transAxes,
                                             ha='center', va='center', fontsize=14, color='gray')
            continue

        unet = load_unet(data_dir, method, dev)
        steps = steps_map[method]
        torch.manual_seed(42)  # same noise for all methods
        gen = generate(unet, dec, method, n_samples, dev, steps)
        del unet

        for row in range(n_samples):
            v = gen[row, 0].cpu().numpy()
            mid = v.shape[0] // 2
            axes[row, col_idx + 1].imshow(v[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axes[row, col_idx + 1].axis('off')
            if row == 0:
                axes[row, col_idx + 1].set_title(method_names[method], fontsize=12, fontweight='bold')

    ds_label = "IXI (Healthy Brain)" if dataset == "ixi" else "BraTS (Glioma)"
    plt.suptitle(f'5-Method Comparison — {ds_label} — Axial View',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'fig_method_comparison_{dataset}'
    plt.savefig(out_dir / f'{fname}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.pdf/png")


# ============================================================
# FIGURE 2: SHORTCUT STEP PROGRESSION
# ============================================================

def make_shortcut_progression(data_dir, dec, vols, dev, out_dir, dataset="ixi"):
    """Shortcut@1, @4, @16, @128 vs FM@50, showing adaptive-step quality scaling."""

    configs = [
        ("shortcut", 1, "Shortcut\n@1 step"),
        ("shortcut", 4, "Shortcut\n@4 steps"),
        ("shortcut", 16, "Shortcut\n@16 steps"),
        ("shortcut", 128, "Shortcut\n@128 steps"),
        ("fm", 50, "FM\n@50 steps"),
    ]

    n_samples = 3
    n_cols = len(configs) + 1  # +1 for real

    fig, axes = plt.subplots(n_samples * 2, n_cols, figsize=(3 * n_cols, 3 * n_samples * 2))
    # 2 views per sample: axial and sagittal

    unet_cache = {}

    for col_idx, (method, steps, label) in enumerate(configs):
        if method not in unet_cache:
            unet_cache[method] = load_unet(data_dir, method, dev)
        unet = unet_cache[method]

        torch.manual_seed(42)
        gen = generate(unet, dec, method, n_samples, dev, steps)

        for row in range(n_samples):
            v = gen[row, 0].cpu().numpy()
            mid = v.shape[0] // 2
            # Axial
            axes[row * 2, col_idx + 1].imshow(v[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axes[row * 2, col_idx + 1].axis('off')
            # Sagittal
            axes[row * 2 + 1, col_idx + 1].imshow(v[mid, :, :], cmap='gray', vmin=0, vmax=1)
            axes[row * 2 + 1, col_idx + 1].axis('off')

            if row == 0:
                axes[0, col_idx + 1].set_title(label, fontsize=11, fontweight='bold')

    del unet_cache

    # Real column
    for row in range(n_samples):
        v = vols[row, 0].numpy()
        mid = v.shape[0] // 2
        axes[row * 2, 0].imshow(v[:, :, mid], cmap='gray', vmin=0, vmax=1)
        axes[row * 2, 0].axis('off')
        axes[row * 2 + 1, 0].imshow(v[mid, :, :], cmap='gray', vmin=0, vmax=1)
        axes[row * 2 + 1, 0].axis('off')
        if row == 0:
            axes[0, 0].set_title('Real', fontsize=11, fontweight='bold')

    # Row labels
    for row in range(n_samples):
        axes[row * 2, 0].set_ylabel(f'Sample {row+1}\nAxial', fontsize=9, rotation=0,
                                      labelpad=40, va='center')
        axes[row * 2 + 1, 0].set_ylabel('Sagittal', fontsize=9, rotation=0,
                                           labelpad=40, va='center')

    ds_label = "IXI" if dataset == "ixi" else "BraTS"
    plt.suptitle(f'Shortcut FM: Adaptive Step-Count Progression — {ds_label}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'fig_shortcut_progression_{dataset}'
    plt.savefig(out_dir / f'{fname}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.pdf/png")


# ============================================================
# FIGURE 3: DIVERSITY SHOWCASE (multiple samples from each method)
# ============================================================

def make_diversity_grid(data_dir, dec, dev, out_dir, dataset="ixi"):
    """8 samples from Shortcut@4 vs Consistency@4 — visually shows mode collapse."""

    n_samples = 8
    fig, axes = plt.subplots(2, n_samples, figsize=(2.5 * n_samples, 5.5))

    for row, (method, steps, label) in enumerate([
        ("shortcut", 4, "Shortcut FM @4 steps"),
        ("consistency", 4, "Consistency @4 steps"),
    ]):
        ckpt = data_dir / method / "final.pt"
        if not ckpt.exists():
            for col in range(n_samples):
                axes[row, col].text(0.5, 0.5, 'N/A', transform=axes[row, col].transAxes,
                                     ha='center', fontsize=12, color='gray')
                axes[row, col].axis('off')
            continue

        unet = load_unet(data_dir, method, dev)
        torch.manual_seed(42)
        gen = generate(unet, dec, method, n_samples, dev, steps)
        del unet

        for col in range(n_samples):
            v = gen[col, 0].cpu().numpy()
            mid = v.shape[0] // 2
            axes[row, col].imshow(v[:, :, mid], cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')

        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=0,
                                  labelpad=100, va='center')

    ds_label = "IXI" if dataset == "ixi" else "BraTS"
    plt.suptitle(f'Diversity Comparison @ 4 Steps — {ds_label}\n'
                 f'(Consistency produces near-identical outputs; Shortcut maintains variety)',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    fname = f'fig_diversity_grid_{dataset}'
    plt.savefig(out_dir / f'{fname}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}.pdf/png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visual quality grids for Paper 3")
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--data-path', type=str, default="data/ixi_preprocessed_64.pt")
    parser.add_argument('--dataset', choices=['ixi', 'brats'], default='ixi')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "pct" in d.name]
    data_dir = subdirs[0] if subdirs else run_dir
    out_dir = data_dir / "visual_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    cfg = {"latent_channels": 8, "num_embeddings": 256, "n_res_blocks": 2}
    ch = cfg["latent_channels"]

    # Load decoder
    p1 = torch.load(data_dir / "phase1_shared.pt", map_location=dev, weights_only=True)
    dec = Decoder3D(1, ch, cfg["n_res_blocks"]).to(dev)
    dec.load_state_dict(p1["dec"]); dec.eval()

    # Load data
    vols = torch.load(args.data_path, weights_only=True)
    print(f"Data: {vols.shape}")

    print(f"\nGenerating visual grids for {args.dataset.upper()}...\n")

    print("Figure 1: 5-Method Comparison")
    make_method_comparison(data_dir, dec, vols, dev, out_dir, args.dataset)

    print("\nFigure 2: Shortcut Step Progression")
    make_shortcut_progression(data_dir, dec, vols, dev, out_dir, args.dataset)

    print("\nFigure 3: Diversity Grid (Shortcut vs Consistency)")
    make_diversity_grid(data_dir, dec, dev, out_dir, args.dataset)

    print(f"\n  All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
