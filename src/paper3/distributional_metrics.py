#!/usr/bin/env python3
"""
Distributional Quality Metrics for Paper 3 (A6)
================================================

Two complementary distributional metrics:
1. Latent FID: Fréchet distance in VQ-GAN encoder space (captures learned features)
2. FRD: Fréchet Radiomic Distance using handcrafted 3D radiomic features (clinically meaningful)

Both measure how well the *distribution* of generated volumes matches real data,
addressing the "SSIM/PSNR insufficient" critique (these are per-sample, not distributional).

Usage:
    python distributional_metrics.py --run-dir results/ixi_benchmark_20260221_045741
    python distributional_metrics.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt
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

warnings.filterwarnings('ignore')


# ============================================================
# MODEL CLASSES (same as other scripts)
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
        h = self.d // 2
        e = math.log(10000) / (h - 1)
        e = torch.exp(torch.arange(h, device=t.device) * -e)
        e = t[:, None] * e[None, :]
        return torch.cat([e.sin(), e.cos()], -1)

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
# RADIOMIC FEATURE EXTRACTION (no external dependencies)
# ============================================================

def extract_radiomic_features(volume):
    """Extract 15 handcrafted 3D radiomic features from a single volume.
    volume: numpy array of shape (D, H, W), values in [0, 1].
    Returns: feature vector of shape (15,)."""
    v = volume.flatten()
    # Threshold to foreground (brain region)
    fg = v[v > 0.05]
    if len(fg) < 100:
        fg = v  # fallback if mostly empty

    features = []

    # First-order statistics (10 features)
    features.append(fg.mean())                                  # 0: mean intensity
    features.append(fg.std())                                   # 1: std
    features.append(float(np.median(fg)))                       # 2: median
    features.append(float(np.percentile(fg, 10)))               # 3: P10
    features.append(float(np.percentile(fg, 90)))               # 4: P90
    features.append(float(np.percentile(fg, 90) - np.percentile(fg, 10)))  # 5: interquartile range
    features.append(float((fg ** 2).mean()))                    # 6: energy
    n, bins = np.histogram(fg, bins=64, density=True)
    n = n / (n.sum() + 1e-8)
    features.append(float(-np.sum(n[n > 0] * np.log2(n[n > 0] + 1e-10))))  # 7: entropy
    from scipy.stats import skew, kurtosis
    features.append(float(skew(fg)))                            # 8: skewness
    features.append(float(kurtosis(fg)))                        # 9: kurtosis

    # Shape/volume features (3 features)
    mask = volume > 0.05
    features.append(float(mask.sum()) / volume.size)            # 10: volume fraction
    # Surface area approximation (count boundary voxels)
    eroded = np.zeros_like(mask)
    eroded[1:-1, 1:-1, 1:-1] = (mask[1:-1, 1:-1, 1:-1] &
                                  mask[:-2, 1:-1, 1:-1] & mask[2:, 1:-1, 1:-1] &
                                  mask[1:-1, :-2, 1:-1] & mask[1:-1, 2:, 1:-1] &
                                  mask[1:-1, 1:-1, :-2] & mask[1:-1, 1:-1, 2:])
    surface = mask.astype(int) - eroded.astype(int)
    features.append(float(surface.sum()) / volume.size)         # 11: surface ratio
    features.append(float(surface.sum()) / max(mask.sum(), 1))  # 12: surface-to-volume

    # Texture features (2 features) — local gradient magnitude
    gx = np.diff(volume, axis=0)
    gy = np.diff(volume, axis=1)
    gz = np.diff(volume, axis=2)
    # Pad to same size
    gx = np.pad(gx, ((0,1),(0,0),(0,0)))
    gy = np.pad(gy, ((0,0),(0,1),(0,0)))
    gz = np.pad(gz, ((0,0),(0,0),(0,1)))
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    features.append(float(grad_mag.mean()))                     # 13: mean gradient
    features.append(float(grad_mag.std()))                      # 14: gradient std

    return np.array(features, dtype=np.float64)


# ============================================================
# FRÉCHET DISTANCE
# ============================================================

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Product of covariances
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1 @ sigma2)
    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(fd, 0))


def compute_frd(real_features, gen_features):
    """Fréchet Radiomic Distance between real and generated feature sets.
    real_features: (N_real, D), gen_features: (N_gen, D)."""
    mu_r = real_features.mean(axis=0)
    mu_g = gen_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6
    sigma_g = np.cov(gen_features, rowvar=False) + np.eye(gen_features.shape[1]) * 1e-6
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def compute_latent_fid(real_latents, gen_latents):
    """Fréchet distance in VQ-GAN latent space.
    real_latents, gen_latents: (N, D) flattened latent vectors."""
    mu_r = real_latents.mean(axis=0)
    mu_g = gen_latents.mean(axis=0)
    sigma_r = np.cov(real_latents, rowvar=False) + np.eye(real_latents.shape[1]) * 1e-6
    sigma_g = np.cov(gen_latents, rowvar=False) + np.eye(gen_latents.shape[1]) * 1e-6
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ============================================================
# GENERATION HELPERS
# ============================================================

@torch.no_grad()
def generate_volumes(unet, dec, method, n, dev, steps, ch=8):
    """Generate n volumes using the specified method and step count."""
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    if method == "shortcut":
        d_val = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * d_val, device=dev)
            d = torch.full((n,), d_val, device=dev)
            z = z + d_val * unet(z, t, d=d)
    elif method == "ddpm":
        T = 1000
        b = torch.linspace(1e-4, 0.02, T); a = 1 - b; ac = torch.cumprod(a, 0)
        ss = max(T // steps, 1); ts = list(range(T - 1, -1, -ss))
        for tv in ts:
            t = torch.full((n,), tv, device=dev, dtype=torch.float32)
            np_ = unet(z, t)
            z = (1/math.sqrt(a[tv])) * (z - (b[tv]/math.sqrt(1-ac[tv])) * np_)
            if tv > 0: z = z + math.sqrt(b[tv]) * torch.randn_like(z)
    elif method == "consistency":
        if steps == 1:
            t = torch.zeros(n, device=dev)
            z = z + unet(z, t)
        else:
            ts_sched = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
            for i, t_val in enumerate(ts_sched):
                t = torch.full((n,), t_val.item(), device=dev)
                v = unet(z, t)
                x_hat = z + (1 - t_val) * v
                if i < steps - 1:
                    t_next = ts_sched[i+1].item()
                    z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
                else:
                    z = x_hat
    else:  # fm, rectified
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=dev)
            z = z + unet(z, t) * dt
    return dec(z).clamp(0, 1)


@torch.no_grad()
def encode_volumes(enc, vq, volumes, dev, batch_size=8):
    """Encode volumes to latent space, return flattened latent vectors."""
    latents = []
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size].to(dev)
        z = enc(batch)
        zq, _, _ = vq(z)
        latents.append(zq.cpu().view(zq.shape[0], -1))
    return torch.cat(latents, 0).numpy()


# ============================================================
# MAIN
# ============================================================

STEP_COUNTS = {
    "ddpm": [1000],
    "fm": [10, 50],
    "rectified": [5, 10],
    "consistency": [1, 4, 50],
    "shortcut": [1, 4, 16, 128],
}

COLORS = {"ddpm": "#F44336", "fm": "#2196F3", "rectified": "#4CAF50",
           "consistency": "#FF9800", "shortcut": "#9C27B0"}
NAMES = {"ddpm": "DDPM", "fm": "Flow Matching", "rectified": "Rectified Flow",
         "consistency": "Consistency", "shortcut": "Shortcut FM"}


def main():
    parser = argparse.ArgumentParser(description="Distributional metrics for Paper 3")
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--data-path', type=str, default="data/ixi_preprocessed_64.pt")
    parser.add_argument('--n-gen', type=int, default=64, help="Number of volumes to generate per config")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "pct" in d.name]
    data_dir = subdirs[0] if subdirs else run_dir
    out_dir = data_dir / "distributional_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    cfg = {"latent_channels": 8, "num_embeddings": 256, "n_res_blocks": 2}
    ch = cfg["latent_channels"]

    # Load data
    print(f"Loading data from {args.data_path}")
    vols = torch.load(args.data_path, weights_only=True)
    print(f"Data: {vols.shape}")

    # Load shared encoder/decoder/VQ
    p1 = torch.load(data_dir / "phase1_shared.pt", map_location=dev, weights_only=True)
    enc = Encoder3D(1, ch, cfg["n_res_blocks"]).to(dev)
    dec = Decoder3D(1, ch, cfg["n_res_blocks"]).to(dev)
    vq = VectorQuantizer(cfg["num_embeddings"], ch).to(dev)
    enc.load_state_dict(p1["enc"]); dec.load_state_dict(p1["dec"]); vq.load_state_dict(p1["vq"])
    enc.eval(); dec.eval(); vq.eval()

    # ── Extract real features ──
    print(f"\nExtracting real features ({len(vols)} volumes)...")
    n_real = min(len(vols), 128)
    real_vols_np = vols[:n_real, 0].numpy()

    print("  Radiomic features...")
    real_radiomic = np.stack([extract_radiomic_features(real_vols_np[i]) for i in range(n_real)])
    print(f"    Shape: {real_radiomic.shape}")

    print("  Latent features...")
    real_latent = encode_volumes(enc, vq, vols[:n_real], dev)
    print(f"    Shape: {real_latent.shape}")

    # ── Evaluate each method ──
    all_results = {}
    methods = ["ddpm", "fm", "rectified", "consistency", "shortcut"]

    for method in methods:
        ckpt = data_dir / method / "final.pt"
        if not ckpt.exists():
            print(f"\n  Skipping {method} (no checkpoint)")
            continue

        print(f"\n{'='*60}")
        print(f"  {method.upper()}")
        print(f"{'='*60}")

        # Load U-Net
        unet = DenoisingUNet3D(ch).to(dev)
        p2 = torch.load(ckpt, map_location=dev, weights_only=True)
        unet_state = {k.replace("unet.", ""): v for k, v in p2.items() if k.startswith("unet.")}
        if unet_state:
            unet.load_state_dict(unet_state)
        else:
            unet.load_state_dict(p2)
        unet.eval()

        step_counts = STEP_COUNTS.get(method, [50])
        method_results = {}

        for steps in step_counts:
            torch.manual_seed(args.seed)
            print(f"  Generating {args.n_gen} volumes @ {steps} steps...")

            # Generate in batches to avoid OOM
            gen_list = []
            remaining = args.n_gen
            while remaining > 0:
                batch_n = min(8, remaining)
                gen_batch = generate_volumes(unet, dec, method, batch_n, dev, steps, ch)
                gen_list.append(gen_batch.cpu())
                remaining -= batch_n
            gen = torch.cat(gen_list, 0)

            # Radiomic features
            gen_np = gen[:, 0].numpy()
            gen_radiomic = np.stack([extract_radiomic_features(gen_np[i]) for i in range(len(gen_np))])

            # Latent features
            gen_latent = encode_volumes(enc, vq, gen, dev)

            # Compute metrics
            frd = compute_frd(real_radiomic, gen_radiomic)
            lfid = compute_latent_fid(real_latent, gen_latent)

            print(f"    FRD={frd:.2f} | Latent-FID={lfid:.2f}")

            method_results[steps] = {
                "frd": frd,
                "latent_fid": lfid,
                "n_generated": len(gen),
            }

        all_results[method] = method_results
        del unet

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  DISTRIBUTIONAL METRICS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<18} {'Steps':>6} {'FRD ↓':>10} {'Lat-FID ↓':>12}")
    print(f"  {'-'*48}")
    for method, mres in all_results.items():
        for steps, r in sorted(mres.items()):
            print(f"  {NAMES.get(method, method):<18} {steps:>6} {r['frd']:>10.2f} {r['latent_fid']:>12.2f}")

    # ── Figure: FRD bar chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect best-step results for bar chart
    bar_data_frd = {}
    bar_data_lfid = {}
    for method, mres in all_results.items():
        for steps, r in mres.items():
            key = f"{NAMES.get(method, method)}\n@{steps}"
            bar_data_frd[key] = r["frd"]
            bar_data_lfid[key] = r["latent_fid"]

    # FRD bars
    keys = list(bar_data_frd.keys())
    frd_vals = [bar_data_frd[k] for k in keys]
    bar_colors = []
    for k in keys:
        for m, name in NAMES.items():
            if name in k:
                bar_colors.append(COLORS[m])
                break
    ax1.bar(range(len(keys)), frd_vals, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(keys, fontsize=8, rotation=45, ha='right')
    ax1.set_ylabel('FRD ↓ (lower = better)', fontsize=12)
    ax1.set_title('(a) Fréchet Radiomic Distance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')

    # Latent FID bars
    lfid_vals = [bar_data_lfid[k] for k in keys]
    ax2.bar(range(len(keys)), lfid_vals, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, fontsize=8, rotation=45, ha='right')
    ax2.set_ylabel('Latent FID ↓ (lower = better)', fontsize=12)
    ax2.set_title('(b) Fréchet Distance in VQ-GAN Latent Space', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'fig_distributional_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig_distributional_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: fig_distributional_metrics.pdf/png")

    # Save JSON
    with open(out_dir / "distributional_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: distributional_results.json")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
