#!/usr/bin/env python3
"""
Ablation Evaluation: Compare shortcut FM variants.
Loads all shortcut* directories from a benchmark run, evaluates with rigorous multi-seed eval.

Usage:
    python evaluate_ablation.py --run-dir results/ixi_benchmark_20260221_045741
    python evaluate_ablation.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt
"""

import os, sys, json, time, math, argparse, warnings
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
# Model classes (exact copy from flow_matching_3d.py)
# ============================================================

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1))
    def forward(self, x): return x + self.net(x)

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
        if d is not None:
            te = te + self.de(d)
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        hm = self.mid(self.d2(h2)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc2(torch.cat([self.u2(hm), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


# ============================================================
# Metrics
# ============================================================

def ssim3d(a, b, C1=0.01**2, C2=0.03**2):
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    sab = ((a - ma) * (b - mb)).mean()
    return float(((2*ma*mb+C1)*(2*sab+C2)) / ((ma**2+mb**2+C1)*(sa**2+sb**2+C2)))

def psnr3d(a, b):
    return float(10 * np.log10(1.0 / max(((a - b)**2).mean(), 1e-10)))

def compute_diversity(samples):
    n = len(samples)
    if n < 2: return 0.0
    dists = []
    for i in range(min(n, 16)):
        for j in range(i+1, min(n, 16)):
            dists.append(float(np.abs(samples[i] - samples[j]).mean()))
    return float(np.mean(dists))


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample_shortcut(unet, dec, n, dev, steps, ch=8):
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    d_val = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * d_val, device=dev)
        d = torch.full((n,), d_val, device=dev)
        z = z + d_val * unet(z, t, d=d)
    return dec(z).clamp(0, 1)

@torch.no_grad()
def sample_fm(unet, dec, n, dev, steps, ch=8):
    """FM sampling — used for sc_ratio=0 ablation (model has d-embed but acts as FM)."""
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    d_val = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * d_val, device=dev)
        d = torch.full((n,), d_val, device=dev)
        # Still pass d to the model (it has d-embedding) but the model
        # was trained with sc_ratio=0, so d-embedding is untrained for d>0
        z = z + d_val * unet(z, t, d=d)
    return dec(z).clamp(0, 1)


# ============================================================
# Evaluation
# ============================================================

def evaluate_variant(unet, dec, vols, dev, step_counts, n_samples=64, seeds=[42, 123, 456], ch=8):
    real = vols[:, 0].numpy()
    results = {}
    
    for steps in step_counts:
        seed_results = {"ssim": [], "psnr": [], "diversity": [], "time": []}
        for seed in seeds:
            torch.manual_seed(seed)
            if dev.type == "mps": torch.mps.manual_seed(seed)
            elif dev.type == "cuda": torch.cuda.manual_seed(seed)
            
            t0 = time.time()
            all_gen = []
            for start in range(0, n_samples, 8):
                nb = min(8, n_samples - start)
                gen = sample_shortcut(unet, dec, nb, dev, steps, ch)
                all_gen.append(gen.cpu())
            gen_all = torch.cat(all_gen, 0)
            elapsed = time.time() - t0
            gen_np = gen_all[:, 0].numpy()
            
            n_compare = min(len(real), len(gen_np))
            ssims = [ssim3d(real[i % len(real)], gen_np[i]) for i in range(n_compare)]
            psnrs = [psnr3d(real[i % len(real)], gen_np[i]) for i in range(n_compare)]
            
            seed_results["ssim"].append(float(np.mean(ssims)))
            seed_results["psnr"].append(float(np.mean(psnrs)))
            seed_results["diversity"].append(compute_diversity(gen_np))
            seed_results["time"].append(elapsed / n_samples)
        
        results[steps] = {}
        for metric in seed_results:
            vals = seed_results[metric]
            results[steps][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        
        r = results[steps]
        print(f"    {steps:>4} steps | SSIM={r['ssim']['mean']:.4f}±{r['ssim']['std']:.4f} | "
              f"PSNR={r['psnr']['mean']:.2f} | Div={r['diversity']['mean']:.4f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for shortcut variants")
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--data-path', type=str, default="data/ixi_preprocessed_64.pt")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    vols = torch.load(args.data_path, weights_only=True)
    print(f"Data: {vols.shape}")
    
    cfg = {"latent_channels": 8, "num_embeddings": 256, "n_res_blocks": 2}
    
    # Find data subdirectory
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "pct" in d.name]
    data_dir = subdirs[0] if subdirs else run_dir
    print(f"Results dir: {data_dir}")
    
    # Load shared decoder
    p1 = torch.load(data_dir / "phase1_shared.pt", map_location=dev, weights_only=True)
    dec = Decoder3D(1, cfg["latent_channels"], cfg["n_res_blocks"]).to(dev)
    dec.load_state_dict(p1["dec"])
    dec.eval()
    
    # Find all shortcut variants
    variants = sorted([d.name for d in data_dir.iterdir()
                       if d.is_dir() and d.name.startswith("shortcut") and (d / "final.pt").exists()])
    
    if not variants:
        print("ERROR: No shortcut variant directories found!")
        return
    
    print(f"\nFound {len(variants)} variants: {', '.join(variants)}")
    step_counts = [1, 2, 4, 8, 16, 50, 128]
    
    print(f"\n{'='*70}")
    print(f"  ABLATION EVALUATION: {args.n_samples} samples × {len(args.seeds)} seeds")
    print(f"{'='*70}\n")
    
    all_results = {}
    for variant in variants:
        print(f"\n  Evaluating: {variant}")
        
        # Load U-Net
        ckpt_path = data_dir / variant / "final.pt"
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        unet = DenoisingUNet3D(cfg["latent_channels"]).to(dev)
        unet_state = {k.replace("unet.", ""): v for k, v in ckpt.items() if k.startswith("unet.")}
        if unet_state:
            unet.load_state_dict(unet_state)
        else:
            try: unet.load_state_dict(ckpt)
            except: pass
        unet.eval()
        
        results = evaluate_variant(unet, dec, vols, dev, step_counts, args.n_samples, args.seeds)
        all_results[variant] = results
    
    # Save results
    out_dir = data_dir / "ablation_eval"
    out_dir.mkdir(exist_ok=True)
    
    json_results = {}
    for variant, res in all_results.items():
        json_results[variant] = {str(s): {m: v for m, v in metrics.items()}
                                  for s, metrics in res.items()}
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # ---- Print comparison table ----
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Variant':<30} {'1-step SSIM':>12} {'4-step SSIM':>12} {'50-step SSIM':>12} "
          f"{'1-step Div':>12} {'50-step Div':>12}")
    print(f"  {'-'*90}")
    
    for variant, results in all_results.items():
        s1 = results.get(1, {})
        s4 = results.get(4, {})
        s50 = results.get(50, {})
        print(f"  {variant:<30} "
              f"{s1.get('ssim', {}).get('mean', 0):.4f}±{s1.get('ssim', {}).get('std', 0):.4f}  "
              f"{s4.get('ssim', {}).get('mean', 0):.4f}±{s4.get('ssim', {}).get('std', 0):.4f}  "
              f"{s50.get('ssim', {}).get('mean', 0):.4f}±{s50.get('ssim', {}).get('std', 0):.4f}  "
              f"{s1.get('diversity', {}).get('mean', 0):.4f}       "
              f"{s50.get('diversity', {}).get('mean', 0):.4f}")
    
    # ---- Generate comparison figure ----
    COLORS = ['#9C27B0', '#F44336', '#2196F3', '#4CAF50', '#FF9800']
    MARKERS = ['s', 'D', 'o', '^', 'v']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (variant, results) in enumerate(all_results.items()):
        steps = sorted(results.keys())
        ssims = [results[s]['ssim']['mean'] for s in steps]
        divs = [results[s]['diversity']['mean'] for s in steps]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        label = variant.replace("shortcut_", "").replace("shortcut", "default (sc=0.25+curr)")
        
        ax1.plot(steps, ssims, f'{marker}-', color=color, linewidth=2, markersize=8, label=label)
        ax2.plot(steps, divs, f'{marker}-', color=color, linewidth=2, markersize=8, label=label)
    
    ax1.set_xlabel('NFE', fontsize=12); ax1.set_ylabel('SSIM ↑', fontsize=12)
    ax1.set_xscale('log', base=2); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) Quality: SSIM vs Steps', fontsize=13, fontweight='bold')
    
    ax2.set_xlabel('NFE', fontsize=12); ax2.set_ylabel('Diversity', fontsize=12)
    ax2.set_xscale('log', base=2); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Diversity vs Steps', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'ablation_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Figures saved to {out_dir}")
    print(f"  - ablation_results.json")
    print(f"  - ablation_comparison.pdf/png")


if __name__ == "__main__":
    main()
