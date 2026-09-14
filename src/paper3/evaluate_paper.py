#!/usr/bin/env python3
"""
Rigorous Evaluation: 5-Method Benchmark (v2 — fixed metrics)
=============================================================

FIXES APPLIED:
  - SSIM: Proper windowed 3D SSIM with 7x7x7 Gaussian kernel (was global)
  - MS-SSIM: DROPPED (was fake multi-scale)
  - Diversity: Uses ALL n_samples (was capped at 16)
  - Tables/figures updated accordingly

Usage:
    python evaluate_paper.py --run-dir results/ixi_benchmark_20260221_045741
    python evaluate_paper.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt
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
# METRICS (FIXED v2)
# ============================================================

def _make_gaussian_kernel_3d(window_size=7, sigma=1.5):
    """Create a 3D Gaussian kernel for windowed SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d = g / g.sum()
    k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
    return (k3d / k3d.sum()).reshape(1, 1, window_size, window_size, window_size)

_SSIM_KERNEL = _make_gaussian_kernel_3d(7, 1.5)


def ssim3d(a, b):
    """Windowed 3D SSIM with 7×7×7 Gaussian kernel (Wang et al. 2004).
    a, b: numpy arrays shape (D,H,W), values in [0,1].
    Returns scalar SSIM averaged over all spatial windows."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    a_t = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0)
    b_t = torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0)
    k = _SSIM_KERNEL
    pad = 3  # 7 // 2
    mu_a = F.conv3d(a_t, k, padding=pad)
    mu_b = F.conv3d(b_t, k, padding=pad)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = F.conv3d(a_t * a_t, k, padding=pad) - mu_a2
    sig_b2 = F.conv3d(b_t * b_t, k, padding=pad) - mu_b2
    sig_ab = F.conv3d(a_t * b_t, k, padding=pad) - mu_ab
    sig_a2 = sig_a2.clamp(min=0)
    sig_b2 = sig_b2.clamp(min=0)
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float((num / den).mean())


def psnr3d(a, b):
    """Peak signal-to-noise ratio for 3D volumes."""
    return float(10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-10)))


def compute_diversity(samples):
    """Average pairwise L1 distance using ALL samples."""
    n = len(samples)
    if n < 2: return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.abs(samples[i] - samples[j]).mean()))
    return float(np.mean(dists))


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(run_dir, method, dev, cfg):
    ch = cfg["latent_channels"]
    enc = Encoder3D(1, ch, cfg["n_res_blocks"]).to(dev)
    dec = Decoder3D(1, ch, cfg["n_res_blocks"]).to(dev)
    vq = VectorQuantizer(cfg["num_embeddings"], ch).to(dev)
    unet = DenoisingUNet3D(ch).to(dev)
    p1 = torch.load(run_dir / "phase1_shared.pt", map_location=dev, weights_only=True)
    enc.load_state_dict(p1["enc"]); dec.load_state_dict(p1["dec"]); vq.load_state_dict(p1["vq"])
    p2 = torch.load(run_dir / method / "final.pt", map_location=dev, weights_only=True)
    unet_state = {k.replace("unet.", ""): v for k, v in p2.items() if k.startswith("unet.")}
    if unet_state: unet.load_state_dict(unet_state)
    else:
        try: unet.load_state_dict(p2)
        except: print(f"  Warning: Could not load U-Net for {method}")
    enc.eval(); dec.eval(); vq.eval(); unet.eval()
    return enc, dec, vq, unet


# ============================================================
# SAMPLING
# ============================================================

@torch.no_grad()
def sample_fm(unet, dec, n, dev, steps, ch=8):
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=dev); z = z + unet(z, t) * dt
    return dec(z).clamp(0, 1)

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
def sample_ddpm(unet, dec, n, dev, steps, ch=8, T=1000):
    b = torch.linspace(1e-4, 0.02, T); a = 1 - b; ac = torch.cumprod(a, 0)
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    ss = max(T // steps, 1); ts = list(range(T - 1, -1, -ss))
    for tv in ts:
        t = torch.full((n,), tv, device=dev, dtype=torch.float32)
        np_ = unet(z, t); a_v, ac_v, b_v = a[tv], ac[tv], b[tv]
        z = (1 / math.sqrt(a_v)) * (z - (b_v / math.sqrt(1 - ac_v)) * np_)
        if tv > 0: z = z + math.sqrt(b_v) * torch.randn_like(z)
    return dec(z).clamp(0, 1)

@torch.no_grad()
def sample_consistency(unet, dec, n, dev, steps, ch=8):
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    if steps == 1:
        t = torch.zeros(n, device=dev); z = z + unet(z, t)
    else:
        ts = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
        for i, t_val in enumerate(ts):
            t = torch.full((n,), t_val.item(), device=dev)
            v = unet(z, t); x_hat = z + (1 - t_val) * v
            if i < steps - 1:
                t_next = ts[i + 1].item()
                z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
            else: z = x_hat
    return dec(z).clamp(0, 1)


def resolve_method(method_name):
    if method_name.startswith("shortcut"): return "shortcut"
    return method_name

SAMPLE_FNS = {
    "ddpm": sample_ddpm, "fm": sample_fm, "rectified": sample_fm,
    "consistency": sample_consistency, "shortcut": sample_shortcut,
}
STEP_COUNTS = {
    "ddpm": [50, 200, 1000], "fm": [10, 25, 50], "rectified": [5, 10, 25],
    "consistency": [1, 2, 4, 8, 16, 50], "shortcut": [1, 2, 4, 8, 16, 50, 128],
}
COLORS = {"ddpm": "#F44336", "fm": "#2196F3", "rectified": "#4CAF50",
           "consistency": "#FF9800", "shortcut": "#9C27B0"}
NAMES = {"ddpm": "DDPM", "fm": "Flow Matching", "rectified": "Rectified Flow",
         "consistency": "Consistency", "shortcut": "Shortcut FM (Ours)"}
MARKERS = {"ddpm": "D", "fm": "o", "rectified": "^", "consistency": "v", "shortcut": "s"}


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(unet, dec, method, vols, dev, step_counts,
                    n_samples=64, seeds=[42, 123, 456], ch=8):
    base_method = resolve_method(method)
    sample_fn = SAMPLE_FNS[base_method]
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
                gen = sample_fn(unet, dec, nb, dev, steps, ch)
                all_gen.append(gen.cpu())
            gen_all = torch.cat(all_gen, 0)
            elapsed = time.time() - t0
            gen_np = gen_all[:, 0].numpy()

            # Per-sample quality (arbitrary pairing — see paper Section 4.3)
            n_cmp = min(len(real), len(gen_np))
            ssims = [ssim3d(real[i % len(real)], gen_np[i]) for i in range(n_cmp)]
            psnrs = [psnr3d(real[i % len(real)], gen_np[i]) for i in range(n_cmp)]

            seed_results["ssim"].append(float(np.mean(ssims)))
            seed_results["psnr"].append(float(np.mean(psnrs)))
            seed_results["diversity"].append(compute_diversity(gen_np))
            seed_results["time"].append(elapsed / n_samples)

        results[steps] = {}
        for metric in seed_results:
            vals = seed_results[metric]
            results[steps][metric] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "seeds": vals}

        print(f"  {method:>10} | {steps:>4} steps | "
              f"SSIM={results[steps]['ssim']['mean']:.4f}±{results[steps]['ssim']['std']:.4f} | "
              f"PSNR={results[steps]['psnr']['mean']:.2f}±{results[steps]['psnr']['std']:.2f} | "
              f"Div={results[steps]['diversity']['mean']:.4f} | "
              f"{results[steps]['time']['mean']:.3f}s/vol")
    return results


# ============================================================
# FIGURES
# ============================================================

def make_paper_figures(all_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: SSIM vs NFE
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, results in all_results.items():
        steps = sorted(results.keys())
        vals = [results[s]['ssim']['mean'] for s in steps]
        stds = [results[s]['ssim']['std'] for s in steps]
        ax.errorbar(steps, vals, yerr=stds,
                    fmt=f'{MARKERS.get(method,"o")}-', color=COLORS.get(method,'#333'),
                    linewidth=2.5, markersize=9, capsize=4,
                    label=NAMES.get(method, method), zorder=5)
    ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=13)
    ax.set_ylabel('SSIM ↑', fontsize=13)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 50, 128, 200, 1000])
    ax.set_xticklabels(['1','2','4','8','16','50','128','200','1K'])
    ax.legend(fontsize=11, loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_title('Generation Quality vs Inference Steps', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig1_ssim_vs_nfe.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig1_ssim_vs_nfe.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 2: Speed-Quality Pareto
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, results in all_results.items():
        steps = sorted(results.keys())
        times = [results[s]['time']['mean'] for s in steps]
        ssims = [results[s]['ssim']['mean'] for s in steps]
        ax.scatter(times, ssims, c=COLORS.get(method,'#333'), s=120, zorder=5,
                   label=NAMES.get(method, method),
                   marker=MARKERS.get(method,'o'), edgecolors='white', linewidth=1.5)
        for s, x, y in zip(steps, times, ssims):
            ax.annotate(f'{s}', (x,y), textcoords="offset points",
                        xytext=(8,5), fontsize=8, color=COLORS.get(method,'#333'))
    ax.set_xlabel('Time per Volume (seconds)', fontsize=13)
    ax.set_ylabel('SSIM ↑', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Speed–Quality Pareto Frontier', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig2_pareto.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig2_pareto.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 3: Quality-Diversity frontier
    fig, ax = plt.subplots(figsize=(10, 7))
    for method, results in all_results.items():
        steps = sorted(results.keys())
        ssim_vals = [results[s]['ssim']['mean'] for s in steps]
        div_vals = [results[s]['diversity']['mean'] for s in steps]
        color = COLORS.get(method, '#333'); marker = MARKERS.get(method, 'o')
        if method == "ddpm":
            idx = [i for i, s in enumerate(steps) if s >= 1000]
            steps_f = [steps[i] for i in idx]
            ssim_f = [ssim_vals[i] for i in idx]
            div_f = [div_vals[i] for i in idx]
        else:
            steps_f, ssim_f, div_f = steps, ssim_vals, div_vals
        if len(ssim_f) > 1:
            ax.plot(div_f, ssim_f, '-', color=color, alpha=0.4, linewidth=1.5, zorder=3)
        ax.scatter(div_f, ssim_f, c=color, s=100, marker=marker,
                   edgecolors='white', linewidth=1.2, label=NAMES.get(method, method), zorder=5)
        for s, x, y in zip(steps_f, div_f, ssim_f):
            if s == steps_f[0] or s == steps_f[-1]:
                ax.annotate(f'{s}', (x,y), textcoords="offset points",
                            xytext=(6,4), fontsize=8, color=color, fontweight='bold')
    ax.set_xlabel('Sample Diversity (mean pairwise L1)', fontsize=13)
    ax.set_ylabel('SSIM ↑', fontsize=13)
    ax.legend(fontsize=10, loc='lower right'); ax.grid(True, alpha=0.2)
    ax.set_title('Quality–Diversity Frontier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'fig4_quality_diversity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig4_quality_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {out_dir}")


def make_latex_table(all_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative comparison of generative methods for 3D brain MRI synthesis.")
    lines.append(r"SSIM: windowed, 7$\times$7$\times$7 Gaussian kernel (Wang et al.\ 2004).")
    lines.append(r"Mean $\pm$ std over 3 seeds (64 samples each).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & NFE & SSIM $\uparrow$ & PSNR $\uparrow$ & Diversity & Time (s) \\")
    lines.append(r"\midrule")

    best_ssim = max(r['ssim']['mean']
                    for results in all_results.values()
                    for r in results.values())

    method_order = ["ddpm", "fm", "rectified", "consistency", "shortcut"]
    for method in method_order:
        if method not in all_results: continue
        results = all_results[method]
        name = NAMES.get(method, method).replace(" (Ours)", "")
        for i, steps in enumerate(sorted(results.keys())):
            r = results[steps]
            m_label = name if i == 0 else ""
            is_best = abs(r['ssim']['mean'] - best_ssim) < 0.001
            ssim_str = (f"\\textbf{{{r['ssim']['mean']:.4f}$\\pm${r['ssim']['std']:.4f}}}"
                       if is_best else
                       f"{r['ssim']['mean']:.4f}$\\pm${r['ssim']['std']:.4f}")
            lines.append(f"{m_label} & {steps} & {ssim_str} & "
                        f"{r['psnr']['mean']:.2f}$\\pm${r['psnr']['std']:.2f} & "
                        f"{r['diversity']['mean']:.4f} & "
                        f"{r['time']['mean']:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)
    with open(out_dir / "table1_main_results.tex", "w") as f:
        f.write(table_str)
    print(f"  LaTeX table saved to {out_dir / 'table1_main_results.tex'}")
    return table_str


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Rigorous evaluation (v2 — fixed metrics)")
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--n-samples', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--data-path', type=str, default="data/ixi_preprocessed_64.pt")
    parser.add_argument('--methods', nargs='+', default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "paper_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    print(f"Loading data from {args.data_path}")
    vols = torch.load(args.data_path, weights_only=True)
    print(f"Data: {vols.shape}")

    cfg = {"latent_channels": 8, "num_embeddings": 256, "n_res_blocks": 2}

    print(f"\nLoading models from {run_dir}")
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "pct" in d.name]
    data_dir = subdirs[0] if subdirs else run_dir
    if subdirs: print(f"  Using subdirectory: {data_dir.name}")

    p1 = torch.load(data_dir / "phase1_shared.pt", map_location=dev, weights_only=True)
    dec = Decoder3D(1, cfg["latent_channels"], cfg["n_res_blocks"]).to(dev)
    dec.load_state_dict(p1["dec"]); dec.eval()

    all_method_names = ["ddpm", "fm", "rectified", "consistency", "shortcut"]
    methods_to_check = args.methods if args.methods else all_method_names

    methods_to_eval = {}
    for method in methods_to_check:
        ckpt = data_dir / method / "final.pt"
        if ckpt.exists():
            print(f"  Loading {method}...")
            _, _, _, unet = load_model(data_dir, method, dev, cfg)
            methods_to_eval[method] = unet
        else:
            print(f"  Skipping {method} (no checkpoint)")

    if not methods_to_eval:
        print("ERROR: No method checkpoints found!"); return

    print(f"\n{'='*70}")
    print(f"  RIGOROUS EVALUATION (v2 — fixed metrics)")
    print(f"  {args.n_samples} samples × {len(args.seeds)} seeds")
    print(f"  Windowed 3D SSIM (7×7×7 Gaussian) | Diversity on all {args.n_samples} samples")
    print(f"  Methods: {', '.join(m.upper() for m in methods_to_eval)}")
    print(f"{'='*70}\n")

    all_results = {}
    for method, unet in methods_to_eval.items():
        step_counts = STEP_COUNTS.get(resolve_method(method), [50])
        print(f"Evaluating {NAMES.get(method, method)}...")
        results = evaluate_method(unet, dec, method, vols, dev,
                                  step_counts, args.n_samples, args.seeds,
                                  cfg["latent_channels"])
        all_results[method] = results
        print()

    # Save
    json_results = {}
    for method, res in all_results.items():
        json_results[method] = {}
        for steps, metrics in res.items():
            json_results[method][str(steps)] = {
                m: {"mean": v["mean"], "std": v["std"]} for m, v in metrics.items()}
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    if len(all_results) >= 2:
        print("Generating paper figures...")
        make_paper_figures(all_results, out_dir)
        print("Generating LaTeX table...")
        table = make_latex_table(all_results, out_dir)
        print(f"\n{table}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for method, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]['ssim']['mean'])
        print(f"  {NAMES.get(method, method):<25} best: {best[0]:>4} steps → SSIM {best[1]['ssim']['mean']:.4f}")
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
