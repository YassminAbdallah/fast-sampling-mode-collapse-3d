#!/usr/bin/env python3
"""
Trajectory Analysis for Paper 3 (Section 3.5 / Figure 4)
=========================================================

Analyzes WHY shortcut flow matching works well in VQ-GAN latent space.

Key analyses:
1. Trajectory straightness: ||x₁ - x₀|| / Σ||x_{t+1} - x_t|| (1.0 = perfectly straight)
2. Pointwise curvature: deviation of each x_t from the straight-line interpolation
3. Step-size error: ||one-2d-step - two-d-steps|| as function of d
4. PCA of intermediate latent states at t = 0, 0.25, 0.5, 0.75, 1.0

Hypothesis: VQ-GAN's discrete codebook creates flatter latent trajectories,
making first-order (Euler) shortcuts more accurate than in pixel space.

Usage:
    python trajectory_analysis.py --run-dir results/ixi_benchmark_20260221_045741
    python trajectory_analysis.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt
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
# MODEL CLASSES (duplicated from flow_matching_3d.py for standalone use)
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
        if d is not None:
            te = te + self.de(d)
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        hm = self.mid(self.d2(h2)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc2(torch.cat([self.u2(hm), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


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
    if unet_state:
        unet.load_state_dict(unet_state)
    else:
        unet.load_state_dict(p2)
    
    enc.eval(); dec.eval(); vq.eval(); unet.eval()
    return enc, dec, vq, unet


# ============================================================
# ANALYSIS 1: TRAJECTORY TRACING
# ============================================================

@torch.no_grad()
def trace_fm_trajectory(unet, n, dev, fine_steps=256, ch=8):
    """Trace FM ODE trajectory at fine granularity, recording all intermediate states.
    Returns: list of (t, z) pairs where z is shape (n, ch, 8, 8, 8)."""
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    dt = 1.0 / fine_steps
    trajectory = [(0.0, z.cpu().clone())]
    for i in range(fine_steps):
        t = torch.full((n,), i * dt, device=dev)
        v = unet(z, t)
        z = z + v * dt
        # Record at selected timepoints
        t_val = (i + 1) * dt
        if (i + 1) % (fine_steps // 32) == 0 or i == fine_steps - 1:
            trajectory.append((t_val, z.cpu().clone()))
    return trajectory


@torch.no_grad()
def trace_shortcut_trajectory(unet, n, dev, steps, ch=8):
    """Trace shortcut trajectory, recording all intermediate states."""
    z = torch.randn(n, ch, 8, 8, 8, device=dev)
    d_val = 1.0 / steps
    trajectory = [(0.0, z.cpu().clone())]
    for i in range(steps):
        t = torch.full((n,), i * d_val, device=dev)
        d = torch.full((n,), d_val, device=dev)
        v = unet(z, t, d=d)
        z = z + d_val * v
        trajectory.append(((i + 1) * d_val, z.cpu().clone()))
    return trajectory


# ============================================================
# ANALYSIS 2: STRAIGHTNESS AND CURVATURE
# ============================================================

def compute_straightness(trajectory):
    """Trajectory straightness: ||x₁ - x₀|| / Σ||x_{t+1} - x_t||
    1.0 = perfectly straight line. <1.0 = curved path.
    Computed per sample, returns array of shape (n,)."""
    states = [s for _, s in trajectory]
    n = states[0].shape[0]
    
    # Flatten spatial dims for distance computation
    x0 = states[0].view(n, -1)    # noise
    x1 = states[-1].view(n, -1)   # data
    
    # Direct distance: ||x₁ - x₀||
    direct = torch.norm(x1 - x0, dim=1)
    
    # Path length: Σ||x_{t+1} - x_t||
    path_len = torch.zeros(n)
    for i in range(len(states) - 1):
        a = states[i].view(n, -1)
        b = states[i + 1].view(n, -1)
        path_len += torch.norm(b - a, dim=1)
    
    straightness = direct / path_len.clamp(min=1e-8)
    return straightness.numpy()


def compute_pointwise_curvature(trajectory):
    """Deviation of each x_t from the straight line connecting x₀ and x₁.
    Returns: (times, mean_deviations) where deviation = ||x_t - lerp(x₀, x₁, t)||."""
    states = [s for _, s in trajectory]
    times = [t for t, _ in trajectory]
    n = states[0].shape[0]
    
    x0 = states[0].view(n, -1)
    x1 = states[-1].view(n, -1)
    
    deviations = []
    for i, (t_val, state) in enumerate(trajectory):
        xt = state.view(n, -1)
        # Linear interpolation: lerp = (1-t)*x₀ + t*x₁
        lerp = (1 - t_val) * x0 + t_val * x1
        dev = torch.norm(xt - lerp, dim=1)
        # Normalize by ||x₁ - x₀|| for scale invariance
        norm = torch.norm(x1 - x0, dim=1).clamp(min=1e-8)
        deviations.append((dev / norm).mean().item())
    
    return times, deviations


# ============================================================
# ANALYSIS 3: STEP-SIZE ERROR (Self-consistency gap)
# ============================================================

@torch.no_grad()
def compute_step_size_error(unet, dev, method, n_samples=16, ch=8):
    """Measure ||one-2d-step - two-d-steps|| as function of d.
    
    For each d in {1/128, 1/64, ..., 1/2}:
      - Start from same x_t at t=0.25 (mid-trajectory)
      - One step:  x_t + 2d * v(x_t, t, 2d)
      - Two steps: x_t + d*v(x_t,t,d) -> x_mid -> x_mid + d*v(x_mid,t+d,d)
      - Error = ||one_step - two_steps|| / ||two_steps - x_t||
    """
    d_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    errors = []
    
    # Create a realistic starting point by running FM for a few steps
    z = torch.randn(n_samples, ch, 8, 8, 8, device=dev)
    # Advance to t=0.25 with fine steps
    dt = 1.0 / 256
    for i in range(64):  # 64/256 = 0.25
        t = torch.full((n_samples,), i * dt, device=dev)
        if method == "shortcut":
            d_zero = torch.zeros(n_samples, device=dev)
            z = z + unet(z, t, d=d_zero) * dt
        else:
            z = z + unet(z, t) * dt
    
    x_t = z.clone()
    t_start = 0.25
    
    for d_val in d_values:
        if t_start + 2 * d_val > 1.0:
            errors.append(float('nan'))
            continue
        
        t_tensor = torch.full((n_samples,), t_start, device=dev)
        
        if method == "shortcut":
            d_big = torch.full((n_samples,), 2 * d_val, device=dev)
            d_small = torch.full((n_samples,), d_val, device=dev)
            
            # One big step: 2d
            v_big = unet(x_t, t_tensor, d=d_big)
            x_one = x_t + 2 * d_val * v_big
            
            # Two small steps: d + d
            v1 = unet(x_t, t_tensor, d=d_small)
            x_mid = x_t + d_val * v1
            t_mid = torch.full((n_samples,), t_start + d_val, device=dev)
            v2 = unet(x_mid, t_mid, d=d_small)
            x_two = x_mid + d_val * v2
        else:
            # Standard FM: no d conditioning, just Euler steps
            v_big = unet(x_t, t_tensor)
            x_one = x_t + 2 * d_val * v_big
            
            v1 = unet(x_t, t_tensor)
            x_mid = x_t + d_val * v1
            t_mid = torch.full((n_samples,), t_start + d_val, device=dev)
            v2 = unet(x_mid, t_mid)
            x_two = x_mid + d_val * v2
        
        # Relative error
        diff = (x_one - x_two).view(n_samples, -1)
        step_mag = (x_two - x_t).view(n_samples, -1)
        error = torch.norm(diff, dim=1) / torch.norm(step_mag, dim=1).clamp(min=1e-8)
        errors.append(error.mean().item())
    
    return d_values, errors


# ============================================================
# ANALYSIS 4: PCA OF INTERMEDIATE STATES
# ============================================================

def pca_trajectory(trajectory, n_components=2):
    """PCA of flattened latent states across trajectory timepoints.
    Returns: (times, projected_points) for visualization."""
    # Collect states at key timepoints
    target_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    collected = []
    collected_times = []
    
    for target_t in target_times:
        # Find closest trajectory point
        best_idx = min(range(len(trajectory)), 
                       key=lambda i: abs(trajectory[i][0] - target_t))
        t_val, state = trajectory[best_idx]
        collected.append(state)
        collected_times.append(t_val)
    
    # Stack and flatten: (n_times * n_samples, latent_dim)
    n_samples = collected[0].shape[0]
    flat = torch.stack([s.view(n_samples, -1) for s in collected])  # (n_times, n_samples, D)
    n_times = flat.shape[0]
    all_points = flat.reshape(-1, flat.shape[-1]).numpy()  # (n_times*n_samples, D)
    
    # PCA
    mean = all_points.mean(axis=0, keepdims=True)
    centered = all_points - mean
    # Use SVD for efficiency
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_components].T  # (n_times*n_samples, 2)
    
    # Reshape back: (n_times, n_samples, 2)
    projected = projected.reshape(n_times, n_samples, n_components)
    
    # Variance explained
    var_explained = (S[:n_components] ** 2) / (S ** 2).sum() * 100
    
    return collected_times, projected, var_explained


# ============================================================
# FIGURE GENERATION
# ============================================================

def make_trajectory_figure(results, out_dir):
    """Generate the 4-panel trajectory analysis figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {"fm": "#2196F3", "shortcut": "#9C27B0", "rectified": "#4CAF50"}
    names = {"fm": "Flow Matching", "shortcut": "Shortcut FM", "rectified": "Rectified Flow"}
    
    # ── Panel (a): Pointwise curvature ──
    ax1 = fig.add_subplot(gs[0, 0])
    for method in ["fm", "shortcut", "rectified"]:
        if method not in results:
            continue
        times, devs = results[method]["curvature"]
        ax1.plot(times, devs, '-', color=colors[method], linewidth=2.5,
                 label=names[method], alpha=0.9)
    ax1.set_xlabel('Time t (noise → data)', fontsize=12)
    ax1.set_ylabel('Normalized Deviation from\nStraight Line', fontsize=12)
    ax1.set_title('(a) Trajectory Curvature in Latent Space', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, 1)
    
    # ── Panel (b): Straightness histogram ──
    ax2 = fig.add_subplot(gs[0, 1])
    for method in ["fm", "shortcut", "rectified"]:
        if method not in results:
            continue
        s = results[method]["straightness"]
        ax2.hist(s, bins=25, alpha=0.5, color=colors[method],
                 label=f'{names[method]} (μ={np.mean(s):.3f})', edgecolor='white')
        ax2.axvline(np.mean(s), color=colors[method], linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Straightness Score (1.0 = perfect)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('(b) Trajectory Straightness Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    # ── Panel (c): Step-size error ──
    ax3 = fig.add_subplot(gs[1, 0])
    for method in ["fm", "shortcut"]:
        if method not in results:
            continue
        d_vals, errs = results[method]["step_error"]
        valid = [(d, e) for d, e in zip(d_vals, errs) if not np.isnan(e)]
        if valid:
            ds, es = zip(*valid)
            ax3.plot(ds, es, 'o-', color=colors[method], linewidth=2.5,
                     markersize=8, label=names[method])
    ax3.set_xlabel('Step Size d', fontsize=12)
    ax3.set_ylabel('Relative Error\n||one(2d) − two(d)|| / ||Δx||', fontsize=12)
    ax3.set_title('(c) First-Order Approximation Error vs Step Size', fontsize=13, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.2, which='both')
    
    # ── Panel (d): PCA visualization ──
    ax4 = fig.add_subplot(gs[1, 1])
    # Show FM trajectory PCA
    method = "fm" if "fm" in results else list(results.keys())[0]
    pca_times, pca_proj, var_exp = results[method]["pca"]
    
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(pca_times)))
    for i, (t_val, color) in enumerate(zip(pca_times, time_colors)):
        points = pca_proj[i]  # (n_samples, 2)
        label = f't={t_val:.2f}' if i in [0, len(pca_times)//2, len(pca_times)-1] else None
        ax4.scatter(points[:, 0], points[:, 1], c=[color], s=40, alpha=0.6,
                    edgecolors='white', linewidth=0.5, label=label, zorder=3)
        # Draw trajectory lines for first few samples
        if i > 0:
            for s in range(min(5, points.shape[0])):
                prev = pca_proj[i-1][s]
                curr = points[s]
                ax4.plot([prev[0], curr[0]], [prev[1], curr[1]],
                         '-', color=color, alpha=0.3, linewidth=1)
    
    ax4.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)', fontsize=12)
    ax4.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)', fontsize=12)
    ax4.set_title(f'(d) PCA of {names[method]} Latent Trajectories', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, shrink=0.8, pad=0.02)
    cbar.set_label('Time t', fontsize=10)
    
    plt.savefig(out_dir / 'fig_trajectory_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_trajectory_analysis.pdf/png")


def make_shortcut_vs_fm_pca(results, out_dir):
    """Side-by-side PCA: FM vs Shortcut trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    for ax, method, title in [(ax1, "fm", "Flow Matching (Euler)"),
                                (ax2, "shortcut", "Shortcut FM (d=1/4)")]:
        if method not in results:
            continue
        pca_times, pca_proj, var_exp = results[method]["pca"]
        time_colors = plt.cm.viridis(np.linspace(0, 1, len(pca_times)))
        
        for i, (t_val, color) in enumerate(zip(pca_times, time_colors)):
            points = pca_proj[i]
            label = f't={t_val:.2f}' if i in [0, len(pca_times)//2, len(pca_times)-1] else None
            ax.scatter(points[:, 0], points[:, 1], c=[color], s=50, alpha=0.6,
                       edgecolors='white', linewidth=0.5, label=label, zorder=3)
            if i > 0:
                for s in range(min(8, points.shape[0])):
                    prev = pca_proj[i-1][s]
                    curr = points[s]
                    ax.plot([prev[0], curr[0]], [prev[1], curr[1]],
                            '-', color=color, alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'fig_pca_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'fig_pca_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_pca_comparison.pdf/png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Trajectory analysis for Paper 3")
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--data-path', type=str, default="data/ixi_preprocessed_64.pt")
    parser.add_argument('--n-trajectories', type=int, default=32,
                        help="Number of trajectories to trace (more = smoother stats)")
    parser.add_argument('--fine-steps', type=int, default=256,
                        help="Fine-grained steps for FM trajectory tracing")
    parser.add_argument('--shortcut-steps', type=int, default=4,
                        help="Step count for shortcut trajectory tracing")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and "pct" in d.name]
    data_dir = subdirs[0] if subdirs else run_dir
    
    out_dir = data_dir / "trajectory_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    
    cfg = {"latent_channels": 8, "num_embeddings": 256, "n_res_blocks": 2}
    n = args.n_trajectories
    torch.manual_seed(args.seed)
    
    # Methods to analyze
    methods_to_run = []
    for m in ["fm", "shortcut", "rectified"]:
        if (data_dir / m / "final.pt").exists():
            methods_to_run.append(m)
    print(f"Methods: {', '.join(methods_to_run)}")
    
    results = {}
    
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"  Analyzing: {method.upper()}")
        print(f"{'='*60}")
        
        _, dec, _, unet = load_model(data_dir, method, dev, cfg)
        
        # Use same noise for all methods
        torch.manual_seed(args.seed)
        
        # 1. Trace trajectory
        print(f"  Tracing {n} trajectories...")
        if method == "shortcut":
            # Trace at multiple step counts for comparison
            traj = trace_shortcut_trajectory(unet, n, dev, args.shortcut_steps)
            # Also trace at fine resolution for curvature analysis (using d=0)
            torch.manual_seed(args.seed)
            traj_fine = trace_shortcut_trajectory(unet, n, dev, args.fine_steps)
        else:
            traj = trace_fm_trajectory(unet, n, dev, args.fine_steps)
            traj_fine = traj
        
        # 2. Straightness
        print(f"  Computing straightness...")
        straightness = compute_straightness(traj_fine)
        print(f"    Mean straightness: {np.mean(straightness):.4f} ± {np.std(straightness):.4f}")
        
        # 3. Pointwise curvature
        print(f"  Computing curvature...")
        times, devs = compute_pointwise_curvature(traj_fine)
        max_dev = max(devs)
        print(f"    Max deviation: {max_dev:.4f} (at t≈{times[devs.index(max_dev)]:.2f})")
        
        # 4. Step-size error
        print(f"  Computing step-size error...")
        torch.manual_seed(args.seed)
        d_vals, errs = compute_step_size_error(unet, dev, method, n_samples=n)
        valid_errs = [e for e in errs if not np.isnan(e)]
        if valid_errs:
            print(f"    Error range: {min(valid_errs):.4f} – {max(valid_errs):.4f}")
        
        # 5. PCA
        print(f"  Computing PCA...")
        pca_times, pca_proj, var_exp = pca_trajectory(traj_fine)
        print(f"    Variance explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")
        
        results[method] = {
            "straightness": straightness,
            "curvature": (times, devs),
            "step_error": (d_vals, errs),
            "pca": (pca_times, pca_proj, var_exp),
        }
    
    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  TRAJECTORY ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'Straightness':>14} {'Max Curvature':>15} {'Min Step Err':>14}")
    print(f"  {'-'*63}")
    for method, r in results.items():
        s = r["straightness"]
        _, devs = r["curvature"]
        _, errs = r["step_error"]
        valid_errs = [e for e in errs if not np.isnan(e)]
        min_err = min(valid_errs) if valid_errs else float('nan')
        print(f"  {method.upper():<20} {np.mean(s):>8.4f}±{np.std(s):.4f}"
              f"  {max(devs):>13.4f}  {min_err:>13.4f}")
    
    # ── Generate figures ──
    print(f"\nGenerating figures...")
    make_trajectory_figure(results, out_dir)
    if "fm" in results and "shortcut" in results:
        make_shortcut_vs_fm_pca(results, out_dir)
    
    # Save numerical results
    save_results = {}
    for method, r in results.items():
        save_results[method] = {
            "straightness_mean": float(np.mean(r["straightness"])),
            "straightness_std": float(np.std(r["straightness"])),
            "max_curvature": float(max(r["curvature"][1])),
            "curvature_times": r["curvature"][0],
            "curvature_values": r["curvature"][1],
            "step_error_d": r["step_error"][0],
            "step_error_values": r["step_error"][1],
            "pca_var_explained": r["pca"][2].tolist(),
        }
    with open(out_dir / "trajectory_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n  All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
