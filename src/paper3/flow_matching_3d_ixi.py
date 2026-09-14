#!/usr/bin/env python3
"""
Benchmark: Modern Generative Paradigms for 3D Brain MRI Synthesis
==================================================================

Systematic comparison of 5 methods in VQ-GAN latent space:
  1. DDPM          - predict noise, 50-200 steps
  2. Flow Matching - predict velocity, 10-50 steps (Euler ODE)
  3. Rectified Flow - FM + reflow distillation, 5-25 steps
  4. Consistency   - distill from FM teacher, 1-4 steps
  5. Shortcut FM   - self-consistent shortcuts, 1-128 steps (adaptive)

All share: same VQ-GAN Phase 1, same 3D U-Net, same dataset.

Usage:
    python flow_matching_3d.py --quick                     # ~2hr sanity
    python flow_matching_3d.py --fraction 1.0              # all methods, 100%
    python flow_matching_3d.py --methods ddpm fm shortcut   # pick methods
    python flow_matching_3d.py --method shortcut            # shortcut only
    python flow_matching_3d.py --full-paper                # all fractions
"""

import os, sys, json, time, math, argparse, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ALL_METHODS = ["ddpm", "fm", "rectified", "consistency", "shortcut"]


# ============================================================
# CONFIGURATION
# ============================================================

def get_config(quick=False, dataset="ixi"):
    dataset_configs = {
        "ixi": {
            "data_dir": "data/ixi",
            "preprocessed_file": "data/ixi_preprocessed_64.pt",
            "max_volumes": 200,
            "modality": None,  # IXI is single-modality T1
        },
        "brats": {
            "data_dir": "data/brats",
            "preprocessed_file": "data/brats_preprocessed_64.pt",
            "max_volumes": 300,
            "modality": "t2f",  # T2-FLAIR; alternatives: t1c, t1n, t2w
        },
    }
    ds = dataset_configs.get(dataset, dataset_configs["ixi"])
    return {
        "dataset": dataset,
        "data_dir": ds["data_dir"],
        "preprocessed_file": ds["preprocessed_file"],
        "modality": ds["modality"],
        "target_size": 64,
        "max_volumes": ds["max_volumes"],
        "latent_channels": 8,
        "num_embeddings": 256,
        "n_res_blocks": 2,
        "epochs_phase1": 15 if quick else 200,
        "epochs_phase2": 15 if quick else 150,
        "epochs_reflow": 10 if quick else 80,
        "epochs_consistency": 10 if quick else 80,
        "epochs_shortcut": 15 if quick else 150,
        "batch_size": 2,
        "lr_gen": 1e-4,
        "lr_disc": 4e-4,
        "num_diffusion_steps": 1000,
        "consistency_s0": 2,
        "consistency_s1": 150,
        "recon_weight": 1.0,
        "vq_weight": 1.0,
        "perceptual_weight": 0.1,
        "gan_weight": 0.1,
        "gan_ramp_epochs": 10 if quick else 50,
        "gp_weight": 10.0,
        "num_samples": 8 if quick else 16,
        "ddpm_sampling_steps": [50, 200, 1000] if not quick else [50],
        "fm_sampling_steps": [10, 25, 50],
        "rectified_sampling_steps": [5, 10, 25],
        "consistency_sampling_steps": [1, 2, 4, 8, 16, 50],
        "shortcut_sampling_steps": [1, 2, 4, 8, 16, 50, 128],
        "shortcut_sc_ratio": 0.25,        # fraction of batches using self-consistency loss
        "shortcut_d_max_schedule": True,   # gradually increase max d during training
    }


# ============================================================
# DATA
# ============================================================

def load_data(cfg):
    cache = Path(cfg["preprocessed_file"])
    if cache.exists():
        print(f"Loading cached: {cache}")
        v = torch.load(cache, weights_only=True)
    else:
        import nibabel as nib, glob
        dd = cfg["data_dir"]
        dataset = cfg.get("dataset", "ixi")
        modality = cfg.get("modality", None)

        if dataset == "brats":
            # BraTS structure: data/brats/BraTS-GLI-XXXXX-XXX/BraTS-GLI-XXXXX-XXX-t2f.nii.gz
            # Also supports: data/brats/BRATS_XXX/ or any nested .nii.gz
            pattern_suffix = f"*-{modality}.nii.gz" if modality else "*.nii.gz"
            files = sorted(glob.glob(f"{dd}/**/{pattern_suffix}", recursive=True))
            if not files:
                # Try flat directory
                files = sorted(glob.glob(f"{dd}/*.nii.gz"))
            # Filter out segmentation files
            files = [f for f in files if "-seg" not in f and "_seg" not in f]
            print(f"BraTS: Found {len(files)} {modality or 'NIfTI'} volumes in {dd}")
        else:
            # IXI: flat directory of .nii.gz files
            files = sorted(glob.glob(f"{dd}/*.nii.gz") + glob.glob(f"{dd}/*.nii"))
            print(f"IXI: Found {len(files)} NIfTI files in {dd}")

        if len(files) == 0:
            raise FileNotFoundError(
                f"No NIfTI files found in {dd}. "
                f"{'For BraTS: download from https://www.synapse.org/#!Synapse:syn51156910 or Kaggle, ' if dataset == 'brats' else 'For IXI: download T1 images, '}"
                f"place .nii.gz files in {dd}/"
            )

        vols = []
        for i, f in enumerate(files[:cfg["max_volumes"]]):
            d = nib.load(f).get_fdata().astype(np.float32)
            # Skip empty or near-empty volumes
            if d.max() - d.min() < 1e-6:
                continue
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            s = cfg["target_size"]
            d = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
            d = F.interpolate(d, size=(s, s, s), mode='trilinear', align_corners=False)
            vols.append(d.squeeze(0))
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i+1}/{min(len(files), cfg['max_volumes'])}")
        v = torch.stack(vols)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(v, cache)
        print(f"  Cached to {cache}")
    print(f"Data: {v.shape} | range [{v.min():.2f}, {v.max():.2f}]")
    return v


class BrainDS(Dataset):
    def __init__(self, vols):
        self.v = vols
    def __len__(self):
        return len(self.v)
    def __getitem__(self, i):
        return {"volume": self.v[i]}


# ============================================================
# MODEL COMPONENTS
# ============================================================

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1))
    def forward(self, x):
        return x + self.net(x)

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, lat_ch=8, n_res=2):
        super().__init__()
        layers = [nn.Conv3d(in_ch, 32, 3, padding=1), nn.SiLU(),
                  nn.Conv3d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
                  nn.Conv3d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
                  nn.Conv3d(128, lat_ch, 4, stride=2, padding=1), nn.SiLU()]
        for _ in range(n_res): layers.append(ResBlock3D(lat_ch))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Upsample3D(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv3d(ci, co, 3, padding=1)
    def forward(self, x):
        return self.conv(self.up(x))

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
    def forward(self, x):
        return self.up(self.res(x))

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

class PatchDisc3D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def block(ci, co, s=2):
            return [nn.Conv3d(ci, co, 4, s, 1), nn.InstanceNorm3d(co), nn.LeakyReLU(0.2)]
        self.blocks = nn.ModuleList([
            nn.Sequential(*block(in_ch, 32)),
            nn.Sequential(*block(32, 64)),
            nn.Sequential(*block(64, 128))])
        self.head = nn.Conv3d(128, 1, 4, padding=1)
    def forward(self, x):
        feats = []
        for b in self.blocks: x = b(x); feats.append(x)
        return self.head(x), feats


# ============================================================
# 3D U-NET (shared backbone, with optional step-size d conditioning)
# ============================================================

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
    """3D U-Net conditioned on timestep t and optionally step-size d.
    
    For standard methods (DDPM, FM, etc): called as unet(x, t)
    For Shortcut FM: called as unet(x, t, d=d) where d is the step size.
    The d-embedding is summed into the time embedding, following the
    Shortcut Models paper (ICLR 2025).
    """
    def __init__(self, ch=8, td=128):
        super().__init__()
        self.te = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        # Step-size d embedding — same architecture as time embedding
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
            te = te + self.de(d)  # sum d-embedding into time embedding
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        hm = self.mid(self.d2(h2)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc2(torch.cat([self.u2(hm), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


# ============================================================
# NOISE SCHEDULER (DDPM)
# ============================================================

class NoiseScheduler:
    def __init__(self, T=1000):
        self.T = T; b = torch.linspace(1e-4, 0.02, T); a = 1 - b
        ac = torch.cumprod(a, 0)
        self.betas, self.alphas, self.ac = b, a, ac
        self.sac, self.s1mac = torch.sqrt(ac), torch.sqrt(1 - ac)
    def add_noise(self, x, noise, t):
        tc = t.cpu(); sa = self.sac[tc].to(x.device); sm = self.s1mac[tc].to(x.device)
        while sa.dim() < x.dim(): sa = sa.unsqueeze(-1); sm = sm.unsqueeze(-1)
        return sa * x + sm * noise


# ============================================================
# FULL MODEL
# ============================================================

class LatentGenerative3D(nn.Module):
    def __init__(self, cfg, method="fm"):
        super().__init__()
        ch = cfg["latent_channels"]; self.method = method; self.cfg = cfg
        self.enc = Encoder3D(1, ch, cfg["n_res_blocks"])
        self.dec = Decoder3D(1, ch, cfg["n_res_blocks"])
        self.vq = VectorQuantizer(cfg["num_embeddings"], ch)
        self.unet = DenoisingUNet3D(ch)
        if method == "ddpm": self.sch = NoiseScheduler(cfg["num_diffusion_steps"])

    def encode(self, x):
        zq, vl, _ = self.vq(self.enc(x)); return zq, vl
    def decode(self, zq):
        return self.dec(zq).clamp(0, 1)
    def forward(self, x):
        zq, vl = self.encode(x); return self.decode(zq), vl

    # ---- DDPM loss ----
    def ddpm_loss(self, x):
        with torch.no_grad(): zq, _, _ = self.vq(self.enc(x))
        B = zq.shape[0]; noise = torch.randn_like(zq)
        t = torch.randint(0, self.sch.T, (B,), device=x.device)
        return F.mse_loss(self.unet(self.sch.add_noise(zq, noise, t), t.float()), noise)

    # ---- Standard FM loss ----
    def fm_loss(self, x):
        with torch.no_grad(): zq, _, _ = self.vq(self.enc(x))
        z0 = torch.randn_like(zq); t = torch.rand(zq.shape[0], device=x.device)
        te = t[:, None, None, None, None]
        xt = (1 - te) * z0 + te * zq
        return F.mse_loss(self.unet(xt, t), zq - z0)

    # ---- Shortcut FM losses ----
    def shortcut_fm_loss(self, x):
        """Standard FM loss with d=0 (velocity field learning).
        Identical to fm_loss but passes d=0 to condition the network."""
        with torch.no_grad(): zq, _, _ = self.vq(self.enc(x))
        z0 = torch.randn_like(zq); t = torch.rand(zq.shape[0], device=x.device)
        te = t[:, None, None, None, None]
        xt = (1 - te) * z0 + te * zq
        d_zero = torch.zeros_like(t)
        target = zq - z0  # velocity
        return F.mse_loss(self.unet(xt, t, d=d_zero), target)

    def shortcut_sc_loss(self, x, d_max=1.0):
        """Self-consistency loss: one step of size 2d ≈ two steps of size d.
        
        s_θ(x_t, t, d) predicts velocity for step size d.
        Consistency constraint:
            x_t + 2d·s_θ(x_t, t, 2d) ≈ x_t + d·s_θ(x_t, t, d) + d·s_θ(x_{t+d}, t+d, d)
        
        Stop-gradient on the two-step (target) side prevents instability.
        d sampled from binary set {1/128, 1/64, ..., 1/2} capped at d_max.
        """
        with torch.no_grad(): zq, _, _ = self.vq(self.enc(x))
        B = zq.shape[0]
        z0 = torch.randn_like(zq)
        t = torch.rand(B, device=x.device)
        te = t[:, None, None, None, None]
        xt = (1 - te) * z0 + te * zq

        # Sample d from binary set {1/128, 1/64, ..., 1/2} capped at d_max
        log2_d = torch.randint(1, 8, (B,), device=x.device)  # 1..7 -> d = 1/2..1/128
        d = (0.5 ** log2_d.float()).clamp(max=d_max)
        # Ensure t + 2d <= 1 (stay within [0,1] interval)
        d = torch.min(d, (1.0 - t) / 2.0).clamp(min=1e-4)
        de = d[:, None, None, None, None]

        # ONE big step: x_t + 2d * s_θ(x_t, t, 2d) [student - gets gradients]
        v_big = self.unet(xt, t, d=2*d)
        x_big_step = xt + 2 * de * v_big

        # TWO small steps: [stop-gradient target]
        with torch.no_grad():
            v1 = self.unet(xt, t, d=d)
            x_mid = xt + de * v1              # first d-step: t -> t+d
            v2 = self.unet(x_mid, t + d, d=d)
            x_two_steps = x_mid + de * v2     # second d-step: t+d -> t+2d

        return F.mse_loss(x_big_step, x_two_steps.detach())

    # ---- Reflow loss ----
    def reflow_loss(self, x, teacher_unet):
        with torch.no_grad():
            zq, _, _ = self.vq(self.enc(x))
            z0 = torch.randn_like(zq); z = z0.clone()
            for i in range(10):
                t_t = torch.full((z.shape[0],), i / 10.0, device=z.device)
                z = z + teacher_unet(z, t_t) * 0.1
            z1 = z
        t = torch.rand(z0.shape[0], device=x.device)
        te = t[:, None, None, None, None]
        xt = (1 - te) * z0 + te * z1
        return F.mse_loss(self.unet(xt, t), z1 - z0)

    # ---- Consistency loss ----
    def consistency_loss(self, x, teacher_unet, n_steps, ema_unet=None):
        """Proper consistency distillation:
        f_theta(x_t, t) = x_t + (1-t)*v_theta(x_t, t) predicts ODE endpoint.
        Loss: ||f_theta(x_tn, tn) - f_ema(x_tn1, tn1)||^2
        where x_tn1 comes from teacher ODE step, f_ema is EMA target."""
        with torch.no_grad(): zq, _, _ = self.vq(self.enc(x))
        B = zq.shape[0]; N = max(n_steps, 2)
        n_idx = torch.randint(1, N, (B,), device=x.device)
        t_n = n_idx.float() / N; t_n1 = (n_idx - 1).float() / N
        z0 = torch.randn_like(zq)
        te_n = t_n[:, None, None, None, None]
        te_n1 = t_n1[:, None, None, None, None]
        # Noisy sample at t_n
        x_tn = (1 - te_n) * z0 + te_n * zq
        # Teacher ODE step: t_n -> t_{n-1} (backward along ODE)
        with torch.no_grad():
            dt = (t_n - t_n1)[:, None, None, None, None]
            x_tn1 = x_tn - teacher_unet(x_tn, t_n) * dt
        # Consistency function: f(x, t) = x + (1-t) * v(x, t)
        # Student prediction at t_n
        v_student = self.unet(x_tn, t_n)
        f_student = x_tn + (1 - te_n) * v_student
        # EMA target prediction at t_{n-1}
        target_net = ema_unet if ema_unet is not None else teacher_unet
        with torch.no_grad():
            v_target = target_net(x_tn1, t_n1)
            f_target = x_tn1 + (1 - te_n1) * v_target
        return F.mse_loss(f_student, f_target)

    # ---- Loss dispatch ----
    def phase2_loss(self, x, **kw):
        if self.method == "ddpm": return self.ddpm_loss(x)
        elif self.method == "fm": return self.fm_loss(x)
        elif self.method == "rectified": return self.reflow_loss(x, kw["teacher_unet"])
        elif self.method == "consistency": return self.consistency_loss(
            x, kw["teacher_unet"], kw["n_steps"], kw.get("ema_unet"))
        elif self.method == "shortcut":
            # Mixed loss — ratio controlled by training loop
            sc_ratio = kw.get("sc_ratio", 0.25)
            d_max = kw.get("d_max", 1.0)
            if torch.rand(1).item() < sc_ratio:
                return self.shortcut_sc_loss(x, d_max=d_max)
            else:
                return self.shortcut_fm_loss(x)

    # ---- DDPM sampling ----
    @torch.no_grad()
    def ddpm_sample(self, n, dev, steps):
        ch = self.cfg["latent_channels"]; z = torch.randn(n, ch, 8, 8, 8, device=dev)
        ss = max(self.sch.T // steps, 1); ts = list(range(self.sch.T - 1, -1, -ss))
        for tv in ts:
            t = torch.full((n,), tv, device=dev, dtype=torch.float32)
            np_ = self.unet(z, t)
            a, ac, b = self.sch.alphas[tv], self.sch.ac[tv], self.sch.betas[tv]
            z = (1 / math.sqrt(a)) * (z - (b / math.sqrt(1 - ac)) * np_)
            if tv > 0: z = z + math.sqrt(b) * torch.randn_like(z)
        return self.decode(z)

    # ---- Euler ODE sampling (FM, Rectified) ----
    @torch.no_grad()
    def ode_sample(self, n, dev, steps):
        ch = self.cfg["latent_channels"]; z = torch.randn(n, ch, 8, 8, 8, device=dev)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=dev); z = z + self.unet(z, t) * dt
        return self.decode(z)

    # ---- Consistency sampling ----
    @torch.no_grad()
    def consistency_sample(self, n, dev, steps):
        """Consistency sampling using f(x,t) = x + (1-t)*v(x,t) -> data estimate.
        1 step: start at t=0 (noise), apply f once to jump to data.
        Multi-step: apply f to get data estimate, re-noise to intermediate t, repeat."""
        ch = self.cfg["latent_channels"]; z = torch.randn(n, ch, 8, 8, 8, device=dev)
        if steps == 1:
            t = torch.zeros(n, device=dev)
            v = self.unet(z, t)
            z = z + v  # f(z, 0) = z + 1.0 * v -> data estimate
        else:
            # Multi-step: iterate from t=0 upward, re-noising between steps
            ts = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
            for i, t_val in enumerate(ts):
                t = torch.full((n,), t_val.item(), device=dev)
                v = self.unet(z, t)
                x_hat = z + (1 - t_val) * v  # data estimate via consistency function
                if i < steps - 1:
                    # Re-noise to next time point using ODE interpolation
                    t_next = ts[i + 1].item()
                    fresh_noise = torch.randn_like(z)
                    z = (1 - t_next) * fresh_noise + t_next * x_hat
                else:
                    z = x_hat  # final step: keep data estimate
        return self.decode(z)

    # ---- Shortcut sampling ----
    @torch.no_grad()
    def shortcut_sample(self, n, dev, steps):
        """Shortcut sampling: use d=1/steps at each step.
        
        The model was trained so s_θ(x_t, t, d) gives the velocity
        for step size d. For N steps: d=1/N, N uniform steps from t=0 to t=1.
        
        For 1 step:   d=1.0, single jump from noise to data.
        For 128 steps: d=1/128, approaches standard FM (Euler ODE).
        """
        ch = self.cfg["latent_channels"]; z = torch.randn(n, ch, 8, 8, 8, device=dev)
        d_val = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * d_val, device=dev)
            d = torch.full((n,), d_val, device=dev)
            v = self.unet(z, t, d=d)
            z = z + d_val * v
        return self.decode(z)

    # ---- Sampling dispatch ----
    def generate(self, n=1, dev="cpu", steps=50):
        if self.method == "ddpm": return self.ddpm_sample(n, dev, steps)
        elif self.method == "consistency": return self.consistency_sample(n, dev, steps)
        elif self.method == "shortcut": return self.shortcut_sample(n, dev, steps)
        else: return self.ode_sample(n, dev, steps)


# ============================================================
# LOSS / TRAINING HELPERS
# ============================================================

def feat_loss(disc, real, fake):
    _, rf = disc(real); _, ff = disc(fake)
    return sum(F.l1_loss(f, r.detach()) for r, f in zip(rf, ff)) / len(rf)

def grad_pen(disc, real, fake, dev):
    a = torch.rand(real.shape[0], 1, 1, 1, 1, device=dev)
    inp = (a * real + (1 - a) * fake).requires_grad_(True)
    p, _ = disc(inp)
    g = torch.autograd.grad(p, inp, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    return ((g.norm(2, dim=(1, 2, 3, 4)) - 1) ** 2).mean()


def train_phase1(model, disc, dl, cfg, dev, rdir):
    ep = cfg["epochs_phase1"]
    print(f"\n{'='*60}\nPHASE 1: VQ-GAN ({ep} epochs)\n{'='*60}")
    gp = list(model.enc.parameters()) + list(model.dec.parameters()) + list(model.vq.parameters())
    og = torch.optim.AdamW(gp, lr=cfg["lr_gen"], betas=(0.5, 0.9))
    od = torch.optim.AdamW(disc.parameters(), lr=cfg["lr_disc"], betas=(0.5, 0.9))
    sg = torch.optim.lr_scheduler.CosineAnnealingLR(og, ep)
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(od, ep)
    hist = []
    for e in range(1, ep + 1):
        st = {"recon": 0, "vq": 0, "gg": 0, "perc": 0, "dl": 0}; nb = 0; t0 = time.time()
        gw = cfg["gan_weight"] * min(1, e / max(cfg["gan_ramp_epochs"], 1))
        model.train(); disc.train()
        for batch in dl:
            x = batch["volume"].to(dev)
            with torch.no_grad(): xr, _ = model(x)
            dr, _ = disc(x); df, _ = disc(xr.detach())
            dloss = (F.relu(1 - dr) + F.relu(1 + df)).mean()
            if nb % 4 == 0: dloss = dloss + cfg["gp_weight"] * grad_pen(disc, x, xr.detach(), dev)
            od.zero_grad(); dloss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0); od.step()
            xr, vql = model(x); rl = F.l1_loss(xr, x)
            dfg, _ = disc(xr); ggl = -dfg.mean(); pl = feat_loss(disc, x, xr)
            gl = cfg["recon_weight"]*rl + cfg["vq_weight"]*vql + gw*ggl + cfg["perceptual_weight"]*pl
            og.zero_grad(); gl.backward()
            torch.nn.utils.clip_grad_norm_(gp, 1.0); og.step()
            st["recon"]+=rl.item(); st["vq"]+=vql.item()
            st["gg"]+=ggl.item(); st["perc"]+=pl.item(); st["dl"]+=dloss.item(); nb+=1
        for k in st: st[k] /= nb
        sg.step(); sd.step(); hist.append(st)
        if e % 5 == 0 or e == 1 or e <= 3:
            print(f"  [{e:3d}/{ep}] rec={st['recon']:.4f} vq={st['vq']:.4f} "
                  f"G={st['gg']:.3f} perc={st['perc']:.4f} D={st['dl']:.3f} | {time.time()-t0:.1f}s")
        if e % 50 == 0 or e == ep:
            rdir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "disc": disc.state_dict()}, rdir/f"p1_ep{e}.pt")
    print("Phase 1 done"); return hist


def freeze_p1(m):
    for p in m.enc.parameters(): p.requires_grad = False
    for p in m.dec.parameters(): p.requires_grad = False
    for p in m.vq.parameters(): p.requires_grad = False

def unfreeze(m):
    for p in m.parameters(): p.requires_grad = True


def train_p2_standard(model, dl, cfg, dev, rdir):
    ep = cfg["epochs_phase2"]; method = model.method.upper()
    print(f"\n{'='*60}\nPHASE 2: {method} ({ep} epochs)\n{'='*60}")
    freeze_p1(model)
    params = list(model.unet.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr_gen"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    losses = []
    for e in range(1, ep + 1):
        el = 0; n = 0; t0 = time.time(); model.train()
        for batch in dl:
            l = model.phase2_loss(batch["volume"].to(dev))
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            el += l.item(); n += 1
        avg = el/n; losses.append(avg); sch.step()
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{ep}] {method.lower()}_loss={avg:.6f} | {time.time()-t0:.1f}s")
    unfreeze(model); torch.save(model.state_dict(), rdir/"final.pt")
    print(f"Phase 2 ({method}) done"); return losses


def train_reflow(model, dl, cfg, dev, rdir, teacher_state):
    ep = cfg["epochs_reflow"]
    print(f"\n{'='*60}\nPHASE 2b: REFLOW ({ep} epochs)\n{'='*60}")
    freeze_p1(model)
    teacher = DenoisingUNet3D(cfg["latent_channels"]).to(dev)
    teacher.load_state_dict(teacher_state); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    params = list(model.unet.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr_gen"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    losses = []
    for e in range(1, ep + 1):
        el = 0; n = 0; t0 = time.time(); model.train()
        for batch in dl:
            l = model.phase2_loss(batch["volume"].to(dev), teacher_unet=teacher)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            el += l.item(); n += 1
        avg = el/n; losses.append(avg); sch.step()
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{ep}] reflow={avg:.6f} | {time.time()-t0:.1f}s")
    unfreeze(model); torch.save(model.state_dict(), rdir/"final.pt")
    del teacher; print("Reflow done"); return losses


def train_consistency(model, dl, cfg, dev, rdir, teacher_state):
    ep = cfg["epochs_consistency"]
    print(f"\n{'='*60}\nPHASE 2c: CONSISTENCY ({ep} epochs)\n{'='*60}")
    freeze_p1(model)
    # FM teacher (frozen, for ODE steps)
    teacher = DenoisingUNet3D(cfg["latent_channels"]).to(dev)
    teacher.load_state_dict(teacher_state); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    # Initialize student from teacher (critical for convergence!)
    model.unet.load_state_dict(teacher_state)
    # EMA copy of student (initialized from teacher too)
    ema_unet = DenoisingUNet3D(cfg["latent_channels"]).to(dev)
    ema_unet.load_state_dict(teacher_state); ema_unet.eval()
    for p in ema_unet.parameters(): p.requires_grad = False
    ema_decay = 0.999

    params = list(model.unet.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr_gen"] * 0.5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    s0, s1 = cfg["consistency_s0"], cfg["consistency_s1"]
    losses = []
    for e in range(1, ep + 1):
        ns = int(s0 + (e / ep) * (s1 - s0))
        el = 0; n = 0; t0 = time.time(); model.train()
        for batch in dl:
            l = model.phase2_loss(batch["volume"].to(dev),
                                  teacher_unet=teacher, n_steps=ns, ema_unet=ema_unet)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            # Update EMA
            with torch.no_grad():
                for p_ema, p_student in zip(ema_unet.parameters(), model.unet.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p_student.data, alpha=1 - ema_decay)
            el += l.item(); n += 1
        avg = el/n; losses.append(avg); sch.step()
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{ep}] consist={avg:.6f} N={ns} | {time.time()-t0:.1f}s")
    unfreeze(model); torch.save(model.state_dict(), rdir/"final.pt")
    del teacher, ema_unet; print("Consistency done"); return losses


def train_shortcut(model, dl, cfg, dev, rdir):
    """Train Shortcut Flow Matching with d_max curriculum.
    
    Key idea from Shortcut Models (Frans et al., ICLR 2025 Oral):
    - ~75% of batches: standard FM loss with d=0 (learn velocity field)
    - ~25% of batches: self-consistency loss with d>0 (learn shortcuts)
    - d_max gradually increases: start with small shortcuts, build to d=1
    
    This is a STANDALONE training procedure — no teacher model needed.
    The self-consistency loss bootstraps from the model's own predictions.
    """
    ep = cfg["epochs_shortcut"]; sc_ratio = cfg["shortcut_sc_ratio"]
    use_schedule = cfg["shortcut_d_max_schedule"]
    print(f"\n{'='*60}\nPHASE 2: SHORTCUT FM ({ep} epochs, SC ratio={sc_ratio})\n{'='*60}")
    freeze_p1(model)
    params = list(model.unet.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr_gen"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    losses = []; fm_losses = []; sc_losses = []
    
    for e in range(1, ep + 1):
        # d_max curriculum: linearly increase from 1/128 to 1.0 over training
        if use_schedule:
            d_max = min(1.0, (1/128) + (e / ep) * (1.0 - 1/128))
        else:
            d_max = 1.0
        
        el = 0; n_fm = 0; n_sc = 0; l_fm = 0; l_sc = 0; t0 = time.time()
        model.train()
        for batch in dl:
            x = batch["volume"].to(dev)
            # Decide FM or SC for this batch
            if torch.rand(1).item() < sc_ratio and d_max > 1/64:
                l = model.shortcut_sc_loss(x, d_max=d_max)
                l_sc += l.item(); n_sc += 1
            else:
                l = model.shortcut_fm_loss(x)
                l_fm += l.item(); n_fm += 1
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            el += l.item()
        
        nb = n_fm + n_sc
        avg = el / nb if nb > 0 else 0
        losses.append(avg)
        fm_losses.append(l_fm / max(n_fm, 1))
        sc_losses.append(l_sc / max(n_sc, 1))
        sch.step()
        
        if e % 5 == 0 or e == 1 or e <= 3:
            print(f"  [{e:3d}/{ep}] total={avg:.6f} fm={fm_losses[-1]:.6f} "
                  f"sc={sc_losses[-1]:.6f} d_max={d_max:.4f} "
                  f"(fm:{n_fm} sc:{n_sc}) | {time.time()-t0:.1f}s")
    
    unfreeze(model); torch.save(model.state_dict(), rdir/"final.pt")
    
    # Save separate loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(fm_losses, label='FM loss', color='#2196F3', linewidth=2)
    ax1.plot(sc_losses, label='SC loss', color='#FF5722', linewidth=2)
    ax1.set_title('Shortcut FM: Component Losses'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(losses, label='Total loss', color='#9C27B0', linewidth=2)
    ax2.set_title('Shortcut FM: Total Loss'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(rdir/'shortcut_losses.png', dpi=150); plt.close()
    
    print("Shortcut FM done"); return losses


# ============================================================
# EVALUATION
# ============================================================

def ssim3d(a, b):
    ma, mb = a.mean(), b.mean(); sa, sb = a.std(), b.std()
    sab = ((a - ma)*(b - mb)).mean(); c1, c2 = 1e-4, 9e-4
    return float(((2*ma*mb+c1)*(2*sab+c2))/((ma**2+mb**2+c1)*(sa**2+sb**2+c2)))

def psnr3d(a, b):
    return float(10*np.log10(1/max(((a-b)**2).mean(), 1e-10)))

def evaluate(model, vols, cfg, dev, rdir, label=""):
    print(f"\n{'='*60}\nEVALUATION: {label}\n{'='*60}")
    model.eval()
    pfx = label.lower().replace(" ","_").replace("+","_").replace("(","").replace(")","")
    # Recon
    with torch.no_grad():
        rx = vols[:16].to(dev); rxr, _ = model(rx)
    rr, rrr = rx.cpu().numpy(), rxr.cpu().numpy()
    recon_ssim = float(np.mean([ssim3d(rr[i,0], rrr[i,0]) for i in range(len(rr))]))
    print(f"   Recon SSIM: {recon_ssim:.4f}")

    step_key = f"{model.method}_sampling_steps"
    step_counts = cfg.get(step_key, [50])
    all_results = {}; ng = cfg["num_samples"]

    for steps in step_counts:
        print(f"\n   Generating {ng} volumes | {steps} steps | {model.method}...")
        t0 = time.time()
        with torch.no_grad(): gen = model.generate(ng, dev, steps)
        tpv = (time.time() - t0) / ng

        n = min(len(vols), len(gen), 16)
        r, g = vols[:n,0].numpy(), gen[:n,0].cpu().numpy()
        ss = [ssim3d(r[i], g[i]) for i in range(n)]
        pp = [psnr3d(r[i], g[i]) for i in range(n)]
        result = {"method": model.method, "steps": steps,
                  "gen_ssim": float(np.mean(ss)), "gen_ssim_std": float(np.std(ss)),
                  "gen_psnr": float(np.mean(pp)), "gen_psnr_std": float(np.std(pp)),
                  "time_per_vol": tpv, "recon_ssim": recon_ssim, "nfe": steps}
        all_results[steps] = result
        print(f"   SSIM: {result['gen_ssim']:.4f} +/- {result['gen_ssim_std']:.4f} "
              f"| PSNR: {result['gen_psnr']:.2f} | {tpv:.2f}s/vol")

        # Grid
        ns = min(16, gen.shape[0]); rows = 4 if ns >= 16 else 2
        fig, ax = plt.subplots(rows, 4, figsize=(12, 3*rows)); af = ax.flatten()
        for i in range(min(ns, len(af))):
            af[i].imshow(gen[i,0,:,:,gen.shape[-1]//2].cpu().numpy(), cmap='gray')
            af[i].set_title(f'S{i+1}', fontsize=9); af[i].axis('off')
        for i in range(ns, len(af)): af[i].axis('off')
        plt.suptitle(f'{label} ({steps} steps)')
        plt.tight_layout(); plt.savefig(rdir/f'{pfx}_{steps}s_grid.png', dpi=150); plt.close()

        # 3-view
        v = gen[0,0].cpu().numpy(); m = v.shape[0]//2
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(v[:,:,m], cmap='gray'); ax[0].set_title('Axial'); ax[0].axis('off')
        ax[1].imshow(v[:,m,:], cmap='gray'); ax[1].set_title('Coronal'); ax[1].axis('off')
        ax[2].imshow(v[m,:,:], cmap='gray'); ax[2].set_title('Sagittal'); ax[2].axis('off')
        plt.suptitle(f'{label} ({steps} steps)')
        plt.tight_layout(); plt.savefig(rdir/f'{pfx}_{steps}s_3views.png', dpi=150); plt.close()

    with open(rdir/f'{pfx}_metrics.json', 'w') as f: json.dump(all_results, f, indent=2)
    return all_results


# ============================================================
# ORCHESTRATION
# ============================================================

def run_shared_phase1(cfg, vols, dev, rdir):
    ds = BrainDS(vols)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    print(f"\n{'#'*60}\n  SHARED PHASE 1: VQ-GAN | {len(vols)} volumes\n{'#'*60}")
    model = LatentGenerative3D(cfg, method="fm").to(dev)
    disc = PatchDisc3D().to(dev)
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    p1h = train_phase1(model, disc, dl, cfg, dev, rdir)
    p1_state = {"enc": model.enc.state_dict(), "dec": model.dec.state_dict(),
                "vq": model.vq.state_dict()}
    torch.save(p1_state, rdir/"phase1_shared.pt")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot([h["recon"] for h in p1h], label='Recon'); a1.plot([h["vq"] for h in p1h], label='VQ')
    a1.legend(); a1.grid(True, alpha=0.3); a1.set_title('Reconstruction')
    a2.plot([h["gg"] for h in p1h], label='G'); a2.plot([h["dl"] for h in p1h], label='D')
    a2.legend(); a2.grid(True, alpha=0.3); a2.set_title('GAN')
    plt.tight_layout(); plt.savefig(rdir/'phase1.png', dpi=150); plt.close()
    return p1_state, p1h


def run_method(cfg, vols, p1_state, dev, rdir, method, fm_unet_state=None):
    ds = BrainDS(vols)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    mdir = rdir / method; mdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*60}\n  METHOD: {method.upper()} | {len(vols)} volumes\n{'#'*60}")
    model = LatentGenerative3D(cfg, method=method).to(dev)
    model.enc.load_state_dict(p1_state["enc"])
    model.dec.load_state_dict(p1_state["dec"])
    model.vq.load_state_dict(p1_state["vq"])

    if method in ("ddpm", "fm"):
        losses = train_p2_standard(model, dl, cfg, dev, mdir)
    elif method == "rectified":
        losses = train_reflow(model, dl, cfg, dev, mdir, fm_unet_state)
    elif method == "consistency":
        losses = train_consistency(model, dl, cfg, dev, mdir, fm_unet_state)
    elif method == "shortcut":
        losses = train_shortcut(model, dl, cfg, dev, mdir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, color='#9C27B0'); ax.set_title(f'{method.upper()} Loss')
    ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(mdir/'loss.png', dpi=150); plt.close()

    results = evaluate(model, vols, cfg, dev, mdir, method.upper())
    unet_state = model.unet.state_dict() if method == "fm" else None
    del model; return results, losses, unet_state


def run_comparison(cfg, vols, dev, base_dir, fraction=1.0, methods=None):
    if methods is None: methods = ALL_METHODS
    n = max(2, int(len(vols) * fraction))
    fvols = vols[:n]; fdir = base_dir / f"{int(fraction*100)}pct_{n}vol"
    fdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  BENCHMARK: {cfg.get('dataset','ixi').upper()} | {n} vol ({fraction*100:.0f}%) | {', '.join(m.upper() for m in methods)}\n{'='*60}")

    p1_state, _ = run_shared_phase1(cfg, fvols, dev, fdir)
    all_results = {}; all_losses = {}; fm_unet = None
    ordered = [m for m in ["ddpm", "fm", "rectified", "consistency", "shortcut"] if m in methods]

    for method in ordered:
        res, losses, unet = run_method(cfg, fvols, p1_state, dev, fdir, method, fm_unet)
        all_results[method] = res; all_losses[method] = losses
        if method == "fm" and unet: fm_unet = unet

    # Summary
    print(f"\n{'='*70}\n  RESULTS: {n} vol ({fraction*100:.0f}%)\n{'='*70}")
    print(f"  {'Method':<18} {'Steps':>6} {'SSIM':>8} {'PSNR':>8} {'Time':>8}")
    print(f"  {'-'*52}")
    for m, res in all_results.items():
        for s, r in res.items():
            print(f"  {m.upper():<18} {s:>6} {r['gen_ssim']:>8.4f} {r['gen_psnr']:>7.2f} {r['time_per_vol']:>7.2f}s")

    # Plots
    colors = {"ddpm":"#F44336","fm":"#2196F3","rectified":"#4CAF50","consistency":"#FF9800","shortcut":"#9C27B0"}
    names = {"ddpm":"DDPM","fm":"Flow Matching","rectified":"Rectified Flow","consistency":"Consistency","shortcut":"Shortcut FM"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for m, losses in all_losses.items():
        ax1.plot(losses, label=names.get(m,m), color=colors.get(m), linewidth=2)
    ax1.set_title(f'Training Loss ({n} vol)'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True, alpha=0.3)

    for m, res in all_results.items():
        sl = sorted(res.keys()); ss = [res[s]["gen_ssim"] for s in sl]
        ax2.plot(sl, ss, 'o-', label=names.get(m,m), color=colors.get(m), linewidth=2, markersize=8)
    ax2.set_title(f'SSIM vs Inference Steps ({n} vol)'); ax2.set_xlabel('NFE'); ax2.set_ylabel('Gen SSIM')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fdir/'comparison.png', dpi=150); plt.close()

    combined = {"fraction": fraction, "n_volumes": n}
    for m, r in all_results.items(): combined[m] = {str(k): v for k, v in r.items()}
    with open(fdir/"benchmark_results.json", "w") as f: json.dump(combined, f, indent=2)
    return combined


def run_full_paper(cfg, vols, dev, base_dir, methods=None):
    fracs = [0.10, 0.25, 0.50, 1.0]; all_res = {}
    for f in fracs:
        all_res[f"{int(f*100)}%"] = run_comparison(cfg, vols, dev, base_dir, f, methods)

    # Master plot
    colors = {"ddpm":"#F44336","fm":"#2196F3","rectified":"#4CAF50","consistency":"#FF9800","shortcut":"#9C27B0"}
    names = {"ddpm":"DDPM","fm":"Flow Matching","rectified":"Rectified Flow","consistency":"Consistency","shortcut":"Shortcut FM"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in ALL_METHODS:
        ssims = []
        for fk in ["10%","25%","50%","100%"]:
            if fk in all_res and m in all_res[fk]:
                best = max((r["gen_ssim"] for r in all_res[fk][m].values() if isinstance(r,dict) and "gen_ssim" in r), default=0)
                ssims.append(best)
            else: ssims.append(0)
        if any(s > 0 for s in ssims):
            ax.plot([10,25,50,100], ssims, 'o-', label=names.get(m,m), color=colors.get(m), linewidth=2, markersize=8)
    ax.set_title('Best SSIM vs Data Fraction'); ax.set_xlabel('Data (%)'); ax.set_ylabel('Gen SSIM')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(base_dir/'ssim_vs_data.png', dpi=150); plt.close()

    with open(base_dir/"full_results.json", "w") as f: json.dump(all_res, f, indent=2)
    return all_res


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="5-Method Benchmark: 3D Brain MRI")
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dataset', choices=['ixi', 'brats'], default='ixi',
                        help='Dataset: ixi (default) or brats')
    parser.add_argument('--method', choices=ALL_METHODS)
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--full-paper', action='store_true')
    parser.add_argument('--epochs-p1', type=int)
    parser.add_argument('--epochs-p2', type=int)
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Resume from existing results dir: reuse VQ-GAN + FM teacher, '
                             'retrain only the specified --method (e.g. consistency)')
    parser.add_argument('--eval-only', action='store_true',
                        help='With --resume-dir: skip training, just re-evaluate existing checkpoint')
    args = parser.parse_args()

    cfg = get_config(args.quick, dataset=args.dataset)
    if args.epochs_p1: cfg["epochs_phase1"] = args.epochs_p1
    if args.epochs_p2:
        cfg["epochs_phase2"] = args.epochs_p2
        cfg["epochs_reflow"] = args.epochs_p2
        cfg["epochs_consistency"] = args.epochs_p2
        cfg["epochs_shortcut"] = args.epochs_p2

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    vols = load_data(cfg)

    # ---- Resume mode: retrain single method using existing VQ-GAN + FM teacher ----
    if args.resume_dir:
        resume_dir = Path(args.resume_dir)
        # Find the data subdirectory (e.g., 100pct_200vol)
        sub = sorted([d for d in resume_dir.iterdir() if d.is_dir() and "pct_" in d.name])
        data_dir = sub[0] if sub else resume_dir
        print(f"Resuming from: {data_dir}")

        method = args.method
        if not method:
            parser.error("--resume-dir requires --method to specify which method to retrain")

        # Load shared VQ-GAN
        p1_path = data_dir / "phase1_shared.pt"
        p1_state = torch.load(p1_path, map_location=dev, weights_only=True)
        print(f"  Loaded VQ-GAN from {p1_path}")

        # Load FM teacher if needed
        fm_unet_state = None
        if method in ("rectified", "consistency"):
            fm_path = data_dir / "fm" / "final.pt"
            fm_full = torch.load(fm_path, map_location=dev, weights_only=True)
            # Extract just the unet keys
            fm_unet_state = {k.replace("unet.", ""): v for k, v in fm_full.items() if k.startswith("unet.")}
            if not fm_unet_state:  # might already be just unet state
                fm_unet_state = fm_full
            print(f"  Loaded FM teacher from {fm_path}")

        # Train the method (or skip if eval-only)
        ds = BrainDS(vols); dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
        mdir = data_dir / method; mdir.mkdir(parents=True, exist_ok=True)
        model = LatentGenerative3D(cfg, method=method).to(dev)
        model.enc.load_state_dict(p1_state["enc"])
        model.dec.load_state_dict(p1_state["dec"])
        model.vq.load_state_dict(p1_state["vq"])

        if args.eval_only:
            # Load existing checkpoint, just re-evaluate
            ckpt_path = mdir / "final.pt"
            if not ckpt_path.exists():
                print(f"  ERROR: No checkpoint found at {ckpt_path}")
                return
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            # Try loading as full model state or just unet
            try:
                model.load_state_dict(ckpt)
            except:
                unet_state = {k.replace("unet.", ""): v for k, v in ckpt.items() if k.startswith("unet.")}
                if unet_state:
                    model.unet.load_state_dict(unet_state)
                else:
                    model.unet.load_state_dict(ckpt)
            print(f"\n{'#'*60}\n  EVAL-ONLY: {method.upper()} | {len(vols)} volumes\n{'#'*60}")
        else:
            print(f"\n{'#'*60}\n  RETRAINING: {method.upper()} | {len(vols)} volumes\n{'#'*60}")

        if not args.eval_only:
            if method in ("ddpm", "fm"):
                losses = train_p2_standard(model, dl, cfg, dev, mdir)
            elif method == "rectified":
                losses = train_reflow(model, dl, cfg, dev, mdir, fm_unet_state)
            elif method == "consistency":
                losses = train_consistency(model, dl, cfg, dev, mdir, fm_unet_state)
            elif method == "shortcut":
                losses = train_shortcut(model, dl, cfg, dev, mdir)

        evaluate(model, vols, cfg, dev, mdir, method.upper())
        print(f"\nDone! {method} results in: {mdir}")
        return

    # ---- Normal mode ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_name = cfg.get("dataset", "ixi")
    base_dir = Path(f"results/{ds_name}_benchmark_{ts}"); base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir/"config.json", "w") as f: json.dump(cfg, f, indent=2)

    if args.method:
        methods = [args.method]
        if args.method in ("rectified","consistency") and "fm" not in methods:
            methods = ["fm"] + methods
    elif args.methods:
        methods = list(args.methods)
        if any(m in methods for m in ("rectified","consistency")) and "fm" not in methods:
            methods = ["fm"] + methods
    else:
        methods = ALL_METHODS

    if args.full_paper: run_full_paper(cfg, vols, dev, base_dir, methods)
    else: run_comparison(cfg, vols, dev, base_dir, args.fraction, methods)
    print(f"\nDone! Results: {base_dir}")


if __name__ == "__main__":
    main()
