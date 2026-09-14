"""
Shared model definitions for Paper 3/4 experiment scripts.
=========================================================

These classes are EXACT copies from flow_matching_3d.py (with conditional extensions).
All experiment scripts import from here to avoid architecture mismatches.

Paper 4 additions (backward compatible):
  - DenoisingUNet3D: num_classes parameter (default 0 = unconditional)
  - sample_latent: class_label parameter (default None = unconditional)

Usage:
    from models_shared import (
        ResBlock3D, Encoder3D, Upsample3D, Decoder3D,
        VectorQuantizer, SinEmb, DenoisingUNet3D
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """3D U-Net conditioned on timestep t, optionally step-size d, and optionally class label.

    Backward compatible: num_classes=0 → identical to Paper 2 (unconditional).
    With num_classes>0: class embedding is summed into the time embedding,
    following standard practice (Ho & Salimans 2022, Dhariwal & Nichol 2021).
    """
    def __init__(self, ch=8, td=128, num_classes=0):
        super().__init__()
        self.num_classes = num_classes
        self.te = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        self.de = nn.Sequential(SinEmb(td), nn.Linear(td, td), nn.GELU(), nn.Linear(td, td))
        # Class conditioning: learned embedding summed into time embedding
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, td)
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

    def forward(self, x, t, d=None, class_label=None):
        te = self.te(t)
        if d is not None:
            te = te + self.de(d)
        if class_label is not None and self.num_classes > 0:
            te = te + self.class_emb(class_label)
        h1 = self.e1(x) + self.tp1(te)[:, :, None, None, None]
        h2 = self.e2(self.d1(h1)) + self.tp2(te)[:, :, None, None, None]
        hm = self.mid(self.d2(h2)) + self.tpm(te)[:, :, None, None, None]
        h = self.dc2(torch.cat([self.u2(hm), h2], 1))
        return self.dc1(torch.cat([self.u1(h), h1], 1))


# ============================================================
# Utility: load checkpoint into U-Net
# ============================================================

def load_unet(ckpt_path, dev, num_classes=0):
    """Load a U-Net from checkpoint, handling both full-model and unet-only state dicts."""
    unet = DenoisingUNet3D(8, num_classes=num_classes).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    unet_state = {k.replace("unet.", ""): v for k, v in ckpt.items() if k.startswith("unet.")}
    if not unet_state:
        unet_state = ckpt
    unet.load_state_dict(unet_state)
    unet.eval()
    return unet


def load_vqgan(p1_path, dev):
    """Load VQ-GAN components from Phase 1 checkpoint."""
    p1_state = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev)
    dec = Decoder3D(1, 8, 2).to(dev)
    vq = VectorQuantizer(256, 8).to(dev)
    enc.load_state_dict(p1_state["enc"])
    dec.load_state_dict(p1_state["dec"])
    vq.load_state_dict(p1_state["vq"])
    enc.eval(); dec.eval(); vq.eval()
    return enc, dec, vq, p1_state


# ============================================================
# Sampling functions
# ============================================================

def sample_latent(unet, method, n, dev, steps, ch=8, class_label=None):
    """Generate latent samples for a given method.

    Args:
        class_label: Optional int or Tensor. If int, broadcast to all samples.
                     If Tensor of shape (n,), use per-sample labels.
                     If None, unconditional generation.
    """
    z = torch.randn(n, ch, 8, 8, 8, device=dev)

    # Prepare class labels
    cl = None
    if class_label is not None:
        if isinstance(class_label, int):
            cl = torch.full((n,), class_label, device=dev, dtype=torch.long)
        elif isinstance(class_label, torch.Tensor):
            cl = class_label.to(dev)
        else:
            cl = torch.tensor([class_label] * n, device=dev, dtype=torch.long)

    with torch.no_grad():
        if method == "shortcut":
            d_val = 1.0 / steps
            for i in range(steps):
                t = torch.full((n,), i * d_val, device=dev)
                d = torch.full((n,), d_val, device=dev)
                z = z + d_val * unet(z, t, d=d, class_label=cl)
        elif method == "consistency":
            if steps == 1:
                t = torch.zeros(n, device=dev)
                z = z + unet(z, t, class_label=cl)
            else:
                ts = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
                for i, t_val in enumerate(ts):
                    t = torch.full((n,), t_val.item(), device=dev)
                    v = unet(z, t, class_label=cl)
                    x_hat = z + (1 - t_val) * v
                    if i < steps - 1:
                        t_next = ts[i + 1].item()
                        z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
                    else:
                        z = x_hat
        elif method == "ddpm":
            b = torch.linspace(1e-4, 0.02, 1000)
            a = 1 - b; ac = torch.cumprod(a, 0)
            ss = max(1000 // steps, 1)
            for tv in range(999, -1, -ss):
                t = torch.full((n,), tv, device=dev, dtype=torch.float32)
                np_ = unet(z, t, class_label=cl)
                z = (1/math.sqrt(a[tv])) * (z - (b[tv]/math.sqrt(1-ac[tv])) * np_)
                if tv > 0:
                    z = z + math.sqrt(b[tv]) * torch.randn_like(z)
        else:  # fm or rectified
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((n,), i * dt, device=dev)
                z = z + unet(z, t, class_label=cl) * dt
    return z
