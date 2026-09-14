#!/usr/bin/env python3
"""
Item 17 — 128³ resolution validation: training pipeline
========================================================

Trains the four-stage pipeline (VQ-GAN → FM teacher → Consistency Distillation
→ Shortcut FM) at 128³ resolution so that the Shortcut > Consistency ordering
observed at 64³ can be tested at a higher (closer to clinical) resolution.

Architecture choice:
  - VQ-GAN encoder/decoder: same architecture as the 64³ run (3 stride-2 downsamplings).
    At 128³ input, this produces 16³ latent (instead of 8³). All convolutions are
    shape-agnostic so no architectural changes are needed.
  - DenoisingUNet3D: same architecture, operates on 16³ latent (4096 → 4096×8 = 32k values
    instead of 512 → 4096 values). Memory grows linearly with latent volume.

Phases (--phase flag):
  vqgan:        Train the shared VQ-GAN encoder/decoder/codebook. Saves phase1_shared.pt.
  fm:           Train Flow Matching teacher. Requires phase1_shared.pt. Saves fm/final.pt.
  consistency:  Distill from FM teacher with EMA target. Requires fm/final.pt. Saves consistency/final.pt.
  shortcut:     Train Shortcut FM (standalone, no teacher). Saves shortcut/final.pt.
  all:          Run vqgan → fm → consistency → shortcut sequentially.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed/train_pipeline_128.py \\
        --data-path data/brats_preprocessed_128.pt \\
        --output-dir results/paper4/brats_128cubed \\
        --phase all

    # or run one phase at a time, e.g.:
    python tier_c/item17_128cubed/train_pipeline_128.py \\
        --data-path data/brats_preprocessed_128.pt \\
        --output-dir results/paper4/brats_128cubed \\
        --phase vqgan

Approximate wall-clock times on Apple M-series (24-32 GB, batch_size=2):
  vqgan:       18-26 hours
  fm:          18-22 hours
  consistency: 10-14 hours
  shortcut:    18-22 hours
  Total:       ~65-85 hours

Reduce --epochs-* flags if you need shorter runs (the script also exposes
shorter defaults than the 64³ pipeline since the goal is a resolution-validation
comparison, not a full retraining match).
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse model definitions from paper-4 (architecture is shape-agnostic)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import (
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D, SinEmb,
)

warnings.filterwarnings("ignore")


# ============================================================
# Dataset
# ============================================================

class VolumeDS(Dataset):
    def __init__(self, volumes):
        self.v = volumes if volumes.dim() == 5 else volumes.unsqueeze(1)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return {"volume": self.v[i].float()}


# ============================================================
# Discriminator for VQ-GAN training
# ============================================================

class Discriminator3D(nn.Module):
    """Simple 3D patch-discriminator for VQ-GAN adversarial loss."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base, base * 2, 4, 2, 1), nn.GroupNorm(8, base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base * 2, base * 4, 4, 2, 1), nn.GroupNorm(8, base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Phase 1: VQ-GAN training
# ============================================================

def train_vqgan(args, dev, vols, output_dir):
    print(f"\n{'='*60}\nPHASE 1: VQ-GAN at 128³ ({args.epochs_vqgan} epochs)\n{'='*60}")

    enc = Encoder3D(in_ch=1, lat_ch=8, n_res=2).to(dev)
    dec = Decoder3D(out_ch=1, lat_ch=8, n_res=2).to(dev)
    vq = VectorQuantizer(ne=256, d=8, beta=0.25).to(dev)
    disc = Discriminator3D(in_ch=1).to(dev)

    params_g = list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters())
    params_d = list(disc.parameters())

    opt_g = torch.optim.AdamW(params_g, lr=1e-4, weight_decay=1e-5)
    opt_d = torch.optim.AdamW(params_d, lr=1e-4, weight_decay=1e-5)

    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, args.epochs_vqgan)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, args.epochs_vqgan)

    ds = VolumeDS(vols)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training set: {len(ds)} volumes, batch {args.batch_size}, {len(dl)} iters/epoch")

    losses = []
    for e in range(1, args.epochs_vqgan + 1):
        t0 = time.time()
        enc.train(); dec.train(); vq.train(); disc.train()
        ep_l_recon, ep_l_vq, ep_l_g_adv, ep_l_d = 0.0, 0.0, 0.0, 0.0
        n_batch = 0
        for it, batch in enumerate(dl):
            x = batch["volume"].to(dev)

            # ---- Train discriminator (every 4 iters to keep G ahead) ----
            if it % 4 == 0:
                with torch.no_grad():
                    z, _, _ = vq(enc(x))
                    x_hat = dec(z).clamp(0, 1)
                d_real = disc(x).mean()
                d_fake = disc(x_hat.detach()).mean()
                l_d = (F.relu(1 - d_real) + F.relu(1 + d_fake)).mean()
                opt_d.zero_grad(); l_d.backward(); opt_d.step()
                ep_l_d += l_d.item()

            # ---- Train generator ----
            z, l_vq, _ = vq(enc(x))
            x_hat = dec(z).clamp(0, 1)
            l_recon = F.l1_loss(x_hat, x)
            l_g_adv = -disc(x_hat).mean()
            l_total = l_recon + l_vq + 0.1 * l_g_adv
            opt_g.zero_grad(); l_total.backward(); opt_g.step()
            ep_l_recon += l_recon.item(); ep_l_vq += l_vq.item(); ep_l_g_adv += l_g_adv.item()
            n_batch += 1

        sch_g.step(); sch_d.step()
        avg_recon = ep_l_recon / n_batch
        avg_vq = ep_l_vq / n_batch
        avg_g_adv = ep_l_g_adv / n_batch
        avg_d = ep_l_d / max(1, n_batch // 4)
        losses.append({"epoch": e, "recon": avg_recon, "vq": avg_vq, "g_adv": avg_g_adv, "d": avg_d,
                       "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_vqgan}] recon={avg_recon:.4f} vq={avg_vq:.4f} "
                  f"g_adv={avg_g_adv:.4f} d={avg_d:.4f} | {time.time()-t0:.1f}s")

    state = {"enc": enc.state_dict(), "dec": dec.state_dict(), "vq": vq.state_dict()}
    torch.save(state, output_dir / "phase1_shared.pt")
    with open(output_dir / "phase1_log.json", "w") as f:
        json.dump({"phase": "vqgan", "epochs": args.epochs_vqgan, "losses": losses,
                   "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"\nSaved {output_dir / 'phase1_shared.pt'}")


# ============================================================
# Phase 2: FM training
# ============================================================

@torch.no_grad()
def encode_quantize(enc, vq, x):
    """Encode a volume to a quantized latent (no-grad)."""
    return vq(enc(x))[0]


def train_fm(args, dev, vols, output_dir):
    print(f"\n{'='*60}\nPHASE 2: FM teacher at 128³ ({args.epochs_fm} epochs)\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint missing: {p1_path}. Run --phase vqgan first.")

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    dec = Decoder3D(1, 8, 2).to(dev); dec.load_state_dict(p1["dec"]); dec.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(dec.parameters()) + list(vq.parameters()):
        p.requires_grad = False

    unet = DenoisingUNet3D(ch=8).to(dev)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_fm)

    ds = VolumeDS(vols)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training set: {len(ds)} volumes, batch {args.batch_size}, {len(dl)} iters/epoch")

    losses = []
    for e in range(1, args.epochs_fm + 1):
        t0 = time.time(); unet.train(); ep_l = 0.0; n = 0
        for batch in dl:
            x = batch["volume"].to(dev)
            zq = encode_quantize(enc, vq, x)  # (B, 8, 16, 16, 16)
            z0 = torch.randn_like(zq)
            t = torch.rand(zq.shape[0], device=dev)
            t_b = t[:, None, None, None, None]
            xt = (1 - t_b) * z0 + t_b * zq
            v_target = zq - z0
            v_pred = unet(xt, t)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "fm_loss": avg, "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_fm}] fm={avg:.6f} | {time.time()-t0:.1f}s")

    (output_dir / "fm").mkdir(exist_ok=True)
    torch.save({"unet": unet.state_dict()}, output_dir / "fm" / "final.pt")
    with open(output_dir / "fm" / "training_log.json", "w") as f:
        json.dump({"phase": "fm", "epochs": args.epochs_fm, "losses": losses,
                   "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"\nSaved {output_dir / 'fm' / 'final.pt'}")


# ============================================================
# Phase 3: Consistency Distillation
# ============================================================

def train_consistency(args, dev, vols, output_dir):
    print(f"\n{'='*60}\nPHASE 3: Consistency Distillation at 128³ ({args.epochs_consistency} epochs)\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    fm_path = output_dir / "fm" / "final.pt"
    if not p1_path.exists() or not fm_path.exists():
        raise FileNotFoundError("VQ-GAN or FM teacher checkpoint missing. Run --phase vqgan and --phase fm first.")

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()): p.requires_grad = False

    fm_state = torch.load(fm_path, map_location=dev, weights_only=True)["unet"]
    teacher = DenoisingUNet3D(ch=8).to(dev); teacher.load_state_dict(fm_state); teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    student = DenoisingUNet3D(ch=8).to(dev); student.load_state_dict(fm_state)
    ema = DenoisingUNet3D(ch=8).to(dev); ema.load_state_dict(fm_state); ema.eval()
    for p in ema.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_consistency)

    ds = VolumeDS(vols)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    s0, s1 = 2, 150
    ema_decay = 0.999

    losses = []
    for e in range(1, args.epochs_consistency + 1):
        t0 = time.time(); student.train(); ep_l = 0.0; n = 0
        ns = int(s0 + (e / args.epochs_consistency) * (s1 - s0))
        for batch in dl:
            x = batch["volume"].to(dev)
            zq = encode_quantize(enc, vq, x)
            B = zq.shape[0]; N = max(ns, 2)
            n_idx = torch.randint(1, N, (B,), device=dev)
            t_n = n_idx.float() / N; t_n1 = (n_idx - 1).float() / N
            te_n = t_n[:, None, None, None, None]; te_n1 = t_n1[:, None, None, None, None]
            z0 = torch.randn_like(zq)
            x_tn = (1 - te_n) * z0 + te_n * zq
            with torch.no_grad():
                dt = (t_n - t_n1)[:, None, None, None, None]
                x_tn1 = x_tn - teacher(x_tn, t_n) * dt
            v_s = student(x_tn, t_n); f_s = x_tn + (1 - te_n) * v_s
            with torch.no_grad():
                v_t = ema(x_tn1, t_n1); f_t = x_tn1 + (1 - te_n1) * v_t
            loss = F.mse_loss(f_s, f_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
            # EMA update
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.data.mul_(ema_decay).add_(ps.data, alpha=1 - ema_decay)
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "consistency_loss": avg, "n_steps": ns, "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_consistency}] cons={avg:.6f} N={ns} | {time.time()-t0:.1f}s")

    (output_dir / "consistency").mkdir(exist_ok=True)
    torch.save({"unet": student.state_dict()}, output_dir / "consistency" / "final.pt")
    with open(output_dir / "consistency" / "training_log.json", "w") as f:
        json.dump({"phase": "consistency", "epochs": args.epochs_consistency, "losses": losses,
                   "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"\nSaved {output_dir / 'consistency' / 'final.pt'}")


# ============================================================
# Phase 4: Shortcut FM (standalone, no teacher)
# ============================================================

def train_shortcut(args, dev, vols, output_dir):
    print(f"\n{'='*60}\nPHASE 4: Shortcut FM at 128³ ({args.epochs_shortcut} epochs)\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint missing: {p1_path}.")

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()): p.requires_grad = False

    # DenoisingUNet3D with d-conditioning support is the same class — d is passed via forward kwargs
    unet = DenoisingUNet3D(ch=8).to(dev)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_shortcut)

    ds = VolumeDS(vols)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    sc_ratio = 0.25  # 25% of batches use SC loss

    losses = []
    for e in range(1, args.epochs_shortcut + 1):
        t0 = time.time(); unet.train(); ep_l = 0.0; n = 0
        d_max = min(1.0, (1 / 128) + (e / args.epochs_shortcut) * (1.0 - 1 / 128))
        for batch in dl:
            x = batch["volume"].to(dev)
            zq = encode_quantize(enc, vq, x)
            use_sc = torch.rand(1).item() < sc_ratio
            if not use_sc:
                # ---- Standard FM loss with d=0 ----
                z0 = torch.randn_like(zq)
                t = torch.rand(zq.shape[0], device=dev)
                t_b = t[:, None, None, None, None]
                xt = (1 - t_b) * z0 + t_b * zq
                v_target = zq - z0
                d_zero = torch.zeros(zq.shape[0], device=dev)
                v_pred = unet(xt, t, d=d_zero) if has_d_kwarg(unet) else unet(xt, t)
                loss = F.mse_loss(v_pred, v_target)
            else:
                # ---- SC loss with d>0 (one big step ≈ two small steps) ----
                z0 = torch.randn_like(zq)
                t = torch.rand(zq.shape[0], device=dev)
                d = torch.empty(zq.shape[0], device=dev).uniform_(1/128, d_max)
                # Cap d so t + 2d <= 1
                d = torch.min(d, (1 - t) / 2).clamp(min=1e-4)
                t_b = t[:, None, None, None, None]
                xt = (1 - t_b) * z0 + t_b * zq
                # Two small steps of size d
                with torch.no_grad():
                    d_b = d[:, None, None, None, None]
                    v1 = unet(xt, t, d=d) if has_d_kwarg(unet) else unet(xt, t)
                    xt_mid = xt + d_b * v1
                    v2 = unet(xt_mid, t + d, d=d) if has_d_kwarg(unet) else unet(xt_mid, t + d)
                    target = (v1 + v2) / 2  # stop-gradient on the two-small-steps result
                # One big step of size 2d
                v_big = unet(xt, t, d=2 * d) if has_d_kwarg(unet) else unet(xt, t)
                loss = F.mse_loss(v_big, target)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "shortcut_loss": avg, "d_max": d_max, "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_shortcut}] shortcut={avg:.6f} d_max={d_max:.4f} | {time.time()-t0:.1f}s")

    (output_dir / "shortcut").mkdir(exist_ok=True)
    torch.save({"unet": unet.state_dict()}, output_dir / "shortcut" / "final.pt")
    with open(output_dir / "shortcut" / "training_log.json", "w") as f:
        json.dump({"phase": "shortcut", "epochs": args.epochs_shortcut, "losses": losses,
                   "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"\nSaved {output_dir / 'shortcut' / 'final.pt'}")


def has_d_kwarg(model):
    """Probes whether the underlying DenoisingUNet3D supports a `d` step-size conditioning input."""
    import inspect
    try:
        sig = inspect.signature(model.forward)
        return "d" in sig.parameters
    except Exception:
        return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/brats_preprocessed_128.pt",
                        help="128³ preprocessed BraTS volumes")
    parser.add_argument("--output-dir", default="results/paper4/brats_128cubed",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--phase", default="all",
                        choices=["vqgan", "fm", "consistency", "shortcut", "all"],
                        help="Which training phase to run")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs-vqgan", type=int, default=80,
                        help="Reduced from 200 in the 64³ run for tractable wall time")
    parser.add_argument("--epochs-fm", type=int, default=80,
                        help="Reduced from 150 in the 64³ run")
    parser.add_argument("--epochs-consistency", type=int, default=60,
                        help="Reduced from 80 in the 64³ run")
    parser.add_argument("--epochs-shortcut", type=int, default=80,
                        help="Reduced from 150 in the 64³ run")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 128³ data
    vols = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if isinstance(vols, dict):
        vols = vols.get("volumes", vols.get("v"))
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    print(f"Loaded {vols.shape[0]} volumes shape {tuple(vols.shape)}, dtype {vols.dtype}")

    phases = ["vqgan", "fm", "consistency", "shortcut"] if args.phase == "all" else [args.phase]
    for ph in phases:
        if ph == "vqgan":
            train_vqgan(args, dev, vols, output_dir)
        elif ph == "fm":
            train_fm(args, dev, vols, output_dir)
        elif ph == "consistency":
            train_consistency(args, dev, vols, output_dir)
        elif ph == "shortcut":
            train_shortcut(args, dev, vols, output_dir)


if __name__ == "__main__":
    main()
