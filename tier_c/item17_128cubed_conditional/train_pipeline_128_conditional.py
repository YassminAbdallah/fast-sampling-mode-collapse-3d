#!/usr/bin/env python3
"""
Conditional 128³ training pipeline (Week 3–5 of the v7.1 → v7.x revision plan)
==============================================================================

Re-trains the four-stage pipeline (VQ-GAN → FM teacher → Consistency
Distillation → Shortcut FM) at 128³ resolution **with binary class
conditioning** (Small tumor vs Large tumor in the BraTS 2023 GLI dataset),
so that the conditional benchmark (Table 6 in the paper) and the
downstream E10a / E10b experiments can be reproduced at higher resolution.

This is the analogue of `tier_c/item17_128cubed/train_pipeline_128.py`
(unconditional), but every generative model that takes a class label
in its forward pass propagates it through training.

Architecture / interface
------------------------
* VQ-GAN is unchanged from the unconditional 128³ pipeline (it doesn't
  consume class labels).
* `DenoisingUNet3D(ch=8, num_classes=2)` is the conditional U-Net from
  `src/paper4/models_shared.py`. With num_classes>0 the class embedding
  is summed into the time embedding and the model accepts
  `forward(x, t, d=None, class_label=...)`.
* All three flow models (FM teacher, Consistency student, Shortcut)
  take a class_label tensor on every forward pass.

Phases
------
    vqgan       — Phase 1 conditional VQ-GAN (same as unconditional; no label).
    fm          — Phase 2 conditional FM teacher.
    consistency — Phase 3 conditional Consistency Distillation from FM.
    shortcut    — Phase 4 conditional Shortcut FM (standalone).
    all         — Run all four sequentially.

Usage
-----
    cd fast-sampling-mode-collapse-3d/

    # Phase 1 (VQ-GAN) — ~24 hours on M-series
    python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \\
        --data-path data/brats_conditional_128.pt \\
        --output-dir results/paper4/brats_128cubed_conditional \\
        --phase vqgan

    # Phase 2 (FM teacher) — ~24 hours
    python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \\
        --data-path data/brats_conditional_128.pt \\
        --output-dir results/paper4/brats_128cubed_conditional \\
        --phase fm

    # Phase 3 (Consistency, distilled from FM)
    python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \\
        --data-path data/brats_conditional_128.pt \\
        --output-dir results/paper4/brats_128cubed_conditional \\
        --phase consistency

    # Phase 4 (Shortcut)
    python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \\
        --data-path data/brats_conditional_128.pt \\
        --output-dir results/paper4/brats_128cubed_conditional \\
        --phase shortcut

    # Or end-to-end:
    python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py \\
        --phase all

Approximate wall-clock on Apple M-series (24–32 GB, batch_size=2)
-----------------------------------------------------------------
    Phase 1 VQ-GAN:           ~24 h
    Phase 2 conditional FM:   ~24 h
    Phase 3 conditional CD:   ~12 h
    Phase 4 conditional Shortcut: ~24 h
    Total:                    ~80–90 h (~4 calendar days running continuously)

You can launch each phase independently as a separate background process.
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse the conditional models from paper4
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models_shared import (  # noqa: E402
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D,
)

# Deterministic seeding helper (added in v7.2)
try:
    from utils.set_seed import set_seed  # noqa: E402
    HAS_SET_SEED = True
except Exception:
    HAS_SET_SEED = False

warnings.filterwarnings("ignore")


# ============================================================
# Dataset (volume + label)
# ============================================================

class ConditionalVolumeDS(Dataset):
    def __init__(self, volumes, labels):
        assert volumes.shape[0] == labels.shape[0], (
            f"volume/label count mismatch: {volumes.shape[0]} vs {labels.shape[0]}"
        )
        self.v = volumes if volumes.dim() == 5 else volumes.unsqueeze(1)
        self.y = labels.long()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return {"volume": self.v[i].float(), "label": self.y[i]}


# ============================================================
# Discriminator (unconditional, reused from item17_128cubed)
# ============================================================

class Discriminator3D(nn.Module):
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
# Helpers
# ============================================================

@torch.no_grad()
def encode_quantize(enc, vq, x):
    return vq(enc(x))[0]


def save_log(path, payload):
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Phase 1: VQ-GAN training (unconditional; labels are not consumed here)
# ============================================================

def train_vqgan(args, dev, vols, labels, output_dir):
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

    ds = ConditionalVolumeDS(vols, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training set: {len(ds)} volumes (labels ignored by VQ-GAN), "
          f"batch {args.batch_size}, {len(dl)} iters/epoch")

    losses = []
    for e in range(1, args.epochs_vqgan + 1):
        t0 = time.time()
        enc.train(); dec.train(); vq.train(); disc.train()
        ep_l_recon, ep_l_vq, ep_l_g_adv, ep_l_d = 0.0, 0.0, 0.0, 0.0
        n_batch = 0
        for it, batch in enumerate(dl):
            x = batch["volume"].to(dev)
            if it % 4 == 0:
                with torch.no_grad():
                    z, _, _ = vq(enc(x))
                    x_hat = dec(z).clamp(0, 1)
                d_real = disc(x).mean()
                d_fake = disc(x_hat.detach()).mean()
                l_d = (F.relu(1 - d_real) + F.relu(1 + d_fake)).mean()
                opt_d.zero_grad(); l_d.backward(); opt_d.step()
                ep_l_d += l_d.item()
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
        avg_vq    = ep_l_vq    / n_batch
        avg_g_adv = ep_l_g_adv / n_batch
        avg_d     = ep_l_d     / max(1, n_batch // 4)
        losses.append({"epoch": e, "recon": avg_recon, "vq": avg_vq,
                       "g_adv": avg_g_adv, "d": avg_d,
                       "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_vqgan}] recon={avg_recon:.4f} "
                  f"vq={avg_vq:.4f} g_adv={avg_g_adv:.4f} d={avg_d:.4f} "
                  f"| {time.time()-t0:.1f}s")

    state = {"enc": enc.state_dict(), "dec": dec.state_dict(), "vq": vq.state_dict()}
    torch.save(state, output_dir / "phase1_shared.pt")
    save_log(output_dir / "phase1_log.json",
             {"phase": "vqgan", "epochs": args.epochs_vqgan, "losses": losses,
              "conditional": True, "num_classes": args.num_classes})
    print(f"\nSaved {output_dir / 'phase1_shared.pt'}")


# ============================================================
# Phase 2: conditional FM teacher
# ============================================================

def train_fm(args, dev, vols, labels, output_dir):
    print(f"\n{'='*60}\nPHASE 2: Conditional FM teacher at 128³ "
          f"({args.epochs_fm} epochs, num_classes={args.num_classes})\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint missing: {p1_path}. Run --phase vqgan first.")

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()):
        p.requires_grad = False

    unet = DenoisingUNet3D(ch=8, num_classes=args.num_classes).to(dev)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_fm)

    ds = ConditionalVolumeDS(vols, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training set: {len(ds)} volumes, batch {args.batch_size}, {len(dl)} iters/epoch")

    losses = []
    for e in range(1, args.epochs_fm + 1):
        t0 = time.time(); unet.train(); ep_l = 0.0; n = 0
        for batch in dl:
            x = batch["volume"].to(dev)
            y = batch["label"].to(dev)
            zq = encode_quantize(enc, vq, x)
            z0 = torch.randn_like(zq)
            t = torch.rand(zq.shape[0], device=dev)
            t_b = t[:, None, None, None, None]
            xt = (1 - t_b) * z0 + t_b * zq
            v_target = zq - z0
            v_pred = unet(xt, t, class_label=y)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "fm_loss": avg, "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_fm}] cond_fm={avg:.6f} | {time.time()-t0:.1f}s")

    (output_dir / "fm").mkdir(exist_ok=True)
    torch.save({"unet": unet.state_dict(), "num_classes": args.num_classes},
               output_dir / "fm" / "final.pt")
    save_log(output_dir / "fm" / "training_log.json",
             {"phase": "fm", "epochs": args.epochs_fm, "losses": losses,
              "conditional": True, "num_classes": args.num_classes})
    print(f"\nSaved {output_dir / 'fm' / 'final.pt'}")


# ============================================================
# Phase 3: Conditional Consistency Distillation
# ============================================================

def train_consistency(args, dev, vols, labels, output_dir):
    print(f"\n{'='*60}\nPHASE 3: Conditional Consistency Distillation at 128³ "
          f"({args.epochs_consistency} epochs)\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    fm_path = output_dir / "fm" / "final.pt"
    if not p1_path.exists() or not fm_path.exists():
        raise FileNotFoundError(
            "VQ-GAN or FM teacher checkpoint missing. Run --phase vqgan and --phase fm first."
        )

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()):
        p.requires_grad = False

    fm_ckpt = torch.load(fm_path, map_location=dev, weights_only=True)
    teacher_state = fm_ckpt["unet"]
    nc = fm_ckpt.get("num_classes", args.num_classes)

    teacher = DenoisingUNet3D(ch=8, num_classes=nc).to(dev)
    teacher.load_state_dict(teacher_state); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = DenoisingUNet3D(ch=8, num_classes=nc).to(dev)
    student.load_state_dict(teacher_state)
    ema = DenoisingUNet3D(ch=8, num_classes=nc).to(dev)
    ema.load_state_dict(teacher_state); ema.eval()
    for p in ema.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_consistency)

    ds = ConditionalVolumeDS(vols, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    s0, s1 = 2, 150
    ema_decay = 0.999

    losses = []
    for e in range(1, args.epochs_consistency + 1):
        t0 = time.time(); student.train(); ep_l = 0.0; n = 0
        ns = int(s0 + (e / args.epochs_consistency) * (s1 - s0))
        for batch in dl:
            x = batch["volume"].to(dev); y = batch["label"].to(dev)
            zq = encode_quantize(enc, vq, x)
            B = zq.shape[0]; N = max(ns, 2)
            n_idx = torch.randint(1, N, (B,), device=dev)
            t_n = n_idx.float() / N
            t_n1 = (n_idx - 1).float() / N
            te_n = t_n[:, None, None, None, None]
            te_n1 = t_n1[:, None, None, None, None]
            z0 = torch.randn_like(zq)
            x_tn = (1 - te_n) * z0 + te_n * zq
            with torch.no_grad():
                dt = (t_n - t_n1)[:, None, None, None, None]
                x_tn1 = x_tn - teacher(x_tn, t_n, class_label=y) * dt
            v_s = student(x_tn, t_n, class_label=y); f_s = x_tn + (1 - te_n) * v_s
            with torch.no_grad():
                v_t = ema(x_tn1, t_n1, class_label=y); f_t = x_tn1 + (1 - te_n1) * v_t
            loss = F.mse_loss(f_s, f_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.data.mul_(ema_decay).add_(ps.data, alpha=1 - ema_decay)
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "consistency_loss": avg, "n_steps": ns,
                       "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_consistency}] cond_cd={avg:.6f} N={ns} "
                  f"| {time.time()-t0:.1f}s")

    (output_dir / "consistency").mkdir(exist_ok=True)
    torch.save({"unet": student.state_dict(), "num_classes": nc},
               output_dir / "consistency" / "final.pt")
    save_log(output_dir / "consistency" / "training_log.json",
             {"phase": "consistency", "epochs": args.epochs_consistency,
              "losses": losses, "conditional": True, "num_classes": nc})
    print(f"\nSaved {output_dir / 'consistency' / 'final.pt'}")


# ============================================================
# Phase 4: Conditional Shortcut FM
# ============================================================

def train_shortcut(args, dev, vols, labels, output_dir):
    print(f"\n{'='*60}\nPHASE 4: Conditional Shortcut FM at 128³ "
          f"({args.epochs_shortcut} epochs)\n{'='*60}")
    p1_path = output_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint missing: {p1_path}.")

    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()):
        p.requires_grad = False

    unet = DenoisingUNet3D(ch=8, num_classes=args.num_classes).to(dev)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs_shortcut)

    ds = ConditionalVolumeDS(vols, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    sc_ratio = 0.25  # 25% of batches use SC loss

    losses = []
    for e in range(1, args.epochs_shortcut + 1):
        t0 = time.time(); unet.train(); ep_l = 0.0; n = 0
        d_max = min(1.0, (1 / 128) + (e / args.epochs_shortcut) * (1.0 - 1 / 128))
        for batch in dl:
            x = batch["volume"].to(dev); y = batch["label"].to(dev)
            zq = encode_quantize(enc, vq, x)
            use_sc = torch.rand(1).item() < sc_ratio
            if not use_sc:
                z0 = torch.randn_like(zq)
                t = torch.rand(zq.shape[0], device=dev)
                t_b = t[:, None, None, None, None]
                xt = (1 - t_b) * z0 + t_b * zq
                v_target = zq - z0
                d_zero = torch.zeros(zq.shape[0], device=dev)
                v_pred = unet(xt, t, d=d_zero, class_label=y)
                loss = F.mse_loss(v_pred, v_target)
            else:
                z0 = torch.randn_like(zq)
                t = torch.rand(zq.shape[0], device=dev)
                d = torch.empty(zq.shape[0], device=dev).uniform_(1 / 128, d_max)
                d = torch.min(d, (1 - t) / 2).clamp(min=1e-4)
                t_b = t[:, None, None, None, None]
                xt = (1 - t_b) * z0 + t_b * zq
                with torch.no_grad():
                    d_b = d[:, None, None, None, None]
                    v1 = unet(xt, t, d=d, class_label=y)
                    xt_mid = xt + d_b * v1
                    v2 = unet(xt_mid, t + d, d=d, class_label=y)
                    target = (v1 + v2) / 2
                v_big = unet(xt, t, d=2 * d, class_label=y)
                loss = F.mse_loss(v_big, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
            ep_l += loss.item(); n += 1
        sch.step()
        avg = ep_l / max(n, 1)
        losses.append({"epoch": e, "shortcut_loss": avg, "d_max": d_max,
                       "elapsed_s": time.time() - t0})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{args.epochs_shortcut}] cond_short={avg:.6f} "
                  f"d_max={d_max:.4f} | {time.time()-t0:.1f}s")

    (output_dir / "shortcut").mkdir(exist_ok=True)
    torch.save({"unet": unet.state_dict(), "num_classes": args.num_classes},
               output_dir / "shortcut" / "final.pt")
    save_log(output_dir / "shortcut" / "training_log.json",
             {"phase": "shortcut", "epochs": args.epochs_shortcut,
              "losses": losses, "conditional": True, "num_classes": args.num_classes})
    print(f"\nSaved {output_dir / 'shortcut' / 'final.pt'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/brats_conditional_128.pt",
                        help="128³ conditional dataset built by prepare_conditional_128.py")
    parser.add_argument("--output-dir", default="results/paper4/brats_128cubed_conditional")
    parser.add_argument("--phase", default="all",
                        choices=["vqgan", "fm", "consistency", "shortcut", "all"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs-vqgan", type=int, default=80)
    parser.add_argument("--epochs-fm", type=int, default=80)
    parser.add_argument("--epochs-consistency", type=int, default=60)
    parser.add_argument("--epochs-shortcut", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducibility (set_seed helper applied if available).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if HAS_SET_SEED:
        set_seed(args.seed)
        print(f"set_seed({args.seed}) applied")
    else:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        print(f"set_seed helper not found; using fallback seeding with seed={args.seed}")

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if isinstance(raw, torch.Tensor):
        sys.exit("ERROR: --data-path looks like an unconditional tensor (no labels). "
                 "Run prepare_conditional_128.py first to produce a conditional dataset.")
    vols = raw["volumes"]
    labels = raw["labels"]
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    print(f"Loaded {vols.shape[0]} conditional volumes shape {tuple(vols.shape)} "
          f"with {args.num_classes} classes: "
          f"{[int((labels==c).sum().item()) for c in range(args.num_classes)]} per class")

    phases = ["vqgan", "fm", "consistency", "shortcut"] if args.phase == "all" else [args.phase]
    for ph in phases:
        if ph == "vqgan":
            train_vqgan(args, dev, vols, labels, output_dir)
        elif ph == "fm":
            train_fm(args, dev, vols, labels, output_dir)
        elif ph == "consistency":
            train_consistency(args, dev, vols, labels, output_dir)
        elif ph == "shortcut":
            train_shortcut(args, dev, vols, labels, output_dir)


if __name__ == "__main__":
    main()
