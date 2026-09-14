#!/usr/bin/env python3
"""
Item 15 — Improved Consistency Distillation (no-EMA variant) — TRAINING
========================================================================

Trains a Consistency Distillation student WITHOUT an EMA target network and
WITH a Pseudo-Huber loss, following the "Improved Techniques for Training
Consistency Models" formulation (Song & Dhariwal, 2023b). Everything else is
held identical to the original Consistency Distillation variant in
src/paper4/flow_matching_3d.py:

  - Same 3D U-Net architecture (DenoisingUNet3D)
  - Same VQ-GAN encoder/decoder/codebook (loaded from phase1_shared.pt)
  - Same FM teacher used to take the ODE step backward
  - Same student initialization (from the FM teacher's weights)
  - Same number of epochs and learning rate schedule
  - Same step-count curriculum (s0 -> s1)
  - Same dataset (BraTS conditional, class label = 1 for large tumor by default)

What's different from original CD:
  - No EMA target network: the target prediction is obtained from a
    stop-gradient copy of the current student itself.
  - Loss: Pseudo-Huber  L = sqrt(||delta||^2 + c^2) - c   instead of  L = ||delta||^2.
    The Pseudo-Huber constant c is chosen as 0.00054 * sqrt(d) where d is
    the per-sample latent dimension (8*8*8*8 = 4096), giving c ≈ 0.0346.

Why we run this:
  The paper claims (§6.2) that Consistency Distillation's mode collapse is
  driven by the EMA target in parameter space. This script directly tests
  that hypothesis by removing the EMA. Two outcomes are both informative:
    a) Improved CD has higher diversity than original CD -> confirms EMA hypothesis.
    b) Improved CD still collapses -> the collapse is more fundamental to
       consistency-style distillation than the EMA component alone.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item15_improved_cd/train_improved_cd.py \\
        --gen-dir results/paper4/brats_benchmark_20260324_154926 \\
        --data-path data/brats_conditional_64.pt \\
        --output-dir results/paper4/improved_cd_brats \\
        --num-classes 2 \\
        --epochs 80

Expected wall time on Apple M-series: ~9 hours.

Output: <output-dir>/improved_cd/final.pt
        <output-dir>/training_log.json   (epoch-level losses, timings)
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make the paper-4 source importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from flow_matching_3d import (
    BrainDS, LatentGenerative3D, get_config, freeze_p1, unfreeze,
    DenoisingUNet3D,
)

warnings.filterwarnings("ignore")


# ============================================================
# IMPROVED CD LOSS — no EMA, Pseudo-Huber
# ============================================================

def improved_cd_loss(model, x, teacher_unet, n_steps, class_label=None, huber_c=None):
    """No-EMA Consistency Distillation loss with Pseudo-Huber.

    Mirrors model.consistency_loss() exactly, except:
      - target = stop-gradient on the student itself (no EMA network)
      - Pseudo-Huber loss: sqrt(||delta||^2 + c^2) - c

    Args:
        model: LatentGenerative3D instance (provides .enc, .vq, .unet, .dec)
        x: batch of input volumes (B, 1, 64, 64, 64)
        teacher_unet: frozen FM teacher U-Net for the backward ODE step
        n_steps: current step-count in the curriculum (integer)
        class_label: optional class label tensor for conditional generation
        huber_c: optional explicit Pseudo-Huber constant. If None, computed
                 from latent dimensions as 0.00054 * sqrt(d).

    Returns:
        Scalar loss tensor.
    """
    with torch.no_grad():
        zq, _, _ = model.vq(model.enc(x))
    B = zq.shape[0]
    N = max(n_steps, 2)
    n_idx = torch.randint(1, N, (B,), device=x.device)
    t_n = n_idx.float() / N
    t_n1 = (n_idx - 1).float() / N
    z0 = torch.randn_like(zq)
    te_n = t_n[:, None, None, None, None]
    te_n1 = t_n1[:, None, None, None, None]

    # Noisy sample at t_n
    x_tn = (1 - te_n) * z0 + te_n * zq

    # One backward ODE step using the teacher
    with torch.no_grad():
        dt = (t_n - t_n1)[:, None, None, None, None]
        x_tn1 = x_tn - teacher_unet(x_tn, t_n, class_label=class_label) * dt

    # Student prediction at t_n (the side we are training)
    v_student = model.unet(x_tn, t_n, class_label=class_label)
    f_student = x_tn + (1 - te_n) * v_student

    # Target = stop-gradient on the student at t_{n-1} (NO EMA)
    with torch.no_grad():
        v_target = model.unet(x_tn1, t_n1, class_label=class_label)
        f_target = x_tn1 + (1 - te_n1) * v_target

    delta = f_student - f_target
    # Pseudo-Huber constant scaled by sqrt of per-sample dim
    if huber_c is None:
        d = delta.numel() / B
        huber_c = 0.00054 * (d ** 0.5)
    sq_sum = (delta * delta).flatten(1).sum(dim=1)
    loss = torch.sqrt(sq_sum + huber_c * huber_c) - huber_c
    return loss.mean()


# ============================================================
# TRAINING LOOP
# ============================================================

def train_improved_cd(model, dl, cfg, dev, rdir, teacher_state, n_epochs):
    """Mirrors src/paper4/flow_matching_3d.py::train_consistency but without EMA."""
    print(f"\n{'='*60}\nIMPROVED CD (no EMA, Pseudo-Huber) — {n_epochs} epochs\n{'='*60}")
    freeze_p1(model)

    num_classes = cfg.get("num_classes", 0)
    teacher = DenoisingUNet3D(cfg["latent_channels"], num_classes=num_classes).to(dev)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize student from teacher (same as original CD)
    model.unet.load_state_dict(teacher_state)

    params = list(model.unet.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr_gen"] * 0.5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    s0, s1 = cfg["consistency_s0"], cfg["consistency_s1"]

    log = []
    for e in range(1, n_epochs + 1):
        ns = int(s0 + (e / n_epochs) * (s1 - s0))
        el = 0.0
        n = 0
        t0 = time.time()
        model.train()
        for batch in dl:
            cl = batch.get("label", None)
            if cl is not None:
                cl = cl.to(dev)
            x = batch["volume"].to(dev)
            l = improved_cd_loss(model, x, teacher, ns, class_label=cl)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            el += l.item()
            n += 1
        avg = el / max(n, 1)
        sch.step()
        elapsed = time.time() - t0
        log.append({"epoch": e, "loss": avg, "n_steps": ns, "elapsed_s": elapsed})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{n_epochs}] improved_cd={avg:.6f} N={ns} | {elapsed:.1f}s")

    unfreeze(model)
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), rdir / "final.pt")
    print(f"\nSaved {rdir/'final.pt'}")
    return log


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", default="results/paper4/brats_benchmark_20260324_154926",
                        help="Existing benchmark directory containing phase1_shared.pt and fm/final.pt")
    parser.add_argument("--data-path", default="data/brats_conditional_64.pt",
                        help="Path to the conditional BraTS tensor")
    parser.add_argument("--output-dir", default="results/paper4/improved_cd_brats",
                        help="Where to save the trained improved-CD model")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of conditioning classes (2 = small/large tumor)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs (default 80 = same as original CD)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--class-label", type=int, default=1,
                        help="Class to bias training toward, kept for parity with the existing pipeline")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")

    # ----- Resolve subfolder structure -----
    gen_dir = Path(args.gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir
    p1_path = data_dir / "phase1_shared.pt"
    fm_path = data_dir / "fm" / "final.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint not found: {p1_path}")
    if not fm_path.exists():
        raise FileNotFoundError(f"FM teacher not found: {fm_path}")

    print(f"VQ-GAN:    {p1_path}")
    print(f"FM teacher:{fm_path}")

    # ----- Load BraTS conditional dataset -----
    d = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols = d.get("volumes", d.get("v"))
        labels = d.get("labels", d.get("y"))
    else:
        vols, labels = d, None
    if vols is None:
        raise ValueError(f"Could not find volumes in {args.data_path}")
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} volumes, shape {tuple(vols.shape)}")
    if labels is not None:
        print(f"Loaded labels, unique values: {sorted(torch.unique(labels).tolist())}")

    # Use the same 180-volume training subset as the existing pipeline
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(vols), generator=g)
    train_idx = perm[:180]
    train_vols = vols[train_idx].float()
    train_labels = labels[train_idx] if labels is not None else None

    # ----- Build config (mirrors get_config defaults for BraTS) -----
    cfg = get_config(quick=False, dataset="brats")
    cfg["num_classes"] = args.num_classes
    cfg["epochs_consistency"] = args.epochs

    # ----- Load VQ-GAN and FM teacher state -----
    p1_state = torch.load(p1_path, map_location=device, weights_only=True)
    fm_full = torch.load(fm_path, map_location=device, weights_only=True)
    fm_unet_state = {k.replace("unet.", ""): v for k, v in fm_full.items() if k.startswith("unet.")}
    if not fm_unet_state:
        fm_unet_state = fm_full

    # ----- Build LatentGenerative3D with method="consistency" -----
    # Sampling uses the consistency function f(x,t) = x + (1-t)v(x,t), identical to original CD.
    # We choose method="consistency" so model.consistency_sample() can be invoked at eval time.
    model = LatentGenerative3D(cfg, method="consistency").to(device)
    model.enc.load_state_dict(p1_state["enc"])
    model.dec.load_state_dict(p1_state["dec"])
    model.vq.load_state_dict(p1_state["vq"])

    # ----- DataLoader -----
    ds = BrainDS(train_vols, labels=train_labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training: {len(ds)} volumes, batch {args.batch_size}, {len(dl)} iters/epoch")

    # ----- Train -----
    rdir = Path(args.output_dir) / "improved_cd"
    log = train_improved_cd(model, dl, cfg, device, rdir, fm_unet_state, args.epochs)

    # ----- Save training log -----
    log_path = Path(args.output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "method": "improved_cd",
            "variant": "no-EMA + Pseudo-Huber",
            "epochs": args.epochs,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch_log": log,
        }, f, indent=2)
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
