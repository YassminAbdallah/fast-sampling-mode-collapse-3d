#!/usr/bin/env python3
"""
2x2 EMA × Loss factorial — TRAINING (the two missing cells)
============================================================

Closes the §5.2 "EMA vs objective" confounding by training the two cells
of the 2x2 factorial that are not already present in the paper:

      Cell           | Loss          | EMA target | Status         | Source
      ---------------+---------------+------------+----------------+--------------
      EMA + L2       | L2            | yes (EMA)  | already trained | original CD in src/paper4/flow_matching_3d.py
      no-EMA +       |               |            |                |
      Pseudo-Huber   | Pseudo-Huber  | no         | already trained | tier_c/item15_improved_cd
      EMA +          |               |            |                |
      Pseudo-Huber   | Pseudo-Huber  | yes (EMA)  | TO TRAIN HERE  | --variant ema_pseudohuber
      no-EMA + L2    | L2            | no         | TO TRAIN HERE  | --variant noema_l2

After this script trains both new cells, the four-cell factorial cleanly
isolates EMA from the loss function in a way the original Improved CD
ablation (which changed both at once) could not.

Architecture: identical to original Consistency Distillation:
  - Same VQ-GAN encoder/decoder/codebook (loaded from phase1_shared.pt)
  - Same FM teacher used for the backward ODE step
  - Same student initialization (from the FM teacher's weights)
  - Same number of epochs and learning-rate schedule
  - Same step-count curriculum (s0 -> s1)
  - Same dataset (BraTS conditional, 180-volume training subset)

Only EMA and loss differ across cells.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/

    # Cell C: EMA + Pseudo-Huber (~9-12h on M-series)
    python tier_c/item15b_2x2_factorial/train_2x2_factorial.py \\
        --variant ema_pseudohuber \\
        --gen-dir results/paper4/brats_benchmark_20260324_154926 \\
        --data-path data/brats_conditional_64.pt \\
        --output-dir results/paper4/cd_2x2 \\
        --epochs 80

    # Cell D: no-EMA + L2 (~9-12h on M-series)
    python tier_c/item15b_2x2_factorial/train_2x2_factorial.py \\
        --variant noema_l2 \\
        --gen-dir results/paper4/brats_benchmark_20260324_154926 \\
        --data-path data/brats_conditional_64.pt \\
        --output-dir results/paper4/cd_2x2 \\
        --epochs 80

Outputs:
    <output-dir>/<variant>/final.pt
    <output-dir>/<variant>/training_log.json
"""

import os, sys, json, time, argparse, warnings
from copy import deepcopy
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

from flow_matching_3d import (  # noqa: E402
    BrainDS, LatentGenerative3D, get_config, freeze_p1, unfreeze,
    DenoisingUNet3D,
)

warnings.filterwarnings("ignore")


# ============================================================
# Variant-aware Consistency Distillation loss
# ============================================================

def cd_loss_variant(model, x, teacher_unet, n_steps, variant,
                     ema_unet=None, class_label=None, huber_c=None):
    """Consistency Distillation loss with selectable EMA and loss function.

    Args:
        model: LatentGenerative3D instance.
        x: batch of input volumes.
        teacher_unet: frozen FM teacher U-Net (for the backward ODE step).
        n_steps: current step-count in the curriculum.
        variant: one of {'ema_pseudohuber', 'noema_l2'}.
                 (The other two cells of the 2x2 — ema_l2 and noema_pseudohuber —
                  are already trained elsewhere; see the module docstring.)
        ema_unet: EMA copy of the student U-Net. Required when variant uses EMA.
        class_label: optional class label tensor for conditional generation.
        huber_c: optional Pseudo-Huber constant override.

    Returns:
        Scalar loss tensor.
    """
    use_ema = variant in ("ema_l2", "ema_pseudohuber")
    use_huber = variant in ("ema_pseudohuber", "noema_pseudohuber")

    if use_ema and ema_unet is None:
        raise ValueError(f"variant={variant} requires --ema_unet to be provided")

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

    # One backward ODE step using the teacher (no grad)
    with torch.no_grad():
        dt = (t_n - t_n1)[:, None, None, None, None]
        x_tn1 = x_tn - teacher_unet(x_tn, t_n, class_label=class_label) * dt

    # Student prediction at t_n (the side we are training)
    v_student = model.unet(x_tn, t_n, class_label=class_label)
    f_student = x_tn + (1 - te_n) * v_student

    # Target prediction at t_{n-1} — either EMA or stop-grad on student itself
    with torch.no_grad():
        if use_ema:
            v_target = ema_unet(x_tn1, t_n1, class_label=class_label)
        else:
            v_target = model.unet(x_tn1, t_n1, class_label=class_label)
        f_target = x_tn1 + (1 - te_n1) * v_target

    delta = f_student - f_target

    if use_huber:
        # Pseudo-Huber loss as in Song & Dhariwal (2023b):
        # L = sqrt(||delta||^2 + c^2) - c
        if huber_c is None:
            d = delta.numel() / B
            huber_c = 0.00054 * (d ** 0.5)
        sq_sum = (delta * delta).flatten(1).sum(dim=1)
        loss = torch.sqrt(sq_sum + huber_c * huber_c) - huber_c
    else:
        # Standard L2 loss
        loss = (delta * delta).flatten(1).mean(dim=1)

    return loss.mean()


# ============================================================
# Training loop
# ============================================================

def train_2x2_cell(model, dl, cfg, dev, rdir, teacher_state, variant, n_epochs,
                    ema_decay=0.999):
    """Train one cell of the 2x2 factorial."""
    print(f"\n{'='*60}\n2x2 FACTORIAL — variant '{variant}' — {n_epochs} epochs\n{'='*60}")
    freeze_p1(model)

    num_classes = cfg.get("num_classes", 0)
    teacher = DenoisingUNet3D(cfg["latent_channels"], num_classes=num_classes).to(dev)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize student from teacher (same as original CD)
    model.unet.load_state_dict(teacher_state)

    # If variant uses EMA, build an EMA target network initialised from teacher
    use_ema = variant in ("ema_l2", "ema_pseudohuber")
    ema_unet = None
    if use_ema:
        ema_unet = DenoisingUNet3D(cfg["latent_channels"], num_classes=num_classes).to(dev)
        ema_unet.load_state_dict(teacher_state)
        ema_unet.eval()
        for p in ema_unet.parameters():
            p.requires_grad = False
        print(f"  EMA target enabled (decay={ema_decay})")
    else:
        print(f"  No EMA: target = stop-gradient on the student itself")

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
            l = cd_loss_variant(model, x, teacher, ns, variant,
                                 ema_unet=ema_unet, class_label=cl)
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            # Update EMA after each optimizer step
            if use_ema:
                with torch.no_grad():
                    for pe, ps in zip(ema_unet.parameters(), model.unet.parameters()):
                        pe.data.mul_(ema_decay).add_(ps.data, alpha=1 - ema_decay)
            el += l.item()
            n += 1
        avg = el / max(n, 1)
        sch.step()
        elapsed = time.time() - t0
        log.append({"epoch": e, "loss": avg, "n_steps": ns, "elapsed_s": elapsed})
        if e % 5 == 0 or e == 1:
            print(f"  [{e:3d}/{n_epochs}] {variant}={avg:.6f} N={ns} | {elapsed:.1f}s")

    unfreeze(model)
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), rdir / "final.pt")
    print(f"\nSaved {rdir/'final.pt'}")
    return log


# ============================================================
# Main
# ============================================================

VARIANT_DESCRIPTIONS = {
    "ema_pseudohuber": "EMA target + Pseudo-Huber loss (Cell C: 'EMA + Pseudo-Huber')",
    "noema_l2":        "No-EMA target + L2 loss          (Cell D: 'no-EMA + L2')",
    "ema_l2":          "EMA target + L2 loss             (Cell A: schedule-matched retrain)",
    # The other two cells are already trained:
    # ema_l2          -> original Consistency Distillation
    # noema_pseudohuber -> Improved CD (item15_improved_cd)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANT_DESCRIPTIONS.keys()),
                        help=" | ".join(f"{k} : {v}" for k, v in VARIANT_DESCRIPTIONS.items()))
    parser.add_argument("--gen-dir", default="results/paper4/brats_benchmark_20260324_154926",
                        help="Existing benchmark directory containing phase1_shared.pt and fm/final.pt")
    parser.add_argument("--data-path", default="data/brats_conditional_64.pt",
                        help="Path to the conditional BraTS tensor")
    parser.add_argument("--output-dir", default="results/paper4/cd_2x2",
                        help="Where to save the variant outputs. A subdirectory named "
                             "after the variant will be created inside.")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80,
                        help="Same as original CD (80) for a clean comparison")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--class-label", type=int, default=1,
                        help="Kept for parity with the existing pipeline; not used at training time "
                             "because the conditional U-Net consumes the actual per-sample label.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay (used only if variant has EMA)")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")
    print(f"Variant: {args.variant}  ({VARIANT_DESCRIPTIONS[args.variant]})")

    # ----- Resolve subfolder structure (same as Improved CD) -----
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

    # ----- Build config -----
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
    model = LatentGenerative3D(cfg, method="consistency").to(device)
    model.enc.load_state_dict(p1_state["enc"])
    model.dec.load_state_dict(p1_state["dec"])
    model.vq.load_state_dict(p1_state["vq"])

    # ----- DataLoader -----
    ds = BrainDS(train_vols, labels=train_labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Training: {len(ds)} volumes, batch {args.batch_size}, {len(dl)} iters/epoch")

    # ----- Train -----
    rdir = Path(args.output_dir) / args.variant
    log = train_2x2_cell(model, dl, cfg, device, rdir, fm_unet_state,
                          args.variant, args.epochs, args.ema_decay)

    # ----- Save training log -----
    log_path = Path(args.output_dir) / f"training_log_{args.variant}.json"
    with open(log_path, "w") as f:
        json.dump({
            "method": "consistency_2x2",
            "variant": args.variant,
            "variant_description": VARIANT_DESCRIPTIONS[args.variant],
            "epochs": args.epochs,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "ema_decay": args.ema_decay,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch_log": log,
        }, f, indent=2)
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
