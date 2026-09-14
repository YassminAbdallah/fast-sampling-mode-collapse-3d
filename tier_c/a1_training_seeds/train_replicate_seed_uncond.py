#!/usr/bin/env python3
"""
A1 unconditional — retrain CD and Shortcut with alternative training seeds
at 64³ UNCONDITIONAL BraTS. Companion to train_replicate_seed.py which is
conditional.

Why both:
    The conditional A1 (Table 6 setup) shows CD variance 34–52% and
    Shortcut 70–81%. But Table 3's headline "CD collapses to ~2%" is
    UNCONDITIONAL. To let the paper speak to whether the unconditional
    2% number is a training-seed artifact or method-inherent, we retrain
    UNCONDITIONALLY as well.

Everything is identical to train_replicate_seed.py except:
    - dataset = brats_preprocessed_64.pt (300 volumes, no labels)
    - benchmark dir = results/paper3/brats_benchmark_20260221_193306/100pct_300vol
    - num_classes = 0 (no conditioning)

Runtime: ~10 min CD, ~15 min Shortcut on M1 (same as conditional).
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.set_seed import set_seed, get_worker_init_fn  # noqa: E402
from flow_matching_3d import (   # noqa: E402
    BrainDS, LatentGenerative3D, get_config,
    train_consistency, train_shortcut,
    freeze_p1, unfreeze,
)

warnings.filterwarnings("ignore")


# ============================================================
# Paths (unconditional BraTS 64^3 — Table 3 source)
# ============================================================

BENCH_DIR = REPO_ROOT / "results/paper3/brats_benchmark_20260221_193306/100pct_300vol"
DATA_PATH = REPO_ROOT / "data/brats_preprocessed_64.pt"

VQGAN_PATH   = BENCH_DIR / "phase1_shared.pt"
TEACHER_PATH = BENCH_DIR / "fm/final.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["consistency", "shortcut"])
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override; default 80 CD, 150 Shortcut")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output-dir",
                    default="results/paper4/train_seed_replication_uncond")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else
                             "cpu")
    print(f"Device: {device}")
    print(f"Method: {args.method} (UNCONDITIONAL)")
    print(f"Training seed: {args.train_seed}")

    # Seed everything
    g_torch = set_seed(args.train_seed)
    worker_init = get_worker_init_fn(args.train_seed)

    for path in (VQGAN_PATH, TEACHER_PATH, DATA_PATH):
        if not path.exists():
            sys.exit(f"ERROR: missing {path}")
    print(f"VQ-GAN:  {VQGAN_PATH}")
    print(f"FM teacher: {TEACHER_PATH}")
    print(f"Data:    {DATA_PATH}")

    # Load unconditional dataset (no labels)
    d = torch.load(DATA_PATH, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols   = d.get("volumes", d.get("v"))
        labels = None  # unconditional
    else:
        vols, labels = d, None
    if vols is None:
        sys.exit(f"ERROR: could not find volumes in {DATA_PATH}")
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} volumes, shape {tuple(vols.shape)}")

    # 180-volume training subset from the same perm(seed=42) as the paper
    g_perm = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(vols), generator=g_perm)
    train_idx = perm[:180]
    train_vols = vols[train_idx].float()

    # cfg with unconditional
    cfg = get_config(quick=False, dataset="brats")
    cfg["num_classes"] = 0
    if args.epochs is not None:
        cfg["epochs_consistency"] = args.epochs
        cfg["epochs_shortcut"]    = args.epochs
    print(f"Epochs: consistency={cfg['epochs_consistency']}, "
          f"shortcut={cfg['epochs_shortcut']}, num_classes=0 (unconditional)")

    # Load VQ-GAN and FM teacher
    p1_state = torch.load(VQGAN_PATH,   map_location=device, weights_only=True)
    fm_full  = torch.load(TEACHER_PATH, map_location=device, weights_only=True)
    fm_unet_state = {k.replace("unet.", ""): v for k, v in fm_full.items()
                     if k.startswith("unet.")}
    if not fm_unet_state:
        fm_unet_state = fm_full

    # Build model
    model = LatentGenerative3D(cfg, method=args.method).to(device)
    model.enc.load_state_dict(p1_state["enc"])
    model.dec.load_state_dict(p1_state["dec"])
    model.vq.load_state_dict(p1_state["vq"])

    # DataLoader (unconditional — labels=None)
    ds = BrainDS(train_vols, labels=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True,
                    generator=g_torch, worker_init_fn=worker_init)
    print(f"Training: {len(ds)} volumes, batch {args.batch_size}, "
          f"{len(dl)} iters/epoch")

    rdir = REPO_ROOT / args.output_dir / f"{args.method}_trainseed{args.train_seed}"
    rdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.method == "consistency":
        losses = train_consistency(model, dl, cfg, device, rdir, fm_unet_state)
    else:
        losses = train_shortcut(model, dl, cfg, device, rdir)
    total = time.time() - t0
    print(f"\nTraining finished in {total:.0f}s = {total/3600:.2f}h")

    log_path = rdir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "method": args.method,
            "train_seed": args.train_seed,
            "conditional": False,
            "epochs": (cfg["epochs_consistency"] if args.method == "consistency"
                       else cfg["epochs_shortcut"]),
            "batch_size": args.batch_size,
            "final_loss": float(losses[-1]) if losses else None,
            "loss_trajectory": [float(l) for l in losses],
            "wall_seconds": total,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "vqgan_source": str(VQGAN_PATH),
            "fm_teacher_source": str(TEACHER_PATH),
            "data_source": str(DATA_PATH),
        }, f, indent=2)
    print(f"Saved training log: {log_path}")
    print(f"Saved model:        {rdir/'final.pt'}")


if __name__ == "__main__":
    main()
