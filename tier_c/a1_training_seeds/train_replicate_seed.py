#!/usr/bin/env python3
"""
A1 — retrain CD and Shortcut with alternative training seeds at 64³.
====================================================================

Motivation: the paper's "5 seeds" vary only the
*sampling* noise, not the *training* initialization. The near-zero
cross-seed std (~0.0001) demonstrates that a *fixed* collapsed model is
deterministic — not that the METHOD collapses across training runs.

This script retrains one of {consistency, shortcut} at 64³ BraTS from a
specified training seed, reusing the shared VQ-GAN and the FM teacher
(both of which are shared across every method in the paper, per §3.3).
Everything else — architecture, epochs, LR schedule, EMA decay,
step-count curriculum — matches the original main-benchmark training
recipe verbatim.

Two additional training seeds per method (100 and 200) plus the original
(implicit-seed) checkpoint = n=3 training instances per method, which
turns the n-of-1-per-method comparison into a proper
mechanism-vs-implementation comparison.

Reuses this repo's own model code so nothing about the training procedure
diverges from what §3 describes.

Usage
-----
    # Train consistency from training seed 100 (~30-45 min on M1):
    python tier_c/a1_training_seeds/train_replicate_seed.py \
        --method consistency --train-seed 100

    # Train shortcut from training seed 100 (~1-2 hrs on M1):
    python tier_c/a1_training_seeds/train_replicate_seed.py \
        --method shortcut --train-seed 100

Output
------
    results/paper4/train_seed_replication/{method}_trainseed{S}/final.pt
    results/paper4/train_seed_replication/{method}_trainseed{S}/training_log.json

Runtime on M1 (approximate, from paper's original epoch counts):
    consistency (80 epochs):  ~30-45 min
    shortcut    (150 epochs): ~1-2 hours

The runbook run_a1_replication.sh drives all four (2 CD + 2 Shortcut)
seeds sequentially and takes ~5-8 hours on M1.
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

# Explicit training seeding (was missing from the original benchmark script;
# retro-fitting it here so training-seed replication is well-defined)
from utils.set_seed import set_seed, get_worker_init_fn  # noqa: E402

# Reuse paper's model + training code so this replication is faithful
from flow_matching_3d import (   # noqa: E402
    BrainDS, LatentGenerative3D, get_config,
    train_consistency, train_shortcut,
    freeze_p1, unfreeze,
)

warnings.filterwarnings("ignore")


# ============================================================
# Paths (BraTS 64^3 main benchmark that produced Table 3)
# ============================================================

BENCH_DIR = REPO_ROOT / "results/paper4/brats_benchmark_20260324_154926/100pct_180vol"
DATA_PATH = REPO_ROOT / "data/brats_conditional_64.pt"

VQGAN_PATH   = BENCH_DIR / "phase1_shared.pt"
TEACHER_PATH = BENCH_DIR / "fm/final.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["consistency", "shortcut"],
                    help="Which method to retrain from a new seed")
    ap.add_argument("--train-seed", type=int, required=True,
                    help="Training seed (100, 200 recommended for A1)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epoch count (default: 80 for CD, 150 for Shortcut)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=2,
                    help="Match original benchmark; conditional BraTS")
    ap.add_argument("--output-dir",
                    default="results/paper4/train_seed_replication")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else
                             "cpu")
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Training seed: {args.train_seed}")

    # ----- Explicit training seed (this is the whole point of A1) -----
    g_torch = set_seed(args.train_seed)
    worker_init = get_worker_init_fn(args.train_seed)

    # ----- Sanity-check inputs -----
    for path in (VQGAN_PATH, TEACHER_PATH, DATA_PATH):
        if not path.exists():
            sys.exit(f"ERROR: missing {path}")
    print(f"VQ-GAN:  {VQGAN_PATH}")
    print(f"FM teacher: {TEACHER_PATH}")
    print(f"Data:    {DATA_PATH}")

    # ----- Load BraTS conditional dataset -----
    d = torch.load(DATA_PATH, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols   = d.get("volumes", d.get("v"))
        labels = d.get("labels",  d.get("y"))
    else:
        vols, labels = d, None
    if vols is None:
        sys.exit(f"ERROR: could not find volumes in {DATA_PATH}")
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    print(f"Loaded {len(vols)} volumes, shape {tuple(vols.shape)}")

    # Same 180-volume training subset as the original benchmark
    # (perm(seed=42) then take first 180 — matches flow_matching_3d.py convention)
    g_perm = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(vols), generator=g_perm)
    train_idx = perm[:180]
    train_vols = vols[train_idx].float()
    train_labels = labels[train_idx] if labels is not None else None

    # ----- Build config -----
    cfg = get_config(quick=False, dataset="brats")
    cfg["num_classes"] = args.num_classes
    if args.epochs is not None:
        cfg["epochs_consistency"] = args.epochs
        cfg["epochs_shortcut"]    = args.epochs
    print(f"Epochs: consistency={cfg['epochs_consistency']}, shortcut={cfg['epochs_shortcut']}")

    # ----- Load VQ-GAN and FM teacher state -----
    p1_state = torch.load(VQGAN_PATH,   map_location=device, weights_only=True)
    fm_full  = torch.load(TEACHER_PATH, map_location=device, weights_only=True)
    fm_unet_state = {k.replace("unet.", ""): v for k, v in fm_full.items()
                     if k.startswith("unet.")}
    if not fm_unet_state:
        fm_unet_state = fm_full

    # ----- Build model -----
    model = LatentGenerative3D(cfg, method=args.method).to(device)
    model.enc.load_state_dict(p1_state["enc"])
    model.dec.load_state_dict(p1_state["dec"])
    model.vq.load_state_dict(p1_state["vq"])

    # ----- DataLoader (seeded generator and worker init) -----
    ds = BrainDS(train_vols, labels=train_labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=True,
                    generator=g_torch, worker_init_fn=worker_init)
    print(f"Training: {len(ds)} volumes, batch {args.batch_size}, "
          f"{len(dl)} iters/epoch")

    # ----- Output dir -----
    rdir = REPO_ROOT / args.output_dir / f"{args.method}_trainseed{args.train_seed}"
    rdir.mkdir(parents=True, exist_ok=True)

    # ----- Train -----
    t0 = time.time()
    if args.method == "consistency":
        losses = train_consistency(model, dl, cfg, device, rdir, fm_unet_state)
    else:  # shortcut
        losses = train_shortcut(model, dl, cfg, device, rdir)
    total = time.time() - t0
    print(f"\nTraining finished in {total:.0f}s = {total/3600:.2f}h")

    # ----- Save training log -----
    log_path = rdir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "method": args.method,
            "train_seed": args.train_seed,
            "epochs": (cfg["epochs_consistency"] if args.method == "consistency"
                       else cfg["epochs_shortcut"]),
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
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
