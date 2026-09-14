#!/usr/bin/env python3
"""
Train a 128³ teacher segmenter for the 128³ E10a downstream pipeline.

Mirrors the 64³ teacher trained by `src/paper4/conditional_segmentation.py::run_e7`
but operates on 128³ real BraTS volumes. The teacher provides pseudo-labels
for synthetic 128³ volumes used in E10a / E10b at 128³.

Architecture: MONAI BasicUNet (spatial_dims=3, features=(32,64,128,256,32,32)),
identical to the 64³ teacher. BasicUNet is shape-agnostic; no architectural
changes are needed for 128³ inputs.

Training:
  * 5000 iterations on real train+val volumes (matches the 64³ teacher protocol)
  * DiceCELoss, AdamW lr=1e-3 wd=1e-5, CosineAnnealingLR
  * 5 seeds (0,1,2,3,4 → seeds 42–46), best-of-5 saved as teacher_best.pt

Usage:
    cd fast-sampling-mode-collapse-3d/
    bash tier_c/item17_128cubed_conditional/run_teacher_128.sh

Estimated wall-clock on Apple M-series: ~6-12 hours.

Outputs:
    results/paper4/brats_128cubed_conditional/e7_teacher_128/
        teacher_best.pt
        teacher_seedN.pt  (for N in 0..4)
        e7_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from monai.networks.nets import BasicUNet
    from monai.losses import DiceCELoss
except ImportError:
    sys.exit("ERROR: MONAI not installed. Run: pip install monai")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# v7.2 deterministic seeding helper
try:
    from utils.set_seed import set_seed
    HAS_SET_SEED = True
except Exception:
    HAS_SET_SEED = False


# ============================================================
# Dataset
# ============================================================

class SegDataset(Dataset):
    """Volume + binary tumor mask, optional flip augmentation."""

    def __init__(self, volumes, segs, augment=False):
        self.v = volumes if volumes.dim() == 5 else volumes.unsqueeze(1)
        self.s = (segs > 0).long()
        if self.s.dim() == 4:
            self.s = self.s.unsqueeze(1)
        self.augment = augment

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        x = self.v[idx].float()
        y = self.s[idx].squeeze(0).long()  # (D, H, W)
        if self.augment:
            for axis in (1, 2, 3):  # spatial axes on the volume (1 channel)
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[axis])
                    y = torch.flip(y, dims=[axis - 1])
        return {"x": x, "y": y}


def create_seg_model(device):
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features=(32, 64, 128, 256, 32, 32),
    ).to(device)
    return model


# ============================================================
# Training
# ============================================================

def evaluate_dice(model, ds, device, batch_size=2):
    model.eval()
    dices = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            x = torch.stack([b["x"] for b in batch]).to(device)
            y = torch.stack([b["y"] for b in batch]).to(device)
            logits = model(x)
            pred = logits.argmax(1)
            intersection = (pred * y).sum(dim=(1, 2, 3))
            union = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dices.extend([float(d) for d in dice])
    return float(np.mean(dices))


def train_one_seed(train_ds, test_ds, device, max_iters, seed):
    if HAS_SET_SEED:
        set_seed(seed)
    else:
        torch.manual_seed(seed); np.random.seed(seed)

    model = create_seg_model(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_iters)

    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                        drop_last=True, generator=g)

    losses, dice_history = [], []
    iter_idx = 0
    model.train()
    while iter_idx < max_iters:
        for batch in loader:
            if iter_idx >= max_iters:
                break
            x = batch["x"].to(device)
            y = batch["y"].unsqueeze(1).to(device)  # (B,1,D,H,W)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            losses.append(loss.item())
            iter_idx += 1
            if iter_idx % 200 == 0 or iter_idx == 1:
                # Quick dice on a few test samples for progress
                model.eval()
                with torch.no_grad():
                    n_eval = min(8, len(test_ds))
                    sub = [test_ds[j] for j in range(n_eval)]
                    xe = torch.stack([s["x"] for s in sub]).to(device)
                    ye = torch.stack([s["y"] for s in sub]).to(device)
                    p = model(xe).argmax(1)
                    inter = (p * ye).sum(dim=(1, 2, 3))
                    u = p.sum(dim=(1, 2, 3)) + ye.sum(dim=(1, 2, 3))
                    d = ((2 * inter + 1e-6) / (u + 1e-6)).mean().item()
                model.train()
                dice_history.append({"iter": iter_idx, "loss": loss.item(),
                                     "test_dice_n8": d})
                print(f"    iter {iter_idx:5d}/{max_iters}  loss={loss.item():.4f}  "
                      f"test_dice@8={d:.4f}")

    # Final full test dice
    final_dice = evaluate_dice(model, test_ds, device, batch_size=2)
    return model, {"seed": seed, "max_iters": max_iters,
                   "final_test_dice": final_dice,
                   "loss_history_last10": losses[-10:],
                   "dice_checkpoints": dice_history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/brats_conditional_128.pt",
                   help="Augmented conditional 128 dataset; must contain "
                        "'volumes' (N,1,128,128,128) and 'segs' (N,1,128,128,128).")
    p.add_argument("--output-dir",
                   default="results/paper4/brats_128cubed_conditional/e7_teacher_128")
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    raw = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict) or "volumes" not in raw:
        sys.exit("ERROR: dataset is not the conditional dict.")
    if "segs" not in raw:
        sys.exit("ERROR: conditional dataset is missing 'segs'. "
                 "Run prepare_segs_128.py first.")
    vols = raw["volumes"]
    segs = raw["segs"]
    splits = raw.get("split_info") or {}

    # If splits weren't carried over from 64³, just use 80/10/10 deterministic.
    if not splits or "train_idx" not in splits:
        n = len(vols)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n).tolist()
        n_tr = int(0.8 * n); n_va = int(0.1 * n)
        splits = {
            "train_idx": idx[:n_tr],
            "val_idx":   idx[n_tr:n_tr + n_va],
            "test_idx":  idx[n_tr + n_va:],
        }
        print(f"  No splits in dataset; using deterministic 80/10/10 = "
              f"{len(splits['train_idx'])}/{len(splits['val_idx'])}/{len(splits['test_idx'])}")

    train_idx = splits["train_idx"] + splits["val_idx"]
    test_idx = splits["test_idx"]
    train_vols = vols[train_idx]
    train_segs = segs[train_idx]
    test_vols  = vols[test_idx]
    test_segs  = segs[test_idx]

    print(f"Train (train+val): {len(train_vols)} volumes")
    print(f"Test:              {len(test_vols)} volumes")

    train_ds = SegDataset(train_vols, train_segs, augment=True)
    test_ds  = SegDataset(test_vols,  test_segs,  augment=False)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    best_dice = -1.0
    for s in range(args.n_seeds):
        seed = 42 + s
        print(f"\n--- Seed {s} (seed={seed}) ---")
        t0 = time.time()
        model, info = train_one_seed(train_ds, test_ds, dev, args.max_iters, seed)
        info["wall_s"] = time.time() - t0
        print(f"  Final test Dice: {info['final_test_dice']:.4f}  "
              f"({info['wall_s']:.0f}s)")
        torch.save(model.state_dict(), outdir / f"teacher_seed{s}.pt")
        results.append(info)
        if info["final_test_dice"] > best_dice:
            best_dice = info["final_test_dice"]
            torch.save(model.state_dict(), outdir / "teacher_best.pt")
            print(f"  → New best teacher (Dice {best_dice:.4f})")

    summary = {
        "all_seeds": results,
        "mean_dice": float(np.mean([r["final_test_dice"] for r in results])),
        "std_dice":  float(np.std([r["final_test_dice"] for r in results])),
        "best_dice": float(best_dice),
        "resolution": 128,
    }
    with open(outdir / "e7_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== 128³ Teacher complete ===")
    print(f"  Best Dice: {summary['best_dice']:.4f}")
    print(f"  Mean ± std over {len(results)} seeds: "
          f"{summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"  Saved teacher_best.pt and per-seed checkpoints to {outdir}")


if __name__ == "__main__":
    main()
