#!/usr/bin/env python3
"""
128³ E10a — Dose-response (diversity isolation experiment) at 128³ resolution.

Adapts `src/paper4/conditional_segmentation.py::run_e10a` to the 128³
conditional pipeline. Same protocol: same model (conditional Shortcut@50),
same step count, same total synthetic volume count (500), only the number
of distinct anatomies varies across conditions {10, 25, 100, 500}, plus
a real-only baseline.

The §5.5 finding at 64³ was that Dice saturates above ~25 unique anatomies
(Table 8). This script tests whether that threshold holds at higher
resolution.

Wall-clock budget (M-series, 128³ MONAI BasicUNet at batch=2):
    - Sample generation: ~30-60 min (500 vols × class 1 × Shortcut@50)
    - Per-condition training: ~6-9 hours per (condition × seed)
    - 4 conditions × 3 seeds + real-only × 3 seeds = 15 training runs ≈ 90-135 hours
    - Total ≈ 4-6 calendar days continuous

Reduce wall-clock by passing `--max-iters 1500` (still well-converged at 128³),
or `--n-seeds 2` if you need to ship faster.

Usage:
    cd fast-sampling-mode-collapse-3d/
    bash tier_c/item17_128cubed_conditional/run_e10a_128.sh
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

try:
    from monai.networks.nets import BasicUNet
    from monai.losses import DiceCELoss
except ImportError:
    sys.exit("ERROR: MONAI not installed. Run: pip install monai")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from models_shared import (  # noqa: E402
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D,
)

try:
    from utils.set_seed import set_seed
    HAS_SET_SEED = True
except Exception:
    HAS_SET_SEED = False


LATENT_SIZE = 16  # 128 / 8 = 16
N_TOTAL = 500     # total synthetic volumes per condition (paper protocol)
DIVERSITY_LEVELS = [500, 100, 25, 10]   # unique counts
TARGET_CLASS = 1  # generate class 1 (large tumor); matches the 64³ E10a protocol


# ============================================================
# Datasets
# ============================================================

class SegDataset(Dataset):
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
        y = self.s[idx].squeeze(0).long()
        if self.augment:
            for a in (1, 2, 3):
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[a])
                    y = torch.flip(y, dims=[a - 1])
        return {"x": x, "y": y}


class SyntheticSegDataset(Dataset):
    """Synthetic volumes with teacher-predicted pseudo-labels (computed lazily)."""

    def __init__(self, syn_volumes, teacher_model, device, batch_size=2, augment=False):
        self.v = syn_volumes if syn_volumes.dim() == 5 else syn_volumes.unsqueeze(1)
        self.device = device
        self.augment = augment

        # Precompute pseudo-labels with teacher
        print(f"    Generating pseudo-labels for {len(self.v)} synthetic volumes...")
        teacher_model.eval()
        labels = []
        n_tumor = 0
        with torch.no_grad():
            for i in range(0, len(self.v), batch_size):
                x = self.v[i:i + batch_size].float().to(device)
                logits = teacher_model(x)
                pred = logits.argmax(1)  # (B, D, H, W)
                labels.append(pred.cpu())
                n_tumor += int((pred.sum(dim=(1, 2, 3)) > 0).sum().item())
        self.s = torch.cat(labels, dim=0).unsqueeze(1)  # (N,1,D,H,W)
        self.tumor_detection_rate = n_tumor / max(1, len(self.v))

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        x = self.v[idx].float()
        y = self.s[idx].squeeze(0).long()
        if self.augment:
            for a in (1, 2, 3):
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[a])
                    y = torch.flip(y, dims=[a - 1])
        return {"x": x, "y": y}


# ============================================================
# Model factories
# ============================================================

def create_seg_model(device):
    model = BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        features=(32, 64, 128, 256, 32, 32),
    ).to(device)
    return model


def load_teacher_128(path, device):
    model = create_seg_model(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_conditional_shortcut_128(ckpt_dir, device, num_classes=2):
    p1 = torch.load(Path(ckpt_dir) / "phase1_shared.pt",
                    map_location=device, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(device); enc.load_state_dict(p1["enc"]); enc.eval()
    dec = Decoder3D(1, 8, 2).to(device); dec.load_state_dict(p1["dec"]); dec.eval()
    vq = VectorQuantizer(256, 8).to(device); vq.load_state_dict(p1["vq"]); vq.eval()

    state = torch.load(Path(ckpt_dir) / "shortcut" / "final.pt",
                       map_location=device, weights_only=True)
    nc = state.get("num_classes", num_classes)
    unet = DenoisingUNet3D(ch=8, num_classes=nc).to(device)
    unet.load_state_dict(state["unet"] if "unet" in state else state)
    unet.eval()
    for m in (enc, dec, vq, unet):
        for p in m.parameters():
            p.requires_grad = False
    return enc, dec, vq, unet, nc


# ============================================================
# Synthetic sampling at 128³
# ============================================================

def _has_d_kwarg(model):
    import inspect
    try:
        return "d" in inspect.signature(model.forward).parameters
    except Exception:
        return False


@torch.no_grad()
def sample_shortcut_conditional_128(unet, dec, n, dev, steps, class_label,
                                     batch_size=2, latent_size=LATENT_SIZE):
    out = []
    rem = n
    has_d = _has_d_kwarg(unet)
    while rem > 0:
        bs = min(batch_size, rem)
        y = torch.full((bs,), class_label, device=dev, dtype=torch.long)
        z = torch.randn(bs, 8, latent_size, latent_size, latent_size, device=dev)
        d_val = 1.0 / steps
        for s in range(steps):
            t = torch.full((bs,), s / steps, device=dev)
            d = torch.full((bs,), d_val, device=dev)
            if has_d:
                v = unet(z, t, d=d, class_label=y)
            else:
                v = unet(z, t, class_label=y)
            z = z + d_val * v
        vols = dec(z).clamp(0, 1).cpu()
        out.append(vols)
        rem -= bs
    return torch.cat(out, dim=0)[:n]


def pairwise_l1_diversity(vol_tensor, n_pairs=1000):
    idx1 = torch.randint(0, len(vol_tensor), (n_pairs,))
    idx2 = torch.randint(0, len(vol_tensor), (n_pairs,))
    return (vol_tensor[idx1] - vol_tensor[idx2]).abs().mean().item()


# ============================================================
# Training (iteration-based, matching the §3.6 protocol)
# ============================================================

def evaluate_dice(model, ds, device, batch_size=2):
    """Mean per-volume Dice across the test set."""
    val, _, _ = evaluate_dice_with_bootstrap(model, ds, device, batch_size, n_bootstrap=0)
    return val


def evaluate_dice_with_bootstrap(model, ds, device, batch_size=2, n_bootstrap=1000):
    """Returns (mean_dice, per_volume_dices, bootstrap_ci_dict).

    Per-volume Dice list is what enables Maier-Hein-style bootstrap CIs
    (arxiv 2307.10926). With 60 test volumes and 1000 bootstrap samples,
    the 95% CI is well-calibrated for segmentation per their paper.
    """
    model.eval()
    # Input nan_to_num to match the training-time scrub.
    dices = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            sub = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            x = torch.stack([b["x"] for b in sub]).to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            y = torch.stack([b["y"] for b in sub]).to(device)
            logits = model(x)
            p = logits.argmax(1)
            inter = (p * y).sum(dim=(1, 2, 3))
            u = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
            d = (2 * inter + 1e-6) / (u + 1e-6)
            dices.extend([float(x_) for x_ in d])
    arr = np.asarray(dices, dtype=float)
    mean_dice = float(arr.mean())
    if n_bootstrap > 0 and len(arr) >= 2:
        rng = np.random.default_rng(42)
        boot_means = np.array([
            arr[rng.integers(0, len(arr), size=len(arr))].mean()
            for _ in range(n_bootstrap)
        ])
        boot_means.sort()
        ci = {
            "lo": float(boot_means[int(n_bootstrap * 0.025)]),
            "hi": float(boot_means[int(n_bootstrap * 0.975)]),
            "n_bootstrap": int(n_bootstrap),
            "n_test_volumes": int(len(arr)),
        }
    else:
        ci = {"lo": mean_dice, "hi": mean_dice,
              "n_bootstrap": 0, "n_test_volumes": int(len(arr))}
    return mean_dice, dices, ci


def train_one_condition(train_ds, test_ds, device, max_iters, seed,
                         warmup_iters=100, peak_lr=3e-4, warmup_lr=3e-5):
    """Train one downstream segmenter for one condition × one seed.

    Numerical robustness (recipe v2, 2026-05-31, after MPS NaN incidents):

    Causes the original code missed:
      * LR 1e-3 was too high for 128³ DiceCELoss + MPS. MONAI maintainer
        rijobro's published guidance is to drop to 1e-5 and tune up.
        Reduced to 3e-4 (10× lower) as a safe middle ground.
      * smooth_nr=1.0 (loss numerator smoothing) actually HURTS stability
        per MONAI's own historical note — reverted to smooth_nr=0.0.
      * Input volumes from the VQ-GAN decoder can contain denormal /
        near-zero values; matt3o's accepted MONAI fix (SignalFillEmpty)
        casts NaN in input data to 0 before the forward pass.

    Safeguards retained from v1:
      * Loss-level NaN guard (skip batch, don't corrupt optimizer state).
      * Gradient-level NaN guard AFTER backward() (zero grads, skip step).
      * Gradient clipping at max_norm=1.0.
      * LR warmup over first `warmup_iters` iterations.
      * Hard abort only after 20 consecutive bad batches (real failure marker).
    """
    if HAS_SET_SEED:
        set_seed(seed)
    else:
        torch.manual_seed(seed); np.random.seed(seed)
    model = create_seg_model(device)
    # Recipe v2: smooth ONLY the denominator. Per MONAI source comment,
    # numerator smoothing was empirically less stable.
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                         smooth_nr=0.0, smooth_dr=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=warmup_lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, max_iters - warmup_iters))
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                        drop_last=True, generator=g)
    model.train()
    iter_idx = 0
    t0 = time.time()
    nan_encountered = False
    skipped_nan_grad_steps = 0
    consecutive_nan_grads = 0
    while iter_idx < max_iters:
        for batch in loader:
            if iter_idx >= max_iters:
                break

            # ----- LR warmup -----
            if iter_idx < warmup_iters:
                lr = warmup_lr + (peak_lr - warmup_lr) * (iter_idx / warmup_iters)
                for pg in opt.param_groups:
                    pg["lr"] = lr

            x = batch["x"].to(device)
            y = batch["y"].unsqueeze(1).to(device)
            # Input-level NaN/Inf scrub (matt3o's SignalFillEmpty pattern, inline).
            # Catches denormal pixels in VQ-GAN-decoded synthetic volumes.
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                skipped_nan_grad_steps += 1
                consecutive_nan_grads += 1
                if consecutive_nan_grads > 20:
                    nan_encountered = True
                    print(f"      iter {iter_idx:5d}/{max_iters} too many consecutive "
                          f"NaN losses ({consecutive_nan_grads}) — aborting this seed")
                    break
                continue
            opt.zero_grad(); loss.backward()
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                opt.zero_grad()
                skipped_nan_grad_steps += 1
                consecutive_nan_grads += 1
                if consecutive_nan_grads > 20:
                    nan_encountered = True
                    print(f"      iter {iter_idx:5d}/{max_iters} too many consecutive "
                          f"NaN gradients ({consecutive_nan_grads}) — aborting this seed")
                    break
                continue
            consecutive_nan_grads = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if iter_idx >= warmup_iters:
                sch.step()
            iter_idx += 1
            if iter_idx % 500 == 0 or iter_idx == 1:
                print(f"      iter {iter_idx:5d}/{max_iters} loss={loss.item():.4f} "
                      f"(lr={opt.param_groups[0]['lr']:.2e}, "
                      f"skipped_nan={skipped_nan_grad_steps}, "
                      f"{(time.time() - t0):.0f}s)")
        if nan_encountered:
            break
    if nan_encountered:
        return model, {"seed": seed, "val_dice": float("nan"),
                       "wall_s": time.time() - t0,
                       "skipped_nan_grad_steps": skipped_nan_grad_steps,
                       "status": "nan_aborted"}
    # Final per-volume Dice + bootstrap CI (Maier-Hein et al. 2025 protocol).
    val_dice, per_vol_dices, bootstrap_ci = evaluate_dice_with_bootstrap(
        model, test_ds, device, batch_size=2, n_bootstrap=1000)
    return model, {"seed": seed, "val_dice": val_dice,
                   "per_volume_dices": per_vol_dices,
                   "bootstrap_ci95": bootstrap_ci,
                   "wall_s": time.time() - t0,
                   "skipped_nan_grad_steps": skipped_nan_grad_steps,
                   "status": "ok"}


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/brats_conditional_128.pt")
    p.add_argument("--ckpt-dir", default="results/paper4/brats_128cubed_conditional",
                   help="Directory containing phase1_shared.pt, shortcut/final.pt.")
    p.add_argument("--teacher-path",
                   default="results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt")
    p.add_argument("--output-dir",
                   default="results/paper4/brats_128cubed_conditional/e10a_128")
    p.add_argument("--max-iters", type=int, default=2000,
                   help="Downstream training iterations per condition×seed. 2000 matches "
                        "the 64³ protocol exactly.")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of seeds. v2: bumped from 3 to 5 to address the peer-review "
                        "punch-list item 'Re-run E10b at 5 seeds'.")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--steps", type=int, default=50,
                   help="NFE for synthetic-volume generation (default matches Shortcut@50).")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    # ----- Load conditional dataset -----
    raw = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict) or "volumes" not in raw or "segs" not in raw:
        sys.exit("ERROR: conditional dataset missing 'volumes' or 'segs'. "
                 "Run prepare_segs_128.py to add 'segs'.")
    vols = raw["volumes"]
    segs = raw["segs"]
    splits = raw.get("split_info") or {}
    if not splits or "train_idx" not in splits:
        n = len(vols); rng = np.random.default_rng(42)
        idx = rng.permutation(n).tolist()
        n_tr = int(0.8 * n); n_va = int(0.1 * n)
        splits = {"train_idx": idx[:n_tr], "val_idx": idx[n_tr:n_tr + n_va],
                  "test_idx": idx[n_tr + n_va:]}

    train_idx = splits["train_idx"]
    test_idx  = splits["test_idx"]
    # 10% real fraction matches the §5.5 protocol exactly
    n_real = max(4, int(len(train_idx) * 0.10))
    real_vols = vols[train_idx[:n_real]]
    real_segs = segs[train_idx[:n_real]]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]
    print(f"Real subset (10% of train_idx): {n_real} volumes")
    print(f"Test: {len(test_vols)} volumes")

    # ----- Step 1: Generate 500 conditional class-1 synthetic volumes at 128³ -----
    print(f"\nStep 1: Generating {N_TOTAL} class-{TARGET_CLASS} samples from "
          f"Shortcut@{args.steps} at 128³ …")
    enc, dec, vq, shortcut_unet, nc = load_conditional_shortcut_128(
        args.ckpt_dir, dev, num_classes=args.num_classes)

    t_gen = time.time()
    syn_vols = sample_shortcut_conditional_128(
        shortcut_unet, dec, N_TOTAL, dev, args.steps,
        class_label=TARGET_CLASS, batch_size=2)
    print(f"  Generated {syn_vols.shape[0]} volumes shape {tuple(syn_vols.shape)} "
          f"({(time.time() - t_gen):.0f}s)")
    # Latent features for centrality ordering (encode the decoded vols back through enc/vq)
    print(f"  Computing latent features for centrality ordering …")
    feats = []
    with torch.no_grad():
        for i in range(0, len(syn_vols), 4):
            x = syn_vols[i:i + 4].to(dev)
            z, _, _ = vq(enc(x))
            feats.append(z.reshape(z.shape[0], -1).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    global_mean = feats.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(feats - global_mean, axis=1)
    sorted_by_centrality = np.argsort(dists)
    # Free the generator (we still need the teacher next)
    del enc, vq, shortcut_unet
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()

    # ----- Step 2: Build the unique-count conditions -----
    print(f"\nStep 2: Building diversity conditions {DIVERSITY_LEVELS} …")
    conditions = {}
    rng = np.random.RandomState(42)
    for n_unique in DIVERSITY_LEVELS:
        label = f"unique_{n_unique}"
        if n_unique >= N_TOTAL:
            cond_vols = syn_vols
        else:
            # Most central n_unique, then replicate to N_TOTAL with replacement
            source = sorted_by_centrality[:n_unique]
            chosen = rng.choice(source, N_TOTAL, replace=True)
            cond_vols = syn_vols[chosen]
        div = pairwise_l1_diversity(cond_vols)
        conditions[label] = cond_vols
        print(f"  {label}: n_total={N_TOTAL}, L1 diversity={div:.4f}")

    # ----- Step 3: Load teacher and run conditions -----
    teacher = load_teacher_128(args.teacher_path, dev)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    test_ds = SegDataset(test_vols, test_segs, augment=False)
    seeds = list(range(42, 42 + args.n_seeds))

    # ----- Resume: load any partial results so a re-run continues from where
    # it left off. This protects against losing days of compute to a crash.
    partial_path = out_dir / "e10a_results_128_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        print(f"\nResuming from {partial_path} — conditions already complete: "
              f"{[k for k in results if k.startswith('unique_') or k == 'real_only']}")
    else:
        results = {}

    import math

    def _is_valid_dice(d):
        """A seed's result is valid if it's a positive finite Dice score."""
        return (d is not None
                and isinstance(d, (int, float))
                and not math.isnan(float(d))
                and float(d) > 0.0)

    for cond_name, cond_vols in conditions.items():
        n_unique = int(cond_name.split("_")[1])

        # Initialise or pad the condition's per-seed arrays to len(seeds).
        cond = results.setdefault(cond_name, {
            "n_unique": n_unique,
            "n_total":  N_TOTAL,
            "dices":              [None] * len(seeds),
            "wall_s_per_seed":    [None] * len(seeds),
            "seed_statuses":      ["pending"] * len(seeds),
            "per_volume_dices":   [None] * len(seeds),
            "bootstrap_ci95":     [None] * len(seeds),
            "diversity_l1":   pairwise_l1_diversity(cond_vols),
        })
        # Backfill structure if loading an older partial JSON.
        for key, default in (("dices", [None] * len(seeds)),
                              ("wall_s_per_seed", [None] * len(seeds)),
                              ("seed_statuses", ["pending"] * len(seeds)),
                              ("per_volume_dices", [None] * len(seeds)),
                              ("bootstrap_ci95", [None] * len(seeds))):
            arr = cond.get(key) or []
            while len(arr) < len(seeds):
                arr.append(default[0])
            cond[key] = arr[:len(seeds)]

        # Decide which seeds need (re-)running.
        seed_status = [
            "skip" if _is_valid_dice(cond["dices"][i]) else "run"
            for i in range(len(seeds))
        ]
        if all(s == "skip" for s in seed_status):
            print(f"\n--- {cond_name.upper()} (all seeds valid, skipping) ---")
            continue

        print(f"\n--- {cond_name.upper()} ---")
        n_run = sum(1 for s in seed_status if s == "run")
        print(f"  Need to run {n_run}/{len(seeds)} seeds "
              f"(others already valid: {[seeds[i] for i, s in enumerate(seed_status) if s == 'skip']})")

        # Build the synthetic dataset only if at least one seed must run.
        syn_ds = SyntheticSegDataset(cond_vols, teacher, dev, batch_size=2, augment=True)
        cond["tumor_detection_rate"] = syn_ds.tumor_detection_rate
        print(f"  Tumor detection rate (teacher on synthetic): "
              f"{syn_ds.tumor_detection_rate:.1%}")
        real_ds = SegDataset(real_vols, real_segs, augment=True)
        combined_ds = ConcatDataset([real_ds, syn_ds])

        for i, seed in enumerate(seeds):
            if seed_status[i] == "skip":
                print(f"  Seed {seed} (already complete, "
                      f"Dice {cond['dices'][i]:.4f}, skipping)")
                continue
            print(f"  Seed {seed}:")
            model, info = train_one_condition(combined_ds, test_ds, dev,
                                              args.max_iters, seed)
            cond["dices"][i] = (None if not _is_valid_dice(info["val_dice"])
                                else float(info["val_dice"]))
            cond["wall_s_per_seed"][i] = float(info["wall_s"])
            cond["seed_statuses"][i] = info.get("status", "ok")
            cond["per_volume_dices"][i] = info.get("per_volume_dices")
            cond["bootstrap_ci95"][i] = info.get("bootstrap_ci95")
            ci = info.get("bootstrap_ci95")
            ci_str = (f"  CI95=[{ci['lo']:.4f}, {ci['hi']:.4f}]"
                      if ci and "lo" in ci else "")
            print(f"    → Dice {info['val_dice']:.4f}{ci_str}  "
                  f"({info['wall_s']:.0f}s = {info['wall_s'] / 3600:.2f}h) "
                  f"[{info.get('status', 'ok')}]")
            del model
            if dev == "mps":
                torch.mps.empty_cache()
            elif dev == "cuda":
                torch.cuda.empty_cache()
            # Per-seed checkpoint — saves after EVERY seed, not just every condition.
            valid = [d for d in cond["dices"] if _is_valid_dice(d)]
            if valid:
                cond["mean"] = float(np.mean(valid))
                cond["std"]  = float(np.std(valid))
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    Checkpoint saved to {partial_path}")

        valid = [d for d in cond["dices"] if _is_valid_dice(d)]
        if valid:
            cond["mean"] = float(np.mean(valid))
            cond["std"]  = float(np.std(valid))
            print(f"  → {cond_name}: {cond['mean']:.4f} ± {cond['std']:.4f} "
                  f"(over {len(valid)} valid seeds)")
        else:
            print(f"  → {cond_name}: NO VALID SEEDS — re-run with different seeds.")

    # ----- Step 4: real-only baseline (per-seed resume) -----
    real_cond = results.setdefault("real_only", {
        "n_unique": 0, "n_total": 0,
        "dices":              [None] * len(seeds),
        "wall_s_per_seed":    [None] * len(seeds),
        "seed_statuses":      ["pending"] * len(seeds),
        "per_volume_dices":   [None] * len(seeds),
        "bootstrap_ci95":     [None] * len(seeds),
    })
    for key, default in (("dices", [None] * len(seeds)),
                          ("wall_s_per_seed", [None] * len(seeds)),
                          ("seed_statuses", ["pending"] * len(seeds)),
                          ("per_volume_dices", [None] * len(seeds)),
                          ("bootstrap_ci95", [None] * len(seeds))):
        arr = real_cond.get(key) or []
        while len(arr) < len(seeds):
            arr.append(default[0])
        real_cond[key] = arr[:len(seeds)]

    seed_status = [
        "skip" if _is_valid_dice(real_cond["dices"][i]) else "run"
        for i in range(len(seeds))
    ]
    if all(s == "skip" for s in seed_status):
        print(f"\n--- REAL ONLY (all seeds valid, skipping) ---")
    else:
        print(f"\n--- REAL ONLY (no synthetic) ---")
        real_only_ds = SegDataset(real_vols, real_segs, augment=True)
        for i, seed in enumerate(seeds):
            if seed_status[i] == "skip":
                print(f"  Seed {seed} (already complete, "
                      f"Dice {real_cond['dices'][i]:.4f}, skipping)")
                continue
            print(f"  Seed {seed}:")
            model, info = train_one_condition(real_only_ds, test_ds, dev,
                                              args.max_iters, seed)
            real_cond["dices"][i] = (None if not _is_valid_dice(info["val_dice"])
                                      else float(info["val_dice"]))
            real_cond["wall_s_per_seed"][i] = float(info["wall_s"])
            real_cond["seed_statuses"][i] = info.get("status", "ok")
            real_cond["per_volume_dices"][i] = info.get("per_volume_dices")
            real_cond["bootstrap_ci95"][i] = info.get("bootstrap_ci95")
            ci = info.get("bootstrap_ci95")
            ci_str = (f"  CI95=[{ci['lo']:.4f}, {ci['hi']:.4f}]"
                      if ci and "lo" in ci else "")
            print(f"    → Dice {info['val_dice']:.4f}{ci_str}  "
                  f"({info['wall_s']:.0f}s = {info['wall_s'] / 3600:.2f}h) "
                  f"[{info.get('status', 'ok')}]")
            del model
            if dev == "mps":
                torch.mps.empty_cache()
            elif dev == "cuda":
                torch.cuda.empty_cache()
            valid = [d for d in real_cond["dices"] if _is_valid_dice(d)]
            if valid:
                real_cond["mean"] = float(np.mean(valid))
                real_cond["std"]  = float(np.std(valid))
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    Checkpoint saved to {partial_path}")

    real_dices = [d for d in real_cond["dices"] if _is_valid_dice(d)]
    real_walls = [w for w, d in zip(real_cond["wall_s_per_seed"], real_cond["dices"])
                  if _is_valid_dice(d)]
    if real_dices:
        real_cond["mean"] = float(np.mean(real_dices))
        real_cond["std"]  = float(np.std(real_dices))
        print(f"  → real_only: {real_cond['mean']:.4f} ± {real_cond['std']:.4f} "
              f"(over {len(real_dices)} valid seeds)")

    # ----- Step 5: paired t-tests (matching v7.2 §5.5 protocol) -----
    # Only run paired tests across seeds that are valid in BOTH conditions.
    from scipy import stats
    stat_tests = []

    def _valid_pairs(a_dices, b_dices):
        out_a, out_b = [], []
        for da, db in zip(a_dices, b_dices):
            if _is_valid_dice(da) and _is_valid_dice(db):
                out_a.append(float(da))
                out_b.append(float(db))
        return out_a, out_b

    sorted_conds = sorted(
        [(k, v) for k, v in results.items()
         if k != "real_only" and "dices" in v],
        key=lambda kv: kv[1]["n_unique"]
    )
    for i in range(len(sorted_conds) - 1):
        lo, hi = sorted_conds[i], sorted_conds[i + 1]
        a, b = _valid_pairs(lo[1]["dices"], hi[1]["dices"])
        if len(a) >= 2:
            t, p = stats.ttest_rel(a, b)
            stat_tests.append({
                "comparison": f"{hi[0]} vs {lo[0]}",
                "test": "paired t-test on per-seed Dice (scipy.stats.ttest_rel)",
                "n_paired_seeds": len(a),
                "t": float(t), "p_value": float(p),
                "dice_diff": float(np.mean(b)) - float(np.mean(a)),
            })
    if len(sorted_conds) >= 2:
        lo, hi = sorted_conds[0], sorted_conds[-1]
        a, b = _valid_pairs(lo[1]["dices"], hi[1]["dices"])
        if len(a) >= 2:
            t, p = stats.ttest_rel(a, b)
            stat_tests.append({
                "comparison": f"{hi[0]} vs {lo[0]} (extremes)",
                "test": "paired t-test on per-seed Dice (scipy.stats.ttest_rel)",
                "n_paired_seeds": len(a),
                "t": float(t), "p_value": float(p),
                "dice_diff": float(np.mean(b)) - float(np.mean(a)),
            })
    real_arr = results["real_only"]["dices"]
    for name, cond in sorted_conds:
        a, b = _valid_pairs(real_arr, cond["dices"])
        if len(a) >= 2:
            t, p = stats.ttest_rel(a, b)
            stat_tests.append({
                "comparison": f"{name} vs real_only",
                "test": "paired t-test on per-seed Dice (scipy.stats.ttest_rel)",
                "n_paired_seeds": len(a),
                "t": float(t), "p_value": float(p),
                "dice_diff": float(np.mean(b)) - float(np.mean(a)),
            })

    results["statistical_tests_paired"] = stat_tests
    results["design"] = {
        "resolution": 128, "n_total": N_TOTAL,
        "diversity_levels": DIVERSITY_LEVELS,
        "n_seeds": args.n_seeds, "max_iters": args.max_iters,
        "target_class": TARGET_CLASS, "steps": args.steps,
        "teacher_path": args.teacher_path,
    }

    out_path = out_dir / "e10a_results_128.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print(f"  128³ E10a dose-response summary (max_iters={args.max_iters}, "
          f"n_seeds={args.n_seeds})")
    print("=" * 70)
    rows = sorted([(k, v) for k, v in results.items()
                   if k in ("real_only",) or k.startswith("unique_")],
                  key=lambda kv: (kv[1].get("n_unique", 0)))
    print(f"{'Condition':>15} {'n_unique':>10} {'Dice':>8} {'± std':>10}")
    for name, r in rows:
        print(f"{name:>15} {r['n_unique']:>10} {r['mean']:>8.4f} "
              f"{r['std']:>10.4f}")

    print("\nPaired t-tests (uncorrected):")
    for st in stat_tests:
        sig = "*" if st["p_value"] < 0.05 else " "
        print(f"  {sig} {st['comparison']:<40} t={st['t']:>7.3f} "
              f"p={st['p_value']:.4g}  ΔDice={st['dice_diff']:+.4f}")


if __name__ == "__main__":
    main()
