#!/usr/bin/env python3
"""
128³ E10b — Practical augmentation comparison at 128-cubed resolution.

Adapts src/paper4/focused_e10b.py to the 128³ conditional pipeline. Same
experimental design as the 64³ E10b (Table 9 in v8): three real-data
fractions × four augmentation strategies × three seeds, totaling 36 runs.

Conditions:
  (a) real_only         — baseline (real volumes only)
  (b) real + classical  — geometric augmentation only (flips, rotations)
  (c) real + shortcut50 — real + 500 Shortcut@50 conditional synthetic
  (d) real + consist4   — real + 500 Consistency@4 conditional synthetic

Real-data fractions: 5%, 10%, 25% of training set.
Seeds per condition × fraction: 3 (42, 43, 44), matching the 64³ E10b protocol
                                so Table 9 at 128³ is directly comparable to
                                Table 9 at 64³ (no methodology divergence to
                                defend in §5.6 / §5.8.4).

Numerical recipe v2 (from Appendix A.7, debugged on E10a 128³):
  * LR 3e-4 with 100-iter warmup from 3e-5
  * DiceCELoss(smooth_nr=0.0, smooth_dr=1.0)
  * Input nan_to_num + clamp before each forward
  * Gradient-level NaN guard after backward
  * Hard abort only after 20 consecutive bad batches
  * 2000 iterations per seed × condition × fraction
  * Same conditional Shortcut + Consistency models trained in §5.8.2
  * Same 128³ teacher segmenter (Seed 0)

Wall-clock at ~7 hours per run × 36 runs = ~10.5 days continuous.
Per-seed-per-condition checkpointing protects against any pause/crash.

Usage:
    cd fast-sampling-mode-collapse-3d/
    bash tier_c/item17_128cubed_conditional/run_e10b_128.sh
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
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


LATENT_SIZE = 16          # 128 / 8
N_SYNTH = 500             # synthetic volumes per method (matches 64³ E10b protocol)
TARGET_CLASS = 1          # generate class 1 (large tumor)
FRACTIONS = [0.05, 0.10, 0.25]
SEEDS = [42, 43, 44]
CONDITIONS = ["real_only", "classical_aug", "shortcut_50", "consistency_4"]


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


class ClassicalAugDataset(Dataset):
    """Real volumes with stronger geometric augmentation (flips + rotations).

    Classical augmentation here means: same volumes, more diverse views per
    epoch. We use random axis-flip in each spatial dim plus a single random
    90-degree rotation per draw. This matches the 64³ E10b protocol; for an
    apples-to-apples comparison we apply the same scheme at 128³.
    """

    def __init__(self, volumes, segs):
        self.v = volumes if volumes.dim() == 5 else volumes.unsqueeze(1)
        self.s = (segs > 0).long()
        if self.s.dim() == 4:
            self.s = self.s.unsqueeze(1)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        x = self.v[idx].float()
        y = self.s[idx].squeeze(0).long()
        # Random flips in every spatial axis
        for a in (1, 2, 3):
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[a])
                y = torch.flip(y, dims=[a - 1])
        # Random 90-degree rotation in a random spatial plane
        if torch.rand(1).item() < 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            planes = [(1, 2), (1, 3), (2, 3)]
            p = planes[int(torch.randint(0, 3, (1,)).item())]
            x = torch.rot90(x, k=k, dims=p)
            # y plane indices shift by -1 because y has no channel dim
            y = torch.rot90(y, k=k, dims=(p[0] - 1, p[1] - 1))
        return {"x": x, "y": y}


class SyntheticSegDataset(Dataset):
    """Synthetic volumes with teacher-predicted pseudo-labels."""

    def __init__(self, syn_volumes, teacher_model, device, batch_size=2, augment=False):
        self.v = syn_volumes if syn_volumes.dim() == 5 else syn_volumes.unsqueeze(1)
        self.device = device
        self.augment = augment

        print(f"    Pseudo-labelling {len(self.v)} synthetic volumes...")
        teacher_model.eval()
        labels = []
        n_tumor = 0
        with torch.no_grad():
            for i in range(0, len(self.v), batch_size):
                x = self.v[i:i + batch_size].float().to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                logits = teacher_model(x)
                pred = logits.argmax(1)
                labels.append(pred.cpu())
                n_tumor += int((pred.sum(dim=(1, 2, 3)) > 0).sum().item())
        self.s = torch.cat(labels, dim=0).unsqueeze(1)
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
    return BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        features=(32, 64, 128, 256, 32, 32),
    ).to(device)


def load_teacher_128(path, device):
    model = create_seg_model(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_cond_pipeline(ckpt_dir, method, device, num_classes=2):
    """Load VQ-GAN decoder + the requested conditional generative U-Net.

    method ∈ {"shortcut", "consistency"}
    """
    p1 = torch.load(Path(ckpt_dir) / "phase1_shared.pt",
                    map_location=device, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(device); enc.load_state_dict(p1["enc"]); enc.eval()
    dec = Decoder3D(1, 8, 2).to(device); dec.load_state_dict(p1["dec"]); dec.eval()
    vq = VectorQuantizer(256, 8).to(device); vq.load_state_dict(p1["vq"]); vq.eval()

    state = torch.load(Path(ckpt_dir) / method / "final.pt",
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
# Sampling
# ============================================================

def _has_d_kwarg(model):
    import inspect
    try:
        return "d" in inspect.signature(model.forward).parameters
    except Exception:
        return False


@torch.no_grad()
def sample_shortcut_cond(unet, dec, n, dev, steps, class_label,
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


@torch.no_grad()
def sample_consistency_cond(unet, dec, n, dev, steps, class_label,
                             batch_size=2, latent_size=LATENT_SIZE):
    out = []
    rem = n
    while rem > 0:
        bs = min(batch_size, rem)
        y = torch.full((bs,), class_label, device=dev, dtype=torch.long)
        z = torch.randn(bs, 8, latent_size, latent_size, latent_size, device=dev)
        t = torch.zeros(bs, device=dev)
        v = unet(z, t, class_label=y)
        x_hat = z + (1.0 - t[:, None, None, None, None]) * v
        for s in range(1, steps):
            t_s = torch.full((bs,), s / max(steps, 1), device=dev)
            noise = torch.randn_like(x_hat)
            te = t_s[:, None, None, None, None]
            x_s = (1 - te) * noise + te * x_hat
            v = unet(x_s, t_s, class_label=y)
            x_hat = x_s + (1 - te) * v
        vols = dec(x_hat).clamp(0, 1).cpu()
        out.append(vols)
        rem -= bs
    return torch.cat(out, dim=0)[:n]


# ============================================================
# Training (recipe v2 — same as E10a 128³)
# ============================================================

def evaluate_dice_with_bootstrap(model, ds, device, batch_size=2, n_bootstrap=1000):
    model.eval()
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
        boot_means = np.array([arr[rng.integers(0, len(arr), size=len(arr))].mean()
                                for _ in range(n_bootstrap)])
        boot_means.sort()
        ci = {"lo": float(boot_means[int(n_bootstrap * 0.025)]),
              "hi": float(boot_means[int(n_bootstrap * 0.975)]),
              "n_bootstrap": int(n_bootstrap),
              "n_test_volumes": int(len(arr))}
    else:
        ci = {"lo": mean_dice, "hi": mean_dice, "n_bootstrap": 0,
              "n_test_volumes": int(len(arr))}
    return mean_dice, dices, ci


def train_one_run(train_ds, test_ds, device, max_iters, seed,
                   warmup_iters=100, peak_lr=3e-4, warmup_lr=3e-5):
    """Train one downstream segmenter for one (fraction, condition, seed)."""
    if HAS_SET_SEED:
        set_seed(seed)
    else:
        torch.manual_seed(seed); np.random.seed(seed)
    model = create_seg_model(device)
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
    skipped_nan = 0
    consec_nan = 0
    while iter_idx < max_iters:
        for batch in loader:
            if iter_idx >= max_iters:
                break
            if iter_idx < warmup_iters:
                lr = warmup_lr + (peak_lr - warmup_lr) * (iter_idx / warmup_iters)
                for pg in opt.param_groups:
                    pg["lr"] = lr
            x = batch["x"].to(device)
            y = batch["y"].unsqueeze(1).to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                skipped_nan += 1; consec_nan += 1
                if consec_nan > 20:
                    nan_encountered = True
                    print(f"      iter {iter_idx:5d}/{max_iters} too many consecutive NaN — aborting seed")
                    break
                continue
            opt.zero_grad(); loss.backward()
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True; break
            if bad_grad:
                opt.zero_grad()
                skipped_nan += 1; consec_nan += 1
                if consec_nan > 20:
                    nan_encountered = True
                    print(f"      iter {iter_idx:5d}/{max_iters} too many consecutive NaN gradients — aborting seed")
                    break
                continue
            consec_nan = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if iter_idx >= warmup_iters:
                sch.step()
            iter_idx += 1
            if iter_idx % 500 == 0 or iter_idx == 1:
                print(f"      iter {iter_idx:5d}/{max_iters} loss={loss.item():.4f} "
                      f"(lr={opt.param_groups[0]['lr']:.2e}, skipped_nan={skipped_nan}, "
                      f"{(time.time() - t0):.0f}s)")
        if nan_encountered:
            break
    if nan_encountered:
        return model, {"seed": seed, "val_dice": float("nan"),
                       "wall_s": time.time() - t0,
                       "skipped_nan": skipped_nan, "status": "nan_aborted"}
    val_dice, pvd, ci = evaluate_dice_with_bootstrap(model, test_ds, device, 2, 1000)
    return model, {"seed": seed, "val_dice": val_dice,
                   "per_volume_dices": pvd, "bootstrap_ci95": ci,
                   "wall_s": time.time() - t0,
                   "skipped_nan": skipped_nan, "status": "ok"}


# ============================================================
# Main
# ============================================================

def _is_valid_dice(d):
    return (d is not None and isinstance(d, (int, float))
            and not math.isnan(float(d)) and float(d) > 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/brats_conditional_128.pt")
    p.add_argument("--ckpt-dir", default="results/paper4/brats_128cubed_conditional",
                   help="Directory with phase1_shared.pt, shortcut/final.pt, "
                        "consistency/final.pt")
    p.add_argument("--teacher-path",
                   default="results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt")
    p.add_argument("--output-dir",
                   default="results/paper4/brats_128cubed_conditional/e10b_128")
    p.add_argument("--max-iters", type=int, default=2000,
                   help="Per (fraction, condition, seed). Matches 64³ protocol.")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Per (fraction, condition). Matches 64³ E10b protocol exactly.")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--steps-shortcut", type=int, default=50)
    p.add_argument("--steps-consistency", type=int, default=4)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")
    seeds = list(range(42, 42 + args.n_seeds))

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

    train_idx_full = splits["train_idx"]
    test_idx = splits["test_idx"]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]
    test_ds = SegDataset(test_vols, test_segs, augment=False)
    print(f"Train pool: {len(train_idx_full)} volumes; Test: {len(test_vols)} volumes")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "e10b_results_128_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        print(f"\nResuming from {partial_path}")
    else:
        results = {}

    # ----- Generate synthetic pools (once, shared across fractions/seeds) -----
    syn_pools = {}
    print("\n=== Generating synthetic pools (one-time) ===")
    for method, steps, key in [
        ("shortcut", args.steps_shortcut, "shortcut_50"),
        ("consistency", args.steps_consistency, "consistency_4"),
    ]:
        cache_path = out_dir / f"syn_pool_{key}.pt"
        if cache_path.exists():
            print(f"  Loading cached {key} pool from {cache_path}")
            syn_pools[key] = torch.load(cache_path, weights_only=False)
            continue
        print(f"  Generating {N_SYNTH} {key} class-{TARGET_CLASS} samples @ NFE={steps}")
        enc, dec, vq, gen_unet, nc = load_cond_pipeline(
            args.ckpt_dir, method, dev, num_classes=args.num_classes)
        t_gen = time.time()
        sampler = sample_shortcut_cond if method == "shortcut" else sample_consistency_cond
        syn = sampler(gen_unet, dec, N_SYNTH, dev, steps,
                       class_label=TARGET_CLASS, batch_size=2)
        print(f"    Generated {syn.shape[0]} volumes shape {tuple(syn.shape)} "
              f"({time.time() - t_gen:.0f}s)")
        syn_pools[key] = syn
        torch.save(syn, cache_path)
        del enc, dec, vq, gen_unet
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()

    # Load teacher (needed for pseudo-labelling synthetic conditions)
    teacher = load_teacher_128(args.teacher_path, dev)

    # ----- Build the experiment matrix -----
    # results[frac_key][cond] = {"dices": [...], "wall_s_per_seed": [...], ...}
    for frac in FRACTIONS:
        frac_key = f"{int(frac * 100)}pct"
        n_real = max(2, int(len(train_idx_full) * frac))
        real_vols = vols[train_idx_full[:n_real]]
        real_segs = segs[train_idx_full[:n_real]]
        print(f"\n========================================")
        print(f"  FRACTION {frac_key} — {n_real} real volumes")
        print(f"========================================")

        frac_results = results.setdefault(frac_key, {})

        for cond in CONDITIONS:
            cond_results = frac_results.setdefault(cond, {
                "dices": [None] * len(seeds),
                "wall_s_per_seed": [None] * len(seeds),
                "seed_statuses": ["pending"] * len(seeds),
                "per_volume_dices": [None] * len(seeds),
                "bootstrap_ci95": [None] * len(seeds),
            })
            for key, default_val in (("dices", None), ("wall_s_per_seed", None),
                                      ("seed_statuses", "pending"),
                                      ("per_volume_dices", None),
                                      ("bootstrap_ci95", None)):
                arr = cond_results.get(key) or []
                while len(arr) < len(seeds):
                    arr.append(default_val)
                cond_results[key] = arr[:len(seeds)]

            seed_status = ["skip" if _is_valid_dice(cond_results["dices"][i])
                           else "run" for i in range(len(seeds))]
            if all(s == "skip" for s in seed_status):
                print(f"  --- {cond.upper()} (all seeds valid, skipping) ---")
                continue

            print(f"  --- {cond.upper()} ---")
            # Build training dataset for this condition (real subset is the same)
            real_ds_aug = SegDataset(real_vols, real_segs, augment=True)
            if cond == "real_only":
                train_ds = real_ds_aug
            elif cond == "classical_aug":
                train_ds = ClassicalAugDataset(real_vols, real_segs)
            elif cond in ("shortcut_50", "consistency_4"):
                syn_vols = syn_pools[cond]
                syn_ds = SyntheticSegDataset(syn_vols, teacher, dev,
                                              batch_size=2, augment=True)
                cond_results["tumor_detection_rate"] = syn_ds.tumor_detection_rate
                print(f"    Tumor detection rate (teacher on {cond} synthetic): "
                      f"{syn_ds.tumor_detection_rate:.1%}")
                train_ds = ConcatDataset([real_ds_aug, syn_ds])
            else:
                sys.exit(f"Unknown condition: {cond}")

            for i, seed in enumerate(seeds):
                if seed_status[i] == "skip":
                    print(f"    Seed {seed} (already complete, Dice "
                          f"{cond_results['dices'][i]:.4f}, skipping)")
                    continue
                print(f"    Seed {seed}:")
                model, info = train_one_run(train_ds, test_ds, dev,
                                             args.max_iters, seed)
                cond_results["dices"][i] = (None if not _is_valid_dice(info["val_dice"])
                                             else float(info["val_dice"]))
                cond_results["wall_s_per_seed"][i] = float(info["wall_s"])
                cond_results["seed_statuses"][i] = info.get("status", "ok")
                cond_results["per_volume_dices"][i] = info.get("per_volume_dices")
                cond_results["bootstrap_ci95"][i] = info.get("bootstrap_ci95")
                ci = info.get("bootstrap_ci95")
                ci_str = (f"  CI95=[{ci['lo']:.4f}, {ci['hi']:.4f}]"
                          if ci and "lo" in ci else "")
                print(f"      → Dice {info['val_dice']:.4f}{ci_str}  "
                      f"({info['wall_s']:.0f}s = {info['wall_s'] / 3600:.2f}h) "
                      f"[{info.get('status', 'ok')}]")
                del model
                if dev == "mps":
                    torch.mps.empty_cache()
                elif dev == "cuda":
                    torch.cuda.empty_cache()
                valid = [d for d in cond_results["dices"] if _is_valid_dice(d)]
                if valid:
                    cond_results["mean"] = float(np.mean(valid))
                    cond_results["std"] = float(np.std(valid))
                with open(partial_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"      Checkpoint saved to {partial_path}")
            valid = [d for d in cond_results["dices"] if _is_valid_dice(d)]
            if valid:
                print(f"  → {frac_key}/{cond}: {np.mean(valid):.4f} "
                      f"± {np.std(valid):.4f} (over {len(valid)} valid seeds)")

    # ----- Save final results + summary -----
    final_path = out_dir / "e10b_results_128.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {final_path}")

    print("\n" + "=" * 80)
    print(f"  128³ E10b summary (max_iters={args.max_iters}, n_seeds={args.n_seeds})")
    print("=" * 80)
    print(f"{'Fraction':>10} {'Condition':>16} {'Dice ± std':>20} {'n valid':>10}")
    for frac in FRACTIONS:
        frac_key = f"{int(frac * 100)}pct"
        for cond in CONDITIONS:
            c = results.get(frac_key, {}).get(cond, {})
            valid = [d for d in c.get("dices", []) if _is_valid_dice(d)]
            if valid:
                print(f"{frac_key:>10} {cond:>16} "
                      f"{np.mean(valid):>10.4f} ± {np.std(valid):.4f}      "
                      f"{len(valid):>5}")

    # ----- Paired t-tests vs real_only within each fraction -----
    from scipy import stats
    print("\nPaired t-tests vs real_only (uncorrected, within each fraction):")
    for frac in FRACTIONS:
        frac_key = f"{int(frac * 100)}pct"
        frac_results = results.get(frac_key, {})
        real_dices = [d for d in frac_results.get("real_only", {}).get("dices", [])
                      if _is_valid_dice(d)]
        for cond in CONDITIONS:
            if cond == "real_only": continue
            cond_dices = [d for d in frac_results.get(cond, {}).get("dices", [])
                          if _is_valid_dice(d)]
            paired = list(zip(real_dices, cond_dices))
            paired = [(a, b) for a, b in paired
                       if _is_valid_dice(a) and _is_valid_dice(b)]
            if len(paired) >= 2:
                a, b = zip(*paired)
                t, p = stats.ttest_rel(a, b)
                dd = float(np.mean(b)) - float(np.mean(a))
                sig = "*" if p < 0.05 else " "
                print(f"  {sig} {frac_key:>5} {cond:>16} vs real_only: "
                      f"t={float(t):>7.3f}  p={float(p):.4g}  ΔDice={dd:+.4f}")


if __name__ == "__main__":
    main()
