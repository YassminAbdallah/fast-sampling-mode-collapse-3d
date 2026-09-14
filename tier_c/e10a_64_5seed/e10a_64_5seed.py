#!/usr/bin/env python3
"""
64-cubed E10a dose-response — 5-seed replication.
==================================================

Motivation: the paper's headline
threshold claim (§5.5, Table 8) is based on n = 3 seeds and none of the
individual comparisons survive Bonferroni. The 128-cubed E10a replication
uses 5 seeds and IS Bonferroni-significant. This script re-runs the 64-cubed
E10a with n = 5 seeds so the primary threshold claim is corrected-significant
in the main setting.

Design (identical to the existing 64-cubed E10a in src/paper4/conditional_segmentation.py::run_e10a,
plus per-seed checkpointing so a mid-run interruption is recoverable):

  * All conditions use 500 total synthetic volumes from Shortcut@50 at 64-cubed
    conditional. The ONLY variable is how many are UNIQUE:
        - unique_500 : full diversity
        - unique_100 : each repeated ~5x  (moderate collapse)
        - unique_25  : each repeated ~20x (severe collapse)
        - unique_10  : each repeated ~50x (extreme collapse)
        - real_only  : no synthetic (baseline)

  * 5 conditions x 5 seeds = 25 downstream segmenter training runs.
    At 64-cubed with max_iters = 2000, wall time per run is ~45-60 min on
    Apple M1. Total: ~20-25 h continuous. Per-seed checkpointing.

  * Same conditional 64-cubed Shortcut FM model, VQ-GAN encoder-decoder, and
    teacher segmenter as the paper's existing 64-cubed E10a (Table 8, §5.5).
    Seeds {42, 43, 44} therefore match the paper's existing 3-seed run
    bit-for-bit (deterministic given the same code and hardware). Seeds
    {45, 46} are the new additions.

  * Same statistics: paired-t on per-seed Dice across conditions, plus
    Bonferroni family size k = 8 to match Appendix A.6 Table A.6b.

Outputs:
  results/paper4/e10a_64_5seed/
    syn_pool_shortcut_50_64.pt         (500 x 1 x 64 x 64 x 64 float, cached)
    e10a_64_5seed_results_partial.json (updated after every seed)
    e10a_64_5seed_results.json         (final; identical schema)

Usage:
  cd fast-sampling-mode-collapse-3d/
  bash tier_c/e10a_64_5seed/run_e10a_64_5seed.sh          # full 5-seed run

Resume: if the run is interrupted, restart the same command. Every seed
already recorded in the partial JSON is skipped.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

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
    load_unet as _load_unet_shared,
    load_vqgan as _load_vqgan_shared,
)

try:
    from utils.set_seed import set_seed
    HAS_SET_SEED = True
except Exception:
    HAS_SET_SEED = False


# ============================================================
# Constants (64-cubed E10a)
# ============================================================

LATENT_SIZE = 8            # 64 / 8 = 8-cubed latent
N_SYNTH = 500
TARGET_CLASS = 1           # class-1 conditioning (large tumor)
UNIQUE_LEVELS = [500, 100, 25, 10]
# 8 seeds total. 42-46 match the previous 5-seed pass; 47-49 are the extension
# added to bring 64-cubed threshold contrasts across Bonferroni α = 0.05 at k = 8.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]


# ============================================================
# Datasets (same as 128-cubed E10b)
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
    """Synthetic volumes with teacher-predicted pseudo-labels."""

    def __init__(self, syn_volumes, teacher_model, device, batch_size=2, augment=True):
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


def load_teacher_64(path, device):
    model = create_seg_model(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_shortcut_64(bench_dir, device, num_classes=2):
    """Load VQ-GAN encoder-decoder + Shortcut FM conditional generator at 64-cubed.

    Uses the paper's canonical loaders (models_shared.load_vqgan / load_unet),
    which handle both the LatentGenerative3D-wrapped state dict (`unet.*`-prefixed
    keys plus enc/dec/vq) and bare UNet state dicts.
    """
    enc, dec, vq, _ = _load_vqgan_shared(Path(bench_dir) / "phase1_shared.pt", device)
    unet = _load_unet_shared(Path(bench_dir) / "shortcut" / "final.pt",
                             device, num_classes=num_classes)
    for m in (enc, dec, vq, unet):
        for p in m.parameters():
            p.requires_grad = False
    return enc, dec, vq, unet, num_classes


# ============================================================
# Shortcut sampling (matches E10b sampler; same as paper's E10a)
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
    """Sample n Shortcut FM volumes at `steps` NFE, class-conditioned to `class_label`.

    Also returns the latent features (flattened) used to feed the 500 samples
    into the feature-centrality routine that builds the diversity conditions.
    """
    out_vols = []
    out_feats = []
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
        out_vols.append(vols)
        out_feats.append(z.view(bs, -1).cpu())
        rem -= bs
    vols = torch.cat(out_vols, dim=0)[:n]
    feats = torch.cat(out_feats, dim=0)[:n]
    return vols, feats


# ============================================================
# Training (recipe v2 — matches 128-cubed E10b)
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
                   batch_size=2, warmup_iters=100, peak_lr=3e-4, warmup_lr=3e-5):
    """Train one segmenter for one (condition, seed). Returns dict with dice + wall time."""
    if HAS_SET_SEED:
        set_seed(seed)
    else:
        torch.manual_seed(seed); np.random.seed(seed)

    model = create_seg_model(device)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=warmup_lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                         smooth_nr=0.0, smooth_dr=1.0)

    t0 = time.time()
    it = 0
    skipped_nan = 0
    consecutive_bad = 0
    dl_iter = iter(dl)
    while it < max_iters:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        it += 1

        # Warmup LR schedule (linear warmup → cosine to zero over remaining iters)
        if it <= warmup_iters:
            lr = warmup_lr + (peak_lr - warmup_lr) * (it / warmup_iters)
        else:
            progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
            lr = peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        x = batch["x"].to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        y = batch["y"].to(device)
        logits = model(x)

        if not torch.isfinite(logits).all():
            skipped_nan += 1
            consecutive_bad += 1
            if consecutive_bad >= 20:
                print(f"      ! 20 consecutive NaN batches; aborting seed.")
                return {"dice": float("nan"), "per_volume_dices": [],
                        "wall_seconds": time.time() - t0, "status": "aborted_nan"}
            continue
        consecutive_bad = 0

        loss = loss_fn(logits, y.unsqueeze(1))
        if not torch.isfinite(loss):
            skipped_nan += 1
            continue
        opt.zero_grad()
        loss.backward()
        # Gradient NaN guard
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            skipped_nan += 1
            continue
        opt.step()

        if it % 500 == 0 or it == 1 or it == max_iters:
            elapsed = time.time() - t0
            print(f"      iter {it:>5}/{max_iters} loss={float(loss):.4f} "
                  f"(lr={lr:.2e}, skipped_nan={skipped_nan}, {elapsed:.0f}s)")

    # Evaluate
    mean_dice, per_vol, ci = evaluate_dice_with_bootstrap(model, test_ds, device)
    total = time.time() - t0
    print(f"      → Dice {mean_dice:.4f}  CI95=[{ci['lo']:.4f}, {ci['hi']:.4f}]  "
          f"({total:.0f}s = {total/3600:.2f}h) [ok]")
    del model
    if device.type == "mps":  torch.mps.empty_cache()
    elif device.type == "cuda": torch.cuda.empty_cache()
    return {"dice": mean_dice, "per_volume_dices": per_vol,
            "ci95_lo": ci["lo"], "ci95_hi": ci["hi"],
            "wall_seconds": total, "skipped_nan": skipped_nan, "status": "ok"}


# ============================================================
# Main driver
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=str,
                    default="results/paper4/brats_benchmark_20260324_154926/100pct_180vol",
                    help="Directory containing phase1_shared.pt + shortcut/final.pt at 64-cubed conditional")
    ap.add_argument("--teacher-path", type=str,
                    default="results/paper4/paper4/e7_teacher/teacher_best.pt",
                    help="Path to 64-cubed teacher segmenter")
    ap.add_argument("--data-path", type=str,
                    default="data/brats_conditional_64.pt",
                    help="Path to 64-cubed conditional BraTS dataset")
    ap.add_argument("--output-dir", type=str,
                    default="results/paper4/e10a_64_5seed",
                    help="Output directory")
    ap.add_argument("--max-iters", type=int, default=2000,
                    help="Segmenter training iterations per seed (default 2000)")
    ap.add_argument("--device", type=str, default=None,
                    help="Device: mps, cuda, cpu. Default: auto-detect.")
    args = ap.parse_args()

    # Device
    if args.device is None:
        dev = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    else:
        dev = torch.device(args.device)
    print(f"Device: {dev}")

    # Paths
    bench_dir = REPO_ROOT / args.bench_dir
    teacher_path = REPO_ROOT / args.teacher_path
    data_path = REPO_ROOT / args.data_path
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (bench_dir, teacher_path, data_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    # Load conditional data
    print(f"\nLoading 64-cubed conditional data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    vols  = data["volumes"]
    segs  = data["segs"]
    splits = data["split_info"]
    train_idx = splits["train_idx"]
    test_idx  = splits["test_idx"]

    # 10% real subset (matches paper's original E10a design)
    n_10 = max(4, int(len(train_idx) * 0.10))
    real_vols = vols[train_idx[:n_10]]
    real_segs = segs[train_idx[:n_10]]
    test_vols = vols[test_idx]
    test_segs = segs[test_idx]
    print(f"  Real subset (10%): {n_10} volumes  |  Test: {len(test_idx)} volumes")

    # Load teacher
    print(f"\nLoading 64-cubed teacher from {teacher_path}...")
    teacher = load_teacher_64(teacher_path, dev)

    # ------------------------------------------------------------------
    # Step 1: Generate + cache 500 Shortcut@50 synthetic volumes
    # ------------------------------------------------------------------
    pool_path = out_dir / "syn_pool_shortcut_50_64.pt"
    feats_path = out_dir / "syn_pool_shortcut_50_64_feats.pt"
    if pool_path.exists() and feats_path.exists():
        print(f"\n[Step 1] Loading cached Shortcut@50 pool from {pool_path.name}")
        all_vols = torch.load(pool_path, map_location="cpu", weights_only=True)
        all_feats = torch.load(feats_path, map_location="cpu", weights_only=True).numpy()
        print(f"  {len(all_vols)} volumes, feat shape {all_feats.shape}")
    else:
        print(f"\n[Step 1] Generating {N_SYNTH} volumes from Shortcut@50 (class {TARGET_CLASS})...")
        _, dec, _, unet, _ = load_shortcut_64(bench_dir, dev)
        # Reproducibility: seed 42 for the synthetic pool
        torch.manual_seed(42)
        if dev.type == "mps": torch.mps.manual_seed(42)
        elif dev.type == "cuda": torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        all_vols, all_feats_t = sample_shortcut_cond(
            unet, dec, N_SYNTH, dev, steps=50,
            class_label=TARGET_CLASS)
        all_feats = all_feats_t.numpy()
        torch.save(all_vols, pool_path)
        torch.save(all_feats_t, feats_path)
        print(f"  Saved {pool_path.name}, {feats_path.name}")
        del unet, dec

    # ------------------------------------------------------------------
    # Step 2: Build diversity conditions from feature-space centrality
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Building diversity conditions from feature centrality...")
    global_mean = all_feats.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(all_feats - global_mean, axis=1)
    order = np.argsort(dists)  # closest-first
    conditions = {}
    rng = np.random.RandomState(42)

    def pairwise_l1_diversity(vol_tensor, n_pairs=1000):
        idx1 = torch.randint(0, len(vol_tensor), (n_pairs,))
        idx2 = torch.randint(0, len(vol_tensor), (n_pairs,))
        return (vol_tensor[idx1] - vol_tensor[idx2]).abs().mean().item()

    for n_unique in UNIQUE_LEVELS:
        label = f"unique_{n_unique}"
        if n_unique >= N_SYNTH:
            syn = all_vols
        else:
            src = rng.choice(N_SYNTH, n_unique, replace=False)
            take = rng.choice(src, N_SYNTH, replace=True)
            syn = all_vols[take]
        div = pairwise_l1_diversity(syn)
        conditions[label] = syn
        print(f"  {label}: {N_SYNTH} total, {n_unique} unique, L1 div = {div:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Resume-aware training loop
    # ------------------------------------------------------------------
    partial_path = out_dir / "e10a_64_5seed_results_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        print(f"\n[Step 3] Resuming from {partial_path.name}")
    else:
        results = {}

    test_ds = SegDataset(test_vols, test_segs, augment=False)

    all_cond_names = list(conditions.keys()) + ["real_only"]
    for cond in all_cond_names:
        print(f"\n========================================")
        print(f"  {cond.upper()}")
        print(f"========================================")

        # Build the training dataset for this condition
        real_ds = SegDataset(real_vols, real_segs, augment=True)
        if cond == "real_only":
            train_ds = real_ds
            n_unique = 0
            tumor_rate = None
        else:
            syn_ds = SyntheticSegDataset(conditions[cond], teacher, dev, augment=True)
            tumor_rate = syn_ds.tumor_detection_rate
            train_ds = ConcatDataset([real_ds, syn_ds])
            n_unique = int(cond.split("_")[1])

        cond_entry = results.get(cond, {
            "n_unique": n_unique,
            "n_total": 0 if cond == "real_only" else N_SYNTH,
            "tumor_detection_rate": tumor_rate,
            "dices": [], "seeds_done": [],
            "per_seed": {},
        })
        if tumor_rate is not None:
            cond_entry["tumor_detection_rate"] = tumor_rate

        for seed in SEEDS:
            if seed in cond_entry.get("seeds_done", []):
                print(f"  Seed {seed}: already done (Dice "
                      f"{cond_entry['per_seed'][str(seed)]['dice']:.4f}), skipping")
                continue

            print(f"  Seed {seed}:")
            res = train_one_run(train_ds, test_ds, dev,
                                max_iters=args.max_iters, seed=seed)
            if res["status"] != "ok":
                print(f"    seed {seed} returned status={res['status']}, skipping in aggregate")
                continue
            cond_entry["per_seed"][str(seed)] = res
            cond_entry["dices"].append(res["dice"])
            cond_entry["seeds_done"].append(seed)

            results[cond] = cond_entry
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    Checkpoint saved to {partial_path.name}")

        if cond_entry["dices"]:
            m  = float(np.mean(cond_entry["dices"]))
            sd = float(np.std(cond_entry["dices"], ddof=1)
                       if len(cond_entry["dices"]) > 1 else 0.0)
            cond_entry["mean"] = m
            cond_entry["std"] = sd
            print(f"  → {cond}: {m:.4f} ± {sd:.4f} (n = {len(cond_entry['dices'])})")

        results[cond] = cond_entry
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Step 4: Statistics
    # ------------------------------------------------------------------
    from scipy import stats

    print(f"\n========================================")
    print(f"  E10a 64³ dose-response (n = {len(SEEDS)} seeds) — summary")
    print(f"========================================")
    print(f"  {'condition':<14} {'n_unique':>8} {'Dice':>10} {'± std':>10} {'n':>4}")
    print(f"  {'-'*50}")
    for cond in ["real_only"] + [f"unique_{n}" for n in UNIQUE_LEVELS]:
        if cond in results and results[cond].get("dices"):
            r = results[cond]
            print(f"  {cond:<14} {r['n_unique']:>8} "
                  f"{r['mean']:>10.4f} {r['std']:>10.4f} {len(r['dices']):>4}")

    # Paired-t tests: adjacent unique_ levels + extremes + each unique_ vs real_only
    tests = []
    ordered_conds = sorted(
        [c for c in results if c.startswith("unique_")],
        key=lambda c: results[c]["n_unique"])

    for i in range(len(ordered_conds) - 1):
        lo, hi = ordered_conds[i], ordered_conds[i + 1]
        r_lo = results[lo]; r_hi = results[hi]
        # Align seeds — only include seeds present in both
        common_seeds = sorted(set(r_lo["seeds_done"]) & set(r_hi["seeds_done"]))
        if len(common_seeds) < 2:
            continue
        d_lo = [r_lo["per_seed"][str(s)]["dice"] for s in common_seeds]
        d_hi = [r_hi["per_seed"][str(s)]["dice"] for s in common_seeds]
        t, p = stats.ttest_rel(d_lo, d_hi)
        tests.append({"comparison": f"{hi} vs {lo}",
                      "n_paired_seeds": len(common_seeds),
                      "t": float(t), "p_uncorrected": float(p),
                      "dice_diff": float(np.mean(d_hi) - np.mean(d_lo))})

    # Extremes: unique_500 vs unique_10
    if len(ordered_conds) >= 2:
        lo = ordered_conds[0]; hi = ordered_conds[-1]
        common = sorted(set(results[lo]["seeds_done"]) & set(results[hi]["seeds_done"]))
        if len(common) >= 2:
            d_lo = [results[lo]["per_seed"][str(s)]["dice"] for s in common]
            d_hi = [results[hi]["per_seed"][str(s)]["dice"] for s in common]
            t, p = stats.ttest_rel(d_lo, d_hi)
            tests.append({"comparison": f"{hi} vs {lo} (extremes)",
                          "n_paired_seeds": len(common),
                          "t": float(t), "p_uncorrected": float(p),
                          "dice_diff": float(np.mean(d_hi) - np.mean(d_lo))})

    # Each unique_ vs real_only
    if "real_only" in results and results["real_only"].get("seeds_done"):
        r_ref = results["real_only"]
        for c in ordered_conds:
            common = sorted(set(r_ref["seeds_done"]) & set(results[c]["seeds_done"]))
            if len(common) < 2:
                continue
            d_ref = [r_ref["per_seed"][str(s)]["dice"] for s in common]
            d_c   = [results[c]["per_seed"][str(s)]["dice"] for s in common]
            t, p = stats.ttest_rel(d_ref, d_c)
            tests.append({"comparison": f"{c} vs real_only",
                          "n_paired_seeds": len(common),
                          "t": float(t), "p_uncorrected": float(p),
                          "dice_diff": float(np.mean(d_c) - np.mean(d_ref))})

    # Bonferroni: family size k = 8 to match Appendix A.6 Table A.6b (or use actual n(tests))
    K = max(8, len(tests))
    print(f"\nPaired-t tests (uncorrected p; * Bonferroni-significant at α=0.05, k = {K})")
    for t in tests:
        star = " *" if t["p_uncorrected"] * K < 0.05 else "  "
        t["p_bonferroni"] = min(1.0, t["p_uncorrected"] * K)
        print(f"  {star}{t['comparison']:<28}  Δ={t['dice_diff']:+.4f}  "
              f"t={t['t']:+.3f}  p={t['p_uncorrected']:.5f}  "
              f"p_bonf={t['p_bonferroni']:.5f}")

    # ------------------------------------------------------------------
    # Step 5: Final JSON
    # ------------------------------------------------------------------
    final = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "purpose": (f"64-cubed E10a dose-response {len(SEEDS)}-seed replication "
                    "to strengthen the threshold claim."),
        "resolution": 64,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "unique_levels": UNIQUE_LEVELS,
        "n_synth_per_condition": N_SYNTH,
        "real_subset_size": n_10,
        "max_iters_per_seed": args.max_iters,
        "results": {k: v for k, v in results.items() if isinstance(v, dict)},
        "paired_t_tests": tests,
        "bonferroni_family_size": K,
    }
    final_path = out_dir / "e10a_64_5seed_results.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved {final_path}")


if __name__ == "__main__":
    main()
