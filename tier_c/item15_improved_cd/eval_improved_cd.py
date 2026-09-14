#!/usr/bin/env python3
"""
Item 15 — Improved Consistency Distillation — EVALUATION
=========================================================

Focused 5-seed evaluation of the trained improved-CD model. Reports SSIM,
PSNR, and pairwise-L1 diversity at NFE in {1, 4, 16, 50}, the same step
counts used for Consistency Distillation in Table 3 of the paper.

Uses the consistency sampling routine (identical to original CD sampling)
since improved CD only differs in the training loss, not the sampler.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item15_improved_cd/eval_improved_cd.py \\
        --improved-cd-dir results/paper4/improved_cd_brats/improved_cd \\
        --gen-dir         results/paper4/brats_benchmark_20260324_154926 \\
        --data-path       data/brats_conditional_64.pt \\
        --output-dir      results/paper4/improved_cd_brats/paper_eval_5seed \\
        --num-classes 2 \\
        --n-samples 64

Estimated wall time on Apple M-series: 30 - 60 minutes.

Output: <output-dir>/eval_results_5seed.json with the same shape as
        results/paper4/.../paper_eval_5seed/eval_results_5seed.json so the
        improved-CD numbers can be merged directly into Table 3 of the paper.
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import (
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D,
    load_unet, load_vqgan, sample_latent,
)

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]
STEP_COUNTS = [1, 4, 16, 50]


# ============================================================
# Metrics — copies of evaluate_paper.py helpers for standalone use
# ============================================================

def ssim3d(a, b):
    """Windowed 3D SSIM with 7x7x7 Gaussian kernel, sigma=1.5."""
    a_t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float()
    b_t = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = 7
    sigma = 1.5
    coords = torch.arange(k).float() - k // 2
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g3d = (g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :])
    g3d = (g3d / g3d.sum()).unsqueeze(0).unsqueeze(0)
    pad = k // 2
    mu_a = F.conv3d(a_t, g3d, padding=pad)
    mu_b = F.conv3d(b_t, g3d, padding=pad)
    s_aa = F.conv3d(a_t * a_t, g3d, padding=pad) - mu_a * mu_a
    s_bb = F.conv3d(b_t * b_t, g3d, padding=pad) - mu_b * mu_b
    s_ab = F.conv3d(a_t * b_t, g3d, padding=pad) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2)
    )
    return float(ssim_map.mean())


def psnr3d(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10 * np.log10(1.0 / mse))


def compute_pairwise_diversity(samples):
    n = len(samples)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.abs(samples[i] - samples[j]).mean()))
    return float(np.mean(dists))


# ============================================================
# Bootstrap 95% CI helper
# ============================================================

def bootstrap_ci(values, n_boot=1000, alpha=0.05, rng=None):
    arr = np.asarray(values, dtype=float)
    if rng is None:
        rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(arr[idx].mean())
    means = np.sort(means)
    lo = float(means[int(n_boot * alpha / 2)])
    hi = float(means[int(n_boot * (1 - alpha / 2))])
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "ci95_lo": lo, "ci95_hi": hi}


# ============================================================
# Generation: improved-CD sampler is identical to original CD
# ============================================================

@torch.no_grad()
def generate_samples_improved_cd(unet, dec, n, dev, steps, class_label=None, batch_size=8, ch=8):
    """Wrapper around sample_latent with method='consistency' (same sampler as original CD)."""
    out = []
    rem = n
    while rem > 0:
        bs = min(batch_size, rem)
        z = sample_latent(unet, "consistency", bs, dev, steps, ch, class_label=class_label)
        vols = dec(z).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


# ============================================================
# Main eval loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--improved-cd-dir", required=True,
                        help="Folder containing the trained final.pt (e.g. results/.../improved_cd/)")
    parser.add_argument("--gen-dir", default="results/paper4/brats_benchmark_20260324_154926",
                        help="Benchmark dir with phase1_shared.pt (VQ-GAN shared with the rest of the paper)")
    parser.add_argument("--data-path", default="data/brats_conditional_64.pt",
                        help="BraTS conditional volumes — used for real-distribution comparison")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write eval_results_5seed.json")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-label", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--steps", nargs="+", type=int, default=STEP_COUNTS,
                        help="NFE values to evaluate")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")

    # ----- Locate VQ-GAN and improved-CD checkpoints -----
    gen_dir = Path(args.gen_dir)
    sub = sorted([d for d in gen_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else gen_dir
    p1_path = data_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"VQ-GAN checkpoint not found: {p1_path}")

    ckpt_path = Path(args.improved_cd_dir) / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Improved-CD checkpoint not found: {ckpt_path}")

    print(f"VQ-GAN:        {p1_path}")
    print(f"Improved CD:   {ckpt_path}")

    # ----- Load models -----
    _, dec, _, _ = load_vqgan(p1_path, device)
    dec.eval()
    unet = load_unet(ckpt_path, device, num_classes=args.num_classes)
    unet.eval()

    # ----- Load real volumes -----
    d = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols = d.get("volumes", d.get("v"))
    else:
        vols = d
    if vols.dim() == 5:
        vols = vols.squeeze(1)
    real = vols.float().numpy()  # (N, 64, 64, 64)
    print(f"Loaded {len(real)} real volumes")

    # ----- Compute real-data diversity baseline -----
    rng = np.random.default_rng(0)
    real_sample = real[rng.choice(len(real), size=min(64, len(real)), replace=False)]
    real_diversity = compute_pairwise_diversity(real_sample)
    print(f"Real data diversity (n={len(real_sample)}): {real_diversity:.4f}")

    # ----- Evaluation loop -----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "method": "improved_cd",
        "variant": "no-EMA + Pseudo-Huber",
        "seeds": SEEDS,
        "n_samples": args.n_samples,
        "real_diversity": real_diversity,
        "by_steps": {},
    }

    t_total = time.time()
    for steps in args.steps:
        print(f"\n----- Evaluating improved_cd @ {steps} steps -----")
        per_seed = {"ssim": [], "psnr": [], "diversity": [], "time_per_vol": []}

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            gen = generate_samples_improved_cd(
                unet, dec, args.n_samples, device, steps,
                class_label=args.class_label, batch_size=8,
            )
            time_per_vol = (time.time() - t0) / args.n_samples
            # (n, 1, 64, 64, 64) -> (n, 64, 64, 64)
            gen_np = gen.squeeze(1) if gen.ndim == 5 else gen

            # SSIM and PSNR vs randomly paired real volumes
            n_compare = min(args.n_samples, len(real_sample))
            ssims = [ssim3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_compare)]
            psnrs = [psnr3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_compare)]
            diversity = compute_pairwise_diversity(gen_np)

            per_seed["ssim"].append(float(np.mean(ssims)))
            per_seed["psnr"].append(float(np.mean(psnrs)))
            per_seed["diversity"].append(float(diversity))
            per_seed["time_per_vol"].append(float(time_per_vol))

            print(f"  seed={seed}: SSIM={np.mean(ssims):.4f}  diversity={diversity:.4f}  ({time_per_vol:.3f}s/vol)")

        # Aggregate
        summary = {
            "per_seed": per_seed,
            "ssim": bootstrap_ci(per_seed["ssim"]),
            "psnr": bootstrap_ci(per_seed["psnr"]),
            "diversity": bootstrap_ci(per_seed["diversity"]),
            "diversity_pct_real": (
                float(np.mean(per_seed["diversity"]) / real_diversity * 100.0) if real_diversity > 0 else None
            ),
        }
        results["by_steps"][str(steps)] = summary
        print(f"  SUMMARY @{steps}: SSIM={summary['ssim']['mean']:.4f}  "
              f"diversity={summary['diversity']['mean']:.4f}  "
              f"({summary['diversity_pct_real']:.1f}% of real)")

    results["timestamp"] = datetime.utcnow().isoformat() + "Z"
    results["total_eval_seconds"] = round(time.time() - t_total, 1)
    out_path = out_dir / "eval_results_5seed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ----- Print Table-3-style summary -----
    print("\n" + "=" * 60)
    print("  Improved CD — Table-3 row preview")
    print("=" * 60)
    print(f"{'NFE':>5}  {'SSIM':>16}  {'Diversity':>16}  {'%Real':>7}")
    for steps in args.steps:
        s = results["by_steps"][str(steps)]
        ssim_str = f"{s['ssim']['mean']:.3f}±{s['ssim']['std']:.3f}"
        div_str = f"{s['diversity']['mean']:.4f}±{s['diversity']['std']:.4f}"
        pct = f"{s['diversity_pct_real']:.0f}%" if s['diversity_pct_real'] is not None else "-"
        print(f"{steps:>5}  {ssim_str:>16}  {div_str:>16}  {pct:>7}")


if __name__ == "__main__":
    main()
