#!/usr/bin/env python3
"""
Item 17 — 128³ resolution validation: evaluation
=================================================

Runs the 5-seed evaluation on the 128³ Shortcut FM and Consistency Distillation
models, producing the comparable SSIM / PSNR / pairwise-L1 diversity numbers used
in Tables 2 and 3 of the paper.

Generates 64 samples × 5 seeds at NFE = 50 from each method, decodes to 128³
volumes, and compares to randomly paired real BraTS volumes (matching the
evaluation protocol used in the main benchmark).

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed/eval_128.py \\
        --data-path data/brats_preprocessed_128.pt \\
        --ckpt-dir results/paper4/brats_128cubed \\
        --output-dir results/paper4/brats_128cubed/paper_eval_5seed \\
        --steps 50

Estimated wall time: 20-40 minutes on M-series (most of the time is decoding
at 128³ rather than U-Net forward passes).

Output: <output-dir>/eval_results_128_5seed.json — same shape as the existing
        eval_results_5seed.json so v6's Table 3 / Table 10 patterns apply.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import (
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D,
)

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]
LATENT_SIZE = 16  # 128 / 8 = 16 for the existing 3-stride-2 encoder


# ============================================================
# Metrics (mirror evaluate_5seed.py)
# ============================================================

def ssim3d(a, b):
    """Windowed 3D SSIM with 7×7×7 Gaussian, σ=1.5 (matches Tables 2/3)."""
    a_t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float()
    b_t = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k, sigma = 7, 1.5
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
    return float(((2 * mu_a * mu_b + C1) * (2 * s_ab + C2) /
                  ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))).mean())


def psnr3d(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 50.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def pairwise_diversity(samples):
    n = len(samples)
    if n < 2: return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.abs(samples[i] - samples[j]).mean()))
    return float(np.mean(dists))


def bootstrap_ci(values, n_boot=1000, alpha=0.05, rng=None):
    arr = np.asarray(values, dtype=float)
    if rng is None: rng = np.random.default_rng(42)
    means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    means = np.sort(means)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "ci95_lo": float(means[int(n_boot * alpha / 2)]),
            "ci95_hi": float(means[int(n_boot * (1 - alpha / 2))])}


# ============================================================
# Generation at 128³
# ============================================================

@torch.no_grad()
def consistency_sample_128(unet, dec, n, dev, steps, batch_size=4, latent_size=LATENT_SIZE):
    """Consistency sampling: f(x,t) = x + (1-t)v(x,t) maps any point to the data endpoint.

    For multi-step consistency sampling, we re-noise after each step and re-apply.
    """
    out = []
    rem = n
    while rem > 0:
        bs = min(batch_size, rem)
        z = torch.randn(bs, 8, latent_size, latent_size, latent_size, device=dev)
        t = torch.zeros(bs, device=dev)  # start from pure noise
        # Single-step consistency function: f(x, 0) directly predicts data endpoint
        v = unet(z, t)
        x_hat = z + (1.0 - t[:, None, None, None, None]) * v
        # Multi-step: re-noise and refine
        for s in range(1, steps):
            t_s = torch.full((bs,), s / max(steps, 1), device=dev)
            noise = torch.randn_like(x_hat)
            te = t_s[:, None, None, None, None]
            x_s = (1 - te) * noise + te * x_hat
            v = unet(x_s, t_s)
            x_hat = x_s + (1 - te) * v
        vols = dec(x_hat).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


@torch.no_grad()
def shortcut_sample_128(unet, dec, n, dev, steps, batch_size=4, latent_size=LATENT_SIZE):
    """Shortcut FM sampling: Euler integration with N=steps step size d=1/N."""
    out = []
    rem = n
    has_d = _has_d_kwarg(unet)
    while rem > 0:
        bs = min(batch_size, rem)
        z = torch.randn(bs, 8, latent_size, latent_size, latent_size, device=dev)
        d_val = 1.0 / steps
        for s in range(steps):
            t = torch.full((bs,), s / steps, device=dev)
            d = torch.full((bs,), d_val, device=dev)
            v = unet(z, t, d=d) if has_d else unet(z, t)
            z = z + d_val * v
        vols = dec(z).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


def _has_d_kwarg(model):
    import inspect
    try:
        return "d" in inspect.signature(model.forward).parameters
    except Exception:
        return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/brats_preprocessed_128.pt",
                        help="128³ real BraTS volumes (for SSIM/PSNR pairing and diversity baseline)")
    parser.add_argument("--ckpt-dir", default="results/paper4/brats_128cubed",
                        help="Directory containing phase1_shared.pt, consistency/final.pt, shortcut/final.pt")
    parser.add_argument("--output-dir", default="results/paper4/brats_128cubed/paper_eval_5seed",
                        help="Where to save eval_results_128_5seed.json")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load VQ-GAN -----
    p1_path = ckpt_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"phase1_shared.pt missing: {p1_path}")
    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    dec = Decoder3D(1, 8, 2).to(dev); dec.load_state_dict(p1["dec"]); dec.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    print(f"VQ-GAN loaded from {p1_path}")

    # ----- Load real volumes -----
    real = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if isinstance(real, dict):
        real = real.get("volumes", real.get("v"))
    if real.dim() == 5:
        real = real.squeeze(1)
    real_np = real.float().numpy()
    print(f"Real BraTS: {len(real_np)} volumes at 128³")

    # Real-data diversity baseline (n=args.n_samples random reals)
    rng = np.random.default_rng(0)
    real_subset = real_np[rng.choice(len(real_np), size=min(args.n_samples, len(real_np)), replace=False)]
    real_diversity = pairwise_diversity(real_subset)
    print(f"Real diversity baseline (n={len(real_subset)}): {real_diversity:.4f}")

    # ----- Evaluate each method -----
    results = {
        "metadata": {
            "seeds": SEEDS, "n_samples": args.n_samples, "steps": args.steps,
            "real_diversity": real_diversity, "resolution": 128, "latent_size": LATENT_SIZE,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
        "results": {},
    }

    method_ckpts = {
        "consistency": ckpt_dir / "consistency" / "final.pt",
        "shortcut": ckpt_dir / "shortcut" / "final.pt",
    }
    samplers = {"consistency": consistency_sample_128, "shortcut": shortcut_sample_128}

    for method, ckpt_path in method_ckpts.items():
        if not ckpt_path.exists():
            print(f"Skipping {method}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n----- Evaluating {method}@{args.steps} -----")
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
        unet = DenoisingUNet3D(ch=8).to(dev)
        unet.load_state_dict(state["unet"] if "unet" in state else state)
        unet.eval()
        sampler = samplers[method]

        per_seed = {"ssim": [], "psnr": [], "diversity": [], "time_per_vol": []}
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            gen = sampler(unet, dec, args.n_samples, dev, args.steps, batch_size=2)
            time_per_vol = (time.time() - t0) / args.n_samples
            gen_np = gen.squeeze(1) if gen.ndim == 5 else gen

            n_compare = min(args.n_samples, len(real_subset))
            ssims = [ssim3d(real_subset[i % len(real_subset)], gen_np[i]) for i in range(n_compare)]
            psnrs = [psnr3d(real_subset[i % len(real_subset)], gen_np[i]) for i in range(n_compare)]
            diversity = pairwise_diversity(gen_np)

            per_seed["ssim"].append(float(np.mean(ssims)))
            per_seed["psnr"].append(float(np.mean(psnrs)))
            per_seed["diversity"].append(float(diversity))
            per_seed["time_per_vol"].append(float(time_per_vol))
            print(f"  seed={seed}: SSIM={np.mean(ssims):.4f}  diversity={diversity:.4f}  ({time_per_vol:.3f}s/vol)")

        summary = {
            "per_seed": per_seed,
            "ssim": bootstrap_ci(per_seed["ssim"]),
            "psnr": bootstrap_ci(per_seed["psnr"]),
            "diversity": bootstrap_ci(per_seed["diversity"]),
            "diversity_pct_real": (float(np.mean(per_seed["diversity"]) / real_diversity * 100.0)
                                   if real_diversity > 0 else None),
        }
        results["results"][method] = summary
        print(f"  SUMMARY @{args.steps}: SSIM={summary['ssim']['mean']:.4f}  "
              f"diversity={summary['diversity']['mean']:.4f}  "
              f"({summary['diversity_pct_real']:.1f}% of real)")

    out_path = output_dir / "eval_results_128_5seed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ----- Print comparison table -----
    print("\n" + "=" * 70)
    print(f"  128³ resolution validation summary (NFE = {args.steps})")
    print("=" * 70)
    print(f"{'Method':>20}  {'SSIM':>16}  {'Diversity':>16}  {'%Real':>7}")
    for method in ["consistency", "shortcut"]:
        if method not in results["results"]: continue
        s = results["results"][method]
        ssim_str = f"{s['ssim']['mean']:.3f}±{s['ssim']['std']:.3f}"
        div_str = f"{s['diversity']['mean']:.4f}±{s['diversity']['std']:.4f}"
        pct = f"{s['diversity_pct_real']:.0f}%" if s['diversity_pct_real'] is not None else "-"
        print(f"{method:>20}  {ssim_str:>16}  {div_str:>16}  {pct:>7}")


if __name__ == "__main__":
    main()
