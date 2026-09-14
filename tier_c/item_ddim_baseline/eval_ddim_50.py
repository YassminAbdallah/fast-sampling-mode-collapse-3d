#!/usr/bin/env python3
"""
DDIM-50 baseline evaluation for Tables 2 and 3
===============================================

Adds the missing DDIM (deterministic) sampler at 50 steps so the
diffusion baseline is reported at its strongest sampling regime. The
existing DDPM-at-50-steps row in Appendix B uses stochastic ancestral
sampling — known to produce noisy outputs at reduced step counts. DDIM
at 50 steps with eta=0 is the fair comparison point against few-step
Shortcut FM and Consistency Distillation.

This script reuses the SAME trained DDPM checkpoint — DDIM is a
sampler, not a separate model. No retraining required.

Protocol mirrors src/paper4/evaluate_5seed.py exactly:
  - 5 seeds: [42, 123, 456, 789, 1337]
  - 64 samples per seed
  - Metrics: SSIM (7×7×7 Gaussian, sigma=1.5), PSNR, pairwise-L1 diversity
  - Bootstrap 95% CI on all metrics

Outputs:
    results/paper4/ddim_baseline_5seed/{dataset}_ddim50.json
        — single-method JSON in the schema of a `by_steps` block from
        the joint benchmark JSON, ready to be merged as a "ddim" row in
        Tables 2 / 3.

Usage:
    cd fast-sampling-mode-collapse-3d/
    python tier_c/item_ddim_baseline/eval_ddim_50.py --dataset brats
    python tier_c/item_ddim_baseline/eval_ddim_50.py --dataset ixi

Wall time on Apple M-series: ~30–60 min per dataset.
"""

import os, sys, json, time, math, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_unet, load_vqgan  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]


# ============================================================
# Default paths per dataset
# ============================================================

DATASETS = {
    "brats": {
        "label": "BraTS 2023 (300 glioma volumes, T2-FLAIR)",
        "data_path":  "data/brats_preprocessed_64.pt",
        "p1_path":    "results/paper3/brats_benchmark_20260221_193306/100pct_300vol/phase1_shared.pt",
        "ddpm_ckpt":  "results/paper3/brats_benchmark_20260221_193306/100pct_300vol/ddpm/final.pt",
    },
    "ixi": {
        "label": "IXI (200 healthy volumes)",
        "data_path":  "data/ixi_preprocessed_64.pt",
        "p1_path":    "results/paper3/ixi_benchmark_20260221_045741/100pct_200vol/phase1_shared.pt",
        "ddpm_ckpt":  "results/paper3/ixi_benchmark_20260221_045741/100pct_200vol/ddpm/final.pt",
    },
}


# ============================================================
# Metrics — identical to evaluate_5seed.py (do not modify)
# ============================================================

def ssim3d(a, b):
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
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2)
    )
    return float(ssim_map.mean())


def psnr3d(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 50.0
    return float(10 * np.log10(1.0 / mse))


def compute_diversity(samples):
    n = len(samples)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.abs(samples[i] - samples[j]).mean()))
    return float(np.mean(dists))


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
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "ci95_lo": lo, "ci95_hi": hi,
        "seeds": values,
    }


# ============================================================
# DDIM-50 sampler (deterministic, eta = 0)
# ============================================================
#
# Given an eps-predictor unet trained with the standard linear noise
# schedule (beta ∈ [1e-4, 0.02], T = 1000), DDIM at S steps picks S
# uniformly-spaced timesteps from {0, …, T−1} and updates by:
#
#     x0_pred(x_t, t) = (x_t − sqrt(1 − ac[t]) * eps(x_t, t)) / sqrt(ac[t])
#     x_{t_prev}     = sqrt(ac[t_prev]) * x0_pred + sqrt(1 − ac[t_prev]) * eps
#
# (eta = 0 → no stochastic term. This is the canonical "DDIM at S steps".)

@torch.no_grad()
def ddim50_sample(unet, n, dev, T=1000, steps=50, ch=8, class_label=None):
    # Schedule (same as src/paper4/models_shared.py ddpm branch)
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, 0)   # alpha_bar

    # Uniformly-spaced timesteps from T-1 down to 0 (closed interval)
    # (so the last step is t=0 → x_0)
    ts = torch.linspace(T - 1, 0, steps + 1, device=dev).round().long()
    # ts has length steps+1; iterate over pairs (ts[i], ts[i+1])

    # Class label handling — identical to sample_latent's prep
    cl = None
    if class_label is not None:
        if isinstance(class_label, int):
            cl = torch.full((n,), class_label, device=dev, dtype=torch.long)
        elif isinstance(class_label, torch.Tensor):
            cl = class_label.to(dev)
        else:
            cl = torch.tensor([class_label] * n, device=dev, dtype=torch.long)

    z = torch.randn(n, ch, 8, 8, 8, device=dev)

    for i in range(steps):
        t_cur = int(ts[i].item())
        t_prev = int(ts[i + 1].item())
        t_in = torch.full((n,), float(t_cur), device=dev, dtype=torch.float32)

        eps = unet(z, t_in, class_label=cl)

        ac_cur  = ac[t_cur]
        ac_prev = ac[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=dev)

        x0_pred = (z - torch.sqrt(1.0 - ac_cur) * eps) / torch.sqrt(ac_cur)
        # Clip in latent space avoided here for parity with existing DDPM code

        if t_prev > 0:
            z = torch.sqrt(ac_prev) * x0_pred + torch.sqrt(1.0 - ac_prev) * eps
        else:
            z = x0_pred  # final step lands on x_0 estimate

    return z


@torch.no_grad()
def generate_ddim50_volumes(unet, dec, n, dev, batch_size=8, ch=8, class_label=None):
    """Returns (n, 1, 64, 64, 64) cpu tensor."""
    out = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = ddim50_sample(unet, bs, dev, steps=50, ch=ch, class_label=class_label)
        gen = dec(z).clamp(0, 1)
        out.append(gen.cpu())
    return torch.cat(out, dim=0)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                   help="brats or ixi")
    p.add_argument("--data-path", default=None,
                   help="Override default data path")
    p.add_argument("--p1-path", default=None,
                   help="Override default VQ-GAN checkpoint path")
    p.add_argument("--ddpm-ckpt", default=None,
                   help="Override default DDPM checkpoint path")
    p.add_argument("--output-dir",
                   default="results/paper4/ddim_baseline_5seed",
                   help="Where to write the per-dataset JSON")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--num-classes", type=int, default=0,
                   help="0 for the unconditional Tables 2/3 DDPM checkpoints")
    p.add_argument("--class-label", type=int, default=None,
                   help="Class label if --num-classes > 0; None for unconditional")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    dev = torch.device(device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({DATASETS[args.dataset]['label']})")

    # ----- Resolve paths -----
    cfg = DATASETS[args.dataset]
    data_path = REPO_ROOT / (args.data_path or cfg["data_path"])
    p1_path   = REPO_ROOT / (args.p1_path or cfg["p1_path"])
    ddpm_ckpt = REPO_ROOT / (args.ddpm_ckpt or cfg["ddpm_ckpt"])

    for path, label in [(data_path, "data"),
                        (p1_path,   "VQ-GAN"),
                        (ddpm_ckpt, "DDPM checkpoint")]:
        if not path.exists():
            sys.exit(f"ERROR: missing {label}: {path}")

    print(f"Data:      {data_path}")
    print(f"VQ-GAN:    {p1_path}")
    print(f"DDPM ckpt: {ddpm_ckpt}")

    # ----- Load models -----
    _, dec, _, _ = load_vqgan(p1_path, dev)
    dec.eval()
    unet = load_unet(ddpm_ckpt, dev, num_classes=args.num_classes)
    unet.eval()

    # ----- Load real volumes -----
    d = torch.load(data_path, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols = d.get("volumes", d.get("v"))
    else:
        vols = d
    if vols.dim() == 5:
        vols = vols.squeeze(1)
    real = vols.float().numpy()  # (N, 64, 64, 64)
    print(f"Loaded {len(real)} real volumes")

    rng = np.random.default_rng(0)
    real_sample = real[rng.choice(len(real), size=min(64, len(real)), replace=False)]
    real_diversity = compute_diversity(real_sample)
    print(f"Real data diversity (n={len(real_sample)}): {real_diversity:.4f}")

    # ----- 5-seed DDIM-50 eval -----
    per_seed = {"ssim": [], "psnr": [], "diversity": [], "time_per_vol": []}

    print(f"\nRunning DDIM-50 on {args.dataset}, {args.n_samples} samples × {len(SEEDS)} seeds")
    for seed in SEEDS:
        torch.manual_seed(seed)
        if dev.type == "mps":
            torch.mps.manual_seed(seed)
        elif dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        t0 = time.time()
        gen = generate_ddim50_volumes(unet, dec, args.n_samples, dev,
                                       class_label=args.class_label)
        elapsed = time.time() - t0

        gen_np = gen[:, 0].numpy()
        n_cmp = min(args.n_samples, len(real_sample))
        ssims = [ssim3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
        psnrs = [psnr3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
        div   = compute_diversity(gen_np)
        tpv   = elapsed / args.n_samples

        per_seed["ssim"].append(float(np.mean(ssims)))
        per_seed["psnr"].append(float(np.mean(psnrs)))
        per_seed["diversity"].append(float(div))
        per_seed["time_per_vol"].append(float(tpv))

        print(f"  seed={seed}: SSIM={np.mean(ssims):.4f}  PSNR={np.mean(psnrs):.2f}  "
              f"div={div:.4f}  ({tpv:.3f}s/vol, {elapsed:.0f}s total)")

    # ----- Aggregate -----
    block_50 = {
        "ssim":      bootstrap_ci(per_seed["ssim"]),
        "psnr":      bootstrap_ci(per_seed["psnr"]),
        "diversity": bootstrap_ci(per_seed["diversity"]),
        "time":      bootstrap_ci(per_seed["time_per_vol"]),
    }
    block_50["diversity_pct_real"] = (
        float(block_50["diversity"]["mean"] / real_diversity * 100.0)
        if real_diversity > 0 else None
    )

    # ----- Print headline -----
    s = block_50
    print("\n" + "=" * 64)
    print(f"  DDIM-50 — {args.dataset} — 5-seed headline")
    print("=" * 64)
    print(f"  SSIM      = {s['ssim']['mean']:.4f} ± {s['ssim']['std']:.4f}  "
          f"[{s['ssim']['ci95_lo']:.4f}, {s['ssim']['ci95_hi']:.4f}]")
    print(f"  PSNR      = {s['psnr']['mean']:.2f} ± {s['psnr']['std']:.2f}  dB")
    print(f"  Diversity = {s['diversity']['mean']:.4f} ± {s['diversity']['std']:.4f}  "
          f"({s['diversity_pct_real']:.1f}% of real)")
    print(f"  Time      = {s['time']['mean']:.3f} s/vol")

    # ----- Save -----
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_ddim50.json"
    payload = {
        "method": "ddim",
        "sampler": "DDIM (deterministic, eta=0)",
        "model_checkpoint": str(ddpm_ckpt),
        "note": ("Same trained DDPM checkpoint as the existing 'ddpm' row in "
                 "Tables 2/3; only the sampling procedure differs. DDIM eta=0 "
                 "gives the fair 50-step diffusion baseline."),
        "dataset": args.dataset,
        "dataset_label": cfg["label"],
        "n_samples": args.n_samples,
        "seeds": SEEDS,
        "real_diversity": real_diversity,
        "by_steps": {"50": block_50},
        "per_seed": per_seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
