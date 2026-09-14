#!/usr/bin/env python3
"""
VQ-GAN reconstruction ceiling for Tables 2 and 3.
==================================================

Every method inherits the shared VQ-GAN's reconstruction fidelity as an
upper bound, so the reconstruction ceiling (real -> encode -> decode
SSIM/PSNR) is reported alongside the generator rows.

This script computes the VQ-GAN reconstruction ceiling on both datasets
and reports TWO SSIM/PSNR flavours:

  (a) "paired"   - reconstruction vs its own original
                   ("true" fidelity of the VQ-GAN encode-decode round-trip)
  (b) "random"   - reconstruction vs a random real (paper's Tables 2/3
                   protocol; directly comparable to generator rows)

Plus diversity of the reconstruction pool (pairwise-L1) to add as the
Diversity column of a "VQ-GAN reconstruction (ceiling)" row.

Why both flavours: interpretation (a) shows the encoder-decoder is not the
bottleneck; interpretation (b) shows what fraction of Tables 2/3's SSIM
is bounded by the random-pair protocol itself.

Protocol matches evaluate_5seed.py exactly (5 seeds, 64 samples/seed).

Outputs:
  results/paper4/vqgan_ceiling/{ixi,brats}_ceiling.json
Runtime: ~5-15 min per dataset on M-series.

Usage:
  cd fast-sampling-mode-collapse-3d/
  python tier_c/vqgan_ceiling/eval_vqgan_ceiling.py --smoke        # ~1 min
  python tier_c/vqgan_ceiling/eval_vqgan_ceiling.py                # full run
  python tier_c/vqgan_ceiling/eval_vqgan_ceiling.py --dataset brats  # just one
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_vqgan  # noqa: E402
from evaluate_5seed import ssim3d, psnr3d, compute_diversity  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]

# Dataset config
DATASETS = {
    "ixi": {
        "data":  "data/ixi_preprocessed_64.pt",
        "p1":    "results/paper3/ixi_benchmark_20260221_045741/100pct_200vol/phase1_shared.pt",
        "real_div_expected": 0.0838,  # from Table 2 caption
    },
    "brats": {
        "data":  "data/brats_preprocessed_64.pt",
        "p1":    "results/paper3/brats_benchmark_20260221_193306/100pct_300vol/phase1_shared.pt",
        "real_div_expected": 0.0461,  # from BraTS 5-seed baseline (0.0500 at 5-seed; 0.0461 at 64³ real data — Tables 2/3 note the baseline directly)
    },
}


@torch.no_grad()
def reconstruct_one_batch(enc, dec, vq, x, dev):
    """Encode + quantize + decode a batch, return decoded output on CPU."""
    x = x.to(dev)
    z = enc(x)          # (B, 8, 8, 8, 8) latent
    zq, _, _ = vq(z)    # quantized (same shape)
    xr = dec(zq).clamp(0, 1).cpu()
    return xr


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([arr[rng.integers(0, len(arr), size=len(arr))].mean()
                     for _ in range(n_boot)])
    boot.sort()
    return float(boot[int(n_boot * alpha / 2)]), float(boot[int(n_boot * (1 - alpha / 2))])


def evaluate_dataset(dataset, args, dev):
    cfg = DATASETS[dataset]
    print(f"\n============================================================")
    print(f"  Dataset: {dataset}")
    print(f"============================================================")
    data_path = REPO_ROOT / cfg["data"]
    p1_path   = REPO_ROOT / cfg["p1"]

    for p in (data_path, p1_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    print(f"  Data:   {data_path}")
    print(f"  VQ-GAN: {p1_path}")

    # Load VQ-GAN
    enc, dec, vq, _ = load_vqgan(p1_path, dev)

    # Load real data
    vols = torch.load(data_path, weights_only=True)
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    N = len(vols)
    real_np = vols[:, 0].numpy()  # (N, 64, 64, 64)
    print(f"  Loaded {N} real volumes, shape {real_np.shape[1:]}")

    seeds = SEEDS[:1] if args.smoke else SEEDS
    n_samples = 8 if args.smoke else args.n_samples

    per_seed = {"ssim_paired": [], "psnr_paired": [],
                "ssim_random": [], "psnr_random": [],
                "diversity":   []}

    for seed in seeds:
        torch.manual_seed(seed)
        if dev.type == "mps":  torch.mps.manual_seed(seed)
        elif dev.type == "cuda": torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Pick n_samples real volumes to reconstruct (seeded selection).
        # Matches evaluate_5seed protocol: np.random.seed(seed) then choose.
        # We use np.random.choice to be explicit and reproducible.
        idx = np.random.choice(N, size=min(n_samples, N), replace=False)
        picked = vols[idx].float()   # (n_samples, 1, 64, 64, 64)
        originals = picked[:, 0].numpy()  # (n_samples, 64, 64, 64)

        # Reconstruct in small batches to control memory
        t0 = time.time()
        recons = []
        bs = args.batch_size
        for start in range(0, len(picked), bs):
            xr = reconstruct_one_batch(enc, dec, vq,
                                        picked[start:start + bs], dev)
            recons.append(xr)
        recons = torch.cat(recons, dim=0)     # (n_samples, 1, 64, 64, 64)
        recons_np = recons[:, 0].numpy()
        elapsed = time.time() - t0

        # (a) Paired reconstruction fidelity
        ssim_p = [ssim3d(originals[i], recons_np[i]) for i in range(len(originals))]
        psnr_p = [psnr3d(originals[i], recons_np[i]) for i in range(len(originals))]

        # (b) Random-pair (matches Tables 2/3 protocol)
        # First n_samples real volumes as the paper does: real[i % len(real)]
        # We use the same idx-shuffled subset for fairness, but pair with a
        # DIFFERENT randomly ordered set of reals so it's truly random.
        rng2 = np.random.default_rng(seed + 1)
        rand_perm = rng2.permutation(N)[:len(recons_np)]
        rand_reals = real_np[rand_perm]
        ssim_r = [ssim3d(rand_reals[i], recons_np[i]) for i in range(len(recons_np))]
        psnr_r = [psnr3d(rand_reals[i], recons_np[i]) for i in range(len(recons_np))]

        # Diversity of the reconstructed pool
        div = compute_diversity(recons_np)

        per_seed["ssim_paired"].append(float(np.mean(ssim_p)))
        per_seed["psnr_paired"].append(float(np.mean(psnr_p)))
        per_seed["ssim_random"].append(float(np.mean(ssim_r)))
        per_seed["psnr_random"].append(float(np.mean(psnr_r)))
        per_seed["diversity"].append(float(div))

        print(f"  seed={seed:>4}: "
              f"paired SSIM={np.mean(ssim_p):.4f} PSNR={np.mean(psnr_p):.2f} | "
              f"random SSIM={np.mean(ssim_r):.4f} PSNR={np.mean(psnr_r):.2f} | "
              f"div={div:.4f} | {elapsed:.0f}s")

    # Real-diversity baseline (same protocol as evaluate_5seed: rng(0).choice)
    rng0 = np.random.default_rng(0)
    real_idx = rng0.choice(N, size=min(64, N), replace=False)
    real_div = compute_diversity(real_np[real_idx])

    # Aggregate
    summary = {}
    for k, vals in per_seed.items():
        arr = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_ci(vals)
        summary[k] = {
            "mean":    float(arr.mean()),
            "std":     float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "ci95_lo": lo, "ci95_hi": hi,
            "seeds":   [float(v) for v in vals],
        }
    summary["diversity_pct_real"] = (
        100.0 * summary["diversity"]["mean"] / real_div if real_div > 0 else None)
    summary["real_diversity"] = float(real_div)

    print(f"\n  SUMMARY {dataset}:")
    print(f"    Paired SSIM (VQ-GAN reconstruction fidelity):    "
          f"{summary['ssim_paired']['mean']:.4f} ± {summary['ssim_paired']['std']:.4f}")
    print(f"    Paired PSNR:                                     "
          f"{summary['psnr_paired']['mean']:.2f} ± {summary['psnr_paired']['std']:.2f}")
    print(f"    Random-pair SSIM (comparable to Tables 2/3):     "
          f"{summary['ssim_random']['mean']:.4f} ± {summary['ssim_random']['std']:.4f}")
    print(f"    Random-pair PSNR:                                "
          f"{summary['psnr_random']['mean']:.2f} ± {summary['psnr_random']['std']:.2f}")
    print(f"    Diversity of reconstruction pool:                "
          f"{summary['diversity']['mean']:.4f} ± {summary['diversity']['std']:.4f}"
          f" ({summary['diversity_pct_real']:.1f}% of real)")
    print(f"    real_diversity baseline (this run):              {real_div:.4f}")
    print(f"    (Reference: expected real_div ~ {cfg['real_div_expected']:.4f})")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ixi", "brats", "both"], default="both")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 8 samples, quick sanity check.")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    datasets = ["ixi", "brats"] if args.dataset == "both" else [args.dataset]

    all_results = {}
    for d in datasets:
        summary = evaluate_dataset(d, args, dev)
        all_results[d] = summary

    if args.smoke:
        print("\nSmoke test complete. Not writing JSON.")
        return

    out_dir = REPO_ROOT / "results/paper4/vqgan_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d, summary in all_results.items():
        out_path = out_dir / f"{d}_ceiling.json"
        payload = {
            "dataset": d,
            "n_samples": args.n_samples,
            "seeds": SEEDS,
            "timestamp": datetime.now().isoformat() + "Z",
            "generator": "tier_c/vqgan_ceiling/eval_vqgan_ceiling.py",
            "purpose": (
                "VQ-GAN reconstruction ceiling for Tables 2/3, "
                "'paired' = reconstruction vs its own original (true fidelity). "
                "'random' = reconstruction vs random-paired real (paper's protocol). "
                "Diversity = pairwise-L1 among reconstructed pool."
            ),
            "results": summary,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
