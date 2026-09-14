#!/usr/bin/env python3
"""A1 UNCONDITIONAL eval — same design as eval_replicated_seeds.py but
scoped to Table 3 (unconditional BraTS). Class_label=None throughout."""

import os, sys, json, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_vqgan, load_unet, sample_latent           # noqa: E402
from evaluate_5seed import ssim3d, psnr3d, compute_diversity, SEEDS       # noqa: E402

warnings.filterwarnings("ignore")

BENCH_DIR = REPO_ROOT / "results/paper3/brats_benchmark_20260221_193306/100pct_300vol"
REPLICA   = REPO_ROOT / "results/paper4/train_seed_replication_uncond"
DATA_PATH = REPO_ROOT / "data/brats_preprocessed_64.pt"

DEFAULT_MODELS = [
    ("consistency", "original", BENCH_DIR / "consistency/final.pt", "implicit"),
    ("consistency", "seed100",  REPLICA / "consistency_trainseed100/final.pt", 100),
    ("consistency", "seed200",  REPLICA / "consistency_trainseed200/final.pt", 200),
    ("shortcut",    "original", BENCH_DIR / "shortcut/final.pt", "implicit"),
    ("shortcut",    "seed100",  REPLICA / "shortcut_trainseed100/final.pt", 100),
    ("shortcut",    "seed200",  REPLICA / "shortcut_trainseed200/final.pt", 200),
]

STEPS_FOR = {"consistency": [4, 50], "shortcut": [4, 50]}


@torch.no_grad()
def generate(unet, dec, method, n, dev, steps, ch=8, batch_size=8):
    """Unconditional sampling (class_label=None)."""
    out = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = sample_latent(unet, method, bs, dev, steps, ch, class_label=None)
        out.append(dec(z).clamp(0, 1).cpu())
    return torch.cat(out, 0)


def eval_one_model(method, ckpt_path, real_np, real_sample, real_div, dev,
                    steps_list, n_samples=64):
    print(f"\n  Loading {method}: {ckpt_path}")
    if not ckpt_path.exists():
        print(f"    MISSING (skip)")
        return None

    unet = load_unet(ckpt_path, dev, num_classes=0)  # unconditional
    unet.eval()
    _, dec, _, _ = load_vqgan(BENCH_DIR / "phase1_shared.pt", dev)
    dec.eval()

    results = {}
    for steps in steps_list:
        per_seed = {"ssim": [], "psnr": [], "diversity": []}
        for seed in SEEDS:
            torch.manual_seed(seed)
            if dev.type == "mps":  torch.mps.manual_seed(seed)
            elif dev.type == "cuda": torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            gen = generate(unet, dec, method, n_samples, dev, steps)
            gen_np = gen[:, 0].numpy()

            n_cmp = min(n_samples, len(real_sample))
            ssims = [ssim3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
            psnrs = [psnr3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
            div = compute_diversity(gen_np)
            per_seed["ssim"].append(float(np.mean(ssims)))
            per_seed["psnr"].append(float(np.mean(psnrs)))
            per_seed["diversity"].append(float(div))

        pct_real = 100.0 * np.mean(per_seed["diversity"]) / real_div
        results[str(steps)] = {
            "ssim_mean": float(np.mean(per_seed["ssim"])),
            "ssim_std":  float(np.std(per_seed["ssim"], ddof=1)),
            "psnr_mean": float(np.mean(per_seed["psnr"])),
            "psnr_std":  float(np.std(per_seed["psnr"], ddof=1)),
            "diversity_mean": float(np.mean(per_seed["diversity"])),
            "diversity_std":  float(np.std(per_seed["diversity"], ddof=1)),
            "diversity_pct_real": float(pct_real),
            "per_seed": per_seed,
        }
        r = results[str(steps)]
        print(f"    NFE={steps:>3}: SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f}  "
              f"div={r['diversity_mean']:.4f} ({pct_real:.1f}% real)")

    del unet, dec
    if dev.type == "mps":  torch.mps.empty_cache()
    elif dev.type == "cuda": torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="results/paper4/train_seed_replication_uncond/a1_eval_summary_uncond.json")
    ap.add_argument("--n-samples", type=int, default=64)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else
                       "cpu")
    print(f"Device: {dev}")

    d = torch.load(DATA_PATH, weights_only=False, map_location="cpu")
    vols = d.get("volumes", d.get("v")) if isinstance(d, dict) else d
    if vols.dim() == 4: vols = vols.unsqueeze(1)
    real_np = vols[:, 0].numpy()
    rng = np.random.default_rng(0)
    real_sample = real_np[rng.choice(len(real_np), size=min(args.n_samples, len(real_np)), replace=False)]
    real_div = compute_diversity(real_sample)
    print(f"real_diversity (unconditional, n={len(real_sample)}) = {real_div:.4f}")

    all_results = {}
    for method, label, ckpt, train_seed_note in DEFAULT_MODELS:
        print(f"\n{'='*60}\n  {method} — {label}  (train_seed: {train_seed_note})\n{'='*60}")
        r = eval_one_model(method, ckpt, real_np, real_sample, real_div, dev,
                           STEPS_FOR[method], n_samples=args.n_samples)
        if r is None: continue
        all_results.setdefault(method, {})[label] = {
            "train_seed": train_seed_note,
            "checkpoint": str(ckpt),
            **r,
        }

    print("\n" + "=" * 70)
    print("A1 UNCONDITIONAL headline: cross-training-seed variance per method")
    print("=" * 70)
    a1_summary = {}
    for method, per_label in all_results.items():
        for steps in STEPS_FOR[method]:
            step_key = str(steps)
            pcts = [per_label[l][step_key]["diversity_pct_real"]
                    for l in per_label if step_key in per_label[l]]
            if len(pcts) >= 2:
                cross_std = float(np.std(pcts, ddof=1))
                cross_mean = float(np.mean(pcts))
                sampling_stds = [per_label[l][step_key]["diversity_std"]
                                 for l in per_label if step_key in per_label[l]]
                mean_sampling_std_pct = 100.0 * float(np.mean(sampling_stds)) / real_div
                key = f"{method}@{steps}"
                a1_summary[key] = {
                    "n_training_seeds": len(pcts),
                    "cross_train_seed_mean_pct_real": cross_mean,
                    "cross_train_seed_std_pct_real":  cross_std,
                    "mean_within_train_seed_sampling_std_pct_real": mean_sampling_std_pct,
                }
                print(f"  {key}: mean {cross_mean:.1f}% real  "
                      f"cross-train-seed std {cross_std:.1f} pp  "
                      f"(vs within-train sampling std {mean_sampling_std_pct:.2f} pp)")

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "conditional": False,
            "real_diversity": real_div,
            "n_samples": args.n_samples,
            "sampling_seeds": list(SEEDS),
            "per_model": all_results,
            "a1_cross_training_seed_summary": a1_summary,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
