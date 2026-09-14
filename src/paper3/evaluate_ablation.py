#!/usr/bin/env python3
"""
Evaluate Ablation Variants
===========================

After running BraTS ablation (EXP 3), evaluates all shortcut ablation
variants and produces a summary table.

Usage:
    python evaluate_ablation.py --run-dir results/brats_benchmark_20260221_193306 --data-path data/brats_preprocessed_64.pt

Output:
    {run-dir}/ablation_results/
        ablation_summary.json
        ablation_table.txt
"""

import os, json, argparse, warnings
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

from models_shared import DenoisingUNet3D, Decoder3D, load_unet, load_vqgan, sample_latent


def ssim3d_windowed(a, b, sigma=1.5):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a = gaussian_filter(a, sigma); mu_b = gaussian_filter(b, sigma)
    sig_a2 = gaussian_filter(a**2, sigma) - mu_a**2
    sig_b2 = gaussian_filter(b**2, sigma) - mu_b**2
    sig_ab = gaussian_filter(a*b, sigma) - mu_a*mu_b
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_a*mu_b+C1)*(2*sig_ab+C2)) / ((mu_a**2+mu_b**2+C1)*(sig_a2+sig_b2+C2))
    return float(ssim_map.mean())


def evaluate_variant(unet, dec, vols, dev, n_samples=64, seeds=[42, 123, 456]):
    step_counts = [1, 4, 16, 128]
    results = {}
    for steps in step_counts:
        ssims_all = []; divs_all = []
        for seed in seeds:
            torch.manual_seed(seed)
            with torch.no_grad():
                z = sample_latent(unet, "shortcut", n_samples, dev, steps)
                gen = dec(z).clamp(0, 1)
            gen_np = gen[:, 0].cpu().numpy()
            real_np = vols[:n_samples, 0].numpy() if vols.dim() == 5 else vols[:n_samples].numpy()
            n = min(len(real_np), len(gen_np))
            ssims_all.append(np.mean([ssim3d_windowed(real_np[i], gen_np[i]) for i in range(n)]))
            flat = gen_np.reshape(gen_np.shape[0], -1)
            dists = [np.mean(np.abs(flat[i]-flat[j])) for i in range(len(flat)) for j in range(i+1, len(flat))]
            divs_all.append(np.mean(dists))
        results[steps] = {
            "ssim_mean": float(np.mean(ssims_all)), "ssim_std": float(np.std(ssims_all)),
            "diversity_mean": float(np.mean(divs_all)), "diversity_std": float(np.std(divs_all)),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and "pct_" in d.name])
    data_dir = sub[0] if sub else run_dir

    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

    _, dec, _, _ = load_vqgan(data_dir / "phase1_shared.pt", dev)

    vols = torch.load(args.data_path, weights_only=True)
    if vols.dim() == 4: vols = vols.unsqueeze(1)

    # Real diversity
    real_np = vols[:64, 0].numpy() if vols.dim() == 5 else vols[:64].numpy()
    flat = real_np.reshape(real_np.shape[0], -1)
    real_div = np.mean([np.mean(np.abs(flat[i]-flat[j])) for i in range(len(flat)) for j in range(i+1, len(flat))])
    print(f"Real diversity: {real_div:.4f}")

    # Find ablation dirs
    ablation_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("shortcut_")])
    base = data_dir / "shortcut"
    if base.exists(): ablation_dirs = [base] + ablation_dirs

    all_results = {}
    out_dir = data_dir / "ablation_results"; out_dir.mkdir(exist_ok=True)

    for adir in ablation_dirs:
        ckpt = adir / "final.pt"
        if not ckpt.exists(): print(f"Skipping {adir.name}"); continue
        name = adir.name if adir.name != "shortcut" else "shortcut (full, SC=0.25)"
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        unet = load_unet(ckpt, dev)
        results = evaluate_variant(unet, dec, vols, dev, n_samples=args.n_samples)
        all_results[name] = results
        del unet
        for s, m in results.items():
            print(f"  @{s:3d}: SSIM={m['ssim_mean']:.4f} Div={m['diversity_mean']:.4f} ({m['diversity_mean']/real_div*100:.0f}%Real)")

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    lines = [f"{'Variant':<35} | {'1-step SSIM':>11} | {'1-step %Real':>12} | {'128-step %Real':>14}"]
    lines.append("-" * 80)
    for name, r in all_results.items():
        s1, s128 = r.get(1, {}), r.get(128, {})
        lines.append(f"{name:<35} | {s1.get('ssim_mean',0):.4f}      | "
                     f"{s1.get('diversity_mean',0)/real_div*100:.0f}%          | "
                     f"{s128.get('diversity_mean',0)/real_div*100:.0f}%")
    table = "\n".join(lines)
    print(f"\n{table}")
    with open(out_dir / "ablation_table.txt", "w") as f: f.write(table)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
