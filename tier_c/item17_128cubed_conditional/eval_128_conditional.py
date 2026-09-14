#!/usr/bin/env python3
"""
Conditional 128³ evaluation: reproduce §5.4 Table 6 at higher resolution.

Loads the conditional Consistency and Shortcut models trained by
`train_pipeline_128_conditional.py` and evaluates them at NFE = 50 with
per-class sampling. Produces SSIM, PSNR, and pairwise-L1 diversity for
each method × class × seed combination.

The §5.4 conditional benchmark at 64³ (Table 6) is the comparison point;
this script lets us assert whether the Shortcut > Consistency diversity
ordering and the Consistency near-zero-variance signature both
replicate under class conditioning at 128³ resolution.

Usage:
    cd fast-sampling-mode-collapse-3d/
    python tier_c/item17_128cubed_conditional/eval_128_conditional.py

Estimated wall-clock on M-series: 30–60 minutes (decoding at 128³ dominates).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import (  # noqa: E402
    Encoder3D, Decoder3D, VectorQuantizer, DenoisingUNet3D,
)

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]
LATENT_SIZE = 16  # 128 / 8 = 16


# ============================================================
# Metrics (mirror eval_128.py)
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
    return float(((2 * mu_a * mu_b + C1) * (2 * s_ab + C2) /
                  ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))).mean())


def psnr3d(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 50.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def pairwise_diversity(samples):
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
    means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    means = np.sort(means)
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "ci95_lo": float(means[int(n_boot * alpha / 2)]),
            "ci95_hi": float(means[int(n_boot * (1 - alpha / 2))])}


# ============================================================
# Conditional sampling at 128³
# ============================================================

def _has_d_kwarg(model):
    import inspect
    try:
        return "d" in inspect.signature(model.forward).parameters
    except Exception:
        return False


@torch.no_grad()
def consistency_sample_128_cond(unet, dec, n, dev, steps, class_label,
                                 batch_size=2, latent_size=LATENT_SIZE):
    """Multi-step Consistency sampling with a fixed class label."""
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
        vols = dec(x_hat).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


@torch.no_grad()
def shortcut_sample_128_cond(unet, dec, n, dev, steps, class_label,
                              batch_size=2, latent_size=LATENT_SIZE):
    """Shortcut FM Euler integration at NFE = steps, fixed class label."""
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
        vols = dec(z).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/brats_conditional_128.pt",
                   help="128³ conditional BraTS dataset (volumes + labels).")
    p.add_argument("--ckpt-dir", default="results/paper4/brats_128cubed_conditional",
                   help="Directory containing phase1_shared.pt, fm/final.pt, "
                        "consistency/final.pt, shortcut/final.pt")
    p.add_argument("--output-dir",
                   default="results/paper4/brats_128cubed_conditional/paper_eval_5seed_conditional")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-class", type=int, default=32,
                   help="Number of samples per (method, class, seed).")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {dev}")

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load VQ-GAN -----
    p1_path = ckpt_dir / "phase1_shared.pt"
    if not p1_path.exists():
        raise FileNotFoundError(f"phase1_shared.pt missing: {p1_path}")
    p1 = torch.load(p1_path, map_location=dev, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(p1["enc"]); enc.eval()
    dec = Decoder3D(1, 8, 2).to(dev); dec.load_state_dict(p1["dec"]); dec.eval()
    vq = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(p1["vq"]); vq.eval()
    print(f"VQ-GAN loaded from {p1_path}")

    # ----- Load conditional dataset (for per-class real diversity baselines) -----
    raw = torch.load(args.data_path, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict) or "volumes" not in raw or "labels" not in raw:
        raise ValueError(
            f"{args.data_path} is not a conditional dataset. Re-run "
            f"prepare_conditional_128.py first.")
    vols_all = raw["volumes"]
    labels_all = raw["labels"]
    if vols_all.dim() == 5:
        vols_np_full = vols_all.squeeze(1).float().numpy()
    else:
        vols_np_full = vols_all.float().numpy()
    print(f"Loaded {len(vols_np_full)} real conditional volumes "
          f"with class counts: "
          f"{[int((labels_all == c).sum().item()) for c in range(args.num_classes)]}")

    # Real-data per-class diversity baselines (n=args.n_per_class random per class)
    real_baselines = {}
    rng = np.random.default_rng(0)
    for c in range(args.num_classes):
        idx_c = np.where(labels_all.numpy() == c)[0]
        chosen = rng.choice(idx_c, size=min(args.n_per_class, len(idx_c)), replace=False)
        real_baselines[c] = {
            "vols": vols_np_full[chosen],
            "diversity": pairwise_diversity(vols_np_full[chosen]),
        }
        print(f"  Real class {c}: diversity baseline (n={len(chosen)}) "
              f"= {real_baselines[c]['diversity']:.4f}")

    # ----- Load conditional models -----
    method_ckpts = {
        "consistency": ckpt_dir / "consistency" / "final.pt",
        "shortcut":    ckpt_dir / "shortcut"    / "final.pt",
    }
    samplers = {
        "consistency": consistency_sample_128_cond,
        "shortcut":    shortcut_sample_128_cond,
    }

    results = {
        "metadata": {
            "seeds": SEEDS,
            "n_per_class": args.n_per_class,
            "steps": args.steps,
            "num_classes": args.num_classes,
            "real_diversity_per_class": {c: real_baselines[c]["diversity"]
                                          for c in real_baselines},
            "resolution": 128,
            "latent_size": LATENT_SIZE,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
        "results": {},  # results[method][class]["per_seed"]["ssim"|"diversity"|...]
    }

    for method, ckpt_path in method_ckpts.items():
        if not ckpt_path.exists():
            print(f"Skipping {method}: checkpoint not found at {ckpt_path}")
            continue
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
        nc = state.get("num_classes", args.num_classes)
        unet = DenoisingUNet3D(ch=8, num_classes=nc).to(dev)
        unet.load_state_dict(state["unet"] if "unet" in state else state)
        unet.eval()
        sampler = samplers[method]

        method_results = {}
        for c in range(args.num_classes):
            print(f"\n----- Evaluating {method}@{args.steps} | class {c} -----")
            real_c = real_baselines[c]["vols"]
            real_div_c = real_baselines[c]["diversity"]

            per_seed = {"ssim": [], "psnr": [], "diversity": [], "time_per_vol": []}
            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed)
                t0 = time.time()
                gen = sampler(unet, dec, args.n_per_class, dev, args.steps,
                              class_label=c, batch_size=2)
                time_per_vol = (time.time() - t0) / args.n_per_class
                gen_np = gen.squeeze(1) if gen.ndim == 5 else gen

                n_compare = min(args.n_per_class, len(real_c))
                ssims = [ssim3d(real_c[i % len(real_c)], gen_np[i]) for i in range(n_compare)]
                psnrs = [psnr3d(real_c[i % len(real_c)], gen_np[i]) for i in range(n_compare)]
                div_c_seed = pairwise_diversity(gen_np)

                per_seed["ssim"].append(float(np.mean(ssims)))
                per_seed["psnr"].append(float(np.mean(psnrs)))
                per_seed["diversity"].append(float(div_c_seed))
                per_seed["time_per_vol"].append(float(time_per_vol))
                print(f"    seed={seed}: SSIM={np.mean(ssims):.4f}  "
                      f"div={div_c_seed:.4f}  ({time_per_vol:.3f} s/vol)")

            method_results[f"class_{c}"] = {
                "per_seed": per_seed,
                "ssim": bootstrap_ci(per_seed["ssim"]),
                "psnr": bootstrap_ci(per_seed["psnr"]),
                "diversity": bootstrap_ci(per_seed["diversity"]),
                "real_diversity_baseline": real_div_c,
                "diversity_pct_real":
                    float(np.mean(per_seed["diversity"]) / real_div_c * 100.0)
                    if real_div_c > 0 else None,
            }
            s = method_results[f"class_{c}"]
            print(f"  SUMMARY {method} class {c}: SSIM={s['ssim']['mean']:.4f}  "
                  f"div={s['diversity']['mean']:.4f}  "
                  f"({s['diversity_pct_real']:.1f}% of real)")

        results["results"][method] = method_results

    out_path = out_dir / "eval_results_128_conditional_5seed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ----- Summary table -----
    print("\n" + "=" * 80)
    print(f"  Conditional 128³ summary (NFE = {args.steps}, "
          f"n = {args.n_per_class}/class/seed × {len(SEEDS)} seeds)")
    print("=" * 80)
    print(f"{'Method':>13} {'Class':>6} {'SSIM':>16} {'Diversity':>16} {'%Real':>7} "
          f"{'CV (std/mean)':>14}")
    for method in ["consistency", "shortcut"]:
        if method not in results["results"]:
            continue
        for c in range(args.num_classes):
            key = f"class_{c}"
            s = results["results"][method][key]
            ssim_str = f"{s['ssim']['mean']:.3f}±{s['ssim']['std']:.3f}"
            div_str = f"{s['diversity']['mean']:.4f}±{s['diversity']['std']:.4f}"
            pct = (f"{s['diversity_pct_real']:.0f}%"
                   if s['diversity_pct_real'] is not None else "-")
            cv = (s['diversity']['std'] / s['diversity']['mean']
                  if s['diversity']['mean'] > 0 else 0.0)
            print(f"{method:>13} {c:>6} {ssim_str:>16} {div_str:>16} {pct:>7} "
                  f"{cv:>14.4f}")

    print("\nWhat to look for:")
    print("  * Consistency class 0 vs class 1 diversity should be SYMMETRIC")
    print("    (within seed std). Asymmetry would falsify P4.")
    print("  * Shortcut diversity should be MUCH higher than Consistency diversity")
    print("    at every class. This replicates the §5.4 ordering.")
    print("  * Consistency cross-seed std on diversity should be near-zero")
    print("    (10^-4 to 10^-3 range), reproducing the collapse signature.")


if __name__ == "__main__":
    main()
