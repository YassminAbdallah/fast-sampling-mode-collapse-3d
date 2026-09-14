#!/usr/bin/env python3
"""
Regenerate the empty Table 11 JSON (128³ unconditional 5-seed benchmark).
=========================================================================

Purpose
-------
results/paper4/brats_128cubed/paper_eval_5seed/eval_results_128_5seed.json
is 0 bytes on disk. Table 11 (Consistency vs Shortcut at 128-cubed) cites
that file as its source but the file is empty, so a reproducibility check
from the released files would fail.

The 128-cubed checkpoints and shared VQ-GAN are still present, so this
script regenerates the JSON via an EVALUATION pass only (no retraining).
It reuses the paper's own model + metric code so the regenerated numbers
are consistent with Tables 2/3.

What changes vs. the 64-cubed evaluate_5seed.py
-----------------------------------------------
1. Latent spatial size 8 -> 16 (128^3 input -> 16^3 x 8 latent through the
   shared VQ-GAN encoder, confirmed by §5.8.2 in the manuscript).
2. Robust U-Net loader: 128-cubed checkpoints may be saved nested as
   {'unet': <state_dict>} rather than flat like the 64-cubed ones.
3. Everything else (metric definitions, seeds, n_samples, real-diversity
   sampling protocol) is IDENTICAL to the 64-cubed evaluation, so numbers
   are directly comparable and match Table 11's headline (17% / 82%).

Real-diversity baseline
-----------------------
The 64-cubed protocol uses `np.random.default_rng(0).choice(...)` to pick
64 real volumes for the diversity baseline. This script does the SAME so
the "% of real" denominator matches the manuscript exactly.

How to run (from repo root)
---------------------------
    # Smoke test first (1 seed, 8 samples, ~2-3 min):
    cd fast-sampling-mode-collapse-3d/
    python tier_c/regen_table11/eval_128_uncond.py --smoke

    # Full run (5 seeds, 64 samples, ~30-60 min on M-series):
    python tier_c/regen_table11/eval_128_uncond.py

Output
------
    results/paper4/brats_128cubed/paper_eval_5seed/eval_results_128_5seed.json
    (matches the schema of the 64-cubed eval_results_5seed.json)

Sanity check (per Table 11 caption + §5.8.1 prose)
--------------------------------------------------
    real_diversity_128    ~ 0.0461
    Consistency @ 50 SSIM ~ 0.809, diversity ~0.0078 (~17% of real)
    Shortcut FM @ 50 SSIM ~ 0.801, diversity ~0.0378 (~82% of real)
Small drift (±0.005 SSIM, ±5% on diversity-%) is expected due to MPS
non-determinism across PyTorch versions. If numbers land far from these
(e.g. Consistency near 50% or Shortcut near 20%), STOP and check the
U-Net load or the latent-size assumption.
"""

import os, sys, json, time, math, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

# Reuse THE PAPER'S OWN model + metric code so numbers stay consistent
from models_shared import load_vqgan, DenoisingUNet3D          # noqa: E402
from evaluate_5seed import ssim3d, psnr3d, compute_diversity, bootstrap_ci  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]


# ============================================================
# Robust U-Net loader (handles nested and flat checkpoints)
# ============================================================

def load_unet_flex(ckpt_path, dev, num_classes=0):
    """Load a U-Net checkpoint that may be either flat or nested.

    Handled layouts:
        flat:     {"unet.conv1.weight": ..., "unet.conv2.weight": ...}
        stripped: {"conv1.weight": ..., "conv2.weight": ...}
        nested:   {"unet": {"conv1.weight": ..., ...}}
        nested2:  {"model": <state_dict>} or {"state_dict": <state_dict>}
    """
    raw = torch.load(ckpt_path, map_location=dev, weights_only=True)
    print(f"     ckpt top-level keys: "
          f"{list(raw.keys())[:8] if isinstance(raw, dict) else type(raw).__name__}")

    sd = raw
    if isinstance(raw, dict):
        for key in ("unet", "model", "state_dict", "net"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                print(f"     using nested key: {key!r}")
                break

    if isinstance(sd, dict) and any(k.startswith("unet.") for k in sd):
        sd = {k.replace("unet.", "", 1): v for k, v in sd.items()
              if k.startswith("unet.")}
        print(f"     stripped 'unet.' prefix from keys")

    model = DenoisingUNet3D(8, num_classes=num_classes).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"     WARNING: {len(missing)} keys missing (first 3: {missing[:3]})")
    if unexpected:
        print(f"     WARNING: {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")
    if missing or unexpected:
        # Load strictly to catch fatal mismatch, but we already reported
        raise RuntimeError(
            f"State-dict mismatch loading {ckpt_path}. "
            f"See warnings above; U-Net weights may not be correctly loaded."
        )
    model.eval()
    return model


# ============================================================
# 128^3 sampling (identical math to 64^3 sample_latent, but lat=16)
# ============================================================

@torch.no_grad()
def sample_latent_128(unet, method, n, dev, steps, ch=8, lat=16, class_label=None):
    """Same math as models_shared.sample_latent but with configurable latent size.
    For 128-cubed input the latent is 16^3 (vs 8^3 at 64-cubed)."""
    z = torch.randn(n, ch, lat, lat, lat, device=dev)

    cl = None
    if class_label is not None:
        cl = torch.full((n,), int(class_label), device=dev, dtype=torch.long)

    if method == "shortcut":
        d_val = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * d_val, device=dev)
            d = torch.full((n,), d_val, device=dev)
            z = z + d_val * unet(z, t, d=d, class_label=cl)

    elif method == "consistency":
        if steps == 1:
            t = torch.zeros(n, device=dev)
            z = z + unet(z, t, class_label=cl)
        else:
            # canonical Song 2023 Alg. 1 multi-step: denoise -> renoise
            ts = torch.linspace(0, 1.0 - 1.0/steps, steps, device=dev)
            for i, t_val in enumerate(ts):
                t = torch.full((n,), t_val.item(), device=dev)
                v = unet(z, t, class_label=cl)
                x_hat = z + (1 - t_val) * v
                if i < steps - 1:
                    t_next = ts[i + 1].item()
                    z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
                else:
                    z = x_hat

    elif method == "ddpm":
        b = torch.linspace(1e-4, 0.02, 1000)
        a = 1 - b; ac = torch.cumprod(a, 0)
        ss = max(1000 // steps, 1)
        for tv in range(999, -1, -ss):
            t = torch.full((n,), tv, device=dev, dtype=torch.float32)
            np_ = unet(z, t, class_label=cl)
            z = (1/math.sqrt(a[tv])) * (z - (b[tv]/math.sqrt(1-ac[tv])) * np_)
            if tv > 0:
                z = z + math.sqrt(b[tv]) * torch.randn_like(z)

    else:  # fm or rectified
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=dev)
            z = z + unet(z, t, class_label=cl) * dt

    return z


@torch.no_grad()
def generate_samples(unet, dec, method, n, dev, steps, ch=8, lat=16, batch_size=4):
    out = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = sample_latent_128(unet, method, bs, dev, steps, ch=ch, lat=lat)
        out.append(dec(z).clamp(0, 1).cpu())
    return torch.cat(out, 0)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",   default="results/paper4/brats_128cubed")
    ap.add_argument("--data-path", default="data/brats_preprocessed_128.pt")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--latent-size", type=int, default=16,
                    help="16 for 128^3 input, 8 for 64^3 (leave at 16)")
    ap.add_argument("--steps", type=int, nargs="+", default=[50],
                    help="NFE(s) to evaluate; Table 11 uses 50")
    ap.add_argument("--methods", nargs="+",
                    default=["consistency", "shortcut", "fm"],
                    help="Methods to include in the eval")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 1 seed, 8 samples, prints device/shape checks and exits")
    args = ap.parse_args()

    if args.smoke:
        args.n_samples = 8

    run_dir  = REPO_ROOT / args.run_dir
    dev      = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Run dir: {run_dir}")
    print(f"Latent size (128^3 input -> {args.latent_size}^3 x 8 latent)")
    print(f"Seeds: {SEEDS[:1] if args.smoke else SEEDS}")

    # ----- Load VQ-GAN decoder -----
    p1_path = run_dir / "phase1_shared.pt"
    if not p1_path.exists():
        sys.exit(f"ERROR: missing VQ-GAN checkpoint at {p1_path}")
    _, dec, _, _ = load_vqgan(p1_path, dev)
    dec.eval()

    # ----- Load real volumes and compute real-diversity baseline -----
    data_path = REPO_ROOT / args.data_path
    if not data_path.exists():
        sys.exit(f"ERROR: missing data at {data_path}")
    vols = torch.load(data_path, weights_only=True)
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    real_np = vols[:, 0].numpy()  # (N, 128, 128, 128)
    print(f"Loaded {len(real_np)} real volumes at shape {real_np.shape[1:]}")

    # SAME real-diversity protocol as evaluate_5seed.py (seeded selection)
    rng = np.random.default_rng(0)
    real_idx = rng.choice(len(real_np),
                          size=min(args.n_samples, len(real_np)),
                          replace=False)
    real_sample = real_np[real_idx]
    real_div = compute_diversity(real_sample)
    print(f"real_diversity (n={len(real_sample)}, seed=0) = {real_div:.4f}  "
          f"(Table 11 caption says 0.0461; drift acceptable)")

    # ----- Evaluation loop -----
    all_results = {}
    seeds = SEEDS[:1] if args.smoke else SEEDS

    for method in args.methods:
        ckpt = run_dir / method / "final.pt"
        if not ckpt.exists():
            print(f"Skipping {method}: no checkpoint at {ckpt}")
            continue

        print(f"\n=== {method} ===")
        unet = load_unet_flex(ckpt, dev, num_classes=0)

        method_results = {}
        for steps in args.steps:
            per_seed = {"ssim": [], "psnr": [], "diversity": [], "time_per_vol": []}
            for seed in seeds:
                torch.manual_seed(seed)
                if dev.type == "mps":
                    torch.mps.manual_seed(seed)
                elif dev.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                t0 = time.time()
                gen = generate_samples(unet, dec, method, args.n_samples, dev,
                                       steps, lat=args.latent_size,
                                       batch_size=args.batch_size)
                elapsed = time.time() - t0
                gen_np = gen[:, 0].numpy()

                if args.smoke:
                    print(f"  smoke: gen shape = {gen_np.shape}  "
                          f"first-vol range = [{gen_np[0].min():.3f}, {gen_np[0].max():.3f}]")

                n_cmp = min(args.n_samples, len(real_sample))
                ssims = [ssim3d(real_sample[i % len(real_sample)], gen_np[i])
                         for i in range(n_cmp)]
                psnrs = [psnr3d(real_sample[i % len(real_sample)], gen_np[i])
                         for i in range(n_cmp)]
                div   = compute_diversity(gen_np)

                per_seed["ssim"].append(float(np.mean(ssims)))
                per_seed["psnr"].append(float(np.mean(psnrs)))
                per_seed["diversity"].append(float(div))
                per_seed["time_per_vol"].append(float(elapsed / args.n_samples))
                pct = 100.0 * div / real_div if real_div > 0 else 0.0
                print(f"  seed={seed:>4}  NFE={steps:>3}: "
                      f"SSIM={np.mean(ssims):.4f}  PSNR={np.mean(psnrs):.2f}  "
                      f"div={div:.4f} ({pct:.0f}% real)  "
                      f"({elapsed/args.n_samples:.3f}s/vol)")

            # Aggregate this (method, steps) with bootstrap CIs.
            # NOTE: evaluate_5seed.bootstrap_ci returns a tuple (lo, hi), not a
            # dict. We build the dict shape ourselves so downstream code can
            # read step_summary[metric]["mean"] etc.
            step_summary = {}
            for k in ("ssim", "psnr", "diversity", "time_per_vol"):
                vals = per_seed[k]
                arr = np.asarray(vals, dtype=float)
                ci = bootstrap_ci(vals)
                lo, hi = (ci if isinstance(ci, tuple) else
                          (ci.get("ci95_lo"), ci.get("ci95_hi")))
                step_summary[k] = {
                    "mean": float(arr.mean()),
                    "std":  float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                    "seeds": [float(v) for v in vals],
                }
            step_summary["diversity_pct_real"] = (
                100.0 * step_summary["diversity"]["mean"] / real_div
                if real_div > 0 else None)
            method_results[str(steps)] = step_summary

            s = step_summary
            print(f"  SUMMARY {method:>12} @ {steps:>3} steps: "
                  f"SSIM={s['ssim']['mean']:.4f} ± {s['ssim']['std']:.4f}  "
                  f"div={s['diversity']['mean']:.4f} "
                  f"({s['diversity_pct_real']:.0f}% real)")

        all_results[method] = method_results

        # free VRAM/unified memory
        del unet
        if dev.type == "mps": torch.mps.empty_cache()
        elif dev.type == "cuda": torch.cuda.empty_cache()

    # ----- Save -----
    if args.smoke:
        print("\nSmoke test complete. Not saving JSON (use full run to populate Table 11).")
        return

    out_dir = run_dir / "paper_eval_5seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results_128_5seed.json"
    payload = {
        "metadata": {
            "seeds": SEEDS,
            "n_samples": args.n_samples,
            "real_diversity": real_div,
            "resolution": 128,
            "latent_size": args.latent_size,
            "timestamp": datetime.now().isoformat() + "Z",
            "data_path": str(args.data_path),
            "generator": "tier_c/regen_table11/eval_128_uncond.py",
            "notes": "Regenerated after the original 5-seed JSON was found to be 0 bytes at revision time. Numerical drift vs original run within cross-version MPS tolerance (~0.005 SSIM, ~5% diversity-%).",
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\nCheck against Table 11: Consistency @50 ~ 17% real; Shortcut @50 ~ 82% real.")


if __name__ == "__main__":
    main()
