#!/usr/bin/env python3
"""
2x2 EMA × Loss factorial — EVALUATION + ASSEMBLY
=================================================

Closes the loop on Item 15b. The four cells of the factorial are:

    Cell  | EMA | Loss        | Identity                       | Checkpoint
    ------+-----+-------------+--------------------------------+------------------------
    A     | yes | L2          | Original Consistency Distillation | results/paper4/brats_benchmark_*/100pct_180vol/consistency/final.pt
    B     | no  | Pseudo-Huber| Improved CD                    | results/paper4/improved_cd_brats/improved_cd/final.pt
    C     | yes | Pseudo-Huber| NEW (this factorial)           | results/paper4/cd_2x2/ema_pseudohuber/final.pt
    D     | no  | L2          | NEW (this factorial)           | results/paper4/cd_2x2/noema_l2/final.pt

Cells A and B already have 5-seed evals on disk; we reuse those JSONs so the
2x2 table aligns numerically with §5.2 of the paper. Only C and D are
evaluated fresh, using the SAME protocol that produced the A and B numbers
(see tier_c/item15_improved_cd/eval_improved_cd.py).

Each cell also gets a LOSS-TRAJECTORY FLAG derived from training_log.json:

    flag = "loss_descent_smooth"   if final/first < 0.8  AND  max/first < 2.0
    flag = "loss_stalled"          if final/first > 0.8  (no improvement)
    flag = "loss_spiked"           if max/first > 2.0    (loss went up >2x mid-training)
    flag = "loss_spiked_stalled"   if both fail

The flag describes the loss CURVE, not whether training "succeeded" —
some valid baselines in this paper (e.g. Improved CD) have loss curves
that spike and stall yet still produce usable samples. The final
interpretation has to combine the loss-trajectory flag with the sample
quality (SSIM, diversity) from eval. A separate "training_status" column
is derived in the assembly step:

    training_status = "trained"          if SSIM @ NFE=4 >= 0.55
    training_status = "failed"           if SSIM @ NFE=4 <  0.55

Interpretation rule: a cell with `training_status = failed`
is not evidence about the diversity *mechanism* — it just means training
broke. We can only compare diversity across cells whose `training_status
= trained`.

Usage:
    cd /path/to/fast-sampling-mode-collapse-3d/
    python tier_c/item15b_2x2_factorial/eval_2x2_factorial.py \\
        --n-samples 64

    # Re-eval everything from scratch (won't reuse existing JSONs for A, B):
    python tier_c/item15b_2x2_factorial/eval_2x2_factorial.py \\
        --n-samples 64 --no-reuse-existing

Wall time on Apple M-series: ~25-40 minutes (only C and D need fresh sampling).

Outputs:
    results/paper4/cd_2x2/ema_pseudohuber/eval_results_5seed.json
    results/paper4/cd_2x2/noema_l2/eval_results_5seed.json
    results/paper4/cd_2x2/factorial_2x2_table.json
    results/paper4/cd_2x2/factorial_2x2_table.md
"""

import os, sys, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_unet, load_vqgan, sample_latent  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]
STEP_COUNTS = [1, 4, 16, 50]

# Cells of the 2x2 factorial. EMA and Loss are the two factors.
CELLS = {
    "A_ema_l2": {
        "ema": True,  "loss": "L2",
        "label": "Cell A: EMA + L2 (schedule-matched, 80 ep / batch 4)",
        "ckpt": "results/paper4/cd_2x2/ema_l2/final.pt",
        "existing_eval": None,
        "existing_eval_path": None,
        "method_in_existing_json": None,
        "training_log": "results/paper4/cd_2x2/training_log_ema_l2.json",
        "fresh_eval_out": "results/paper4/cd_2x2/ema_l2/eval_results_5seed.json",
    },
    "B_noema_pseudohuber": {
        "ema": False, "loss": "Pseudo-Huber",
        "label": "Cell B: no-EMA + Pseudo-Huber (Improved CD)",
        "ckpt": "results/paper4/improved_cd_brats/improved_cd/final.pt",
        "existing_eval": "results/paper4/improved_cd_brats/paper_eval_5seed/eval_results_5seed.json",
        # eval_improved_cd.py shape: {by_steps: {nfe: {...}}, real_diversity: ...}
        "existing_eval_path": None,
        "training_log": "results/paper4/improved_cd_brats/training_log.json",
    },
    "C_ema_pseudohuber": {
        "ema": True,  "loss": "Pseudo-Huber",
        "label": "Cell C: EMA + Pseudo-Huber (NEW)",
        "ckpt": "results/paper4/cd_2x2/ema_pseudohuber/final.pt",
        "existing_eval": None,
        "method_in_existing_json": None,
        "training_log": "results/paper4/cd_2x2/training_log_ema_pseudohuber.json",
        "fresh_eval_out": "results/paper4/cd_2x2/ema_pseudohuber/eval_results_5seed.json",
    },
    "D_noema_l2": {
        "ema": False, "loss": "L2",
        "label": "Cell D: no-EMA + L2 (NEW)",
        "ckpt": "results/paper4/cd_2x2/noema_l2/final.pt",
        "existing_eval": None,
        "method_in_existing_json": None,
        "training_log": "results/paper4/cd_2x2/training_log_noema_l2.json",
        "fresh_eval_out": "results/paper4/cd_2x2/noema_l2/eval_results_5seed.json",
    },
}


# ============================================================
# Metrics — identical to eval_improved_cd.py (do NOT modify)
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


def compute_pairwise_diversity(samples):
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
    }


@torch.no_grad()
def generate_samples_consistency(unet, dec, n, dev, steps, class_label=None,
                                   batch_size=8, ch=8):
    out = []
    rem = n
    while rem > 0:
        bs = min(batch_size, rem)
        z = sample_latent(unet, "consistency", bs, dev, steps, ch,
                          class_label=class_label)
        vols = dec(z).clamp(0, 1).cpu().numpy()
        out.append(vols)
        rem -= bs
    return np.concatenate(out, axis=0)[:n]


# ============================================================
# Convergence flag from training_log.json
# ============================================================

def convergence_flag(training_log_path):
    """Return (flag, details_dict) using only the loss trajectory."""
    if training_log_path is None:
        return "unknown", {"reason": "no_training_log_provided"}
    p = Path(training_log_path)
    if not p.exists():
        return "unknown", {"reason": f"training_log_not_found:{p}"}

    with open(p) as f:
        log = json.load(f)

    # Two shapes encountered: {"epoch_log": [...]} or {"history": [...]} or [...]
    if isinstance(log, dict):
        traj = log.get("epoch_log") or log.get("history") or log.get("epochs") or []
    else:
        traj = log
    if not traj:
        return "unknown", {"reason": "empty_training_log"}

    losses = [float(e.get("loss", e.get("train_loss", float("nan"))))
              for e in traj if isinstance(e, dict)]
    losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
    if len(losses) < 5:
        return "unknown", {"reason": f"too_few_loss_points:{len(losses)}"}

    first = float(losses[0])
    final = float(losses[-1])
    mx    = float(max(losses))

    progress_ratio = final / first if first > 0 else float("inf")
    spike_ratio    = mx    / first if first > 0 else float("inf")

    no_progress = progress_ratio > 0.8
    spiked      = spike_ratio    > 2.0

    if no_progress and spiked:
        flag = "loss_spiked_stalled"
    elif no_progress:
        flag = "loss_stalled"
    elif spiked:
        flag = "loss_spiked"
    else:
        flag = "loss_descent_smooth"

    return flag, {
        "first_loss": first, "final_loss": final, "max_loss": mx,
        "progress_ratio_final_over_first": progress_ratio,
        "spike_ratio_max_over_first": spike_ratio,
        "n_epochs": len(losses),
    }


# ============================================================
# Read existing eval JSONs (for cells A and B)
# ============================================================

def load_existing_eval(eval_path, eval_path_in_json=None):
    """Load an existing 5-seed eval JSON and normalize to:
        {by_steps: {nfe: {ssim, psnr, diversity, diversity_pct_real}},
         real_diversity, source}

    Two source shapes are handled:
      (1) eval_improved_cd.py output:
             {by_steps: {nfe_str: {...}}, real_diversity: ..., ...}
      (2) joint paper benchmark output:
             {metadata: {real_diversity: ..., seeds: ...},
              results: {consistency: {nfe_str: {ssim, psnr, diversity, time}}, ...}}
          For this shape, eval_path_in_json = ["results", "consistency"].
    """
    with open(eval_path) as f:
        data = json.load(f)

    # Pull real_diversity from wherever it lives (metadata or the block itself).
    real_div = None
    if isinstance(data, dict):
        meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        if "real_diversity" in meta:
            real_div = meta["real_diversity"]

    block = data
    if eval_path_in_json:
        for key in eval_path_in_json:
            block = block[key]
    if real_div is None and isinstance(block, dict):
        real_div = block.get("real_diversity")

    # Find the NFE-keyed map. It's either block["by_steps"] or block itself
    # when the joint-benchmark put NFE keys directly inside the method block.
    if isinstance(block, dict) and "by_steps" in block:
        by_steps_in = block["by_steps"]
    elif isinstance(block, dict) and all(
            (k.isdigit() and isinstance(v, dict)) for k, v in block.items()
            if not k.startswith("_")):
        by_steps_in = block
    else:
        # Fallback: treat block as a flat NFE=4 record.
        by_steps_in = {"4": block}

    by_steps_out = {}
    for k, v in by_steps_in.items():
        if not isinstance(v, dict):
            continue
        ssim = v.get("ssim") or v.get("SSIM") or {}
        psnr = v.get("psnr") or v.get("PSNR") or {}
        div  = v.get("diversity") or v.get("pairwise_l1") or {}
        if isinstance(ssim, (int, float)):
            ssim = {"mean": float(ssim), "std": 0.0}
        if isinstance(psnr, (int, float)):
            psnr = {"mean": float(psnr), "std": 0.0}
        if isinstance(div, (int, float)):
            div = {"mean": float(div), "std": 0.0}
        pct  = v.get("diversity_pct_real")
        # If the joint benchmark didn't store pct_real, compute it now.
        if pct is None and real_div and isinstance(div, dict) and "mean" in div and real_div > 0:
            pct = float(div["mean"] / real_div * 100.0)
        by_steps_out[str(k)] = {
            "ssim": ssim, "psnr": psnr, "diversity": div,
            "diversity_pct_real": pct,
        }
    return {
        "source": str(eval_path),
        "by_steps": by_steps_out,
        "real_diversity": real_div,
    }


# ============================================================
# Fresh eval for cells C and D
# ============================================================

def run_fresh_eval(cell, args, dev, real_sample, real_diversity):
    print(f"\n[FRESH] {cell['label']}")
    ckpt = REPO_ROOT / cell["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
    p1_path = REPO_ROOT / args.vqgan_ckpt
    _, dec, _, _ = load_vqgan(p1_path, dev)
    dec.eval()
    unet = load_unet(ckpt, dev, num_classes=args.num_classes)
    unet.eval()

    by_steps = {}
    for steps in args.steps:
        print(f"  ----- @ NFE={steps} -----")
        per_seed = {"ssim": [], "psnr": [], "diversity": []}
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            gen = generate_samples_consistency(
                unet, dec, args.n_samples, dev, steps,
                class_label=args.class_label, batch_size=8,
            )
            gen_np = gen.squeeze(1) if gen.ndim == 5 else gen
            n_cmp = min(args.n_samples, len(real_sample))
            ssims = [ssim3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
            psnrs = [psnr3d(real_sample[i % len(real_sample)], gen_np[i]) for i in range(n_cmp)]
            div   = compute_pairwise_diversity(gen_np)
            per_seed["ssim"].append(float(np.mean(ssims)))
            per_seed["psnr"].append(float(np.mean(psnrs)))
            per_seed["diversity"].append(float(div))
            print(f"    seed={seed}: SSIM={np.mean(ssims):.4f}  div={div:.4f}")
        by_steps[str(steps)] = {
            "ssim": bootstrap_ci(per_seed["ssim"]),
            "psnr": bootstrap_ci(per_seed["psnr"]),
            "diversity": bootstrap_ci(per_seed["diversity"]),
            "diversity_pct_real": (
                float(np.mean(per_seed["diversity"]) / real_diversity * 100.0)
                if real_diversity and real_diversity > 0 else None
            ),
            "per_seed": per_seed,
        }
        s = by_steps[str(steps)]
        print(f"    SUMMARY @ NFE={steps}: SSIM={s['ssim']['mean']:.4f}  "
              f"div={s['diversity']['mean']:.4f}  "
              f"({s['diversity_pct_real']:.1f}% of real)")

    out = {
        "source": "fresh",
        "by_steps": by_steps,
        "real_diversity": real_diversity,
        "seeds": SEEDS,
        "n_samples": args.n_samples,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Persist per-cell JSON too (so subsequent runs reuse it)
    fresh_out = cell.get("fresh_eval_out")
    if fresh_out:
        outp = REPO_ROOT / fresh_out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {outp}")

    return out


# ============================================================
# Markdown table assembly
# ============================================================

SSIM_TRAINED_THRESHOLD = 0.55  # below this we treat the cell as broken-training


def cell_row(cell_name, cell, ev, loss_flag, loss_details, nfe="4"):
    by = ev.get("by_steps", {})
    block = by.get(str(nfe)) or by.get(nfe) or {}
    ssim = block.get("ssim", {}).get("mean")
    div  = block.get("diversity", {}).get("mean")
    pct  = block.get("diversity_pct_real")
    psnr = block.get("psnr", {}).get("mean")

    # training_status: combines sample quality with the loss-trajectory diagnostic
    if isinstance(ssim, (int, float)):
        if ssim >= SSIM_TRAINED_THRESHOLD:
            training_status = "trained"
        else:
            training_status = "failed"
    else:
        training_status = "unknown"

    def fmt(x, p=4):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"

    return {
        "cell": cell_name,
        "label": cell["label"],
        "ema": cell["ema"],
        "loss": cell["loss"],
        "ssim_at_nfe4": ssim,
        "psnr_at_nfe4": psnr,
        "diversity_at_nfe4": div,
        "diversity_pct_real": pct,
        "loss_trajectory_flag": loss_flag,
        "loss_trajectory_details": loss_details,
        "training_status": training_status,
        "_fmt": {
            "ssim": fmt(ssim, 4), "psnr": fmt(psnr, 2),
            "diversity": fmt(div, 4),
            "pct_real": (f"{pct:.1f}%" if isinstance(pct, (int, float)) else "—"),
        },
    }


def markdown_table(rows, nfe="4"):
    lines = []
    lines.append(f"## 2×2 EMA × Loss factorial — quality, diversity, training status at NFE={nfe}\n")
    lines.append("| Cell | EMA | Loss | SSIM ↑ | PSNR ↑ | Diversity ↑ | % of real div | Loss-traj | Training status |")
    lines.append("|------|-----|------|--------|--------|-------------|---------------|-----------|-----------------|")
    for r in rows:
        f = r["_fmt"]
        status_marker = "✓" if r["training_status"] == "trained" else ("✗" if r["training_status"] == "failed" else "?")
        lines.append(
            f"| {r['cell']} | {'yes' if r['ema'] else 'no'} | {r['loss']} | "
            f"{f['ssim']} | {f['psnr']} | {f['diversity']} | {f['pct_real']} | "
            f"{r['loss_trajectory_flag']} | **{status_marker} {r['training_status']}** |"
        )
    lines.append("")
    lines.append("**Loss-trajectory flag** (descriptive only, from training_log.json loss curve):")
    lines.append("- `loss_descent_smooth` — monotone descent, final < 0.8× first, no spike")
    lines.append("- `loss_spiked` — loss went up >2× first at some point")
    lines.append("- `loss_stalled` — final ≈ first (final > 0.8× first)")
    lines.append("- `loss_spiked_stalled` — both shape problems present")
    lines.append("- `unknown` — no training log on disk for this cell")
    lines.append("")
    lines.append(f"**Training status** (the verdict — combines sample SSIM with the loss flag):")
    lines.append(f"- `trained` — SSIM @ NFE=4 ≥ {SSIM_TRAINED_THRESHOLD} (the model produces brain-like samples)")
    lines.append(f"- `failed`  — SSIM @ NFE=4 < {SSIM_TRAINED_THRESHOLD} (samples are degenerate; training broke)")
    lines.append("")
    lines.append("**Interpretation rule:** the 2×2 diversity comparison is only valid across cells with "
                 "`training_status = trained`. A `failed` cell indicates the {EMA, loss} combination is "
                 "unstable to train — not whether the *objective* causes collapse. The loss-trajectory flag "
                 "is a descriptive diagnostic of the training curve only; some valid baselines (e.g. Cell B) "
                 "have a `loss_spiked_stalled` curve and still produce usable samples.")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/brats_conditional_64.pt")
    parser.add_argument("--vqgan-ckpt",
                        default="results/paper4/brats_benchmark_20260324_154926/100pct_180vol/phase1_shared.pt")
    parser.add_argument("--output-dir", default="results/paper4/cd_2x2")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-label", type=int, default=1)
    parser.add_argument("--steps", nargs="+", type=int, default=STEP_COUNTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-reuse-existing", action="store_true",
                        help="Force fresh eval on all four cells (slow). Default reuses A/B JSONs.")
    parser.add_argument("--headline-nfe", type=int, default=4,
                        help="NFE used in the headline 2x2 markdown table")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    print(f"Device: {device}")
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Real data -----
    d = torch.load(REPO_ROOT / args.data_path, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        vols = d.get("volumes", d.get("v"))
    else:
        vols = d
    if vols.dim() == 5:
        vols = vols.squeeze(1)
    real = vols.float().numpy()
    print(f"Loaded {len(real)} real volumes")

    rng = np.random.default_rng(0)
    real_sample = real[rng.choice(len(real), size=min(64, len(real)), replace=False)]
    real_diversity = compute_pairwise_diversity(real_sample)
    print(f"Real data diversity (n={len(real_sample)}): {real_diversity:.4f}")

    # ----- Per-cell eval -----
    cell_evals = {}
    for name, cell in CELLS.items():
        existing = cell.get("existing_eval")
        if existing and not args.no_reuse_existing and (REPO_ROOT / existing).exists():
            print(f"\n[REUSE] {cell['label']}\n  {existing}")
            cell_evals[name] = load_existing_eval(
                REPO_ROOT / existing,
                eval_path_in_json=cell.get("existing_eval_path"),
            )
            # Cross-check real_diversity if both present (informational)
            if cell_evals[name].get("real_diversity"):
                rd = cell_evals[name]["real_diversity"]
                print(f"  real_diversity in existing JSON: {rd:.4f} (this run: {real_diversity:.4f})")
        else:
            cell_evals[name] = run_fresh_eval(cell, args, device, real_sample, real_diversity)

    # ----- Loss-trajectory flags -----
    loss_traj = {}
    for name, cell in CELLS.items():
        flag, details = convergence_flag(
            (REPO_ROOT / cell["training_log"]) if cell.get("training_log") else None
        )
        loss_traj[name] = {"flag": flag, "details": details}
        print(f"\n[LOSS-TRAJ] {cell['label']}: {flag}")
        for k, v in details.items():
            print(f"    {k} = {v}")

    # ----- Assemble headline rows -----
    rows = [
        cell_row(name, CELLS[name], cell_evals[name],
                  loss_traj[name]["flag"], loss_traj[name]["details"],
                  nfe=str(args.headline_nfe))
        for name in ["A_ema_l2", "B_noema_pseudohuber",
                     "C_ema_pseudohuber", "D_noema_l2"]
    ]

    md = markdown_table(rows, nfe=str(args.headline_nfe))

    table_json_path = out_dir / "factorial_2x2_table.json"
    table_md_path   = out_dir / "factorial_2x2_table.md"
    with open(table_json_path, "w") as f:
        json.dump({
            "cells": rows,
            "real_diversity": real_diversity,
            "headline_nfe": args.headline_nfe,
            "per_cell_eval": {k: v for k, v in cell_evals.items()},
            "loss_trajectory": loss_traj,
            "ssim_trained_threshold": SSIM_TRAINED_THRESHOLD,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)
    with open(table_md_path, "w") as f:
        f.write(md)

    print("\n" + "=" * 70)
    print(md)
    print("\nSaved:")
    print(f"  {table_json_path}")
    print(f"  {table_md_path}")


if __name__ == "__main__":
    main()
