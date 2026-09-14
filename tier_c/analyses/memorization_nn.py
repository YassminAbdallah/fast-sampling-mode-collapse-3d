#!/usr/bin/env python3
"""
Nearest-neighbor memorization analysis at 128³.

Computes, for each synthetic volume in the cached 128³ pools, the distance
to its nearest neighbor in the real BraTS training set, in VQ-GAN encoder
feature space (the same space used for Precision/Recall/Latent-FID). Outputs
per-method distance distributions. Anomalously small distances indicate the
model is reproducing real volumes from memory rather than sampling the
distribution. Strong precedent: Dar et al. 2023/2024 ("Unconditional Latent
Diffusion Models Memorize Patient Imaging Data").

Defaults to CPU mode so it does NOT contend with any concurrent MPS/CUDA
training. Use --device mps or --device cuda to run faster when no training
is in flight.

Outputs:
    results/analyses/memorization_nn.csv     — per-synthetic-volume NN distances
    results/analyses/memorization_nn.png     — distribution figure
    results/analyses/memorization_nn.json    — per-method summary stats

Usage:
    cd fast-sampling-mode-collapse-3d/
    python tier_c/analyses/memorization_nn.py                # CPU mode (~30 min)
    python tier_c/analyses/memorization_nn.py --device mps   # MPS mode (~5 min)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT = Path(__file__).resolve().parent
REPO = SCRIPT.parent.parent
sys.path.insert(0, str(REPO / "src" / "paper4"))

from models_shared import Encoder3D, VectorQuantizer  # noqa: E402


# ============================================================
# Helpers
# ============================================================

def load_vqgan(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    enc = Encoder3D(1, 8, 2).to(device)
    enc.load_state_dict(state["enc"]); enc.eval()
    vq = VectorQuantizer(256, 8).to(device)
    vq.load_state_dict(state["vq"]); vq.eval()
    for p in list(enc.parameters()) + list(vq.parameters()):
        p.requires_grad = False
    return enc, vq


@torch.no_grad()
def encode_volumes(vols, enc, vq, device, batch_size=2):
    """Encode a tensor of volumes to flattened latent feature vectors."""
    feats = []
    for i in range(0, len(vols), batch_size):
        x = vols[i:i + batch_size].float().to(device)
        # nan-safe: synthetic VQ-GAN decodes occasionally contain denormals
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        z, _, _ = vq(enc(x))
        feats.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def nearest_neighbor_distances(syn_feats, real_feats):
    """For each row in syn_feats, compute (NN distance, NN index) in real_feats.
    Returns tuple (nn_dist, nn_idx), both shape (n_syn,).
    Tracking the index lets us count how many distinct real volumes the synthetic
    pool clusters around (mode-collapse signature)."""
    nn_dist = np.empty(len(syn_feats), dtype=np.float64)
    nn_idx = np.empty(len(syn_feats), dtype=np.int64)
    batch = 32
    for i in range(0, len(syn_feats), batch):
        s = syn_feats[i:i + batch]
        diffs = s[:, None, :] - real_feats[None, :, :]
        d = np.sqrt((diffs * diffs).sum(axis=-1))
        nn_dist[i:i + batch] = d.min(axis=-1)
        nn_idx[i:i + batch] = d.argmin(axis=-1)
    return nn_dist, nn_idx


def real_to_real_loo_distances(real_feats):
    """Leave-one-out nearest-neighbor distance among the real volumes themselves.
    For each real volume, find its NN among the OTHER real volumes (excluding
    itself). This is the baseline for what 'closeness to the real distribution'
    looks like under genuine real-data sampling."""
    n = len(real_feats)
    dist = np.empty(n, dtype=np.float64)
    batch = 32
    for i in range(0, n, batch):
        s = real_feats[i:i + batch]
        diffs = s[:, None, :] - real_feats[None, :, :]
        d = np.sqrt((diffs * diffs).sum(axis=-1))
        # Mask out self-distances (set to +inf)
        for bi, ri in enumerate(range(i, min(i + batch, n))):
            d[bi, ri] = np.inf
        dist[i:i + batch] = d.min(axis=-1)
    return dist


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu",
                   help="cpu (default, safe) | mps | cuda")
    p.add_argument("--vqgan",
                   default="results/paper4/brats_128cubed_conditional/phase1_shared.pt")
    p.add_argument("--data",
                   default="data/brats_conditional_128.pt")
    p.add_argument("--syn-dir",
                   default="results/paper4/brats_128cubed_conditional/e10b_128")
    p.add_argument("--output-dir", default="results/analyses")
    args = p.parse_args()

    dev = args.device
    print(f"Device: {dev}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- Load VQ-GAN -----
    vqgan_path = REPO / args.vqgan
    if not vqgan_path.exists():
        sys.exit(f"ERROR: missing VQ-GAN at {vqgan_path}")
    print(f"Loading VQ-GAN: {vqgan_path}")
    enc, vq = load_vqgan(vqgan_path, dev)

    # ----- Load real training volumes -----
    data_path = REPO / args.data
    if not data_path.exists():
        sys.exit(f"ERROR: missing data at {data_path}")
    print(f"Loading data: {data_path}")
    raw = torch.load(data_path, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict):
        sys.exit("ERROR: data is not a conditional dict")
    vols = raw["volumes"]
    splits = raw.get("split_info") or {}
    train_idx = splits.get("train_idx") or list(range(len(vols)))
    # Restrict to the same training pool the generators were trained on
    train_vols = vols[train_idx]
    if train_vols.dim() == 4:
        train_vols = train_vols.unsqueeze(1)
    print(f"  Real training pool: {len(train_vols)} volumes (shape {tuple(train_vols.shape)})")

    # ----- Encode real training pool (with disk cache so re-runs are fast) -----
    real_cache = out / "_cache_real_feats.npy"
    if real_cache.exists():
        print(f"Loading cached real features: {real_cache}")
        real_feats = np.load(real_cache)
        print(f"  Real features shape {real_feats.shape}  (from cache)")
    else:
        print(f"Encoding real training pool...")
        t0 = time.time()
        real_feats = encode_volumes(train_vols, enc, vq, dev, batch_size=2)
        print(f"  Real features shape {real_feats.shape}  ({time.time()-t0:.0f}s)")
        np.save(real_cache, real_feats)
        print(f"  Cached to {real_cache}")

    # ----- Real-to-real LOO baseline -----
    # Calibrates 'what does closeness to the real distribution look like
    # when the samples ARE real?' Without this, the absolute distance
    # numbers from synthetic-to-real are uninterpretable.
    print(f"Computing real-to-real LOO baseline (calibration)...")
    t_real = time.time()
    real_to_real = real_to_real_loo_distances(real_feats)
    print(f"  Done ({time.time()-t_real:.0f}s)")
    print(f"  Real-to-real NN distance: min={real_to_real.min():.3f}  "
          f"median={np.median(real_to_real):.3f}  "
          f"mean={real_to_real.mean():.3f}  "
          f"p95={np.percentile(real_to_real, 95):.3f}")

    # ----- Encode each synthetic pool and compute NN distances -----
    methods = [
        ("shortcut_50", "Shortcut FM @ NFE=50"),
        ("consistency_4", "Consistency Distillation @ NFE=4"),
    ]
    per_volume_rows = []
    summary = {}
    distances_per_method = {}

    for key, pretty in methods:
        syn_path = REPO / args.syn_dir / f"syn_pool_{key}.pt"
        if not syn_path.exists():
            print(f"  Skipping {key} (no cached pool at {syn_path})")
            continue
        print(f"\nLoading {pretty}: {syn_path}")
        syn = torch.load(syn_path, weights_only=False, map_location="cpu")
        if syn.dim() == 4:
            syn = syn.unsqueeze(1)
        print(f"  Synthetic pool: {tuple(syn.shape)}")

        # Encode with disk cache so re-runs of this analysis are near-instant
        syn_cache = out / f"_cache_syn_feats_{key}.npy"
        if syn_cache.exists():
            print(f"  Loading cached synthetic features: {syn_cache}")
            syn_feats = np.load(syn_cache)
            print(f"    Shape {syn_feats.shape}  (from cache)")
        else:
            print(f"  Encoding {len(syn)} synthetic volumes...")
            t1 = time.time()
            syn_feats = encode_volumes(syn, enc, vq, dev, batch_size=2)
            print(f"    Encoded ({time.time()-t1:.0f}s)")
            np.save(syn_cache, syn_feats)
            print(f"    Cached to {syn_cache}")

        print(f"  Computing nearest-neighbor distances to real pool...")
        t2 = time.time()
        nn_d, nn_idx = nearest_neighbor_distances(syn_feats, real_feats)
        print(f"    Done ({time.time()-t2:.0f}s)")

        distances_per_method[key] = nn_d
        # --- Left-tail check: candidate copies (below real-to-real min) ---
        real_min = float(real_to_real.min())
        real_p5 = float(np.percentile(real_to_real, 5))
        n_below_real_min = int((nn_d < real_min).sum())
        n_below_real_p5 = int((nn_d < real_p5).sum())
        # --- Neighbor-collapse check: how many distinct real volumes are
        #     hit as nearest neighbor by the synthetic pool? ---
        distinct_nn = int(np.unique(nn_idx).size)
        nn_counts = np.bincount(nn_idx, minlength=len(real_feats))
        top_5_share = float(np.sort(nn_counts)[-5:].sum()) / len(nn_d)

        summary[key] = {
            "label": pretty,
            "n_synthetic": int(len(nn_d)),
            "n_real_train": int(len(real_feats)),
            "nn_distance_min": float(nn_d.min()),
            "nn_distance_p5": float(np.percentile(nn_d, 5)),
            "nn_distance_median": float(np.median(nn_d)),
            "nn_distance_mean": float(nn_d.mean()),
            "nn_distance_p95": float(np.percentile(nn_d, 95)),
            "nn_distance_max": float(nn_d.max()),
            "nn_distance_std": float(nn_d.std()),
            # Disambiguation diagnostics:
            "n_below_real_to_real_min": n_below_real_min,
            "n_below_real_to_real_p5": n_below_real_p5,
            "distinct_real_neighbors_used": distinct_nn,
            "top_5_real_neighbors_share_of_pool": top_5_share,
        }
        for i, (d, idx) in enumerate(zip(nn_d, nn_idx)):
            per_volume_rows.append({
                "method": key,
                "synthetic_idx": i,
                "nn_distance_to_real_train": float(d),
                "nn_real_train_index": int(idx),
            })

    if not distances_per_method:
        sys.exit("No synthetic pools were found.")

    # Stash the real-to-real baseline INTO summary BEFORE writing JSON,
    # so downstream scripts (e.g. memorization_visual_check.py) can read it.
    summary["_real_to_real_baseline"] = {
        "n_real": int(len(real_to_real)),
        "min": float(real_to_real.min()),
        "p5": float(np.percentile(real_to_real, 5)),
        "median": float(np.median(real_to_real)),
        "mean": float(real_to_real.mean()),
        "p95": float(np.percentile(real_to_real, 95)),
        "max": float(real_to_real.max()),
        "std": float(real_to_real.std()),
    }

    # ----- Save outputs (JSON first — the smallest + most important) -----
    json_path = out / "memorization_nn.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {json_path}")

    # ----- CSV (per-volume distances) — fieldnames cover all keys -----
    import csv
    csv_path = out / "memorization_nn.csv"
    fieldnames = sorted({k for r in per_volume_rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_volume_rows)
    print(f"Saved {csv_path}")

    # ----- Plot distance distributions WITH real-to-real baseline -----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"shortcut_50": "#9b59b6", "consistency_4": "#e74c3c"}
    # Baseline first (drawn underneath)
    ax.hist(real_to_real, bins=40, alpha=0.45,
            label=f"Real-to-real (LOO, n={len(real_to_real)})",
            color="#7f8c8d", edgecolor="white", linewidth=0.5)
    for key, d in distances_per_method.items():
        label = summary[key]["label"]
        ax.hist(d, bins=40, alpha=0.55, label=label,
                color=colors.get(key, "gray"), edgecolor="white", linewidth=0.5)
    ax.axvline(float(real_to_real.min()), color="black", linestyle="--",
               linewidth=0.8, label=f"Real-to-real min = {real_to_real.min():.2f}")
    ax.set_xlabel("Nearest-neighbor distance (VQ-GAN encoder space)")
    ax.set_ylabel("# volumes")
    ax.set_title("Sample-space geometry: synthetic vs real-to-real NN distances\n"
                 "(synthetic-to-real shifted left of real-to-real baseline ⇒ closer to the data distribution)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = out / "memorization_nn.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")
    # (baseline summary already stashed into `summary` before JSON save)

    # ----- Summary print -----
    print("\nReal-to-real baseline:")
    b = summary["_real_to_real_baseline"]
    print(f"  min    = {b['min']:.3f}")
    print(f"  p5     = {b['p5']:.3f}")
    print(f"  median = {b['median']:.3f}")
    print(f"  mean   = {b['mean']:.3f}")
    print(f"  p95    = {b['p95']:.3f}")

    print("\nPer-method NN-distance to real training pool:")
    for key, s in summary.items():
        if key.startswith("_"):
            continue
        print(f"  {s['label']}:")
        print(f"    n_synthetic = {s['n_synthetic']}, n_real_train = {s['n_real_train']}")
        print(f"    min    = {s['nn_distance_min']:.3f}  (real-to-real min = {b['min']:.3f})")
        print(f"    p5     = {s['nn_distance_p5']:.3f}")
        print(f"    median = {s['nn_distance_median']:.3f}  (real-to-real median = {b['median']:.3f})")
        print(f"    mean   = {s['nn_distance_mean']:.3f}")
        print(f"    p95    = {s['nn_distance_p95']:.3f}")
        print(f"    Left-tail (candidate memorization):")
        print(f"      synthetic samples below real-to-real min ({b['min']:.3f}): {s['n_below_real_to_real_min']} / {s['n_synthetic']}")
        print(f"      synthetic samples below real-to-real p5  ({b['p5']:.3f}): {s['n_below_real_to_real_p5']} / {s['n_synthetic']}")
        print(f"    Neighbor-collapse (mode concentration):")
        print(f"      distinct real neighbors used: {s['distinct_real_neighbors_used']} / {s['n_real_train']}")
        print(f"      top-5 real neighbors capture {s['top_5_real_neighbors_share_of_pool']*100:.1f}% of pool")

    print("\nVerdict guide (collapse vs memorization):")
    print("  - If a method's median is BELOW the real-to-real median but its")
    print("    left-tail count (samples below real-to-real min) is near zero,")
    print("    that's collapse-to-centroid: samples closer to the data mean")
    print("    than real samples are to each other, but NOT copies of any")
    print("    specific real volume. Frame as collapse, not memorization.")
    print("  - If left-tail count is materially > 0 (e.g. >5%), there are")
    print("    candidate near-copies of specific real volumes. That is the")
    print("    privacy/memorization signature.")
    print("  - Low 'distinct neighbors used' or high 'top-5 share' reinforces")
    print("    collapse: many synthetic samples land near the same few real")
    print("    volumes (mode collapse, not patient memorization).")

    if len(distances_per_method) >= 2:
        from scipy import stats as sst
        keys = list(distances_per_method.keys())
        u, p = sst.mannwhitneyu(distances_per_method[keys[0]],
                                 distances_per_method[keys[1]],
                                 alternative="two-sided")
        print(f"\nMann-Whitney U test ({keys[0]} vs {keys[1]}):")
        print(f"  U = {u:.0f}, two-sided p = {p:.4g}")


if __name__ == "__main__":
    main()
