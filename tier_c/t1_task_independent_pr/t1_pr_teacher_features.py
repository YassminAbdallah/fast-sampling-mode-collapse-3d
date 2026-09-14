#!/usr/bin/env python3
"""
Task-independent P/R/D/C confirmation.
======================================

Motivation: the paper's Precision/Recall/Density/Coverage metrics
in §5.2 are computed in the same VQ-GAN encoder feature space that every
generator decodes through. Standard practice uses a task-independent feature
extractor.

This script re-runs P/R/D/C for the two focal methods (Consistency@50 and
Shortcut@50) on both BraTS unconditional and IXI unconditional, using the
64-cubed BraTS tumour-segmentation teacher's encoder path (conv_0 → down_1 →
down_2 → down_3 → down_4) as a task-independent feature space. The teacher is
not part of any generative pipeline, is trained for a different task, and is
applied identically across every (dataset, method) combination.

Reuses kynkaanniemi_precision_recall and naeem_density_coverage from
src/paper4/precision_recall_metrics.py, so the P/R/D/C definitions are
literally identical to §5.2 — only the feature extractor differs.

Protocol: 5 sampling seeds × 64 samples per seed (matches §5.2 evaluate_5seed).

Architecture note: to keep MPS-side memory pressure low and avoid multi-model
crashes reported in the initial MPS run, this script processes one
(dataset, method) at a time and never keeps a generator UNet and the teacher
segmenter loaded on the accelerator simultaneously. Real features for each
dataset are cached to a small .npz once and reused across methods.

Runtime: ~30 min per dataset on Apple M1 MPS; ~60 min total for both.

Usage:
  cd <repo-root>/
  bash tier_c/t1_task_independent_pr/run_t1_pr.sh
"""

import argparse, gc, json, os, sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# NOTE: we do NOT set PYTORCH_MPS_HIGH_WATERMARK_RATIO here. Setting it to 0.0
# in an earlier iteration coincided with segfaults on some PyTorch versions;
# the default value is more stable. Memory pressure is instead managed by
# loading the teacher once upfront (on CPU) and never reloading models after
# MPS has been active.

try:
    from monai.networks.nets import BasicUNet
except ImportError:
    sys.exit("ERROR: MONAI not installed. Run: pip install monai")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "paper4"))

from models_shared import load_vqgan, load_unet, sample_latent  # noqa: E402


# ============================================================
# Custom (pure-numpy) P/R/D/C — bypasses sklearn.NearestNeighbors
# ============================================================
#
# sklearn.neighbors.NearestNeighbors on macOS routes through Apple's Accelerate
# BLAS and has been observed to segfault on some (PyTorch, sklearn, macOS)
# combinations for feature matrices in the 64 x 2048 size range. Our sample
# sizes are small enough that computing distance matrices directly in numpy is
# trivial and completely bypasses that failure surface.
#
# The definitions below are the same ones used in src/paper4/precision_recall_metrics.py:
#   Precision & Recall — Kynkäänniemi et al., NeurIPS 2019
#   Density & Coverage — Naeem et al., ICML 2020
# Only the internal distance-search implementation differs.

def _pairwise_l2(a, b):
    """Return the [len(a), len(b)] pairwise Euclidean distance matrix."""
    # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2, then sqrt with clamp.
    a2 = (a * a).sum(axis=1, keepdims=True)
    b2 = (b * b).sum(axis=1, keepdims=True).T
    sq = a2 - 2.0 * a @ b.T + b2
    return np.sqrt(np.maximum(sq, 0.0))


def _knn_radii(feats, k):
    """For each sample, distance to its k-th nearest neighbour among the others."""
    D = _pairwise_l2(feats, feats)
    np.fill_diagonal(D, np.inf)  # exclude self
    D.sort(axis=1)
    return D[:, k - 1]  # k-th neighbour distance (k is 1-indexed among "others")


def kynkaanniemi_precision_recall(real_features, gen_features, k=3):
    """Improved Precision and Recall (Kynkäänniemi et al., 2019). Same definition
    as src/paper4/precision_recall_metrics.py; pure-numpy implementation."""
    real_radii = _knn_radii(real_features, k)
    gen_radii  = _knn_radii(gen_features,  k)
    D_gr = _pairwise_l2(gen_features, real_features)  # [n_gen, n_real]
    # For each gen sample, distance and index to its nearest real:
    idx_gen_to_real = D_gr.argmin(axis=1)
    dist_gen_to_real = D_gr.min(axis=1)
    precision = float(
        np.mean(dist_gen_to_real <= real_radii[idx_gen_to_real]))
    D_rg = D_gr.T  # [n_real, n_gen]
    idx_real_to_gen = D_rg.argmin(axis=1)
    dist_real_to_gen = D_rg.min(axis=1)
    recall = float(
        np.mean(dist_real_to_gen <= gen_radii[idx_real_to_gen]))
    return precision, recall


def naeem_density_coverage(real_features, gen_features, k=5):
    """Density and Coverage (Naeem et al., 2020). Same definition as
    src/paper4/precision_recall_metrics.py; pure-numpy implementation."""
    real_radii = _knn_radii(real_features, k)
    D = _pairwise_l2(real_features, gen_features)  # [n_real, n_gen]
    # For each real: how many gen samples fall within its k-NN radius?
    within = D <= real_radii[:, None]
    counts = within.sum(axis=1)
    coverage = float((counts > 0).mean())
    density  = float(counts.mean() / k)
    return density, coverage

warnings.filterwarnings("ignore")

SEEDS = [42, 123, 456, 789, 1337]
NFE = 50
LATENT_CH = 8
LATENT_SIZE = 8

DATASETS = {
    "brats": {
        "data":  "data/brats_preprocessed_64.pt",
        "bench": "results/paper3/brats_benchmark_20260221_193306/100pct_300vol",
    },
    "ixi": {
        "data":  "data/ixi_preprocessed_64.pt",
        "bench": "results/paper3/ixi_benchmark_20260221_045741/100pct_200vol",
    },
}
METHODS = ["consistency", "shortcut", "fm", "rectified", "ddpm"]
# Per-method NFE, matching the representative rows of Tables 2/3.
# DDPM at 50 steps is noise-dominated (§5.1) and must run at 1000.
METHOD_NFE = {"consistency": 50, "shortcut": 50, "fm": 50,
              "rectified": 10, "ddpm": 1000}


# ============================================================
# Cleanup helpers
# ============================================================

def free_accelerator():
    """Aggressive cleanup: gc + empty MPS/CUDA caches."""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def seed_all(seed, device):
    """Reproducible seeding; guarded against MPS RNG bugs on older PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "mps":
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Teacher-encoder feature extraction (no forward hooks)
# ============================================================

def build_teacher(teacher_path, device):
    """Load the 64³ teacher segmenter onto `device` in eval mode."""
    model = BasicUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        features=(32, 64, 128, 256, 32, 32),
    ).to(device)
    model.load_state_dict(torch.load(teacher_path, map_location=device,
                                     weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def teacher_encoder_features(model, volumes, device, batch_size=2):
    """Run the teacher's encoder path (conv_0 → down_1 → ... → down_4) and
    return the flattened bottleneck features, with NaN/Inf sanitisation on
    both the input volumes and the output features.

    Rationale: collapsed generators (esp. Consistency Distillation at some
    step counts) can occasionally produce NaN/Inf voxels; .clamp() does not
    remove those; unsanitised NaNs propagate through the teacher and then
    crash sklearn's NearestNeighbors in the downstream P/R/D/C computation.
    """
    volumes = volumes.detach().to("cpu").contiguous().float()
    # Scrub NaN / Inf at the voxel level before feeding the teacher.
    volumes = torch.nan_to_num(volumes, nan=0.0, posinf=1.0, neginf=0.0)
    volumes = volumes.clamp(0.0, 1.0).contiguous()
    if volumes.dim() == 4:
        volumes = volumes.unsqueeze(1)
    N = len(volumes)
    n_batches = (N + batch_size - 1) // batch_size
    out = []
    for bi, i in enumerate(range(0, N, batch_size)):
        x = volumes[i:i+batch_size].contiguous()
        try:
            x0 = model.conv_0(x)
            x1 = model.down_1(x0)
            x2 = model.down_2(x1)
            x3 = model.down_3(x2)
            x4 = model.down_4(x3)
        except Exception as e:
            print(f"        ! teacher forward failed at batch {bi+1}/{n_batches}: "
                  f"{type(e).__name__}: {e}", flush=True)
            raise
        feats = x4.reshape(x4.shape[0], -1).numpy()
        out.append(feats)
        del x, x0, x1, x2, x3, x4
    feats_all = np.concatenate(out, axis=0)
    n_nan = int(np.isnan(feats_all).sum())
    n_inf = int(np.isinf(feats_all).sum())
    if n_nan or n_inf:
        # Scrub any residual NaN/Inf so downstream sklearn kNN does not crash.
        feats_all = np.nan_to_num(feats_all, nan=0.0, posinf=1.0, neginf=-1.0)
    print(f"        features: shape={feats_all.shape} "
          f"nan={n_nan} inf={n_inf} "
          f"range=[{feats_all.min():.3g}, {feats_all.max():.3g}]",
          flush=True)
    return feats_all


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_unconditional(unet, dec, method, n, dev, steps,
                            ch=LATENT_CH, batch_size=8):
    out = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = sample_latent(unet, method, bs, dev, steps, ch, class_label=None)
        vols = dec(z).clamp(0, 1).cpu()
        out.append(vols)
        del z
    return torch.cat(out, dim=0)


# ============================================================
# Sequential-load evaluator: never keeps two model families on the
# accelerator at once.
# ============================================================

def sample_all_seeds(dataset_name, method, n_samples, device, teacher_path):
    """Load VQ-GAN + generator UNet, sample seeds' worth of gen volumes,
    also collect the seed-matched real volume batches. Return everything on
    CPU. Generator is FULLY UNLOADED before returning.
    """
    ds = DATASETS[dataset_name]
    bench_dir = REPO_ROOT / ds["bench"]
    data_path = REPO_ROOT / ds["data"]

    print(f"    [step 1/2] Loading VQ-GAN + {method} generator")
    enc, dec, vq, _ = load_vqgan(bench_dir / "phase1_shared.pt", device)
    unet = load_unet(bench_dir / method / "final.pt", device, num_classes=0)

    print(f"    [step 1/2] Loading real dataset {data_path.name}")
    vols = torch.load(data_path, weights_only=True)
    if vols.dim() == 4: vols = vols.unsqueeze(1)
    N = len(vols)

    real_batches = []
    gen_batches = []
    for seed in SEEDS:
        seed_all(seed, device)
        idx = np.random.choice(N, size=min(n_samples, N), replace=False)
        # Fully detach from any MPS context and re-materialise as clean CPU tensors
        r = vols[idx].detach().to("cpu").contiguous().float()
        real_batches.append(r)
        gen_batch = generate_unconditional(unet, dec, method, n_samples,
                                             device, METHOD_NFE[method])
        gen_batches.append(gen_batch.detach().to("cpu").contiguous().float())
        print(f"      seed={seed:>4}: sampled {n_samples} real + {n_samples} gen")

    # Fully unload the generator side before returning to the teacher pass
    del enc, dec, vq, unet, vols
    free_accelerator()

    return real_batches, gen_batches


def score_all_seeds(real_batches, gen_batches, teacher, cpu_dev):
    """Extract features via the pre-loaded CPU teacher, compute P/R/D/C.

    Rationale: after MPS has been active, even a CPU-side torch.load can
    segfault because unpickling still touches MPS metadata. So the teacher
    is loaded ONCE at startup (before any MPS work) and reused across all
    (dataset, method) combinations.
    """
    print(f"    [step 2/2] Extracting task-independent features (CPU teacher)")

    per_seed = {"precision": [], "recall": [], "density": [], "coverage": []}
    for i, (real, gen) in enumerate(zip(real_batches, gen_batches)):
        seed = SEEDS[i]
        real_feats = teacher_encoder_features(teacher, real, cpu_dev)
        gen_feats  = teacher_encoder_features(teacher, gen,  cpu_dev)
        prec, rec = kynkaanniemi_precision_recall(real_feats, gen_feats, k=3)
        dens, cov = naeem_density_coverage(real_feats, gen_feats, k=5)
        per_seed["precision"].append(prec)
        per_seed["recall"].append(rec)
        per_seed["density"].append(dens)
        per_seed["coverage"].append(cov)
        print(f"      seed={seed:>4}: P={prec:.3f}  R={rec:.4f}  "
              f"D={dens:.3f}  C={cov:.3f}")

    summary = {}
    for k, v in per_seed.items():
        arr = np.asarray(v, dtype=float)
        summary[k] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            "per_seed": [float(x) for x in v],
        }
    return summary



# ============================================================
# Driver
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", type=str,
                    default="results/paper4/paper4/e7_teacher/teacher_best.pt")
    ap.add_argument("--n-samples", type=int, default=64,
                    help="Samples per seed (matches §5.2 evaluate_5seed).")
    ap.add_argument("--output", type=str,
                    default="results/paper4/t1_pr_task_independent/t1_pr_results.json")
    ap.add_argument("--dataset", type=str, choices=["brats", "ixi", "both"],
                    default="both")
    ap.add_argument("--force-cpu", action="store_true",
                    help="Skip MPS/CUDA and run entirely on CPU (slower but "
                         "totally deterministic; use if MPS segfaults recur).")
    args = ap.parse_args()

    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    print(f"Device: {device}")

    teacher_path = REPO_ROOT / args.teacher_path
    if not teacher_path.exists():
        sys.exit(f"ERROR: teacher checkpoint missing at {teacher_path}")

    # Load the teacher ONCE, upfront, BEFORE any MPS work happens.
    # After MPS has been active, even a CPU-side torch.load has been observed
    # to segfault on some PyTorch/macOS combinations because unpickling still
    # touches MPS metadata. Loading upfront eliminates that surface entirely.
    cpu_dev = torch.device("cpu")
    print(f"Loading task-independent feature extractor (teacher segmenter) on CPU:")
    print(f"  {teacher_path}")
    teacher = build_teacher(teacher_path, cpu_dev)
    print(f"  Teacher loaded (CPU); encoder path will be traversed manually.")

    datasets = ["brats", "ixi"] if args.dataset == "both" else [args.dataset]

    all_results = {}
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"  {dataset_name.upper()} × {method} @ NFE={METHOD_NFE[method]}")
            print(f"{'='*60}")
            real_batches, gen_batches = sample_all_seeds(
                dataset_name, method, args.n_samples, device, teacher_path)
            all_results[dataset_name][method] = score_all_seeds(
                real_batches, gen_batches, teacher, cpu_dev)
            del real_batches, gen_batches
            free_accelerator()

    # Save
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "purpose": (
            "T1: task-independent "
            "Precision/Recall/Density/Coverage for Consistency@50 and "
            "Shortcut@50 on IXI + BraTS 64-cubed, using the BraTS "
            "tumour-segmentation teacher's encoder-path bottleneck features."
        ),
        "resolution": 64,
        "seeds": SEEDS,
        "n_samples_per_seed": args.n_samples,
        "nfe": METHOD_NFE,
        "feature_space": (
            "MONAI BasicUNet segmentation teacher (32, 64, 128, 256, 32, 32); "
            "encoder path traversed manually (conv_0 -> down_1 -> down_2 -> "
            "down_3 -> down_4); bottleneck flattened to 2,048 dimensions"
        ),
        "teacher_path": str(teacher_path.relative_to(REPO_ROOT)),
        "pr_definitions": (
            "Kynkäänniemi et al. 2019 P/R at k=3; "
            "Naeem et al. 2020 D/C at k=5. Same definitions as §5.2; "
            "only the feature extractor differs."
        ),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print headline
    print(f"\n{'='*60}")
    print("  T1 headline (task-independent teacher-encoder features)")
    print(f"{'='*60}")
    print(f"  {'dataset':<8} {'method':<12} {'Precision':>16} {'Recall':>16} "
          f"{'Density':>16} {'Coverage':>16}")
    print(f"  {'-'*84}")
    for d in datasets:
        for m in METHODS:
            r = all_results[d][m]
            print(f"  {d:<8} {m:<12} "
                  f"{r['precision']['mean']:>7.3f} ± {r['precision']['std']:>.3f}  "
                  f"{r['recall']['mean']:>7.4f} ± {r['recall']['std']:>.4f}  "
                  f"{r['density']['mean']:>7.3f} ± {r['density']['std']:>.3f}  "
                  f"{r['coverage']['mean']:>7.3f} ± {r['coverage']['std']:>.3f}")


if __name__ == "__main__":
    main()
