#!/usr/bin/env python3
"""
128-cubed manifold metrics, distributional metrics and diversity.
================================================================

Computes at 128-cubed the same feature-space and distributional metrics that
the paper reports at 64-cubed, from the existing 128-cubed checkpoints:

    results/paper4/brats_128cubed/phase1_shared.pt          (VQ-GAN)
    results/paper4/brats_128cubed/{consistency,shortcut,fm} (generators)
    results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_best.pt

Inference only; no training.

WHAT IT COMPUTES (all at 128-cubed, NFE=50, 5 seeds x 64 samples)
  1. Precision / Recall (k=3) and Density / Coverage (k=5) in the
     task-independent teacher-encoder space.
  2. The same four metrics in the generator's own VQ-GAN space, to show the
     two spaces disagree at 128-cubed exactly as they do at 64-cubed.
  3. Pairwise-L1 diversity, as a percentage of real-data diversity.
  4. Frechet Radiomic Distance (15 first-order radiomic features).
  5. Latent FID on a PCA-128 projection of the VQ-GAN latent space.

ONE METHODOLOGICAL DECISION, STATED EXPLICITLY
  The BasicUNet bottleneck is 32 channels at input/16 resolution:

      64-cubed  ->  32 x  4^3 =  2,048 dims   (what the paper reports)
      128-cubed ->  32 x  8^3 = 16,384 dims

  Computing k-NN metrics on 16,384 dims from 64 samples per seed would be a
  curse-of-dimensionality artefact, not a finding, and would NOT be comparable
  to the 64-cubed numbers. The bottleneck is therefore average-pooled to 4^3,
  giving exactly 2,048 dims at both resolutions. Pass --no-pool to also emit
  the unpooled 16,384-dim values as a robustness check.

RUNTIME
  ~35-50 min on Apple M1 MPS. 3 methods x 5 seeds x 64 samples = 960 volumes
  at the measured 128-cubed cost of ~1.0 s/volume, plus feature extraction.

USAGE
  python3 fixes/07_manifold_metrics_128.py
  python3 fixes/07_manifold_metrics_128.py --no-pool     # + 16,384-dim check
  python3 fixes/07_manifold_metrics_128.py --n-samples 16 --seeds 42   # smoke test
"""
import argparse, gc, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from monai.networks.nets import BasicUNet
except ImportError:
    sys.exit("ERROR: MONAI not installed.  pip install monai")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "paper4"))
from models_shared import load_vqgan, DenoisingUNet3D  # noqa: E402

# ---------------------------------------------------------------------------
# Reuse the 64-cubed radiomic + Frechet definitions EXACTLY.
#
# A hand-written copy of these features was wrong in a way that only shows up at
# 128-cubed: "energy" was implemented as (fg**2).sum() rather than (fg**2).mean().
# The sum is EXTENSIVE, so it grows with voxel count, and at 8x the voxels it
# reached ~1e6, drove covariance entries to ~1e12, overflowed the matmuls, and
# produced FRD = 2.8e9 against a 64-cubed value of ~1.4. Importing the original
# definition removes the whole class of error and guarantees the 64-cubed and
# 128-cubed FRD numbers are computed by identical code.
#
# Loaded by file path because src/paper3 and src/paper4 each contain a
# models_shared.py, and putting src/paper3 on sys.path would shadow the one
# already imported above.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_dm64", ROOT / "src" / "paper3" / "distributional_metrics.py")
_dm64 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dm64)
extract_radiomic_features = _dm64.extract_radiomic_features   # the 15 features
compute_frd = _dm64.compute_frd
compute_latent_fid = _dm64.compute_latent_fid


def load_unet_flex(ckpt_path, dev, num_classes=0):
    """Load a U-Net checkpoint whatever layout it was saved in.

    models_shared.load_unet only handles flat keys with a 'unet.' prefix. The
    128-cubed checkpoints were saved NESTED, as {"unet": <state_dict>}, so that
    loader raises. This is the same flexible loader already used by
    tier_c/regen_table11/eval_128_uncond.py.

    Handled layouts:
        flat:     {"unet.conv1.weight": ...}
        stripped: {"conv1.weight": ...}
        nested:   {"unet": {...}}   <-- what the 128-cubed runs produced
        nested2:  {"model"|"state_dict"|"net": {...}}
    """
    raw = torch.load(ckpt_path, map_location=dev, weights_only=True)
    sd = raw
    if isinstance(raw, dict):
        for key in ("unet", "model", "state_dict", "net"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                break
    if isinstance(sd, dict) and any(k.startswith("unet.") for k in sd):
        sd = {k.replace("unet.", "", 1): v for k, v in sd.items()
              if k.startswith("unet.")}
    unet = DenoisingUNet3D(8, num_classes=num_classes).to(dev)
    unet.load_state_dict(sd)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    return unet

SEEDS = [42, 123, 456, 789, 1337]      # identical to the 64-cubed protocol
METHODS = ["consistency", "shortcut", "fm"]   # the three trained at 128-cubed
NFE = 50
LATENT = 16          # 128 / 8
LATENT_CH = 8


# ----------------------------------------------------------------------------
# P/R/D/C — pure numpy (sklearn's NearestNeighbors segfaults on some macOS
# Accelerate/PyTorch combinations at these matrix sizes). Definitions are
# byte-for-byte those of src/paper4/precision_recall_metrics.py.
# ----------------------------------------------------------------------------
def _sanitise(f, name=""):
    """Force float64 and scrub non-finite values, reporting loudly if any exist.

    A single inf in a feature matrix poisons every k-NN radius derived from it,
    which would silently corrupt Precision/Recall. Better to know.
    """
    f = np.asarray(f, dtype=np.float64)
    bad = ~np.isfinite(f)
    if bad.any():
        print(f"      ! {name}: {bad.sum()} non-finite values scrubbed "
              f"({100*bad.mean():.3f}% of entries)", flush=True)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f


def _pairwise_l2(a, b):
    """Exact pairwise Euclidean distance, float64.

    NOT the (|a|^2 + |b|^2 - 2ab) expansion: that cancels catastrophically in
    float32 at high dimension and was emitting inf/NaN warnings at 128-cubed.
    scipy.cdist computes the distances directly.
    """
    from scipy.spatial.distance import cdist
    return cdist(np.asarray(a, dtype=np.float64),
                 np.asarray(b, dtype=np.float64), metric="euclidean")


def _knn_radii(f, k):
    d = _pairwise_l2(f, f)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def precision_recall(real, gen, k=3):
    rr, gr = _knn_radii(real, k), _knn_radii(gen, k)
    d = _pairwise_l2(gen, real)
    precision = float(np.mean((d <= rr[None, :]).any(axis=1)))
    recall = float(np.mean((d.T <= gr[None, :]).any(axis=1)))
    return precision, recall


def density_coverage(real, gen, k=5):
    rr = _knn_radii(real, k)
    d = _pairwise_l2(gen, real)
    density = float((d <= rr[None, :]).sum() / (k * len(gen)))
    coverage = float(np.mean((d <= rr[None, :]).any(axis=0)))
    return density, coverage


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def build_teacher(path, device):
    m = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                  features=(32, 64, 128, 256, 32, 32)).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def teacher_features(model, vols, device, pool=True, bs=2):
    """Encoder path conv_0 -> down_1 -> ... -> down_4, flattened.

    pool=True average-pools the bottleneck to 4^3 so the dimensionality is
    2,048, exactly matching the 64-cubed protocol.
    """
    v = torch.nan_to_num(vols.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0)
    v = v.clamp(0, 1)
    if v.dim() == 4:
        v = v.unsqueeze(1)
    out = []
    for i in range(0, len(v), bs):
        x = v[i:i + bs].to(device)
        x = model.down_4(model.down_3(model.down_2(model.down_1(model.conv_0(x)))))
        if pool and x.shape[-1] > 4:
            # Reduce the bottleneck to 4^3 -> 32 x 4^3 = 2,048 dims, matching the
            # 64-cubed protocol exactly.
            #
            # NOTE: F.adaptive_avg_pool3d is NOT implemented on Apple MPS. At
            # 128-cubed the bottleneck is exactly 8^3, so the reduction to 4^3 is
            # an exact factor-2 average pool, which avg_pool3d does natively and
            # which is mathematically identical to the adaptive call. The general
            # case falls back to CPU rather than silently changing the feature
            # definition.
            f = x.shape[-1] // 4
            if x.shape[-1] % 4 == 0:
                x = F.avg_pool3d(x, kernel_size=f, stride=f)
            else:
                x = F.adaptive_avg_pool3d(x.cpu(), 4).to(x.device)
        out.append(x.reshape(x.shape[0], -1).cpu().numpy())
        del x
    return _sanitise(np.concatenate(out, 0), 'teacher features')


@torch.no_grad()
def vqgan_features(enc, vq, vols, device, bs=2):
    out = []
    for i in range(0, len(vols), bs):
        x = vols[i:i + bs].to(device).float()
        if x.dim() == 4:
            x = x.unsqueeze(1)
        z = enc(torch.nan_to_num(x, nan=0.0).clamp(0, 1))
        out.append(z.reshape(z.shape[0], -1).cpu().numpy())
        del x, z
    return _sanitise(np.concatenate(out, 0), 'VQ-GAN features')


# ----------------------------------------------------------------------------
# Distributional metrics
# ----------------------------------------------------------------------------
def radiomic_features(vols):
    """The 15 first-order radiomic features, computed by the SAME function the
    64-cubed results used (src/paper3/distributional_metrics.py)."""
    out = [extract_radiomic_features(np.nan_to_num(v.squeeze().numpy().astype(np.float64), nan=0.0))
           for v in vols]
    return _sanitise(np.asarray(out, dtype=np.float64), "radiomic features")




def pairwise_l1(vols, n_pairs=2016):
    x = vols.reshape(len(vols), -1).float()
    n = len(x)
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += (x[i] - x[j]).abs().mean().item()
            cnt += 1
            if cnt >= n_pairs:
                return tot / cnt
    return tot / max(cnt, 1)


# ----------------------------------------------------------------------------
@torch.no_grad()
def sample_128(unet, dec, method, n, dev, steps=NFE, bs=2):
    """128-cubed sampler.

    models_shared.sample_latent() hardcodes an 8^3 latent grid, so it cannot be
    used at 128-cubed (latent 16^3). The per-method update rules below are
    transcribed EXACTLY from models_shared.sample_latent so that the only
    difference from the 64-cubed protocol is the latent size. In particular
    Consistency at steps > 1 re-noises between steps, which is the behaviour
    the 64-cubed numbers were produced with.
    """
    out = []
    rem = n
    while rem > 0:
        b = min(bs, rem)
        z = torch.randn(b, LATENT_CH, LATENT, LATENT, LATENT, device=dev)

        if method == "shortcut":
            d_val = 1.0 / steps
            for i in range(steps):
                t = torch.full((b,), i * d_val, device=dev)
                d = torch.full((b,), d_val, device=dev)
                z = z + d_val * unet(z, t, d=d)

        elif method == "consistency":
            if steps == 1:
                t = torch.zeros(b, device=dev)
                z = z + unet(z, t)
            else:
                ts = torch.linspace(0, 1.0 - 1.0 / steps, steps, device=dev)
                for i, t_val in enumerate(ts):
                    t = torch.full((b,), t_val.item(), device=dev)
                    v = unet(z, t)
                    x_hat = z + (1 - t_val) * v
                    if i < steps - 1:
                        t_next = ts[i + 1].item()
                        z = (1 - t_next) * torch.randn_like(z) + t_next * x_hat
                    else:
                        z = x_hat

        else:  # fm (and rectified)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((b,), i * dt, device=dev)
                z = z + unet(z, t) * dt

        x = dec(z).clamp(0, 1).cpu()
        out.append(x)
        rem -= b
        del z, x
    return torch.cat(out, 0)


def free(dev):
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/paper4/brats_128cubed")
    ap.add_argument("--data-path", default="data/brats_preprocessed_128.pt")
    ap.add_argument("--teacher", default="results/paper4/brats_128cubed_conditional/"
                                         "e7_teacher_128/teacher_best.pt")
    ap.add_argument("--out", default="results/paper4/manifold_metrics_128/"
                                     "manifold_metrics_128.json")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-pool", action="store_true",
                    help="also report the unpooled 16,384-dim teacher space")
    a = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {dev}")

    run = ROOT / a.run_dir
    for p in [run / "phase1_shared.pt", ROOT / a.data_path, ROOT / a.teacher]:
        if not p.exists():
            sys.exit(f"MISSING: {p}")

    # ---- real data ---------------------------------------------------------
    blob = torch.load(ROOT / a.data_path, map_location="cpu", weights_only=False)
    real = blob["volumes"] if isinstance(blob, dict) else blob
    if real.dim() == 4:
        real = real.unsqueeze(1)
    real = real[:128].float()
    print(f"  real: {len(real)} volumes @ {tuple(real.shape[2:])}")

    enc, dec, vq, _ = load_vqgan(run / "phase1_shared.pt", dev)
    teacher = build_teacher(ROOT / a.teacher, dev)

    real_t = teacher_features(teacher, real, dev, pool=not a.no_pool)
    real_v = vqgan_features(enc, vq, real, dev)
    real_r = radiomic_features(real)
    real_div = pairwise_l1(real[:64])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(128, len(real_v))).fit(real_v)   # fitted on real only, once
    print(f"  teacher feature dim: {real_t.shape[1]}   real diversity: {real_div:.5f}")

    results = {}
    for meth in METHODS:
        ck = run / meth / "final.pt"
        if not ck.exists():
            print(f"  !! no checkpoint for {meth}, skipping"); continue
        unet = load_unet_flex(ck, dev)
        per_seed = {k: [] for k in
                    ["precision", "recall", "density", "coverage",
                     "vq_precision", "vq_recall", "diversity", "frd", "latent_fid"]}
        for sd in a.seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            gen = sample_128(unet, dec, meth, a.n_samples, dev)

            gt = teacher_features(teacher, gen, dev, pool=not a.no_pool)
            p, r = precision_recall(real_t, gt, k=3)
            d, c = density_coverage(real_t, gt, k=5)

            gv = vqgan_features(enc, vq, gen, dev)
            vp, vr = precision_recall(real_v, gv, k=3)

            gr = radiomic_features(gen)

            per_seed["precision"].append(p);  per_seed["recall"].append(r)
            per_seed["density"].append(d);    per_seed["coverage"].append(c)
            per_seed["vq_precision"].append(vp); per_seed["vq_recall"].append(vr)
            per_seed["diversity"].append(pairwise_l1(gen))
            per_seed["frd"].append(compute_frd(real_r, gr))
            per_seed["latent_fid"].append(
                compute_latent_fid(pca.transform(real_v), pca.transform(gv)))
            print(f"    {meth:12s} seed {sd:4d}  P={p:.3f} R={r:.3f} "
                  f"D={d:.3f} C={c:.3f} | VQ-space R={vr:.3f}", flush=True)
            del gen, gt, gv; free(dev)

        results[meth] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                             "seeds": [float(x) for x in v]}
                         for k, v in per_seed.items()}
        results[meth]["diversity_pct_real"] = 100.0 * np.mean(per_seed["diversity"]) / real_div
        del unet; free(dev)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "128-cubed manifold metrics in a task-independent feature space. "
                   "Closes the gap where the paper's own recommendation (measure Recall "
                   "outside the generator) was demonstrated only at 64-cubed.",
        "resolution": 128,
        "nfe": NFE,
        "seeds": a.seeds,
        "n_samples_per_seed": a.n_samples,
        "feature_space": ("MONAI BasicUNet 128-cubed tumour-segmentation teacher; encoder "
                          "bottleneck average-pooled to 4^3 -> 2,048 dims, matching the "
                          "64-cubed protocol exactly"
                          if not a.no_pool else
                          "unpooled bottleneck, 16,384 dims"),
        "teacher_path": a.teacher,
        "real_diversity": real_div,
        "pr_definitions": "Kynkaanniemi et al. 2019 P/R at k=3; Naeem et al. 2020 D/C at k=5.",
        "results": results,
    }, open(out, "w"), indent=1)

    print("\n" + "=" * 74)
    print("  128-CUBED, TASK-INDEPENDENT TEACHER SPACE")
    print("=" * 74)
    print(f"  {'method':14s} {'Precision':>9s} {'Recall':>8s} {'Coverage':>9s} {'Div %real':>10s}")
    for m, v in results.items():
        print(f"  {m:14s} {v['precision']['mean']:9.3f} {v['recall']['mean']:8.3f} "
              f"{v['coverage']['mean']:9.3f} {v['diversity_pct_real']:9.1f}%")
    print("\n  For comparison, the 64-cubed result (BraTS):")
    print("    consistency    Recall = 0.000   |   shortcut/fm/rectified/ddpm = 0.52 to 0.73")
    print(f"\n  written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
