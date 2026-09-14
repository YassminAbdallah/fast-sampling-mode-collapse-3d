#!/usr/bin/env python3
"""
Re-evaluate the 64-cubed BraTS teacher segmenter on the 60-volume held-out test
set and write its metrics to results/paper4/paper4/e7_teacher/teacher_metrics.json.

The original training script saved the teacher checkpoint but did not store its
test Dice; this script recomputes it from the saved checkpoint so that the value
reported in the paper (Section 4.4, Appendix A.5) traces to a result file like
every other number. Inference only; no training.

Usage:
    python3 fixes/06_recover_teacher_dice.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import BasicUNet

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results/paper4/paper4/e7_teacher/teacher_best.pt"
DATA = ROOT / "data/brats_conditional_64.pt"
OUT  = ROOT / "results/paper4/paper4/e7_teacher/teacher_metrics.json"

# Same split as §4.1 and as used by conditional_segmentation.py: 180 / 60 / 60.
N_TRAIN, N_VAL, N_TEST = 180, 60, 60
SPLIT_SEED = 42


def dice(pred, gt, eps=1e-6):
    """Binary Dice on one volume."""
    p = (pred > 0.5).float().flatten()
    g = (gt > 0.5).float().flatten()
    inter = (p * g).sum()
    return ((2 * inter + eps) / (p.sum() + g.sum() + eps)).item()


def main():
    for f in (CKPT, DATA):
        if not f.exists():
            raise SystemExit(f"Missing: {f}")

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {dev}")

    # --- data -------------------------------------------------------------
    blob = torch.load(DATA, map_location="cpu", weights_only=False)
    vols = blob["volumes"] if isinstance(blob, dict) else blob[0]
    segs = blob["segs"]    if isinstance(blob, dict) else blob[1]
    if vols.dim() == 4:
        vols = vols.unsqueeze(1)
    if segs.dim() == 4:
        segs = segs.unsqueeze(1)
    print(f"  loaded {len(vols)} volumes, shape {tuple(vols.shape[2:])}")

    # Reproduce the split used to train the teacher.
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(vols))
    test_idx = idx[N_TRAIN + N_VAL: N_TRAIN + N_VAL + N_TEST]
    print(f"  test set: {len(test_idx)} volumes (split seed {SPLIT_SEED})")

    # --- model ------------------------------------------------------------
    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                      features=(32, 64, 128, 256, 32, 32)).to(dev)
    state = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state
                          else state["state_dict"])
    model.eval()

    # --- evaluate ---------------------------------------------------------
    scores = []
    with torch.no_grad():
        for i in test_idx:
            x = vols[i].unsqueeze(0).to(dev).float()
            y = segs[i].unsqueeze(0).to(dev).float()
            logits = model(x)
            pred = torch.softmax(logits, dim=1)[:, 1]      # tumour channel
            scores.append(dice(pred.cpu(), y.cpu()))

    scores = np.array(scores)
    mean = float(scores.mean())

    # bootstrap CI, matching the protocol used elsewhere
    boot = np.array([np.mean(np.random.choice(scores, len(scores), replace=True))
                     for _ in range(1000)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "purpose": "Recovered test Dice for the 64-cubed BraTS teacher segmenter. "
                   "The original training run saved the checkpoint but never stored "
                   "this metric, leaving the paper's reported 0.828 unverifiable.",
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "n_test_volumes": len(test_idx),
        "split_seed": SPLIT_SEED,
        "test_dice": mean,
        "test_dice_std": float(scores.std(ddof=1)),
        "ci95": [lo, hi],
        "per_volume_dice": scores.tolist(),
    }, open(OUT, "w"), indent=1)

    print()
    print("=" * 62)
    print(f"  RECOVERED test Dice : {mean:.4f}   (95% CI {lo:.4f} to {hi:.4f})")
    print(f"  PAPER reports       : 0.828")
    print("=" * 62)
    if abs(mean - 0.828) < 0.003:
        print("  MATCHES. The paper is correct. Cite:")
        print(f"    {OUT.relative_to(ROOT)}")
    else:
        print("  DOES NOT MATCH.")
        print(f"  Update the paper to {mean:.3f} in Section 4.4 and Appendix A.5.")
        print("  Report what this prints. Do not adjust it to fit the manuscript.")
    print()
    print(f"  written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
