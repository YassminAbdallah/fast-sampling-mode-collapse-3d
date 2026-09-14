# Item 16 — Multi-class BraTS dose-response

This is the first Tier C addition: re-run the E10a diversity-isolation experiment with multi-class segmentation (4 classes) instead of binary, and report the standard BraTS WT / TC / ET Dice.

## Why this matters

The binary E10a established that ~25 unique synthetic anatomies are sufficient for binary tumor segmentation at 64³. A natural question is whether the threshold holds for harder tasks. Multi-class BraTS is the natural harder task: it must distinguish necrotic core (label 1), edema (label 2), and enhancing tumor (label 3) — the standard BraTS evaluation protocol.

The result tells us one of two things, both useful for the paper:
- **Threshold preserved (~25 unique still saturates):** the diversity threshold is robust to task complexity at 64³.
- **Threshold shifts upward (e.g., to 50–100 unique):** harder tasks need more diversity, a quantifiable result we can report.

## What you run

A single Python script. It runs three phases internally:

| Phase | Description | Time on M-series |
|-------|-------------|------------------|
| 1 | Train multi-class teacher segmenter on all real BraTS (5000 iters) | ~2 hours |
| 2 | Generate 500 Shortcut@50 synthetic volumes (cached after first run) | ~5 minutes |
| 3 | Use teacher to produce multi-class pseudo-labels for the 500 vols | ~5 minutes |
| 4 | Dose-response: 4 unique-counts × 3 seeds + real-only × 3 seeds (15 segmenter runs × ~70 min) | ~14 hours |
| **Total** | | **~17 hours** |

Best run overnight. The script saves intermediate state after each phase, so if it crashes mid-run you can restart and most work will be preserved.

## Prerequisites

Before launching, verify you have:

- The repository structure intact.
- `data/brats_conditional_64.pt` — already there (300 volumes with class labels).
- `data/brats_seg_preprocessed_64.pt` — already there (300 multi-class segmentations).
- A trained Shortcut FM checkpoint at `results/paper4/brats_benchmark_20260324_154926/shortcut/final.pt`. Confirm it exists with:
  ```bash
  ls results/paper4/brats_benchmark_20260324_154926/shortcut/final.pt
  ```
- The corresponding VQ-GAN checkpoint at `results/paper4/brats_benchmark_20260324_154926/vqgan_final.pt`.
- Python dependencies: `pip install monai torch numpy matplotlib scikit-learn`

## How to run

From the repository root:

```bash
python tier_c/item16_multiclass/multiclass_e10a.py \
    --data-path data/brats_conditional_64.pt \
    --seg-path  data/brats_seg_preprocessed_64.pt \
    --gen-dir   results/paper4/brats_benchmark_20260324_154926 \
    --output-dir results/paper4/paper4/e10a_isolation_multiclass \
    --n-seeds 3 \
    --teacher-iters 5000 \
    --seg-iters 2000
```

Best practice: run inside `tmux` or `screen` so a disconnection doesn't kill it.

```bash
# Optional: run in tmux for safety
tmux new -s tier_c_item16
# then inside the session:
python tier_c/item16_multiclass/multiclass_e10a.py [args as above] 2>&1 | tee item16_run.log
# Detach with Ctrl-B then D. Re-attach later with: tmux attach -t tier_c_item16
```

## What gets produced

Inside `results/paper4/paper4/e10a_isolation_multiclass/`:

| File | Content |
|------|---------|
| `teacher_multiclass.pt` | Trained multi-class teacher segmenter |
| `teacher_metrics.json` | Teacher's WT/TC/ET Dice on the test set (sanity check) |
| `shortcut50_500vols.pt` | Cached 500 Shortcut@50 synthetic volumes |
| `e10a_multiclass_results.json` | **The main result** — WT/TC/ET Dice per condition, all seeds |
| `e10a_multiclass_dose_response.png` and `.pdf` | Dose-response figure with 4 panels (WT, TC, ET, Mean) |

## Sanity checks to do once it finishes

1. **Teacher Dice should be reasonable.** Multi-class BraTS at 64³ is hard. Expected ranges (rough): WT ~0.75–0.85, TC ~0.50–0.70, ET ~0.30–0.55, mean ~0.55–0.65. A mean Dice below 0.40 indicates a problem with the labels or the training.
2. **Real-only baseline.** Should be roughly similar to the binary E10a real-only (~0.76 for WT, lower for TC/ET). If it's near zero on TC/ET, the multi-class labels may not have loaded correctly.
3. **Curve shape.** Across the 4 unique-count conditions, mean Dice should be non-decreasing. A monotonic plateau above ~25 unique would reproduce the binary finding.

## Output

The result file:

```
results/paper4/paper4/e10a_isolation_multiclass/e10a_multiclass_results.json
```

That's all I need to write the v4 paper update. Optional but helpful: also send the `.png` figure and the `teacher_metrics.json`.

## If something fails

| Symptom | Likely cause | Quick fix |
|---------|--------------|-----------|
| `ModuleNotFoundError: monai` | MONAI not installed | `pip install monai` |
| `RuntimeError: MPS out of memory` | Batch size too large for shared M-series memory | Add `--batch-size 2` (edit the script to expose this flag) |
| `FileNotFoundError: shortcut/final.pt` | The Shortcut FM checkpoint path is wrong | Verify the actual location and pass `--gen-dir <correct-path>` |
| Multi-class teacher trains to very low Dice (<0.30) | Labels may not be in {0,1,2,3} as expected | Check `torch.unique(segs).tolist()` on the data file and remap the labels if needed |
| Run dies after 8 hours with no log | Background process killed (e.g., laptop slept) | Run inside `tmux` and add `caffeinate -i` prefix on macOS to prevent sleep |

