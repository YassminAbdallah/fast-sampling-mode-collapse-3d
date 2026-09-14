# Item 18 — Cross-dataset generalization (UPenn-GBM)

The third Tier C addition: tests whether augmentation using the existing **BraTS-trained** Shortcut@50 model transfers to a different glioma dataset (UPenn-GBM). **No new generative training** is performed — we re-use the existing trained Shortcut FM checkpoint at `results/paper4/brats_benchmark_20260324_154926/100pct_180vol/shortcut/final.pt`.

## Why this matters

The paper currently shows that Shortcut@50 augmentation improves tumor segmentation Dice by up to 7 points on BraTS 2023 GLI. Item 18 tests whether this transfers to a different glioma dataset.

Two outcomes are both useful for the paper:

- **Augmentation transfers (similar Dice gain on UPenn-GBM)** — strong evidence of dataset-agnostic generalization, expanding the practical reach of the augmentation claim.
- **Augmentation does not transfer (or transfers partially)** — honest scope limit; a measured negative result is more credible than no test at all.

## Three-phase workflow

```
Step 1 — Download UPenn-GBM           (you)        ~30 minutes
Step 2 — Preprocess to 64³            (script)     ~20 minutes
Step 3 — Run cross-dataset experiment (script)     ~10-12 hours
```

---

## Step 1 — Download UPenn-GBM from TCIA

UPenn-GBM is a public glioblastoma dataset hosted on The Cancer Imaging Archive:

- Collection page: **https://www.cancerimagingarchive.net/collection/upenn-gbm/**
- Direct download: requires the NBIA Data Retriever tool (or the TCIA REST API).
- License: CC BY 4.0 (open access, free for research).
- For our experiment we only need ~60 subjects with **T2-FLAIR** + **segmentation mask**.

**Recommended:** download the "imaging" subset only (skip the genomic/clinical metadata). This is ~3 GB.

Folder structure should end up looking like:

```
/path/to/UPenn-GBM/
├── UPENN-GBM-00001/
│   ├── UPENN-GBM-00001_FLAIR.nii.gz
│   ├── UPENN-GBM-00001_T1.nii.gz
│   ├── UPENN-GBM-00001_T1GD.nii.gz
│   ├── UPENN-GBM-00001_T2.nii.gz
│   └── UPENN-GBM-00001_automated_approx_segm.nii.gz
├── UPENN-GBM-00002/
│   └── ...
...
```

The preprocessing script auto-detects the FLAIR and segmentation files within each subject folder — variant filename conventions are tolerated.

Smallest viable subset: ~60 subjects (≈3 GB). More is better but ~60 gives statistically meaningful test results.

---

## Step 2 — Preprocess to 64³

From the repository root:

```bash
cd <path-to-repo>/

python tier_c/item18_cross_dataset/preprocess_upenn.py \
    --input-dir /path/to/UPenn-GBM/ \
    --output-dir data/ \
    --max-subjects 100
```

This script:

- Iterates subjects, finds T2-FLAIR + segmentation per subject
- Resamples to 64×64×64 (trilinear for volumes, nearest-neighbor for masks)
- Min-max normalizes intensity to [0, 1] using non-zero voxels
- Binarizes the segmentation (any tumor label → 1) to match the existing pipeline
- Saves `data/upenn_volumes_64.pt` and `data/upenn_seg_64.pt`

Expected wall time: ~20 minutes for ~60 subjects.

If the script fails on filename detection, it prints what it found so you can either point it at a subdirectory or use the `--subj-pattern` argument.

Dependencies: `nibabel` (`pip install nibabel`), `torch`, `numpy`.

---

## Step 3 — Run the cross-dataset experiment

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item18_cross_dataset/cross_dataset_upenn.py \
    --upenn-vols data/upenn_volumes_64.pt \
    --upenn-segs data/upenn_seg_64.pt \
    --gen-dir results/paper4/brats_benchmark_20260324_154926 \
    --output-dir results/paper4/cross_dataset_upenn \
    --num-classes-gen 2 \
    --n-seeds 3 \
    --teacher-iters 3000 \
    --seg-iters 2000
```

This runs four phases internally:

| Phase | Description | Time on M-series |
|-------|-------------|------------------|
| 1 | Train a UPenn-GBM teacher segmenter on the train portion (~36 vol) | ~1.5–2 hours |
| 2 | Generate 500 Shortcut@50 synthetic volumes from existing BraTS checkpoint | ~1–2 minutes |
| 3 | Pseudo-label the 500 synthetic volumes with the UPenn teacher | ~2 minutes |
| 4 | Three conditions × 3 seeds × 2000 iters = 9 segmenter runs | ~7–9 hours |
| **Total** | | **~10–12 hours** |

The script saves intermediate state after each phase, so a crash mid-run loses at most one phase.

Best run overnight, inside `tmux` or `caffeinate -i` so the laptop doesn't sleep.

---

## What gets produced

Inside `results/paper4/cross_dataset_upenn/`:

| File | Contents |
|------|----------|
| `upenn_teacher.pt` | Trained UPenn-GBM binary teacher segmenter |
| `teacher_metrics.json` | UPenn teacher's test Dice (sanity check) |
| `shortcut50_500vols.pt` | Cached 500 synthetic volumes from BraTS Shortcut@50 |
| `upenn_results.json` | **The main result.** Per-condition Dice + per-seed values for all three conditions |

The script also prints a summary table at the end:

```
============================================================
  UPenn-GBM cross-dataset summary (10% real = N vol)
============================================================
              real_only  Dice = 0.XXXX ± 0.XXXX
    real_classical_aug  Dice = 0.XXXX ± 0.XXXX
       real_shortcut50  Dice = 0.XXXX ± 0.XXXX
```

That's the key information for the paper.

---

## Output

The result file:

```
results/paper4/cross_dataset_upenn/upenn_results.json
```


## Sanity checks before launching

```bash
# Verify the BraTS Shortcut FM checkpoint exists where the script expects
ls -la results/paper4/brats_benchmark_20260324_154926/100pct_180vol/shortcut/final.pt
# Verify VQ-GAN exists
ls -la results/paper4/brats_benchmark_20260324_154926/100pct_180vol/phase1_shared.pt
# Verify preprocessed UPenn data
ls -la data/upenn_volumes_64.pt data/upenn_seg_64.pt
```

All four files must be present.

## If something fails

| Symptom | Likely cause | Quick fix |
|---------|--------------|-----------|
| Preprocess script finds 0 subjects | Wrong `--input-dir` level | Run `ls` on the input dir; if subjects are one level deeper, point `--input-dir` at that subdirectory |
| Preprocess script logs "missing FLAIR or seg" for most subjects | UPenn-GBM file naming differs from defaults | Pass a matching `--subj-pattern` |
| UPenn teacher Dice <0.40 after 3000 iters | Teacher needs more iterations OR data quality issues | Try `--teacher-iters 5000` |
| Cross-dataset training crashes with OOM | Edit script to use batch_size 2 |  |
| Cross-dataset finishes with all three conditions at similar Dice (no augmentation effect) | Possible: BraTS-trained generator doesn't transfer | This IS a result — send the JSON and we'll write it up honestly |
