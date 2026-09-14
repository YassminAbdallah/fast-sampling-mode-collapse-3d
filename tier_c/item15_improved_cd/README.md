# Item 15 — Improved Consistency Distillation (no-EMA variant)

The second Tier C addition: re-train Consistency Distillation **without an EMA target network** and **with a Pseudo-Huber loss**, following Song & Dhariwal (2023b) "Improved Techniques for Training Consistency Models." Everything else (architecture, dataset, FM teacher, training epochs, learning rate schedule, step-count curriculum) is held identical to the original Consistency Distillation variant already in the paper.

## Why this matters

Section 6.2 of the paper claims that Consistency Distillation's mode collapse is driven by **EMA in parameter space** during distillation. Right now that claim rests on the four falsifiable predictions P1–P4 (class symmetry, monotonic step-wise degradation, etc.). Item 15 lets us add a direct test: remove the EMA, keep everything else the same, and see what happens.

Two outcomes, both useful:

- **Improved CD has higher diversity than original CD** → direct confirmation of the EMA hypothesis. Strongest possible mechanistic evidence for §6.2.
- **Improved CD still collapses** → the collapse is more fundamental to consistency-style distillation than the EMA component alone. Strengthens the "fast distillation is intrinsically diversity-limited" framing.

There is no "bad" outcome for the paper.

## What you run

Two scripts, in sequence:

### Step 1 — Train improved CD

```bash
cd <path-to-repo>/

PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item15_improved_cd/train_improved_cd.py \
    --gen-dir results/paper4/brats_benchmark_20260324_154926 \
    --data-path data/brats_conditional_64.pt \
    --output-dir results/paper4/improved_cd_brats \
    --num-classes 2 \
    --epochs 80 \
    > tier_c/item15_improved_cd/train.log 2>&1 &

# Watch progress
tail -f tier_c/item15_improved_cd/train.log
```

The script will:
- Auto-detect the `100pct_180vol/` subfolder inside `--gen-dir`
- Load `phase1_shared.pt` (VQ-GAN) and `fm/final.pt` (FM teacher) from there
- Initialize the student from the FM teacher (same as original CD)
- Train for 80 epochs with no-EMA + Pseudo-Huber loss
- Save the checkpoint to `results/paper4/improved_cd_brats/improved_cd/final.pt`
- Save a training log to `results/paper4/improved_cd_brats/training_log.json`

**Expected wall time on Apple M-series: ~9 hours**. You should see progress lines every 5 epochs:

```
[  1/80] improved_cd=0.012345 N=2 | 8.4s
[  5/80] improved_cd=0.008765 N=10 | 8.2s
[ 10/80] improved_cd=0.006543 N=20 | 8.1s
...
```

### Step 2 — Evaluate improved CD (after training finishes)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item15_improved_cd/eval_improved_cd.py \
    --improved-cd-dir results/paper4/improved_cd_brats/improved_cd \
    --gen-dir results/paper4/brats_benchmark_20260324_154926 \
    --data-path data/brats_conditional_64.pt \
    --output-dir results/paper4/improved_cd_brats/paper_eval_5seed \
    --num-classes 2 \
    --n-samples 64 \
    > tier_c/item15_improved_cd/eval.log 2>&1 &

tail -f tier_c/item15_improved_cd/eval.log
```

The evaluation generates 64 volumes × 5 seeds × 4 step counts (1, 4, 16, 50) and computes the same metrics as the existing `evaluate_5seed.py`: windowed 3D SSIM, PSNR, pairwise-L1 diversity, and bootstrap 95% confidence intervals.

**Expected wall time: 30–60 minutes**.

## What it produces

| File | Contents |
|------|----------|
| `results/paper4/improved_cd_brats/improved_cd/final.pt` | Trained improved-CD checkpoint |
| `results/paper4/improved_cd_brats/training_log.json` | Per-epoch losses and timings |
| `results/paper4/improved_cd_brats/paper_eval_5seed/eval_results_5seed.json` | **The main result.** Same structure as `eval_results_5seed.json` for the existing methods |
| `tier_c/item15_improved_cd/train.log`, `eval.log` | Stdout logs |

The eval script will also print a Table-3-style preview at the end, showing what the row should look like in the paper:

```
============================================================
  Improved CD — Table-3 row preview
============================================================
  NFE              SSIM         Diversity   %Real
    1   0.787±0.001  0.0058±0.0003     12%
    4   0.789±0.001  0.0048±0.0002     10%
   16   0.787±0.001  0.0042±0.0002      8%
   50   0.787±0.001  0.0039±0.0002      8%
```

(The numbers above are placeholders — actual values are what we're measuring.)

## Output

The result file:

```
results/paper4/improved_cd_brats/paper_eval_5seed/eval_results_5seed.json
```


## Sanity checks before launching

```bash
# Verify the FM teacher exists where the script expects
ls -la results/paper4/brats_benchmark_20260324_154926/100pct_180vol/fm/final.pt
# Verify VQ-GAN exists
ls -la results/paper4/brats_benchmark_20260324_154926/100pct_180vol/phase1_shared.pt
```

Both files must be present. If either is missing the script will fail immediately with a clear FileNotFoundError.

## If something fails

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError: phase1_shared.pt` | Wrong `--gen-dir` path | Use the same gen-dir that worked for Item 16 |
| `ModuleNotFoundError: flow_matching_3d` | Run from wrong directory | Always run from the repository root |
| Loss explodes (>10) after epoch 5 | Pseudo-Huber c too small | Increase the Pseudo-Huber constant |
| Loss does not decrease | Teacher state didn't load correctly | Check that the printed teacher path matches the existing FM teacher |
| MPS OOM | Reduce `--batch-size 4` to `--batch-size 2` | Just edit the command |
