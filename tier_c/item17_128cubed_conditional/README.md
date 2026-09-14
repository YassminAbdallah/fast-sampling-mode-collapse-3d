# Conditional 128³ retraining (Weeks 3–5)

This subdirectory adds the **conditional** companion to the unconditional 128³ pipeline in `tier_c/item17_128cubed/`. The unconditional run reproduces the §5.8 Shortcut-vs-Consistency comparison at higher resolution; this conditional run reproduces the §5.4 conditional benchmark (Table 6) at higher resolution and, downstream, enables the §5.5 dose-response (E10a) and §5.6 augmentation comparison (E10b) at 128³.

Together with the existing unconditional 128³ models, completing this pipeline gives the v7.x revision the full set of 128³ checkpoints needed for §5.4 (conditional benchmark), §5.5 (E10a dose-response), and §5.6 (E10b augmentation comparison) — i.e. the experiments the peer review identified as the gating item for top-tier Q1 (MedIA / IEEE TMI) submission.

## Critical-path timeline

On an Apple M-series laptop GPU (24–32 GB unified memory, batch_size = 2):

| Step | Wall-clock | Notes |
|------|-----------|-------|
| 0. `prepare_conditional_128.py` | ~5 minutes | combines existing 128³ volumes + 64³ labels |
| 1. Phase 1 — VQ-GAN | ~24 h | identical training to unconditional Phase 1 — could reuse existing checkpoint instead |
| 2. Phase 2 — Conditional FM teacher | ~24 h | |
| 3. Phase 3 — Conditional Consistency Distillation | ~12 h | distills from Phase 2 |
| 4. Phase 4 — Conditional Shortcut FM | ~24 h | independent of Phase 3, can be parallelized on a second device |
| **Total** | **~85 h** (~4 calendar days continuous) | |

Followed by:

| Step | Wall-clock | Notes |
|------|-----------|-------|
| E10a at 128³ (4 unique-count × 5 seeds) | ~36–48 h | needs Phase 4 to be complete |
| Partial E10b at 128³ (Shortcut@50 vs real-only, 10% data, 5 seeds) | ~30 h | needs Phase 4 |

So end-to-end the conditional 128³ pipeline plus the downstream re-runs lands at roughly **3 weeks of wall-clock** if you run continuously. Launching Phase 1 today therefore sets the schedule for the final paper.

## Shortcut: reuse the existing unconditional VQ-GAN

The unconditional Phase 1 already produced a working VQ-GAN at 128³ in `results/paper4/brats_128cubed/phase1_shared.pt`. The VQ-GAN itself does not consume class labels (only image reconstruction matters), so you can save ~24 hours by symlinking the existing checkpoint into the conditional output directory before launching Phase 2:

```bash
mkdir -p results/paper4/brats_128cubed_conditional/
ln -sf $(pwd)/results/paper4/brats_128cubed/phase1_shared.pt \
       results/paper4/brats_128cubed_conditional/phase1_shared.pt
```

If you do this, skip Phase 1 in the steps below and start with Phase 2.

## Steps

### 0. Build the conditional 128³ dataset

```bash
cd fast-sampling-mode-collapse-3d/
python tier_c/item17_128cubed_conditional/prepare_conditional_128.py
```

This pairs the existing `data/brats_preprocessed_128.pt` (300 volumes at 128³) with the labels from `data/brats_conditional_64.pt`, producing `data/brats_conditional_128.pt`. The pairing is positional and assumes both preprocessing pipelines selected subjects in the same alphabetical-sort order (which they do — see §5.8 v7.1 cohort footnote). If you'd rather rederive labels at 128³ directly from the segmentation masks, pass `--rederive-labels` (requires `data/brats_seg_preprocessed_128.pt`); the labels usually match the 64³-derived labels exactly because tumor-volume median split is highly stable under 64↔128 resampling.

### 1. Phase 1 — VQ-GAN (or skip via symlink, see above)

```bash
bash tier_c/item17_128cubed_conditional/run_phase1_vqgan.sh
```

### 2. Phase 2 — Conditional FM teacher

```bash
bash tier_c/item17_128cubed_conditional/run_phase2_fm.sh
```

### 3. Phase 3 — Conditional Consistency Distillation

```bash
bash tier_c/item17_128cubed_conditional/run_phase3_consistency.sh
```

### 4. Phase 4 — Conditional Shortcut FM

```bash
bash tier_c/item17_128cubed_conditional/run_phase4_shortcut.sh
```

Phase 4 is **independent of Phase 3**. If you have access to a second machine or can fit both at the same time, run Phases 3 and 4 in parallel.

## What gets saved

```
results/paper4/brats_128cubed_conditional/
├── phase1_shared.pt              # VQ-GAN enc/dec/vq
├── phase1_log.json
├── fm/
│   ├── final.pt                  # Conditional FM teacher
│   └── training_log.json
├── consistency/
│   ├── final.pt                  # Conditional Consistency student
│   └── training_log.json
└── shortcut/
    ├── final.pt                  # Conditional Shortcut FM
    └── training_log.json
```

## Reproducibility

All four training scripts call the `set_seed(seed)` helper from `src/utils/set_seed.py` (added in v7.2) with `--seed 42` as the default. This sets Python/numpy/torch CPU/CUDA/MPS RNGs and enables `torch.use_deterministic_algorithms(True)` plus the cuDNN deterministic flags. On Apple MPS, some reductions are still non-deterministic; the cross-backend tolerance for these runs is documented in `requirements.txt`.

## Sleep prevention on macOS

Every `run_phaseN_*.sh` script wraps Python in `PYTHONUNBUFFERED=1 caffeinate -i python -u …`. The `caffeinate -i` prevents idle sleep so multi-day training survives overnight unattended; the display is still allowed to sleep (saves power). Do **not** run `python …` directly without `caffeinate` for these long jobs — the Mac will idle-sleep after ~10 minutes and the training process freezes mid-epoch (it doesn't die outright, but no further progress occurs until you wake the machine, which silently extends every multi-day job by however long the laptop slept). If you'd rather run the python command manually, prefix it with `caffeinate -i`:

```bash
caffeinate -i python tier_c/item17_128cubed_conditional/train_pipeline_128_conditional.py …
```

You can also use `caffeinate -is` (also prevents sleep when on battery), but the default `-i` is correct for plugged-in laptops.

## Sanity checks after each phase

After each phase, before moving on, confirm:

* **Phase 1 VQ-GAN.** Final recon loss should be in the 0.020–0.030 range, similar to the unconditional 128³ run (recon 0.024 at epoch 80).
* **Phase 2 FM teacher.** Training loss should converge from ~0.55 to ~0.08 over 80 epochs. If it plateaus high (> 0.20), check that `class_label` is actually reaching the U-Net forward call.
* **Phase 3 Consistency.** Loss should drop several orders of magnitude (e.g. 8e-4 → 1e-5 region) as in the unconditional 128³ run. EMA-driven near-zero loss is the expected signature.
* **Phase 4 Shortcut.** Training loss should land in the 0.07–0.10 range, with `d_max` advancing from 0.02 → 1.0 across the schedule.

If any phase looks badly off, stop and inspect the conditional dataset (`prepare_conditional_128.py` output) before running the next phase.

## Why this matters for the paper

The §5.4–§5.6 results are at 64³ resolution only. The §5.8 unconditional 128³ result rules out a basic "all at 64³" artifact, but only for the upstream Shortcut-vs-Consistency comparison. Completing this pipeline closes that gap on the conditional Table 6 numbers, and unlocks the downstream re-runs that genuinely demonstrate the augmentation result at clinically more credible resolution.
