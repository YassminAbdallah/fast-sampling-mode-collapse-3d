# 64-cubed E10a dose-response — 5-seed replication

Motivation: the paper's headline threshold claim (§5.5, Table 8) rests on n = 3 seeds and none of the individual comparisons survive Bonferroni. This runs the same experiment at n = 5 so the primary claim becomes corrected-significant in the main setting.

## Design

Identical to the paper's existing 64³ E10a (`src/paper4/conditional_segmentation.py::run_e10a`), just with 5 seeds instead of 3:

- **5 conditions**: `real_only`, `unique_10`, `unique_25`, `unique_100`, `unique_500`
- **5 seeds**: 42, 43, 44, 45, 46 (seeds 42–44 match the paper's existing run bit-for-bit; 45–46 are new)
- **Fixed 500 synthetic volumes per condition** — only the number of *unique* anatomies varies (10, 25, 100, 500), so total volume count is not a confound
- **Same shared 64³ VQ-GAN, same Shortcut FM model, same teacher segmenter** as the paper's E10a
- **Statistics**: paired-t on per-seed Dice, Bonferroni family size k = 8 (matches Appendix A.6 Table A.6b)

Total: 5 × 5 = 25 downstream segmenter training runs.

## Required paths (already set as defaults in the script)

| Path | Purpose |
|---|---|
| `data/brats_conditional_64.pt` | 64³ conditional BraTS |
| `results/paper4/brats_benchmark_20260324_154926/100pct_180vol/phase1_shared.pt` | Shared VQ-GAN |
| `results/paper4/brats_benchmark_20260324_154926/100pct_180vol/shortcut/final.pt` | 64³ conditional Shortcut FM |
| `results/paper4/paper4/e7_teacher/teacher_best.pt` | 64³ teacher segmenter |

The script will exit early with a clear error message if any of these is missing.

## Wall time

On Apple M1 (24 GB, MPS backend): **~45–60 min per seed**, so **20–25 h total** for the full 25 runs. Runs unattended with `caffeinate -s`, no need to keep the laptop awake.

If you want a smoke test first, override `MAX_ITERS`:

```bash
MAX_ITERS=200 bash tier_c/e10a_64_5seed/run_e10a_64_5seed.sh
```

That completes all 25 runs in ~2 h and lets you catch any pipeline issue before committing to the full 25 h.

## Running it

```bash
cd fast-sampling-mode-collapse-3d/
bash tier_c/e10a_64_5seed/run_e10a_64_5seed.sh
```

Output goes to `results/paper4/e10a_64_5seed/`:

- `syn_pool_shortcut_50_64.pt` — 500 cached synthetic volumes (generated once, ~1–2 GB)
- `syn_pool_shortcut_50_64_feats.pt` — features used for the diversity conditions
- `e10a_64_5seed_results_partial.json` — updated after **every seed**
- `e10a_64_5seed_stdout.log` — full training log
- `e10a_64_5seed_results.json` — final summary with paired-t + Bonferroni

## Resuming after interruption

Just run the same command again. Every (condition, seed) already recorded in the partial JSON is skipped. The synthetic pool is cached to disk so it isn't regenerated. No state to reset.

## What to send back

The single file `e10a_64_5seed_results.json` (~50 KB when done) is all that's needed for the paper update. The `.pt` files (synth pool + features) are only needed if you want to re-run further analyses on the same pool.
