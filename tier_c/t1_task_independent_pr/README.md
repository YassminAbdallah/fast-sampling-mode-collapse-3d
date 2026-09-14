# T1: task-independent P/R/D/C

Motivation: the paper's Precision/Recall/Density/Coverage metrics in §5.2 are computed in the same VQ-GAN encoder feature space that every generator decodes through. Standard practice for these metrics uses a task-independent feature extractor. This run confirms whether the §5.2 ordering (Consistency = collapsed, Shortcut = diverse) holds under a fully independent feature space.

## What the script does

For each `(dataset, method)` in `{BraTS, IXI} × {Consistency@50, Shortcut@50}`:

1. Loads the paper's unconditional 64³ Shortcut and Consistency generators from `results/paper3/*_benchmark_*/100pct_*vol/{consistency,shortcut}/final.pt`.
2. Samples 64 volumes unconditionally at NFE = 50 (5 sampling seeds).
3. Feeds real and synthetic volumes through the **BraTS tumour-segmentation teacher** (`results/paper4/paper4/e7_teacher/teacher_best.pt`), a MONAI BasicUNet trained for a different task and independent of any generative pipeline.
4. Extracts the **bottleneck feature map** (`down_4` block, flattened to 2,048 dimensions) via a forward hook.
5. Computes P/R/D/C using the same Kynkäänniemi (k=3) and Naeem (k=5) definitions as §5.2 — **only the feature extractor differs**.

**5 seeds × 64 samples × 4 conditions = 20 P/R/D/C measurements**, each producing a JSON entry with mean ± std.

## Why the teacher is a defensible task-independent choice

- It's a segmentation model, not part of any of the five generative pipelines
- It was trained on a different task (tumour segmentation, not synthesis)
- It's applied identically to every method and both datasets (any bias is common-mode)

## Runtime

On Apple M1 (24 GB, MPS backend): **~30 min per dataset**, so **~60 min total** for both datasets. Runs unattended with `caffeinate -s`.

## Running it

```bash
cd fast-sampling-mode-collapse-3d/
bash tier_c/t1_task_independent_pr/run_t1_pr.sh
```

Output goes to `results/paper4/t1_pr_task_independent/`:

- `t1_pr_results.json` — the full 4-row result set (~5 KB)
- `t1_pr_stdout.log` — full run log

## What we expect

If the §5.2 ordering is genuine (not an encoder-space artefact), we expect:

- **Consistency Distillation:** near-zero Recall and low Coverage on both datasets — matching the encoder-space finding.
- **Shortcut FM:** measurably higher Recall and Coverage on both datasets, comparable to (or above) 30% Recall on IXI.

If instead the ordering flips or Consistency's Recall becomes non-zero under this independent extractor, that would indicate the encoder-space P/R/D/C finding is not robust — a genuine finding worth reporting.

## Output

The single file `t1_pr_results.json` (~5 KB), reported in Appendix A.4 alongside the encoder-space table.
