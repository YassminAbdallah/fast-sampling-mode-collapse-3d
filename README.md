# Mode Collapse in Fast-Sampling 3D Medical Synthesis

**Detection, Mechanism, and Downstream Impact on Tumor Segmentation**

Yassmin Abdallah Ahmed, Abdel-Badeeh M. Salem, Taymoor Nazmy

Faculty of Computer and Information Sciences, Ain Shams University, Cairo, Egypt

Code, per-seed result files and figures for the paper, prepared for submission to *Computerized Medical Imaging and Graphics*. This work forms part of the first author's PhD research at Ain Shams University. Preprint: *to be added once posted on arXiv*.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps every table and figure in the paper to the result file it was computed from.

## Summary

We compare five generative methods (DDPM, Flow Matching, Rectified Flow, Consistency Distillation, Shortcut Flow Matching) for 3D brain MRI synthesis, all sharing the same VQ-GAN latent space and the same 3D U-Net, so differences between them come from the generative training procedure and not the compression stage. Data is IXI (200 healthy) and BraTS 2023 (300 glioma), at 64³ and 128³.

The headline result: the standard quality metrics pick the wrong method. Consistency Distillation wins SSIM and PSNR at both resolutions, and Precision and Density at 64³, but it has undergone near-total mode collapse. Its Recall in a feature space independent of the generator is exactly zero at 64³ (0.027 at 128³); every other method scores 0.52 to 0.82.

| Method (BraTS, 64³) | Precision | Recall | Coverage |
|---|---|---|---|
| Consistency Distillation | 1.000 | 0.000 | 0.125 |
| Shortcut FM | 0.844 | 0.597 | 0.944 |
| Flow Matching | 0.856 | 0.578 | 0.969 |
| Rectified Flow | 0.856 | 0.522 | 0.938 |
| DDPM @1000 | 0.850 | 0.600 | 0.953 |

A few more things we found along the way. Recall computed in the generator's own VQ-GAN latent space is degenerate for every method at 64³ (even DDPM at 1000 steps), so manifold metrics need an independent feature extractor. A count-matched dose-response, holding the synthetic pool at 500 volumes while varying unique anatomies over {10, 25, 100, 500}, puts the knee in downstream Dice at about 25 unique anatomies; below it, at 128³, augmentation is worse than none at all. And a schedule-matched 2×2 factorial shows the EMA target, not the loss function, is what restrains the collapse: with EMA both losses reach ~50% of real diversity, without it diversity falls to 3.4% or training fails outright.

## Repository layout

```
├── src/                 Training and evaluation code
│   ├── paper3/          Unconditional benchmark: VQ-GAN, the five generators,
│   │                    5-seed evaluation, ablations, trajectory analysis
│   ├── paper4/          Conditional generation, teacher segmenter,
│   │                    downstream augmentation experiments (E10a, E10b)
│   └── utils/           Seeding and determinism helpers
├── tier_c/              Follow-up experiments, one folder each with its own README:
│                        2×2 factorial, training-seed replication, 128³ pipelines,
│                        cross-dataset transfer (UPenn-GBM), independent-encoder
│                        manifold metrics, VQ-GAN ceiling, DDIM control, analyses
├── fixes/               Verification and regeneration scripts (see below)
├── results/             Per-seed result JSONs, logs and analysis outputs
├── figures/             All paper figures and the scripts that draw them
├── REPRODUCIBILITY.md   Table/figure → result-file map
└── requirements.txt     Exact package versions used for every reported number
```

The `src/paper3` and `src/paper4` names are internal labels for the two development phases of the code base (unconditional benchmark, then conditional and downstream work). They are kept as-is because the result files and REPRODUCIBILITY.md refer to them.

## Installation

Python 3.10 is required (3.10.13 was used). All package versions are pinned exactly:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The radiomic features used for the Fréchet Radiomic Distance are computed directly in NumPy; no radiomics package is required.

## Reproducing the numbers

Every value reported in the paper can be recomputed from the released JSONs, no GPU needed:

```bash
python3 fixes/05_verify_all_numbers.py
```

This prints the headline numbers straight from the source files: dose-response means and the eight corrected p-values, the factorial cells, training-seed distributions, manifold metrics in both feature spaces, teacher-segmenter Dice, and the real-diversity denominators used throughout.

To check a single result by hand, for example the dose-response threshold:

```python
import json, numpy as np
from scipy import stats

d = json.load(open('results/paper4/e10a_64_5seed/e10a_64_5seed_results.json'))
M = {k: np.array(v['dices']) for k, v in d['results'].items()}
t, p = stats.ttest_rel(M['unique_25'], M['unique_10'])
print(f'unique_25 vs unique_10: t={t:.2f}, p={p:.4f}')   # t=2.86, p=0.0245
```

The other scripts in `fixes/` regenerate specific artefacts from the result files or checkpoints: the schedule-matched factorial cell (`01`, `01b`), the five-method manifold metrics in the teacher-encoder space (`02`), Figure 9 (`03`), the ablation-table normalisation (`04`), the teacher segmenter's test Dice (`06`) and the 128³ manifold metrics (`07`). Each has a docstring stating its inputs and outputs.

## Re-running the experiments

Training from scratch requires the datasets (below) and a GPU. The unconditional benchmark is driven by `src/paper3/run_all_experiments.sh`; the conditional generators and downstream segmentation experiments by `src/paper4/conditional_segmentation.py`. Each follow-up experiment under `tier_c/` has a README with the exact command, expected runtime and output file. Every entry point seeds Python, NumPy and PyTorch through `src/utils/set_seed.py`.

## Data

The datasets are not included in this repository, but all three are public: BraTS 2023 GLI from [synapse.org](https://www.synapse.org/), IXI from [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/), and UPenn-GBM from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/).

Preprocessing is intensity normalisation and trilinear resampling to 64³ or 128³. Note that BraTS ships skull-stripped and IXI does not.

## Model checkpoints

Checkpoints (~0.9 GB) are not tracked in git. They are archived on Zenodo: *DOI to be added*.

The synthetic volume pools (~10 GB) are not archived, since they can be regenerated from the checkpoints.

## Hardware

All experiments were run on a single Apple M-series GPU via PyTorch MPS. The 128³ downstream training uses a hardened numerical recipe (reduced learning rate with warmup, `DiceCELoss(smooth_nr=0)`, NaN-gradient guard, gradient clipping) documented in Appendix A.7 of the paper. It is required on MPS and not on CUDA. Reproduction on CUDA is expected to agree within ≤ 0.005 SSIM, ≤ 0.5 dB PSNR and ≤ 0.001 pairwise-L1 diversity.

## Citation

```bibtex
@article{ahmed2026modecollapse,
  title   = {Mode Collapse in Fast-Sampling 3D Medical Synthesis: Detection,
             Mechanism, and Downstream Impact on Tumor Segmentation},
  author  = {Ahmed, Yassmin Abdallah and Salem, Abdel-Badeeh M. and Nazmy, Taymoor},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Yassmin Abdallah Ahmed — yassminabdallah@gmail.com

## License

Code released under the MIT License. The datasets retain their original licences.
