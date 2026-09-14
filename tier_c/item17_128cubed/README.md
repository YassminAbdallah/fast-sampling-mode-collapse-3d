# Item 17 — 128³ resolution validation

The fourth (and final) Tier C addition: re-train the Shortcut FM and Consistency Distillation pipeline at 128³ resolution and check whether the Shortcut > Consistency ordering observed at 64³ also holds at higher resolution.

The most common concern about a 64³ study is whether the findings hold at higher resolution; a 128³ comparison addresses it.

## What this experiment shows

At 128³ resolution, comparing two methods at 50 NFE:

- **Consistency Distillation** — uses an FM teacher distilled with EMA-target consistency loss
- **Shortcut Flow Matching** — single model with self-consistency loss + d-conditioning

We measure SSIM (windowed 7×7×7 Gaussian, σ=1.5), PSNR, pairwise-L1 diversity, and % of real-data diversity. The 64³ result (Table 3) was: Consistency@50 SSIM 0.787, diversity 2% of real; Shortcut@50 SSIM 0.771, diversity 50% of real. If the ordering replicates at 128³, the paper's central claim about Shortcut > Consistency holds at higher resolution.

## Four sequential phases

The pipeline has four training phases. They must run in order:

| Phase | Script entry | What is trained | M-series wall time |
|-------|--------------|-----------------|--------------------|
| 0 | `preprocess_brats_128.py` | Just preprocesses the BraTS zip to 128³ | ~45 minutes |
| 1 | `train_pipeline_128.py --phase vqgan` | VQ-GAN encoder/decoder/codebook at 128³ | ~18–26 hours |
| 2 | `train_pipeline_128.py --phase fm` | FM teacher on 16³ latent | ~18–22 hours |
| 3 | `train_pipeline_128.py --phase consistency` | Consistency distillation from FM teacher | ~10–14 hours |
| 4 | `train_pipeline_128.py --phase shortcut` | Shortcut FM (standalone) | ~18–22 hours |
| 5 | `eval_128.py` | Generates samples, computes metrics | ~20–40 minutes |
| **Total** | | | **~65–85 hours** |

For a 5-day deadline: start Phase 0 immediately, run all phases continuously, finish by day 4–5. Leaves 1–2 days for paper update + buffer.

## Exact commands

### Phase 0 — Preprocess BraTS to 128³ (~45 min)

```bash
cd <path-to-repo>/

PYTHONUNBUFFERED=1 python -u tier_c/item17_128cubed/preprocess_brats_128.py \
    --zip-path datasets/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
    --output data/brats_preprocessed_128.pt \
    --max-subjects 300
```

This streams the BraTS zip directly (no full unzip needed) and writes a (N, 1, 128, 128, 128) tensor at `data/brats_preprocessed_128.pt`. Approximate disk size: ~2 GB.

**Sanity check** before moving to Phase 1:

```bash
python3 -c "
import torch
v = torch.load('data/brats_preprocessed_128.pt', weights_only=False)
print(f'Volumes: {tuple(v.shape)} dtype={v.dtype}')
print(f'Range: [{v.min():.3f}, {v.max():.3f}]')
print(f'Total disk: {v.numel() * 4 / 1e9:.2f} GB')
"
```

Expected output:
- Shape `(N, 1, 128, 128, 128)` where N is around 300
- dtype `torch.float32`
- Range `[0.000, 1.000]`

### Phase 1 — Train VQ-GAN at 128³ (~18–26 hours)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/train_pipeline_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --output-dir results/paper4/brats_128cubed \
    --phase vqgan \
    --batch-size 2 \
    --epochs-vqgan 80
```

You should see lines every 5 epochs:
```
[  5/80] recon=0.024 vq=0.121 g_adv=-1.4 d=0.4 | 12m 34s
[ 10/80] recon=0.018 vq=0.087 g_adv=-1.1 d=0.5 | 12m 18s
...
```

When done, `results/paper4/brats_128cubed/phase1_shared.pt` should exist. Spot-check VQ-GAN reconstruction:

```bash
python3 -c "
import sys, torch
from pathlib import Path
sys.path.insert(0, 'src/paper4')
from models_shared import Encoder3D, Decoder3D, VectorQuantizer
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
s = torch.load('results/paper4/brats_128cubed/phase1_shared.pt', map_location=dev, weights_only=True)
enc = Encoder3D(1, 8, 2).to(dev); enc.load_state_dict(s['enc']); enc.eval()
dec = Decoder3D(1, 8, 2).to(dev); dec.load_state_dict(s['dec']); dec.eval()
vq  = VectorQuantizer(256, 8).to(dev); vq.load_state_dict(s['vq']); vq.eval()
v = torch.load('data/brats_preprocessed_128.pt', weights_only=False)
with torch.no_grad():
    x = v[:1].float().to(dev)
    z, _, _ = vq(enc(x))
    x_hat = dec(z).clamp(0, 1)
    err = (x_hat - x).abs().mean().item()
print(f'Mean reconstruction L1 error: {err:.4f}')
print('Healthy if < 0.05. Higher means VQ-GAN needs more training.')
"
```

### Phase 2 — Train FM teacher (~18–22 hours)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/train_pipeline_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --output-dir results/paper4/brats_128cubed \
    --phase fm \
    --batch-size 2 \
    --epochs-fm 80
```

When done: `results/paper4/brats_128cubed/fm/final.pt`.

### Phase 3 — Distill Consistency student (~10–14 hours)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/train_pipeline_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --output-dir results/paper4/brats_128cubed \
    --phase consistency \
    --batch-size 2 \
    --epochs-consistency 60
```

When done: `results/paper4/brats_128cubed/consistency/final.pt`.

### Phase 4 — Train Shortcut FM (~18–22 hours)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/train_pipeline_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --output-dir results/paper4/brats_128cubed \
    --phase shortcut \
    --batch-size 2 \
    --epochs-shortcut 80
```

When done: `results/paper4/brats_128cubed/shortcut/final.pt`.

### Phase 5 — Evaluate (~20–40 min)

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/eval_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --ckpt-dir results/paper4/brats_128cubed \
    --output-dir results/paper4/brats_128cubed/paper_eval_5seed \
    --steps 50 \
    --n-samples 64
```

You'll see for each method:
- 5 lines (one per seed) with SSIM / diversity per seed
- A summary line with mean and % of real diversity

At the very end:
```
============================================================
  128³ resolution validation summary (NFE = 50)
============================================================
              Method             SSIM         Diversity     %Real
        consistency  0.XXX±0.XXX     0.XXXX±0.XXXX     XX%
           shortcut  0.XXX±0.XXX     0.XXXX±0.XXXX     XX%
```

Result file: `results/paper4/brats_128cubed/paper_eval_5seed/eval_results_128_5seed.json`.

## All-in-one launch (run sequentially)

If you want to launch everything and walk away:

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/train_pipeline_128.py \
    --data-path data/brats_preprocessed_128.pt \
    --output-dir results/paper4/brats_128cubed \
    --phase all \
    --batch-size 2 \
    --epochs-vqgan 80 \
    --epochs-fm 80 \
    --epochs-consistency 60 \
    --epochs-shortcut 80 \
    2>&1 | tee tier_c/item17_128cubed/train_128.log
```

Then eval after it finishes:

```bash
PYTHONUNBUFFERED=1 caffeinate -i python -u tier_c/item17_128cubed/eval_128.py \
    --ckpt-dir results/paper4/brats_128cubed \
    --output-dir results/paper4/brats_128cubed/paper_eval_5seed
```

## Output

The single JSON file at:

```
results/paper4/brats_128cubed/paper_eval_5seed/eval_results_128_5seed.json
```

This gives:
- Whether Shortcut > Consistency ordering replicates at 128³
- The SSIM and diversity numbers at 128³ (to compare against Table 3's 64³ numbers)
- The bootstrap CIs


## What to do if something fails

| Symptom | Likely cause | Fix |
|---|---|---|
| MPS OOM at 128³ batch=2 | Memory budget exceeded | Drop to `--batch-size 1` (twice slower but works on 16-32 GB) |
| VQ-GAN reconstruction L1 > 0.10 after 80 epochs | Undertrained — needs more epochs | Re-run `--phase vqgan --epochs-vqgan 120` (resumes from scratch — back up old checkpoint if you want) |
| FM loss never drops below 0.5 | FM not converging — likely VQ-GAN issue | Inspect VQ-GAN reconstruction first |
| Consistency loss spikes mid-training | EMA decay too aggressive at low data — known fragility | Reduce LR by half |
| Shortcut SC loss explodes | d sampling unstable | Continue — usually self-corrects after a few epochs of curriculum warm-up |
| Generated 128³ volumes look like noise | Generator didn't converge — UNet probably too small for 16³ latent | Scale up the U-Net base channels |


## Strategic note

The honest expectation: at 128³ with reduced epochs, both methods will have **lower absolute quality** than at 64³. That's fine. The paper's claim is about the *relative ordering* of methods, not absolute quality. As long as Shortcut > Consistency in the ordering and the trend of Consistency-collapses-with-more-steps replicates, the paper's headline finding holds at higher resolution.

If the ordering does NOT replicate at 128³ (e.g., Consistency outperforms Shortcut), that would be a real and important finding. We'd write it up honestly — "at higher resolution, the collapse pattern shifts" — and the paper would qualify its claims to 64³ only. That's still a publishable result, just a different framing. We will not fabricate.

Submit the JSON when done and we'll write v7 from whatever the numbers actually show.
