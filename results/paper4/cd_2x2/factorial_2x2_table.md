## 2×2 EMA × Loss factorial — quality, diversity, training status at NFE=4

| Cell | EMA | Loss | SSIM ↑ | PSNR ↑ | Diversity ↑ | % of real div | Loss-traj | Training status |
|------|-----|------|--------|--------|-------------|---------------|-----------|-----------------|
| A_ema_l2 | yes | L2 | 0.7986 | 19.34 | 0.0225 | 49.8% | loss_descent_smooth | **✓ trained** |
| B_noema_pseudohuber | no | Pseudo-Huber | 0.6896 | 13.63 | 0.0015 | 3.4% | loss_stalled | **✓ trained** |
| C_ema_pseudohuber | yes | Pseudo-Huber | 0.7988 | 19.34 | 0.0225 | 49.9% | loss_descent_smooth | **✓ trained** |
| D_noema_l2 | no | L2 | 0.0384 | 1.18 | 0.0095 | 21.1% | loss_spiked_stalled | **✗ failed** |

**Loss-trajectory flag** (descriptive only, from training_log.json loss curve):
- `loss_descent_smooth` — monotone descent, final < 0.8× first, no spike
- `loss_spiked` — loss went up >2× first at some point
- `loss_stalled` — final ≈ first (final > 0.8× first)
- `loss_spiked_stalled` — both shape problems present
- `unknown` — no training log on disk for this cell

**Training status** (the verdict — combines sample SSIM with the loss flag):
- `trained` — SSIM @ NFE=4 ≥ 0.55 (the model produces brain-like samples)
- `failed`  — SSIM @ NFE=4 < 0.55 (samples are degenerate; training broke)

**Interpretation rule:** the 2×2 diversity comparison is only valid across cells with `training_status = trained`. A `failed` cell indicates the {EMA, loss} combination is unstable to train — not whether the *objective* causes collapse. The loss-trajectory flag is a descriptive diagnostic of the training curve only; some valid baselines (e.g. Cell B) have a `loss_spiked_stalled` curve and still produce usable samples.