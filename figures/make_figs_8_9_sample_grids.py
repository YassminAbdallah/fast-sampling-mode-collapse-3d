"""
Figures 8 and 9: Visual sample comparisons.

Fig 8: All five methods at their best step count, plus real volumes.
Fig 9: Shortcut FM step progression (1, 4, 16, 128) plus FM@50.

Both figures show axial mid-slices of 64^3 volumes.

Usage:
    python make_figs_8_9_sample_grids.py \
        --run-dir results/ixi_benchmark_20260221_045741/100pct_200vol \
        --data-path data/ixi_preprocessed_64.pt \
        --output-dir figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# The script assumes the flow_matching_3d.py model definitions are importable.
# Add the directory containing flow_matching_3d.py to your PYTHONPATH, or run
# this script from that directory.
try:
    from flow_matching_3d import VQGAN3D, DenoisingUNet3D, sample_fm, sample_shortcut, sample_ddpm, sample_consistency
except ImportError:
    print(
        "Note: could not import model definitions from flow_matching_3d.\n"
        "Run this script from the directory containing flow_matching_3d.py, or\n"
        "add it to PYTHONPATH first."
    )
    raise


def load_vqgan_and_unet(phase1_path, method_path, device):
    """Load the shared VQ-GAN and a method-specific U-Net checkpoint."""
    vqgan = VQGAN3D().to(device)
    vqgan_ckpt = torch.load(phase1_path, map_location=device)
    vqgan.load_state_dict(vqgan_ckpt["vqgan"])
    vqgan.eval()

    unet = DenoisingUNet3D().to(device)
    unet_ckpt = torch.load(method_path, map_location=device)
    unet.load_state_dict(unet_ckpt["unet"] if "unet" in unet_ckpt else unet_ckpt)
    unet.eval()

    return vqgan, unet


def mid_axial_slice(volume):
    """Return the middle axial slice of a 64^3 volume as a 2D numpy array."""
    # volume: (1, 1, D, H, W) or (1, D, H, W) or (D, H, W)
    v = volume.squeeze().detach().cpu().numpy()
    d = v.shape[0]
    return v[d // 2]


@torch.no_grad()
def sample_method(method, vqgan, unet, n_samples, steps, device, seed=42):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 8, 8, 8, 8, device=device)  # (B, C, D, H, W)
    if method == "ddpm":
        z1 = sample_ddpm(unet, z, steps=steps)
    elif method == "fm":
        z1 = sample_fm(unet, z, steps=steps)
    elif method == "rectified":
        z1 = sample_fm(unet, z, steps=steps)
    elif method == "consistency":
        z1 = sample_consistency(unet, z, steps=steps)
    elif method == "shortcut":
        z1 = sample_shortcut(unet, z, steps=steps)
    else:
        raise ValueError(method)
    vols = vqgan.decode(z1)
    return vols  # (B, 1, 64, 64, 64)


def make_grid(slices, row_labels=None, col_labels=None, title=None, figsize=None):
    n_rows = len(slices)
    n_cols = len(slices[0])
    if figsize is None:
        figsize = (1.5 * n_cols, 1.5 * n_rows + 0.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes]
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c] if n_cols > 1 else axes[r]
            ax.imshow(slices[r][c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/"))
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    phase1_path = args.run_dir / "phase1_shared.pt"

    # ---- Figure 8: all five methods at best step count ----
    method_best = [
        ("ddpm", 1000, "DDPM\n(1000 steps)"),
        ("fm", 50, "FM\n(50 steps)"),
        ("rectified", 10, "Rectified\n(10 steps)"),
        ("consistency", 50, "Consistency\n(50 steps)"),
        ("shortcut", 50, "Shortcut\n(50 steps)"),
    ]

    # Load real volumes
    real_data = torch.load(args.data_path, map_location="cpu")
    real_volumes = real_data[:args.n_samples] if isinstance(real_data, torch.Tensor) else real_data["volumes"][:args.n_samples]

    slices_fig8 = []
    # Real row
    real_slices = [mid_axial_slice(v) for v in real_volumes]
    slices_fig8.append(real_slices)

    for method, steps, label in method_best:
        ckpt_path = args.run_dir / method / "final.pt"
        if not ckpt_path.exists():
            print(f"[warn] {ckpt_path} not found — skipping")
            continue
        vqgan, unet = load_vqgan_and_unet(phase1_path, ckpt_path, device)
        vols = sample_method(method, vqgan, unet, args.n_samples, steps, device)
        slices_fig8.append([mid_axial_slice(v) for v in vols])

    row_labels_fig8 = ["Real"] + [lab for _, _, lab in method_best[:len(slices_fig8) - 1]]
    fig8 = make_grid(
        slices_fig8,
        row_labels=row_labels_fig8,
        title="Fig. 8 — Five methods at best step count. Single-sample quality looks similar; diversity differences are invisible here.",
    )
    out_pdf = args.output_dir / "fig8_best_step_comparison.pdf"
    fig8.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig8.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_pdf}")
    plt.close(fig8)

    # ---- Figure 9: Shortcut FM step progression ----
    shortcut_ckpt = args.run_dir / "shortcut" / "final.pt"
    fm_ckpt = args.run_dir / "fm" / "final.pt"
    if not (shortcut_ckpt.exists() and fm_ckpt.exists()):
        print("[warn] shortcut or fm checkpoint missing — skipping Fig 9")
        return

    vqgan_sc, unet_sc = load_vqgan_and_unet(phase1_path, shortcut_ckpt, device)
    _, unet_fm = load_vqgan_and_unet(phase1_path, fm_ckpt, device)

    slices_fig9 = []
    row_labels_fig9 = []

    # Real
    slices_fig9.append([mid_axial_slice(v) for v in real_volumes])
    row_labels_fig9.append("Real")

    # Shortcut at each step count
    for steps in [1, 4, 16, 128]:
        vols = sample_method("shortcut", vqgan_sc, unet_sc, args.n_samples, steps, device)
        slices_fig9.append([mid_axial_slice(v) for v in vols])
        row_labels_fig9.append(f"Shortcut\n({steps} steps)")

    # FM baseline
    vols = sample_method("fm", vqgan_sc, unet_fm, args.n_samples, 50, device)
    slices_fig9.append([mid_axial_slice(v) for v in vols])
    row_labels_fig9.append("FM\n(50 steps)")

    fig9 = make_grid(
        slices_fig9,
        row_labels=row_labels_fig9,
        title="Fig. 9 — Shortcut FM at 1, 4, 16, 128 steps (same weights) vs FM at 50 steps.",
    )
    out_pdf = args.output_dir / "fig9_shortcut_progression.pdf"
    fig9.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig9.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out_pdf}")
    plt.close(fig9)


if __name__ == "__main__":
    main()
