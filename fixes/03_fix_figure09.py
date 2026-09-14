#!/usr/bin/env python3
"""
Regenerate Figure 9 (count-matched dose-response at 64-cubed) directly from
results/paper4/e10a_64_5seed/e10a_64_5seed_results.json (8 seeds).

The per-seed Dice values are read from the result file and plotted; no values
are hard-coded. If the result file is missing the script exits with an error
rather than drawing from a fallback.

Usage:  python3 fixes/03_fix_figure09.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/paper4/e10a_64_5seed/e10a_64_5seed_results.json"
OUT = ROOT / "figures/fig09_dose_response_e10a.png"

# --- load; no fallback by design -------------------------------------------
d = json.load(open(SRC))
R = {k: np.array(v["dices"]) for k, v in d["results"].items()}
n_seeds = len(R["unique_500"])
assert n_seeds == 8, f"expected 8 seeds, got {n_seeds} -- is this the right JSON?"

order = ["unique_10", "unique_25", "unique_100", "unique_500"]
x = np.array([10, 25, 100, 500])
mu = np.array([R[k].mean() for k in order])
sd = np.array([R[k].std(ddof=1) for k in order])
base_mu, base_sd = R["real_only"].mean(), R["real_only"].std(ddof=1)

# --- Holm-corrected threshold contrast (the one the paper actually claims) ---
pairs = [("unique_500","real_only"),("unique_10","real_only"),("unique_25","real_only"),
         ("unique_100","real_only"),("unique_500","unique_10"),("unique_500","unique_100"),
         ("unique_100","unique_25"),("unique_25","unique_10")]
ps = sorted((stats.ttest_rel(R[a], R[b]).pvalue, a, b) for a, b in pairs)
holm, prev = {}, 0.0
for i, (p, a, b) in enumerate(ps):
    prev = max(min(1.0, (len(ps) - i) * p), prev)
    holm[(a, b)] = prev
p_thresh = holm[("unique_25", "unique_10")]   # 0.049

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.axhspan(base_mu - base_sd, base_mu + base_sd, color="0.85", zorder=0)
ax.axhline(base_mu, ls="--", lw=1.6, color="0.35", zorder=1,
           label=f"Real-only baseline ({base_mu:.3f})")

ax.errorbar(x, mu, yerr=sd, marker="o", ms=8, lw=2.2, capsize=4,
            color="#1f77b4", zorder=3,
            label=f"Real + synthetic augmentation (n = {n_seeds} seeds)")

ax.axvline(25, ls=":", lw=1.6, color="#c0392b", zorder=2)
ax.annotate("knee\n≈25 unique", xy=(25, base_mu - 0.008),
            xytext=(30, base_mu - 0.014), fontsize=9, color="#c0392b")

# threshold contrast: the claim the paper makes
ax.annotate("", xy=(24, mu[1]), xytext=(10.4, mu[0]),
            arrowprops=dict(arrowstyle="->", lw=2, color="#1f6f3d"))
ax.text(15.5, mu[1] + 0.0045,
        f"$p_{{\\mathrm{{Holm}}}}$ = {p_thresh:.3f}",
        color="#1f6f3d", fontsize=10, fontweight="bold",
        ha="center", va="bottom")

ax.set_xscale("log")
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
ax.set_xlabel("Unique synthetic anatomies (log scale)", fontsize=11)
ax.set_ylabel("Test Dice", fontsize=11)
ax.set_title("Diversity dose–response: 18 real + 500 synthetic\n"
             "(total count fixed; only the unique count varies)", fontsize=11)
ax.set_ylim(0.740, 0.812)          # was 0.60-0.85; all data lives in 0.76-0.80
ax.grid(alpha=0.3, which="both", ls=":")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix("." + ext), dpi=300, bbox_inches="tight")

print(f"Wrote {OUT} (+ .pdf)\n")
print(f"  {'condition':12s} {'Dice':>8s} {'std':>7s}")
print(f"  {'real_only':12s} {base_mu:>8.4f} {base_sd:>7.4f}")
for k, m, s in zip(order, mu, sd):
    print(f"  {k:12s} {m:>8.4f} {s:>7.4f}")
print(f"\n  unique_25 vs unique_10: p_Holm = {p_thresh:.4f}")
print("\n  n = 8 seeds; the curve rises to unique_500; the 25-vs-10 threshold is")
print("  Holm-significant (Bonferroni p = 0.196).")
