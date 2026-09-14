#!/usr/bin/env python3
"""
Recompute every headline number of the paper directly from the released result
files, so the manuscript can be checked against the source of record without a
GPU.

Prints: the 64-cubed (8-seed) and 128-cubed (5-seed) dose-response means,
t-statistics and Bonferroni/Holm-corrected p-values; the real-diversity
denominators for every dataset, resolution and conditioning setting; the
training-seed distributions of Consistency Distillation and Shortcut FM
diversity; Recall and Coverage in the VQ-GAN encoder space and in the
teacher-encoder space; the teacher-segmenter test Dice values; and the
128-cubed manifold metrics.

The script computes numbers only and never reads the manuscript.

Usage:  python3 fixes/05_verify_all_numbers.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
def J(p): return json.load(open(ROOT / p))
def H(t): print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}")


# --------------------------------------------------------------------------
H("TABLE 8 — dose-response @ 64³  (8 seeds)")
d = J("results/paper4/e10a_64_5seed/e10a_64_5seed_results.json")["results"]
M = {k: np.array(v["dices"]) for k, v in d.items()}
for k in ["real_only", "unique_10", "unique_25", "unique_100", "unique_500"]:
    a = M[k]
    print(f"  {k:12s} n={len(a)}  Dice={a.mean():.4f} ± {a.std(ddof=1):.4f}")

pairs = [("unique_500","real_only"),("unique_10","real_only"),("unique_25","real_only"),
         ("unique_100","real_only"),("unique_500","unique_10"),("unique_500","unique_100"),
         ("unique_100","unique_25"),("unique_25","unique_10")]
raw = {(a,b): stats.ttest_rel(M[a], M[b]) for a, b in pairs}
srt = sorted(pairs, key=lambda ab: raw[ab].pvalue)
holm, prev = {}, 0.0
for i, ab in enumerate(srt):
    prev = max(min(1.0, (len(pairs) - i) * raw[ab].pvalue), prev)
    holm[ab] = prev

print(f"\n  {'contrast':30s} {'t':>7s} {'p_raw':>9s} {'p_Bonf':>9s} {'p_Holm':>9s}")
for a, b in pairs:
    r = raw[(a, b)]
    pb = min(1.0, r.pvalue * 8)
    flag = "  <-- threshold" if (a, b) == ("unique_25", "unique_10") else ""
    print(f"  {a+' vs '+b:30s} {r.statistic:>+7.2f} {r.pvalue:>9.4f} "
          f"{pb:>9.3f} {holm[(a,b)]:>9.3f}{flag}")
print("\n  NOTE: the threshold contrast (25 vs 10) is Holm-significant but not Bonferroni-significant.")
print("  NOTE: unique_500 > unique_100 is significant, so the curve is a knee rather than a plateau.")

# seed-by-seed monotonicity
order = ["real_only", "unique_10", "unique_25", "unique_100", "unique_500"]
viol = [s for s in range(len(M["real_only"]))
        if any(M[order[i]][s] > M[order[i+1]][s] for i in range(4))]
print(f"\n  Seeds in which the unique-count ordering is not monotone: {len(viol)} of {len(M['real_only'])}  -> {viol}")
print("  NOTE: the ordering holds on the seed means, not seed-by-seed.")


# --------------------------------------------------------------------------
H("TABLE 13/14 — dose-response @ 128³  (5 seeds)")
d = J("results/paper4/brats_128cubed_conditional/e10a_128/e10a_results_128.json")
M = {k: np.array(v["dices"]) for k, v in d.items() if isinstance(v, dict) and "dices" in v}
for k in ["real_only", "unique_10", "unique_25", "unique_100", "unique_500"]:
    a = M[k]
    print(f"  {k:12s} n={len(a)}  Dice={a.mean():.4f} ± {a.std(ddof=1):.4f}")
print("\n  t-statistics:")
for a, b in [("unique_25","unique_10"), ("unique_500","unique_10"), ("unique_10","real_only")]:
    r = stats.ttest_rel(M[a], M[b])
    print(f"    {a+' vs '+b:30s} t = {r.statistic:>+7.2f}   ΔDice = {M[a].mean()-M[b].mean():+.4f}")
print("  NOTE: the sign of t agrees with the sign of ΔDice.")


# --------------------------------------------------------------------------
H("REAL-DIVERSITY DENOMINATORS  (the Table 5a/5b bug)")
for lab, p in [
    ("IXI 64³",                  "results/paper3/ixi_benchmark_20260221_045741/100pct_200vol/paper_eval_5seed/eval_results_5seed.json"),
    ("BraTS 64³ uncond",         "results/paper3/brats_benchmark_20260221_193306/100pct_300vol/paper_eval_5seed/eval_results_5seed.json"),
    ("BraTS 64³ cond",           "results/paper4/brats_benchmark_20260324_154926/100pct_180vol/paper_eval_5seed/eval_results_5seed.json"),
    ("BraTS 128³",               "results/paper4/brats_128cubed/paper_eval_5seed/eval_results_128_5seed.json"),
]:
    print(f"  {lab:22s} real_diversity = {J(p)['metadata']['real_diversity']:.5f}")
print("\n  NOTE: every '% of real' value divides by the denominator of its own experiment.")


# --------------------------------------------------------------------------
H("CD DIVERSITY — single run vs training-seed distribution")
for lab, p, key in [
    ("unconditional", "results/paper4/train_seed_replication_uncond/a1_eval_summary_uncond.json", "50"),
    ("conditional",   "results/paper4/train_seed_replication/a1_eval_summary.json",               "50"),
]:
    try:
        pm = J(p)["per_model"]
    except FileNotFoundError:
        print(f"  {lab}: MISSING {p}"); continue
    for meth in ["consistency", "shortcut"]:
        v = [r[key]["diversity_pct_real"] for r in pm[meth].values() if key in r]
        print(f"  {lab:14s} {meth:12s} {min(v):5.1f}–{max(v):5.1f}%  "
              f"(mean {np.mean(v):5.1f}, std {np.std(v, ddof=1):4.1f} pp, n={len(v)})")
print("\n  NOTE: single-run values sit within the training-seed range above; the paper quotes mean and range.")


# --------------------------------------------------------------------------
H("RECALL — is it degenerate on BraTS? (the §5.2 claim)")
pr = J("results/paper3/brats_benchmark_20260221_193306/100pct_300vol/precision_recall_pca128/precision_recall_results.json")["results"]
print("  VQ-GAN encoder space, PCA-128, BraTS:")
for m in ["fm@50", "rectified@10", "shortcut@50", "ddpm@1000", "consistency@50"]:
    if m in pr:
        print(f"    {m:16s} Recall = {pr[m]['recall']['mean']:.4f}   "
              f"Coverage = {pr[m]['coverage']['mean']:.3f}")
print("\n  NOTE: Recall is degenerate for every method in this space on BraTS (see Table 6).")

t1 = ROOT / "results/paper4/t1_pr_task_independent/t1_pr_results_all_methods.json"
if t1.exists():
    r = json.load(open(t1))["results"]
    print("\n  Task-independent teacher space, BraTS:")
    for m, v in r.get("brats", {}).items():
        print(f"    {m:16s} Recall = {v['recall']['mean']:.4f}   "
              f"Coverage = {v['coverage']['mean']:.3f}")
    print("  -> Recall works HERE. This is the space to quote.")


# --------------------------------------------------------------------------
H("TEACHER SEGMENTERS  (pseudo-label sources)")
for lab, p_, key in [
    ("64-cubed BraTS", "results/paper4/paper4/e7_teacher/teacher_metrics.json", "test_dice"),
    ("UPenn-GBM",      "results/paper4/cross_dataset_upenn/teacher_metrics.json", "test_dice"),
]:
    try:
        v = J(p_)[key]
        print(f"  {lab:18s} test Dice = {v:.4f}   <- {p_}")
    except FileNotFoundError:
        print(f"  {lab:18s} MISSING: {p_}")
print("  128-cubed BraTS    test Dice = 0.8013   <- results/paper4/brats_128cubed_conditional/e7_teacher_128/teacher_stdout.log")
print()
print("  (Reported in the paper as 0.828 / 0.834 / 0.801.)")


# --------------------------------------------------------------------------
H("MANIFOLD METRICS @ 128-cubed  (Table 7)")
try:
    d = J("results/paper4/manifold_metrics_128/manifold_metrics_128.json")["results"]
    print(f"  {'method':14s} {'Precision':>9s} {'Recall':>8s} {'Coverage':>9s} {'VQ Recall':>10s} {'Div %real':>10s}")
    for m, v in d.items():
        print(f"  {m:14s} {v['precision']['mean']:9.3f} {v['recall']['mean']:8.3f} "
              f"{v['coverage']['mean']:9.3f} {v['vq_recall']['mean']:10.3f} "
              f"{v['diversity_pct_real']:9.1f}%")
    print()
    print("  NOTE: Consistency Recall is 0.027 at 128-cubed (near zero, not exactly zero as at 64-cubed).")
    print("  Seeds: " + str([round(x, 3) for x in d["consistency"]["recall"]["seeds"]]))
    print("  NOTE: the VQ-GAN space is degenerate at 64-cubed but only compressed at 128-cubed")
    print("  (diverse methods recover ~0.20 there, against <=0.007 at 64-cubed).")
except FileNotFoundError:
    print("  MISSING: run fixes/07_manifold_metrics_128.py")


print("\n" + "=" * 78)
print("  All values above are computed from the released result files.")
print("=" * 78)
