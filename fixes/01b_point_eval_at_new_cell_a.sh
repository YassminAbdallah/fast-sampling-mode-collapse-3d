#!/usr/bin/env bash
# =============================================================================
# Point the factorial evaluator at the schedule-matched Cell A checkpoint
# (results/paper4/cd_2x2/ema_l2/final.pt) and re-run the evaluation, so that
# all four cells are evaluated at 80 epochs / batch 4. The original
# 150-epoch checkpoint is untouched.
#
# RUNTIME  ~10 min (sampling only, no training).
#
# Usage:  bash fixes/01b_point_eval_at_new_cell_a.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL="tier_c/item15b_2x2_factorial/eval_2x2_factorial.py"
NEW_CKPT="results/paper4/cd_2x2/ema_l2/final.pt"

[ -f "$NEW_CKPT" ] || { echo "ERROR: $NEW_CKPT not found. Run fixes/01_retrain_cell_a.sh first."; exit 1; }

# --- stash the ORIGINAL (150-epoch) Cell A numbers before we overwrite -------
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("results/paper4/cd_2x2/factorial_2x2_table.json")
if p.exists():
    old = {c["cell"]: {"ssim_at_nfe4": c["ssim_at_nfe4"],
                       "diversity_pct_real": c["diversity_pct_real"]}
           for c in json.load(open(p))["cells"]}
    out = pathlib.Path("results/paper4/cd_2x2/factorial_2x2_table_ORIGINAL_150ep_cellA.json")
    out.write_text(json.dumps(old, indent=1))
    print(f"  archived pre-fix table -> {out}")
    print(f"  original Cell A (150 ep, batch 2): "
          f"SSIM {old['A_ema_l2']['ssim_at_nfe4']:.4f}, "
          f"{old['A_ema_l2']['diversity_pct_real']:.1f}% of real")
PY

# --- repoint Cell A ---------------------------------------------------------
python3 - "$EVAL" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]); src = p.read_text()

if "cd_2x2/ema_l2/final.pt" in src:
    print("  evaluator already points at the retrained Cell A"); sys.exit(0)

old_block = re.search(r'    "A_ema_l2": \{.*?\n    \},\n', src, re.S)
if not old_block:
    sys.exit("ERROR: could not locate the A_ema_l2 block -- patch by hand.")

new_block = '''    "A_ema_l2": {
        "ema": True,  "loss": "L2",
        "label": "Cell A: EMA + L2 (schedule-matched, 80 ep / batch 4)",
        "ckpt": "results/paper4/cd_2x2/ema_l2/final.pt",
        "existing_eval": None,
        "existing_eval_path": None,
        "method_in_existing_json": None,
        "training_log": "results/paper4/cd_2x2/training_log_ema_l2.json",
        "fresh_eval_out": "results/paper4/cd_2x2/ema_l2/eval_results_5seed.json",
    },
'''
p.write_text(src.replace(old_block.group(0), new_block, 1))
print("  patched: Cell A -> results/paper4/cd_2x2/ema_l2/final.pt")
PY

echo ""
echo "=== Re-evaluating all four cells under one protocol ==="
python3 "$EVAL" --n-samples 64 --no-reuse-existing \
    2>&1 | tee results/paper4/cd_2x2/cell_a_reeval_MATCHED_stdout.log

echo ""
python3 - <<'PY'
import json, pathlib
cells = json.load(open("results/paper4/cd_2x2/factorial_2x2_table.json"))["cells"]
orig = pathlib.Path("results/paper4/cd_2x2/factorial_2x2_table_ORIGINAL_150ep_cellA.json")
old = json.load(open(orig)) if orig.exists() else {}

print("=" * 74)
print("  2x2 FACTORIAL — all four cells now 80 epochs / batch 4")
print("=" * 74)
print(f"  {'cell':24s} {'SSIM':>8s} {'% of real':>11s} {'status':>10s}")
print("  " + "-" * 70)
for c in cells:
    print(f"  {c['cell']:24s} {c['ssim_at_nfe4']:>8.4f} "
          f"{c['diversity_pct_real']:>10.1f}% {c['training_status']:>10s}")

A = next(c for c in cells if c["cell"] == "A_ema_l2")
C = next(c for c in cells if c["cell"] == "C_ema_pseudohuber")
a_new, c_new = A["diversity_pct_real"], C["diversity_pct_real"]
a_old = old.get("A_ema_l2", {}).get("diversity_pct_real")

print()
if a_old:
    print(f"  Cell A: {a_old:.1f}%  (150 ep, batch 2 — CONFOUNDED)")
    print(f"       -> {a_new:.1f}%  (80 ep,  batch 4 — schedule-matched)")
print()
print("  THE CLAIM UNDER TEST — §5.2 / §6.2:")
print("    'Given EMA, Pseudo-Huber improves diversity over L2 (49.9% vs 32.7%)'")
print()
gap = c_new - a_new
if abs(gap) < 5:
    print(f"    A = {a_new:.1f}%, C = {c_new:.1f}%  -> gap {gap:+.1f} pp.")
    print("    THE CLAIM DISSOLVES. The old 'Pseudo-Huber beats L2' effect was the")
    print("    training schedule, not the loss. Rewrite §5.2 and §6.2: once training")
    print("    is matched, the loss function is NOT a meaningful lever. EMA is.")
    print("    (The EMA effect — C 49.9% vs B 3.4% — is unaffected and still holds.)")
else:
    print(f"    A = {a_new:.1f}%, C = {c_new:.1f}%  -> gap {gap:+.1f} pp SURVIVES matching.")
    print("    The claim stands, and is now properly controlled. Update the numbers.")
print()
print("  Either way: Appendix A.3 must state BOTH recipes —")
print("    main benchmark generators: 150 epochs, batch 2")
print("    2x2 factorial cells:        80 epochs, batch 4")
print("=" * 74)
PY
