#!/usr/bin/env bash
# =============================================================================
# Compute Precision/Recall/Density/Coverage in the teacher-encoder feature
# space for all five methods (the original T1 run covered Consistency and
# Shortcut only). Per-method NFE matches the representative rows of the
# benchmark tables:
#     consistency 50 | shortcut 50 | fm 50 | rectified 10 | ddpm 1000
# (DDPM at reduced step counts is noise-dominated; see Appendix B.)
#
# Output: results/paper4/t1_pr_task_independent/t1_pr_results_all_methods.json
# RUNTIME  ~20-30 min (DDPM@1000 dominates).
#
# Usage:  bash fixes/02_rerun_t1_all_methods.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

T1="tier_c/t1_task_independent_pr/t1_pr_teacher_features.py"
OUT="results/paper4/t1_pr_task_independent/t1_pr_results_all_methods.json"

# ---------------------------------------------------------------------------
# Patch: extend METHODS, and give each method its own NFE.
# The original hardcodes `NFE = 50` and passes it at the generate call site.
# ---------------------------------------------------------------------------
python3 - "$T1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); src = p.read_text()

if "METHOD_NFE" in src:
    print("  already patched - skipping"); sys.exit(0)

# 1. all five methods, each with the NFE used in Tables 2/3
src = src.replace(
    'METHODS = ["consistency", "shortcut"]',
    'METHODS = ["consistency", "shortcut", "fm", "rectified", "ddpm"]\n'
    '# Per-method NFE, matching the representative rows of Tables 2/3.\n'
    '# DDPM at 50 steps is noise-dominated (§5.1) and must run at 1000.\n'
    'METHOD_NFE = {"consistency": 50, "shortcut": 50, "fm": 50,\n'
    '              "rectified": 10, "ddpm": 1000}',
    1)

# 2. the generate call site: NFE -> METHOD_NFE[method]
src = src.replace(
    "gen_batch = generate_unconditional(unet, dec, method, n_samples,\n"
    "                                             device, NFE)",
    "gen_batch = generate_unconditional(unet, dec, method, n_samples,\n"
    "                                             device, METHOD_NFE[method])",
    1)

# 3. the progress banner
src = src.replace(
    'print(f"  {dataset_name.upper()} × {method} @ NFE={NFE}")',
    'print(f"  {dataset_name.upper()} × {method} @ NFE={METHOD_NFE[method]}")',
    1)

# 4. record the per-method NFE in the output payload
src = src.replace('"nfe": NFE,', '"nfe": METHOD_NFE,', 1)

p.write_text(src)
print("  patched: 5 methods + per-method NFE")
PY

echo ""
echo "=== Running T1 across all five methods, both datasets ==="
python3 "$T1" \
    --dataset both \
    --n-samples 64 \
    --output "$OUT" \
    2>&1 | tee results/paper4/t1_pr_task_independent/t1_all_methods_stdout.log

echo ""
python3 - "$OUT" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))["results"]
print("=" * 66)
print("  TASK-INDEPENDENT P/R/D/C  (teacher-encoder features)")
print("=" * 66)
print(f"  {'dataset':8s} {'method':13s} {'Precision':>10s} {'Recall':>9s} {'Coverage':>10s}")
print("  " + "-" * 62)
for ds in d:
    for m, v in d[ds].items():
        print(f"  {ds:8s} {m:13s} {v['precision']['mean']:>10.3f} "
              f"{v['recall']['mean']:>9.4f} {v['coverage']['mean']:>10.3f}")
print()
print("  READ THIS: if FM / Rectified / DDPM show Recall > 0 on BraTS here,")
print("  then §5.2's 'every other method recovers measurable Recall' is TRUE")
print("  in the task-independent space -- say so, and name the space.")
print("  If they are also ~0, the sentence is false and must be cut.")
print("  Either way, add these rows to Table A.4b and state the feature")
print("  space wherever you quote Precision/Recall/Density/Coverage.")
PY
