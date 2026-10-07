#!/usr/bin/env bash
# Regenerate the shipped TsT-RF IBP removal lists (Tables 10, 13, 14) with the IBP CLI and check
# that they match reproduce/data/ibp/rf/ exactly.
#
# Usage: bash reproduce/regenerate_ibp.sh [OUT_DIR] [JOBS]
#   OUT_DIR  where the runs are written (default: outputs/ibp_regenerated)
#   JOBS     runs in parallel (default: 1; about 30 minutes in total on a 96-core machine,
#            each Random Forest already uses all cores)
#
# Needs network access the first time (VSI-Bench at the pinned revision bc96b17); exact
# agreement needs the locked numpy / pandas / scikit-learn versions (uv.lock, constraints.txt).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-outputs/ibp_regenerated}"
JOBS="${2:-1}"
PY="${PYTHON:-python}"
SHIPPED="$HERE/data/ibp/rf"
V1="$HERE/data/vsi_bench_debiased_v1_removed_ids.txt"
mkdir -p "$OUT"

# name | CLI arguments (all: VSI-Bench @ bc96b17, leak-free default features, ties broken by id)
RUNS=(
  "pf_b200|--alloc per_format --budget 200 --batch_size 25"
  "pf_b500|--alloc per_format --budget 500"
  "pf_b1000|--alloc per_format --budget 1000"
  "pf_b1500|--alloc per_format --budget 1500"
  "pf_b2000|--alloc per_format --budget 2000"
  "pf_b2500|--alloc per_format --budget 2500"
  "pt_uniform54|--alloc per_type --frac 0.54"
  "pt_v1budgets_s42|--alloc per_type --budgets_from V1"
  "pt_v1budgets_s1|--alloc per_type --budgets_from V1 --random_state 1"
)

run_one() {
  local name="${1%%|*}" args="${1#*|}"
  # shellcheck disable=SC2206
  local argv=($args)
  for i in "${!argv[@]}"; do [ "${argv[$i]}" = V1 ] && argv[$i]="$V1"; done  # paths may contain spaces
  "$PY" -m TsT.debiasing --benchmark vsi --mode rf --revision bc96b17 "${argv[@]}" --output_dir "$OUT/$name" \
    > "$OUT/$name.log" 2>&1 && echo "done: $name" || { echo "FAILED: $name (see $OUT/$name.log)"; return 1; }
}
export -f run_one
export PY OUT V1

printf '%s\n' "${RUNS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

status=0
for entry in "${RUNS[@]}"; do
  name="${entry%%|*}"
  if cmp -s <(tr -d '\r' < "$OUT/$name/removed_ids.txt" | sort) <(tr -d '\r' < "$SHIPPED/$name/removed_ids.txt" | sort); then
    echo "match:    $name ($(wc -l < "$OUT/$name/removed_ids.txt") removed)"
  else
    echo "MISMATCH: $name"; status=1
  fi
done
exit $status
