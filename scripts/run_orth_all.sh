#!/usr/bin/env bash
# All ⊥AdamW experiments in order: E1, choose the lr from E1, then E2, E3, E4, then summaries.
#   bash scripts/run_orth_all.sh <gpu_jobs> <cpu_jobs>
set -euo pipefail
cd "$(dirname "$0")/.."
G="${1:-8}"; C="${2:-4}"
PY="${PYTHON:-python}"
"$PY" sweep.py orth_e1 --with_reference --device cuda --jobs "$G" --cpu_jobs "$C"
LR=$("$PY" scripts/pick_best_lr.py)
echo "best ⊥AdamW lr from E1: $LR"
for g in orth_e2 orth_e3 orth_e4; do
  "$PY" sweep.py "$g" --best_lr "$LR" --device cuda --jobs "$G" --cpu_jobs "$C"
done
for g in orth_e1 orth_e2 orth_e3 orth_e4; do "$PY" summarize.py "$g" > /dev/null || true; done
echo "all ⊥AdamW experiments finished"
