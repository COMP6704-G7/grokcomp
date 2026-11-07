#!/usr/bin/env bash
# P0: Exp1 Modular Addition p=97 (SGD/EGD/ColNorm) + Exp3 Efficiency (RSVD)
# 3 seeds each. Uses GPU 0 by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source /home/wgs/anaconda3/etc/profile.d/conda.sh
conda activate grok
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUT="$ROOT/experiments/results"
LOG="$ROOT/experiments/logs"
mkdir -p "$OUT" "$LOG"

SEEDS=(0 1 2)
# Exp1 + shared Exp3 methods
METHODS=(vanilla col_norm egd egd_rsvd)

run_one() {
  local method="$1" seed="$2" rank_args=()
  local tag="p97_${method}_seed${seed}"
  if [[ "$method" == "egd_rsvd" ]]; then
    rank_args=(--rank 64)
    tag="p97_${method}_r64_seed${seed}"
  fi
  echo "===== START $tag ====="
  python experiments/run_mod_add.py \
    --p 97 \
    --method "$method" \
    --seed "$seed" \
    --epochs 15000 \
    --stop_at_95 \
    --out_dir "$OUT" \
    "${rank_args[@]}" \
    2>&1 | tee "$LOG/${tag}.log"
  echo "===== DONE $tag ====="
}

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one "$method" "$seed"
  done
done

echo "All P0 runs finished."
python experiments/summarize.py --results_dir "$OUT" --task modadd --p 97 || true
