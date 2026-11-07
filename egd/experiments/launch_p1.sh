#!/usr/bin/env bash
# P1: Exp1 extension (p=79,127 SGD vs EGD) + Exp2 Sparse Parity
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

echo "===== Exp1 extension: p in {79,127}, SGD vs EGD ====="
for p in 79 127; do
  for method in vanilla egd; do
    for seed in "${SEEDS[@]}"; do
      tag="p${p}_${method}_seed${seed}"
      echo "===== START $tag ====="
      python experiments/run_mod_add.py \
        --p "$p" --method "$method" --seed "$seed" \
        --epochs 15000 --stop_at_95 --out_dir "$OUT" \
        2>&1 | tee "$LOG/${tag}.log"
      echo "===== DONE $tag ====="
    done
  done
done

echo "===== Exp2: Sparse Parity k=2,3,4 ====="
for k in 2 3 4; do
  for method in vanilla egd; do
    for seed in "${SEEDS[@]}"; do
      tag="parity_k${k}_${method}_seed${seed}"
      echo "===== START $tag ====="
      python experiments/run_parity.py \
        --k "$k" --method "$method" --seed "$seed" \
        --stop_at_95 --out_dir "$OUT" \
        2>&1 | tee "$LOG/${tag}.log"
      echo "===== DONE $tag ====="
    done
  done
done

echo "===== Exp4 optional: RSVD rank ablation on p=97 ====="
for rank in 32 64 128; do
  for seed in "${SEEDS[@]}"; do
    # skip r=64 seed* if already have from P0 (same filename)
    tag="p97_egd_rsvd_r${rank}_seed${seed}"
    echo "===== START $tag ====="
    python experiments/run_mod_add.py \
      --p 97 --method egd_rsvd --rank "$rank" --seed "$seed" \
      --epochs 15000 --stop_at_95 --out_dir "$OUT" \
      2>&1 | tee "$LOG/${tag}.log"
    echo "===== DONE $tag ====="
  done
done

echo "All P1(+P2 ablation) runs finished."
python experiments/summarize.py --results_dir "$OUT" || true
