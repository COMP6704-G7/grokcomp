#!/usr/bin/env bash
# Resume remaining P1/P2 after parity EGD numerical fix
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

echo "===== Resume: parity k=3,4 EGD + k=4 vanilla ====="
for k in 3 4; do
  for method in vanilla egd; do
    for seed in "${SEEDS[@]}"; do
      tag="parity_k${k}_${method}_seed${seed}"
      # skip if already done (vanilla k3 already exists)
      if [[ -f "$OUT/${tag}.json" ]]; then
        echo "[skip] $tag"
        continue
      fi
      echo "===== START $tag ====="
      python experiments/run_parity.py \
        --k "$k" --method "$method" --seed "$seed" \
        --stop_at_95 --out_dir "$OUT" \
        2>&1 | tee "$LOG/${tag}.log"
      echo "===== DONE $tag ====="
    done
  done
done

echo "===== Resume: RSVD rank ablation r=32,128 ====="
for rank in 32 128; do
  for seed in "${SEEDS[@]}"; do
    tag="p97_egd_rsvd_r${rank}_seed${seed}"
    echo "===== START $tag ====="
    python experiments/run_mod_add.py \
      --p 97 --method egd_rsvd --rank "$rank" --seed "$seed" \
      --epochs 15000 --stop_at_95 --out_dir "$OUT" \
      2>&1 | tee "$LOG/${tag}.log"
    echo "===== DONE $tag ====="
  done
done

echo "===== All resumed runs finished ====="
python experiments/summarize.py --results_dir "$OUT" || true
python experiments/plot_figures.py --results_dir "$OUT" --fig_dir "$ROOT/experiments/figures" || true
