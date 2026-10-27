#!/usr/bin/env bash
# E0 · Reproduce the anchor paper's own headline figures with the authors' code, unchanged.
# Prieto et al., "Grokking at the Edge of Numerical Stability", ICLR 2025.
#
#   bash repro/anchor/run_anchor.sh [cuda|cpu] [fig1|fig6|all]
#
# Clones the authors' repository at a pinned commit into repro/anchor/upstream/ (git-ignored)
# and runs the Figure 1 and Figure 6 commands from their run_main_experiments.sh, writing
# their metrics.csv files under repro/anchor/upstream/loggs/. Then:
#   python repro/anchor/summarize_anchor.py
set -euo pipefail
DEVICE_KIND="${1:-cuda}"
WHICH="${2:-all}"
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
UP="$HERE/upstream"
COMMIT=720d244

if [ ! -d "$UP/.git" ]; then
  git clone -q https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability.git "$UP"
fi
git -C "$UP" checkout -q "$COMMIT"
DEV="cpu"; [ "$DEVICE_KIND" = "cuda" ] && DEV="cuda:0"
cd "$UP"
run() { echo ">>> $*"; "$PY" grokking_experiments.py --device "$DEV" "$@"; }

if [ "$WHICH" = "fig1" ] || [ "$WHICH" = "all" ]; then
  # MLP, (a + b) mod 113, 40% train, full batch, no weight decay.
  # (a) StableMax alone groks; (b) ⊥AdamW alone groks within a few hundred epochs;
  # (c) plain softmax in float64 for comparison.
  run --lr 0.01 --num_epochs 80000 --log_frequency 5000 --train_fraction 0.4 --loss_function stablemax --cross_entropy_dtype float64 --beta2 0.999
  run --lr 0.01 --num_epochs 300 --log_frequency 10 --train_fraction 0.4 --beta2 0.99 --orthogonal_gradients
  run --lr 0.01 --num_epochs 80000 --log_frequency 5000 --train_fraction 0.4 --cross_entropy_dtype float64
fi

if [ "$WHICH" = "fig6" ] || [ "$WHICH" = "all" ]; then
  # 1-layer transformer: ⊥AdamW vs AdamW with weight decay 1.5 vs AdamW without; then the MLP pair.
  run --lr 0.001 --num_epochs 5000 --log_frequency 200 --train_fraction 0.4 --orthogonal_gradients --use_transformer
  run --lr 0.001 --num_epochs 5000 --log_frequency 200 --train_fraction 0.4 --use_transformer --weight_decay 1.5
  run --lr 0.001 --num_epochs 5000 --log_frequency 200 --train_fraction 0.4 --use_transformer
  run --lr 0.005 --num_epochs 500 --log_frequency 20 --train_fraction 0.4
  run --lr 0.005 --num_epochs 500 --log_frequency 20 --train_fraction 0.4 --orthogonal_gradients
fi
echo "done; now: python repro/anchor/summarize_anchor.py"
