#!/usr/bin/env bash
# The whole unified experiment, start to finish:
#   tests -> lr tuning (seed 100) -> pick lr (+ up to 2 steps past a grid edge) -> main runs (seeds 1-5)
#   + decay off/on (seeds 1-3) -> timing and peak-memory runs (one per method, alone) -> analysis.
# Every step skips runs whose summary already exists, so the script can be re-run after an interruption.
#
#   bash scripts/run_all.sh OUT_DIR [PARALLEL_RUNS]      (default 12 runs at a time on one GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?usage: bash scripts/run_all.sh OUT_DIR [PARALLEL_RUNS]}"
J="${2:-12}"
PY="${PYTHON:-python}"
DEV="${DEVICE:-cuda}"
S=(--out_dir "$OUT" --device "$DEV")
log() { echo "[$(date '+%F %T')] $*"; }

log "unified experiment -> $OUT ($J runs in parallel on $DEV)"
"$PY" -m pytest -q tests
"$PY" sweep.py uni_tune "${S[@]}" --jobs "$J"
"$PY" scripts/pick_unified.py --out_dir "$OUT"
for round in 1 2; do
  "$PY" sweep.py uni_extend "${S[@]}" --jobs "$J"
  "$PY" scripts/pick_unified.py --out_dir "$OUT"
done
log "learning rates frozen: $OUT/results/uni_selected.json"
"$PY" sweep.py uni_main_decay "${S[@]}" --jobs "$J"
log "main and decay runs finished; timing and memory runs, one at a time"
"$PY" sweep.py uni_timing "${S[@]}" --jobs 1
"$PY" scripts/analyze_unified.py --out_dir "$OUT"
log "done: tables and figures in $OUT/analysis"
