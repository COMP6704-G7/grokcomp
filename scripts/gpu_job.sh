#!/usr/bin/env bash
# One entry point for GPU jobs, so a queue/scheduler only ever runs `bash scripts/gpu_job.sh ...`.
#
#   bash scripts/gpu_job.sh test                  # unit tests
#   bash scripts/gpu_job.sh timing                # 1 run vs 10 parallel runs, prints steps/s
#   bash scripts/gpu_job.sh anchor                # E0: the anchor paper's own figures (authors' code)
#   bash scripts/gpu_job.sh sweep orth_e1 [args]  # any sweep.py grid
#   bash scripts/gpu_job.sh summarize orth_e1     # tables and figures
#   bash scripts/gpu_job.sh all 8 4               # E1 → best lr → E2, E3, E4 (8 GPU + 4 CPU workers)
#
# Set PYTHON to the interpreter of the environment to use (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHON="$PY"
# pip-installed PyTorch ships its own CUDA libraries; a system CUDA on LD_LIBRARY_PATH
# (e.g. /usr/local/cuda/lib64 from a login shell) shadows them and breaks `import torch`.
unset LD_LIBRARY_PATH
what="${1:?usage: gpu_job.sh test|timing|anchor|sweep <grid>|summarize <prefix>}"; shift || true
echo "[$(date '+%F %T')] $what $* on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

case "$what" in
  test)      "$PY" -m pytest -q tests ;;
  timing-cpu)
    common=(--p 97 --batch_size 512 --optimizer OrthoAdamW --loss stablemax --loss_dtype float64 --weight_decay 0 --budget 1000 --device cpu --raw_label)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    t0=$(date +%s); "$PY" main.py "${common[@]}" --label timing_cpu_single >/dev/null 2>&1; t1=$(date +%s)
    echo "single CPU run (1 thread), 1000 steps: $((t1 - t0)) s"
    n=$(( $(nproc --all) - 1 ))
    for k in $(seq 1 $n); do "$PY" main.py "${common[@]}" --seed $k --label "timing_cpu_$k" >/dev/null 2>&1 & done; wait
    t2=$(date +%s); echo "$n parallel CPU runs, 1000 steps each: $((t2 - t1)) s"
    ;;
  timing)
    common=(--p 97 --batch_size 512 --optimizer OrthoAdamW --loss stablemax --loss_dtype float64 --weight_decay 0 --budget 1000 --device cuda --raw_label)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    t0=$(date +%s); "$PY" main.py "${common[@]}" --label timing_single >/dev/null 2>&1; t1=$(date +%s)
    echo "single run, 1000 steps: $((t1 - t0)) s"
    n=10; t1=$(date +%s)
    for k in $(seq 1 $n); do "$PY" main.py "${common[@]}" --seed $k --label "timing_par${n}_$k" >/dev/null 2>&1 & done; wait
    t2=$(date +%s); echo "$n parallel GPU runs, 1000 steps each: $((t2 - t1)) s"
    # same work on the CPU only, one thread per run (nproc --all ignores OMP_NUM_THREADS)
    n=$(( $(nproc --all) - 1 )); t1=$(date +%s)
    for k in $(seq 1 $n); do "$PY" main.py "${common[@]/cuda/cpu}" --seed $k --label "timing_cpu_$k" >/dev/null 2>&1 & done; wait
    t2=$(date +%s); echo "$n parallel CPU runs, 1000 steps each: $((t2 - t1)) s"
    "$PY" - <<'EOF'
import glob, json
for f in sorted(glob.glob("results/summary_timing_*.json")):
    s = json.load(open(f)); print(s["label"], "grok", s["grok_step"], "sec/step %.4f" % s["sec_per_step"])
EOF
    ;;
  anchor)
    bash repro/anchor/run_anchor.sh cuda all
    "$PY" repro/anchor/summarize_anchor.py
    ;;
  sweep)     "$PY" sweep.py "$@" --device cuda ;;
  all)       bash scripts/run_orth_all.sh "$@" ;;
  summarize) "$PY" summarize.py "$@" ;;
  *) echo "unknown: $what"; exit 2 ;;
esac
echo "[$(date '+%F %T')] finished $what"
