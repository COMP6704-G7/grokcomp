"""Run grids of the unified experiment. Every run writes results/summary_<label>.json; a run whose
summary already exists is skipped, so an interrupted grid can simply be started again.

    python sweep.py uni_tune --out_dir OUT --device cuda --jobs 12   # lr tuning on seed 100
    python sweep.py uni_main_decay --out_dir OUT --jobs 12           # seeds 1-5 + decay off/on
    python sweep.py uni_timing --out_dir OUT --jobs 1                # one run per method alone
    python sweep.py uni_tune --dry_run                               # print the commands only

scripts/run_all.sh runs every step in order. Labels encode method, lr, decay and seed.
"""
import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Settings shared by every run.
COMMON = ["--p", "97", "--batch_size", "512", "--beta1", "0.9", "--beta2", "0.98"]

# x*y mod 97, 50% split, 20k steps, no early stop. Decoupled decay is set as eta*lambda (lr * weight_decay),
# so it means the same shrink per step for every method that uses it.
UNI_ETA_LAMBDA = 1e-3          # AdamW lr 1e-3, weight decay 1: the standard grokking setting
UNI_TUNE_SEED = 100            # tuning only; never reused for the comparison
UNI_SEEDS = [1, 2, 3, 4, 5]
UNI_DECAY_SEEDS = UNI_SEEDS[:3]
# method: (optimizer, loss, loss_dtype, eta*lambda, lr grid, extra flags)
UNI_ARMS = {
    "adam":     ("AdamW", "softmax", "float32", 0.0, [1e-3], []),          # reference without decay, not tuned
    "adamw":    ("AdamW", "softmax", "float32", UNI_ETA_LAMBDA, [5e-4, 1e-3, 2e-3, 4e-3, 8e-3], []),
    "lion":     ("Lion", "softmax", "float32", UNI_ETA_LAMBDA, [3e-5, 5e-5, 1e-4, 2e-4, 3e-4], ["--beta2", "0.99"]),
    "orth":     ("OrthoAdamW", "stablemax", "float64", 0.0, [1e-4, 3e-4, 1e-3], []),
    "fpd":      ("FPD", "softmax", "float32", 0.0, [5e-4, 1e-3, 2e-3, 4e-3, 8e-3], []),  # fpd.py defaults s=.1 T=25 K=20
    # EGD in its paper's transformer setup: U V^T on the MLP weights, then Adam with coupled weight decay.
    "egd_adam": ("EGDAdam", "softmax", "float32", None, [5e-4, 1e-3, 2e-3, 4e-3, 8e-3], []),
}
# Methods whose decay is a fixed weight_decay (coupled L2 inside Adam) rather than eta*lambda.
UNI_FIXED_WD = {"egd_adam": 0.005}
# Decay ablation: each method with decay off and on, at its frozen lr (cells equal to the main runs are reused).
UNI_DECAY_EL = [0.0, UNI_ETA_LAMBDA]


def run(label, optimizer, loss, lr, wd, seed, frac=0.5, loss_dtype="float32", extra=()):
    return label, ["--label", label, "--raw_label", "--optimizer", optimizer, "--loss", loss,
                   "--loss_dtype", loss_dtype, "--lr", str(lr), "--weight_decay", str(wd),
                   "--seed", str(seed), "--train_frac", str(frac), *extra]


def uni_run(label, arm, lr, seed, eta_lambda=None):
    opt, loss, dt, el, _, extra = UNI_ARMS[arm]
    if arm in UNI_FIXED_WD:  # eta_lambda here means the weight_decay value itself
        wd = UNI_FIXED_WD[arm] if eta_lambda is None else eta_lambda
        return run(label, opt, loss, lr, wd, seed, loss_dtype=dt, extra=extra)
    el = el if eta_lambda is None else eta_lambda
    return run(label, opt, loss, lr, el / lr, seed, loss_dtype=dt, extra=extra)


def uni_selected(path="results/uni_selected.json"):
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run `python scripts/pick_unified.py` after uni_tune")
    return json.load(open(path))


def lr_tag(lr):
    return f"{lr:.0e}".replace("-0", "-")


def selected(sel, arm):
    return UNI_ARMS[arm][4][0] if arm == "adam" else sel[arm]["lr"]


def grid(name, a):
    out = []
    if name == "uni_tune":
        for arm, spec in UNI_ARMS.items():
            if arm == "adam":
                continue
            for lr in spec[4]:
                out.append(uni_run(f"uni_tune_{arm}_lr{lr_tag(lr)}_s{UNI_TUNE_SEED}", arm, lr, UNI_TUNE_SEED))
    elif name == "uni_extend":
        # lr values one step beyond a grid edge, proposed by scripts/pick_unified.py
        for arm, s in uni_selected().items():
            if arm not in UNI_ARMS or not isinstance(s, dict):
                continue
            for lr in s.get("extend_lr") or []:
                out.append(uni_run(f"uni_tune_{arm}_lr{lr_tag(lr)}_s{UNI_TUNE_SEED}", arm, lr, UNI_TUNE_SEED))
    elif name == "uni_main":
        sel = uni_selected()
        for arm, s in itertools.product(UNI_ARMS, UNI_SEEDS):
            lr = selected(sel, arm)
            out.append(uni_run(f"uni_main_{arm}_lr{lr_tag(lr)}_s{s}", arm, lr, s))
    elif name == "uni_decay":
        sel = uni_selected()
        for arm, s in itertools.product([a for a in UNI_ARMS if a != "adam"], UNI_DECAY_SEEDS):
            lr = selected(sel, arm)
            if arm in UNI_FIXED_WD:  # decay off; decay on (the paper's value) is the main runs
                out.append(uni_run(f"uni_decay_{arm}_lr{lr_tag(lr)}_wd0_s{s}", arm, lr, s, eta_lambda=0.0))
                continue
            for el in UNI_DECAY_EL:
                if el != UNI_ARMS[arm][3]:
                    out.append(uni_run(f"uni_decay_{arm}_lr{lr_tag(lr)}_el{el:g}_s{s}", arm, lr, s, eta_lambda=el))
    elif name == "uni_main_decay":  # one queue, so no GPU slot idles when one grid finishes first
        out += grid("uni_main", a) + grid("uni_decay", a)
    elif name == "uni_timing":
        # wall-clock and peak GPU memory: one run per method at its selected lr, seed 1, with --jobs 1
        sel = uni_selected()
        for arm in UNI_ARMS:
            lr = selected(sel, arm)
            out.append(uni_run(f"uni_timing_{arm}_lr{lr_tag(lr)}_s1", arm, lr, 1))
    else:
        sys.exit(f"unknown grid {name!r}; choose uni_tune, uni_extend, uni_main, uni_decay, uni_main_decay, uni_timing")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel on --device")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--budget", default="20000", help="optimiser steps per run")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_dir", default=".", help="directory whose results/ receives the runs")
    a = ap.parse_args()
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    os.makedirs(a.out_dir, exist_ok=True)
    os.chdir(a.out_dir)

    os.makedirs("results", exist_ok=True)
    todo = []
    for label, args in grid(a.grid, a):
        if os.path.exists(f"results/summary_{label}.json"):
            continue
        cmd = [sys.executable, main_py, *COMMON, *args, "--budget", a.budget, "--stop_after", "0"]
        todo.append((label, cmd))
    print(f"{a.grid}: {len(todo)} runs to do")

    if a.dry_run:
        for _, cmd in todo:
            print(shlex.join(cmd + ["--device", a.device]))
        return

    # Each run is a tiny model driven by a Python loop, so it is CPU-bound. Many runs in parallel
    # only scale if each one uses a single CPU thread (PyTorch otherwise starts one per core).
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    queue = list(todo)
    lock = threading.Lock()

    def worker(device):
        while True:
            with lock:
                if not queue:
                    return
                label, cmd = queue.pop(0)
            # one retry: a CUDA context occasionally fails to start when many runs launch at once
            for attempt in range(2):
                with open(f"results/stdout_{label}.txt", "w") as f:
                    rc = subprocess.call(cmd + ["--device", device], stdout=f, stderr=subprocess.STDOUT, env=env)
                if rc == 0:
                    break
            print(("done " if rc == 0 else f"FAILED ({rc}) ") + f"{label} [{device}]", flush=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(worker, [a.device] * a.jobs))


if __name__ == "__main__":
    main()
