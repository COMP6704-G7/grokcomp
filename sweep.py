"""Experiment grids. Every run writes results/summary_<label>.json, and a run whose summary already
exists is skipped, so an interrupted sweep can simply be started again.

    python sweep.py orth_e1 --dry_run          # print the commands only
    python sweep.py orth_e1 --jobs 10          # run 10 at a time on one GPU (~300 MB each)
    python sweep.py orth_e2 --best_lr 1e-3     # E2/E3 need the lr picked from E1
    python summarize.py orth_                  # table + figures

Labels encode the configuration and end in _s<seed>, so summarize.py can average over seeds.
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

SEEDS = [0, 1, 2]
LRS = [3e-4, 1e-3, 3e-3]

# Settings shared by every run (the team's protocol).
COMMON = ["--p", "97", "--batch_size", "512", "--beta1", "0.9", "--beta2", "0.98"]

# Method presets: (optimizer, loss, extra flags)
ORTH = ("OrthoAdamW", "stablemax", [])
ADAMW = ("AdamW", "softmax", [])


def run(label, optimizer, loss, lr, wd, seed, frac=0.5, loss_dtype="float64", extra=()):
    return label, ["--label", label, "--raw_label", "--optimizer", optimizer, "--loss", loss,
                   "--loss_dtype", loss_dtype, "--lr", str(lr), "--weight_decay", str(wd),
                   "--seed", str(seed), "--train_frac", str(frac), *extra]


def lr_tag(lr):
    return f"{lr:.0e}".replace("-0", "-")


def grid(name, a):
    out = []
    if name == "orth_e1":
        # Main comparison: ⊥AdamW + StableMax as designed (no weight decay), 3 lr x 3 seeds.
        for lr, s in itertools.product(LRS, SEEDS):
            out.append(run(f"orth_e1_orth_lr{lr_tag(lr)}_s{s}", *ORTH[:2], lr, 0.0, s))
        if a.with_reference:
            for lr, s in itertools.product(LRS, SEEDS):
                out.append(run(f"orth_e1_adamw_lr{lr_tag(lr)}_s{s}", *ADAMW[:2], lr, a.ref_wd, s,
                               loss_dtype="float32"))
    elif name == "orth_e2":
        # Weight decay: does ⊥ still help when wd is also on, and does it need it?
        for wd, s in itertools.product([0.0, 0.1, 1.0], SEEDS):
            out.append(run(f"orth_e2_wd{wd:g}_s{s}", *ORTH[:2], a.best_lr, wd, s))
    elif name == "orth_e3":
        # Training fraction: less data makes grokking slower.
        for frac, s in itertools.product([0.3, 0.5, 0.7], SEEDS):
            out.append(run(f"orth_e3_frac{frac:g}_s{s}", *ORTH[:2], a.best_lr, 0.0, s, frac=frac))
    elif name == "orth_e4":
        # Which piece does the work: ⊥ on/off x StableMax on/off x wd on/off.
        for orth, stable, wd, s in itertools.product([True, False], [True, False], [0.0, 1.0], SEEDS):
            opt = "OrthoAdamW" if orth else "AdamW"
            loss = "stablemax" if stable else "softmax"
            tag = f"{'orth' if orth else 'adamw'}_{loss}_wd{wd:g}"
            out.append(run(f"orth_e4_{tag}_s{s}", opt, loss, a.best_lr, wd, s))
        # Implementation choices of the projection.
        for variant, extra in [("global", ["--orth_scope", "global"]),
                               ("norescale", ["--orth_no_rescale"]),
                               ("skip1d", ["--orth_skip_1d"])]:
            for s in SEEDS:
                out.append(run(f"orth_e4_var-{variant}_s{s}", *ORTH[:2], a.best_lr, 0.0, s, extra=extra))
        # Softmax Collapse demo: plain softmax, no wd, float32 vs float64 loss.
        for dt, s in itertools.product(["float32", "float64"], SEEDS):
            out.append(run(f"orth_e4_sc-{dt}_s{s}", "AdamW", "softmax", a.best_lr, 0.0, s, loss_dtype=dt))
    else:
        sys.exit(f"unknown grid {name!r}; choose orth_e1, orth_e2, orth_e3 or orth_e4")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel on --device")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu_jobs", type=int, default=0,
                    help="extra runs in parallel on the CPU alongside the --jobs device runs")
    ap.add_argument("--budget", default="100000", help="max optimiser steps per run")
    ap.add_argument("--stop_after", default="50", help="epochs at >= 99%% val acc before stopping early")
    ap.add_argument("--best_lr", type=float, default=1e-3)
    ap.add_argument("--with_reference", action="store_true", help="E1: also run the AdamW reference line")
    ap.add_argument("--ref_wd", type=float, default=1.0, help="weight decay of the AdamW reference")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    todo = []
    for label, args in grid(a.grid, a):
        if os.path.exists(f"results/summary_{label}.json"):
            continue
        cmd = [sys.executable, "main.py", *COMMON, *args,
               "--budget", a.budget, "--stop_after", a.stop_after]
        todo.append((label, cmd))
    print(f"{a.grid}: {len(todo)} runs to do")

    if a.dry_run:
        for _, cmd in todo:
            print(shlex.join(cmd + ["--device", a.device]))
        return

    # Each run is a tiny model driven by a Python loop, so it is CPU-bound. Many runs in parallel
    # only scale if each one uses a single CPU thread (PyTorch otherwise starts one per core).
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    # One shared queue; each worker owns a device. Longest-expected runs first is not known
    # in advance, so runs go in grid order.
    queue = list(todo)
    lock = threading.Lock()

    def worker(device):
        while True:
            with lock:
                if not queue:
                    return
                label, cmd = queue.pop(0)
            # one retry: a CUDA context occasionally fails to start ("device busy") when many
            # runs launch on the same GPU at once
            for attempt in range(2):
                with open(f"results/stdout_{label}.txt", "w") as f:
                    rc = subprocess.call(cmd + ["--device", device], stdout=f, stderr=subprocess.STDOUT, env=env)
                if rc == 0:
                    break
            print(("done " if rc == 0 else f"FAILED ({rc}) ") + f"{label} [{device}]", flush=True)

    devices = [a.device] * a.jobs + ["cpu"] * a.cpu_jobs
    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(worker, devices))


if __name__ == "__main__":
    main()
