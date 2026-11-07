#!/usr/bin/env python3
"""Summarize experiment JSON results into tables for the EGD plan."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: Path):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"skip {path}: {e}")
    return rows


def fmt(x, nd=1):
    if x is None:
        return "DNF"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summarize_modadd(rows, p=None):
    rows = [r for r in rows if r.get("task") == "modular_addition"]
    if p is not None:
        rows = [r for r in rows if r.get("p") == p]

    groups = defaultdict(list)
    for r in rows:
        key = (r["p"], r["method"], r.get("rank"))
        groups[key].append(r)

    print("\n=== Modular Addition Summary ===")
    print(
        f"{'p':>4} {'method':>10} {'rank':>5} {'n':>2} "
        f"{'test95(mean±std)':>18} {'delay':>12} {'t/ep(s)':>10} {'wall95(s)':>12} {'reach':>6}"
    )
    for key in sorted(groups):
        p_, method, rank = key
        rs = groups[key]
        t95 = [r["metrics"]["epochs_to_test_95"] for r in rs]
        delay = [r["metrics"]["grokking_delay"] for r in rs if r["metrics"]["grokking_delay"] is not None]
        tep = [r["metrics"]["time_per_epoch_mean_s"] for r in rs if r["metrics"]["time_per_epoch_mean_s"] is not None]
        wall = [r["metrics"]["wall_clock_to_test_95_s"] for r in rs if r["metrics"]["wall_clock_to_test_95_s"] is not None]
        reached = sum(1 for x in t95 if x is not None)

        t95_ok = [x for x in t95 if x is not None]
        t95_s = f"{np.mean(t95_ok):.0f}±{np.std(t95_ok):.0f}" if t95_ok else "DNF"
        delay_s = f"{np.mean(delay):.0f}±{np.std(delay):.0f}" if delay else "DNF"
        tep_s = f"{np.mean(tep):.4f}" if tep else "-"
        wall_s = f"{np.mean(wall):.1f}±{np.std(wall):.1f}" if wall else "DNF"
        print(
            f"{p_:>4} {method:>10} {str(rank) if rank else '-':>5} {len(rs):>2} "
            f"{t95_s:>18} {delay_s:>12} {tep_s:>10} {wall_s:>12} {reached}/{len(rs):>4}"
        )


def summarize_parity(rows):
    rows = [r for r in rows if r.get("task") == "sparse_parity"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["k"], r["method"], r.get("rank"))].append(r)

    print("\n=== Sparse Parity Summary ===")
    print(
        f"{'k':>2} {'method':>10} {'rank':>5} {'n':>2} "
        f"{'test95(mean±std)':>18} {'wall95(s)':>12} {'reach':>6}"
    )
    for key in sorted(groups):
        k, method, rank = key
        rs = groups[key]
        t95 = [r["metrics"]["epochs_to_test_95"] for r in rs]
        wall = [r["metrics"]["wall_clock_to_test_95_s"] for r in rs if r["metrics"]["wall_clock_to_test_95_s"] is not None]
        t95_ok = [x for x in t95 if x is not None]
        t95_s = f"{np.mean(t95_ok):.0f}±{np.std(t95_ok):.0f}" if t95_ok else "DNF"
        wall_s = f"{np.mean(wall):.1f}" if wall else "DNF"
        reached = sum(1 for x in t95 if x is not None)
        print(
            f"{k:>2} {method:>10} {str(rank) if rank else '-':>5} {len(rs):>2} "
            f"{t95_s:>18} {wall_s:>12} {reached}/{len(rs)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--task", choices=["all", "modadd", "parity"], default="all")
    ap.add_argument("--p", type=int, default=None)
    args = ap.parse_args()
    rows = load_results(Path(args.results_dir))
    if not rows:
        print("No results yet.")
        return
    if args.task in ("all", "modadd"):
        summarize_modadd(rows, args.p)
    if args.task in ("all", "parity"):
        summarize_parity(rows)


if __name__ == "__main__":
    main()
