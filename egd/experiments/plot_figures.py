#!/usr/bin/env python3
"""Generate unified Figure 1–4 for the COMP6704 EGD experimental plan."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_STYLE = {
    "vanilla": {"label": "SGD", "color": "#4C78A8", "ls": "-"},
    "col_norm": {"label": "Column Norm", "color": "#F58518", "ls": "--"},
    "egd": {"label": "EGD (Full SVD)", "color": "#E45756", "ls": "-"},
    "egd_rsvd": {"label": "EGD (RSVD)", "color": "#54A24B", "ls": "-."},
}


def load_all(results_dir: Path):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception:
            pass
    return rows


def mean_curve(runs, key_epochs="eval_epochs", key_vals="test_acc", xmax=None):
    """Average curves on a common epoch grid via interpolation."""
    grids, vals = [], []
    for r in runs:
        e = np.asarray(r["curves"][key_epochs], dtype=float)
        v = np.asarray(r["curves"][key_vals], dtype=float)
        if len(e) < 2:
            continue
        grids.append(e)
        vals.append(v)
    if not grids:
        return None, None, None
    lo = max(g.min() for g in grids)
    hi = min(g.max() for g in grids)
    if xmax is not None:
        hi = min(hi, xmax)
    if hi <= lo:
        return None, None, None
    grid = np.linspace(lo, hi, 300)
    stacked = np.stack([np.interp(grid, g, v) for g, v in zip(grids, vals)])
    return grid, stacked.mean(0), stacked.std(0)


def fig1_grokking_curves(rows, out: Path, p=97):
    """Train/Test accuracy curves for SGD vs EGD (+ ColNorm) on mod-add p."""
    subset = [
        r
        for r in rows
        if r.get("task") == "modular_addition" and r.get("p") == p and r.get("method") in ("vanilla", "egd", "col_norm")
    ]
    by_method = defaultdict(list)
    for r in subset:
        by_method[r["method"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("train_acc", "test_acc"),
        ("Train Accuracy", "Test Accuracy"),
    ):
        for method in ("vanilla", "col_norm", "egd"):
            runs = by_method.get(method, [])
            if not runs:
                continue
            style = METHOD_STYLE[method]
            grid, mean, std = mean_curve(runs, key_vals=metric)
            if grid is None:
                continue
            ax.plot(grid, mean, color=style["color"], ls=style["ls"], lw=2, label=style["label"])
            ax.fill_between(grid, mean - std, mean + std, color=style["color"], alpha=0.15)
        ax.axhline(0.95, color="gray", ls=":", lw=1, label="95% threshold")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle(f"Figure 1 — Grokking Curves (Modular Addition, p={p})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig1_grokking_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig1_grokking_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_generalization_speed(rows, out: Path):
    """Epochs to 95% across tasks/settings."""
    # Modular addition across p
    mod = [r for r in rows if r.get("task") == "modular_addition" and r.get("method") in ("vanilla", "egd")]
    parity = [r for r in rows if r.get("task") == "sparse_parity" and r.get("method") in ("vanilla", "egd")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: mod-add by p
    ax = axes[0]
    ps = sorted({r["p"] for r in mod})
    width = 0.35
    x = np.arange(len(ps))
    for i, method in enumerate(("vanilla", "egd")):
        means, stds = [], []
        for p in ps:
            vals = [
                r["metrics"]["epochs_to_test_95"]
                for r in mod
                if r["p"] == p and r["method"] == method and r["metrics"]["epochs_to_test_95"] is not None
            ]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        style = METHOD_STYLE[method]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=style["label"],
            color=style["color"],
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in ps])
    ax.set_ylabel("Epochs to 95% Test Acc")
    ax.set_title("Modular Addition")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Right: parity by k
    ax = axes[1]
    ks = sorted({r["k"] for r in parity})
    x = np.arange(len(ks))
    for i, method in enumerate(("vanilla", "egd")):
        means, stds = [], []
        for k in ks:
            vals = [
                r["metrics"]["epochs_to_test_95"]
                for r in parity
                if r["k"] == k and r["method"] == method and r["metrics"]["epochs_to_test_95"] is not None
            ]
            # DNF -> use nan and annotate later
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        style = METHOD_STYLE[method]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=style["label"],
            color=style["color"],
            capsize=3,
        )
        # mark DNF
        for j, (m, k) in enumerate(zip(means, ks)):
            n_runs = sum(1 for r in parity if r["k"] == k and r["method"] == method)
            n_ok = sum(
                1
                for r in parity
                if r["k"] == k and r["method"] == method and r["metrics"]["epochs_to_test_95"] is not None
            )
            if n_runs and n_ok == 0:
                ax.text(j + (i - 0.5) * width, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 1, "DNF", ha="center", fontsize=8, color=style["color"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Epochs to 95% Test Acc")
    ax.set_title("Sparse Parity")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 2 — Generalization Speed (Epochs to 95% Test Accuracy)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig2_generalization_speed.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig2_generalization_speed.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_wallclock(rows, out: Path, p=97):
    """Wall-clock to 95% for SGD / Full SVD / RSVD."""
    subset = [
        r
        for r in rows
        if r.get("task") == "modular_addition"
        and r.get("p") == p
        and r.get("method") in ("vanilla", "egd", "egd_rsvd", "col_norm")
    ]
    # Prefer RSVD r=64 for main efficiency figure
    methods_order = [
        ("vanilla", None),
        ("col_norm", None),
        ("egd", None),
        ("egd_rsvd", 64),
    ]
    labels, means, stds, tep_means = [], [], [], []
    for method, rank in methods_order:
        runs = [
            r
            for r in subset
            if r["method"] == method and (rank is None or r.get("rank") == rank)
        ]
        if method == "egd_rsvd":
            runs = [r for r in runs if r.get("rank") == 64]
        walls = [r["metrics"]["wall_clock_to_test_95_s"] for r in runs if r["metrics"]["wall_clock_to_test_95_s"] is not None]
        teps = [r["metrics"]["time_per_epoch_mean_s"] for r in runs if r["metrics"]["time_per_epoch_mean_s"] is not None]
        if not walls:
            continue
        if method == "egd_rsvd":
            labels.append("EGD-RSVD (r=64)")
        else:
            labels.append(METHOD_STYLE[method]["label"])
        means.append(np.mean(walls))
        stds.append(np.std(walls))
        tep_means.append(np.mean(teps) if teps else np.nan)

    colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B"][: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Wall-clock to 95% Test Acc (s)")
    ax.set_title("Total Time to Target")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f"{m:.1f}s", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(x, [t * 1000 for t in tep_means], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Time per Epoch (ms)")
    ax.set_title("Per-Epoch Cost")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Figure 3 — Wall-clock Efficiency (Modular Addition, p={p})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig3_wallclock_efficiency.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig3_wallclock_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_ablation(rows, out: Path, p=97):
    """SVD strategy + RSVD rank ablation."""
    subset = [r for r in rows if r.get("task") == "modular_addition" and r.get("p") == p]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: strategy comparison
    ax = axes[0]
    order = [
        ("vanilla", None, "SGD"),
        ("col_norm", None, "Column Norm"),
        ("egd_rsvd", 64, "EGD-RSVD r=64"),
        ("egd", None, "EGD Full SVD"),
    ]
    means, stds, labels, colors = [], [], [], []
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for (method, rank, lab), c in zip(order, palette):
        vals = [
            r["metrics"]["epochs_to_test_95"]
            for r in subset
            if r["method"] == method
            and (rank is None or r.get("rank") == rank)
            and r["metrics"]["epochs_to_test_95"] is not None
        ]
        if not vals:
            continue
        labels.append(lab)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        colors.append(c)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Epochs to 95% Test Acc")
    ax.set_title("Normalization Strategy")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: RSVD rank sweep (+ full SVD reference)
    ax = axes[1]
    rank_runs = defaultdict(list)
    for r in subset:
        if r["method"] == "egd_rsvd" and r.get("rank") is not None:
            if r["metrics"]["epochs_to_test_95"] is not None:
                rank_runs[r["rank"]].append(r["metrics"]["epochs_to_test_95"])
    ranks = sorted(rank_runs)
    if ranks:
        rm = [np.mean(rank_runs[r]) for r in ranks]
        rs = [np.std(rank_runs[r]) for r in ranks]
        ax.errorbar(ranks, rm, yerr=rs, marker="o", color="#54A24B", lw=2, capsize=4, label="EGD-RSVD")
    full = [
        r["metrics"]["epochs_to_test_95"]
        for r in subset
        if r["method"] == "egd" and r["metrics"]["epochs_to_test_95"] is not None
    ]
    if full:
        ax.axhline(np.mean(full), color="#E45756", ls="--", lw=2, label=f"Full SVD ({np.mean(full):.0f})")
    ax.set_xlabel("RSVD Rank r")
    ax.set_ylabel("Epochs to 95% Test Acc")
    ax.set_title("RSVD Rank Ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Figure 4 — Ablation Study (Modular Addition, p={p})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig4_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig4_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_table(rows, out: Path):
    lines = ["# EGD Experimental Summary\n"]
    lines.append("## Modular Addition — Epochs / Wall-clock to 95% Test Accuracy\n")
    lines.append("| p | Method | Rank | Seeds | Epochs→95% | Delay | t/ep (ms) | Wall→95% (s) | Reach |\n")
    lines.append("|---|--------|------|-------|------------|-------|-----------|--------------|-------|\n")

    groups = defaultdict(list)
    for r in rows:
        if r.get("task") != "modular_addition":
            continue
        groups[(r["p"], r["method"], r.get("rank"))].append(r)
    for key in sorted(groups):
        p, method, rank = key
        rs = groups[key]
        t95 = [x["metrics"]["epochs_to_test_95"] for x in rs]
        ok = [x for x in t95 if x is not None]
        delay = [x["metrics"]["grokking_delay"] for x in rs if x["metrics"]["grokking_delay"] is not None]
        tep = [x["metrics"]["time_per_epoch_mean_s"] for x in rs if x["metrics"]["time_per_epoch_mean_s"] is not None]
        wall = [x["metrics"]["wall_clock_to_test_95_s"] for x in rs if x["metrics"]["wall_clock_to_test_95_s"] is not None]
        t95_s = f"{np.mean(ok):.0f}±{np.std(ok):.0f}" if ok else "DNF"
        delay_s = f"{np.mean(delay):.0f}±{np.std(delay):.0f}" if delay else "DNF"
        tep_s = f"{1000*np.mean(tep):.1f}" if tep else "-"
        wall_s = f"{np.mean(wall):.1f}±{np.std(wall):.1f}" if wall else "DNF"
        lines.append(
            f"| {p} | {method} | {rank or '-'} | {len(rs)} | {t95_s} | {delay_s} | {tep_s} | {wall_s} | {len(ok)}/{len(rs)} |\n"
        )

    lines.append("\n## Sparse Parity\n")
    lines.append("| k | n | Method | Seeds | Epochs→95% | Wall→95% (s) | Reach |\n")
    lines.append("|---|---|--------|-------|------------|--------------|-------|\n")
    groups = defaultdict(list)
    for r in rows:
        if r.get("task") != "sparse_parity":
            continue
        groups[(r["k"], r["n"], r["method"])].append(r)
    for key in sorted(groups):
        k, n, method = key
        rs = groups[key]
        ok = [x["metrics"]["epochs_to_test_95"] for x in rs if x["metrics"]["epochs_to_test_95"] is not None]
        wall = [x["metrics"]["wall_clock_to_test_95_s"] for x in rs if x["metrics"]["wall_clock_to_test_95_s"] is not None]
        t95_s = f"{np.mean(ok):.0f}±{np.std(ok):.0f}" if ok else "DNF"
        wall_s = f"{np.mean(wall):.1f}±{np.std(wall):.1f}" if wall else "DNF"
        lines.append(f"| {k} | {n} | {method} | {len(rs)} | {t95_s} | {wall_s} | {len(ok)}/{len(rs)} |\n")

    (out / "summary_tables.md").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--fig_dir", default="experiments/figures")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all(results_dir)
    if not rows:
        print("No results found.")
        return

    fig1_grokking_curves(rows, fig_dir)
    fig2_generalization_speed(rows, fig_dir)
    fig3_wallclock(rows, fig_dir)
    fig4_ablation(rows, fig_dir)
    write_summary_table(rows, fig_dir)
    print(f"Wrote figures to {fig_dir}")
    for p in sorted(fig_dir.glob("fig*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
