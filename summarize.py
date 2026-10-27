"""Average runs over seeds and draw the standard figures.

    python summarize.py orth_e1          # every label starting with orth_e1
Writes results/table_<prefix>.md / .csv and results/fig_<prefix>_*.png.
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(prefix):
    groups = defaultdict(list)
    for path in sorted(glob.glob(f"results/summary_{prefix}*.json")):
        with open(path) as f:
            s = json.load(f)
        groups[re.sub(r"_s\d+$", "", s["label"])].append(s)
    return groups


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.0f}"
    return f"{statistics.mean(xs):.0f} ± {statistics.stdev(xs):.0f}"


def table(groups, prefix):
    cols = ["config", "grokked", "fit_step", "grok_step", "sustained_step", "delay", "sec/step"]
    rows = []
    for cfg, runs in sorted(groups.items()):
        n_grok = sum(r["sustained_step"] is not None for r in runs)
        rows.append([cfg.replace(prefix + "_", ""), f"{n_grok}/{len(runs)}",
                     mean_std([r["fit_step"] for r in runs]), mean_std([r["grok_step"] for r in runs]),
                     mean_std([r["sustained_step"] for r in runs]), mean_std([r["delay"] for r in runs]),
                     f"{statistics.mean(r['sec_per_step'] for r in runs):.4f}"])
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    md += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    with open(f"results/table_{prefix}.md", "w") as f:
        f.write(md)
    with open(f"results/table_{prefix}.csv", "w") as f:
        f.write(",".join(cols) + "\n" + "".join(",".join(r) + "\n" for r in rows))
    print(md)


def curves(groups, prefix, key, ylabel, logy=False):
    plt.figure(figsize=(6.4, 4))
    for cfg, runs in sorted(groups.items()):
        for k, r in enumerate(runs):
            path = f"results/log_{r['label']}.jsonl"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rows = [json.loads(line) for line in f]
            plt.plot([x["step"] for x in rows], [x[key] for x in rows], lw=1, alpha=0.8,
                     color=f"C{sorted(groups).index(cfg) % 10}", label=cfg.replace(prefix + "_", "") if k == 0 else None)
    plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel("optimisation steps")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"results/fig_{prefix}_{key}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "orth_"
    groups = load(prefix)
    if not groups:
        sys.exit(f"no results/summary_{prefix}*.json yet")
    table(groups, prefix)
    for key, label, logy in [("val_acc", "validation accuracy", False), ("w_norm", "weight norm ‖θ‖", True),
                             ("cos_gw", "cos(gradient, weights) before ⊥", False),
                             ("softmax_collapse", "share of samples in Softmax Collapse", False)]:
        curves(groups, prefix, key, label, logy)
    print(f"figures: results/fig_{prefix}_*.png")
