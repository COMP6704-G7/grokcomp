"""Summarise the E0 anchor-paper runs: final and best test accuracy per run, plus curves.

    python repro/anchor/summarize_anchor.py
Reads repro/anchor/upstream/loggs/*/metrics.csv (the authors' format: one row per
epoch × train/test × metric) and writes repro/anchor/anchor_summary.md and anchor_curves.png.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def short(args):
    bits = ["transformer" if args.get("use_transformer") else "MLP",
            "⊥AdamW" if args.get("orthogonal_gradients") else args.get("optimizer", "AdamW"),
            args.get("loss_function", "cross_entropy").replace("cross_entropy", "softmax"),
            f"wd={args.get('weight_decay', 0)}", f"loss {args.get('cross_entropy_dtype', 'float32')}",
            f"lr={args.get('lr')}", f"{args.get('num_epochs')} ep"]
    return " · ".join(bits)


def main():
    rows_md = ["| run | final test acc | best test acc | first epoch test ≥ 99% |", "|---|---|---|---|"]
    plt.figure(figsize=(7, 4.2))
    for k, d in enumerate(sorted(glob.glob(os.path.join(HERE, "upstream", "loggs", "*")))):
        with open(os.path.join(d, "args.json")) as f:
            args = json.load(f)
        # the authors' logger stores the log index (epoch // log_frequency) in the "epoch"
        # column and accuracy as a fraction in [0, 1]
        freq = int(args.get("log_frequency", 1))
        acc = defaultdict(dict)
        with open(os.path.join(d, "metrics.csv")) as f:
            for r in csv.DictReader(f):
                if r["metric_name"] == "accuracy":
                    acc[r["input_type"]][int(float(r["epoch"])) * freq] = 100 * float(r["value"])
        test = acc.get("test", {})
        if not test:
            continue
        ep = sorted(test)
        vals = [test[e] for e in ep]
        hit = next((e for e in ep if test[e] >= 99), None)
        name = short(args)
        rows_md.append(f"| {name} | {vals[-1]:.1f}% | {max(vals):.1f}% | {hit if hit is not None else 'never'} |")
        plt.plot(ep, vals, marker="o", ms=3, label=name, color=f"C{k}")
    plt.xscale("symlog")
    plt.xlabel("epoch (full batch = one step)")
    plt.ylabel("test accuracy (%)")
    plt.legend(fontsize=6)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "anchor_curves.png"), dpi=150)
    md = "\n".join(rows_md) + "\n"
    with open(os.path.join(HERE, "anchor_summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
