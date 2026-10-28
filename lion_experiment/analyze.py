"""Generate the three requested figures and censored, paired result tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import mean_std
from .train import ROOT, atomic_json


def load_runs(root, phase):
    runs = []
    for path in sorted((root / phase).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["phase"] != phase:
            raise ValueError(f"Incorrect phase in {path}")
        config = json.loads((path.parent / "config.json").read_text())
        metadata = json.loads((path.parent / "metadata.json").read_text())
        rows = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
        runs.append((summary, rows, config, metadata))
    return runs


def verify_pairs(runs):
    for seed in {r[0]["seed"] for r in runs}:
        pair = [r for r in runs if r[0]["seed"] == seed]
        if len(pair) != 2 or {r[0]["optimizer"] for r in pair} != {"lion", "adamw"}:
            raise ValueError(f"Incomplete or duplicate pair for seed {seed}")
        for key in ["split_sha256", "initial_weights_sha256"]:
            if pair[0][0][key] != pair[1][0][key]:
                raise ValueError(f"Pair differs in {key}")
        if pair[0][2]["protocol"] != pair[1][2]["protocol"]:
            raise ValueError("Protocol differs between paired methods")
        for key in ["gpu", "torch", "cuda", "source_sha256"]:
            if pair[0][3][key] != pair[1][3][key]:
                raise ValueError(f"Execution environment differs in {key}")
    if runs:
        for item in runs[1:]:
            if item[2]["protocol"] != runs[0][2]["protocol"]:
                raise ValueError("Formal study contains mixed protocols")


def format_stat(stat, precision=1):
    if not stat["n"]:
        return "—"
    value = f"{stat['mean']:.{precision}f}"
    return value + (f" ± {stat['std']:.{precision}f}" if stat["std"] is not None else " (n=1)")


def analyze(root):
    root = Path(root)
    formal, tuning = load_runs(root, "formal"), load_runs(root, "tuning")
    if not formal:
        raise ValueError("No completed formal results; pilot logs are excluded")
    verify_pairs(formal)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    flat = []
    for summary, _, _, _ in formal:
        item = {key: summary[key] for key in ["optimizer", "seed", "lr", "weight_decay", "eta_lambda",
            "status", "steps", "max_steps", "final_train_acc", "final_val_acc", "mean_step_ms",
            "peak_memory_bytes", "optimizer_state_bytes", "wall_seconds", "segments", "generalization_delay"]}
        for name in ["train99", "val95", "val99"]:
            for key in ["step", "wall_seconds", "confirmed_step"]:
                item[f"{name}_{key}"] = summary[name][key] if summary[name] else None
        flat.append(item)
    with (out / "per_run.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    comparison = []
    for opt in ["lion", "adamw"]:
        runs = [r for r in flat if r["optimizer"] == opt]
        if not runs:
            continue
        result = {"optimizer": opt, "n": len(runs),
                  "val95_successes": sum(r["val95_step"] is not None for r in runs),
                  "val99_successes": sum(r["val99_step"] is not None for r in runs),
                  "val99_success_rate": sum(r["val99_step"] is not None for r in runs) / len(runs)}
        for key in ["train99_step", "val95_step", "val99_step", "val99_wall_seconds",
                    "generalization_delay", "mean_step_ms", "peak_memory_bytes",
                    "optimizer_state_bytes", "final_val_acc"]:
            result[key] = mean_std([r[key] for r in runs if r[key] is not None])
        comparison.append(result)
    atomic_json(out / "comparison.json", comparison)
    with (out / "comparison.csv").open("w") as stream:
        flattened = []
        for row in comparison:
            item = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    item.update({f"{key}_{k}": v for k, v in value.items()})
                else:
                    item[key] = value
            flattened.append(item)
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"lion": "#176B87", "adamw": "#D86828"}
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    seen = set()
    for summary, rows, _, _ in formal:
        opt = summary["optimizer"]
        for column, style, label in [("train_acc", "--", "train"), ("val_acc", "-", "validation")]:
            key = (opt, label)
            ax.plot([r["step"] for r in rows], [r[column] for r in rows],
                    style, color=colors[opt], alpha=.65, linewidth=1.3,
                    label=f"{opt.upper()} {label}" if key not in seen else None)
            seen.add(key)
    ax.set(xlabel="Optimizer steps", ylabel="Accuracy", ylim=(0, 1.025),
           title="Mod-97 multiplication · formal seeds 1, 2, 3")
    ax.grid(alpha=.18)
    ax.legend(loc="lower right", ncols=2)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"accuracy_steps.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    seen = set()
    for summary, rows, _, _ in formal:
        opt = summary["optimizer"]
        ax.plot([r["wall_seconds"] for r in rows], [r["val_acc"] for r in rows],
                color=colors[opt], alpha=.65, label=opt.upper() if opt not in seen else None)
        seen.add(opt)
    ax.axhline(.99, color="gray", linestyle=":", linewidth=1)
    ax.set(xlabel="Elapsed wall time including evaluation (s)", ylabel="Validation accuracy",
           ylim=(0, 1.025), title="Identical A100 · one run at a time · FP32, TF32 disabled")
    ax.grid(alpha=.18)
    ax.legend()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"validation_time.{ext}", dpi=180)
    plt.close(fig)
    if tuning:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        for summary, rows, _, _ in sorted(tuning, key=lambda r: r[0]["lr"]):
            label = f"lr={summary['lr']:.0e}"
            ax1.plot([r["step"] for r in rows], [r["val_acc"] for r in rows], label=label)
            hit = summary["val99"]
            ax2.scatter(summary["lr"], hit["step"] if hit else summary["max_steps"],
                        marker="o" if hit else "^", s=60)
            ax2.annotate(str(hit["step"]) if hit else f">{summary['max_steps']}",
                         (summary["lr"], hit["step"] if hit else summary["max_steps"]),
                         xytext=(0, 7), textcoords="offset points", ha="center")
        ax1.set(xlabel="Optimizer steps", ylabel="Validation accuracy", ylim=(0, 1.025))
        ax1.legend()
        ax2.set(xlabel="Lion learning rate", ylabel="First step in 3-checkpoint val ≥99% streak")
        ax2.set_xticks([r[0]["lr"] for r in tuning], [f"{r[0]['lr']:.0e}" for r in tuning])
        ax2.margins(x=.3, y=.3)
        for ax in [ax1, ax2]:
            ax.grid(alpha=.18)
        fig.suptitle("Learning-rate sensitivity · exploratory seed 0 only · weight decay=5")
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"lr_sensitivity.{ext}", dpi=180)
        plt.close(fig)

    lines = ["# Lion / AdamW 实验结果", "",
        "数据来自真实训练日志；仅汇总 formal 阶段，pilot 与单 seed 学习率筛选不混入正式统计。", "",
        "| 方法 | val95 达标 | val99 达标 | Ttrain99 | Tval95 | Tval99 | ΔT | val99 时间/s | 每步/ms |",
        "|---|---|---|---|---|---|---|---|---|"]
    for item in comparison:
        cells = [item["optimizer"].upper(), f"{item['val95_successes']}/{item['n']}",
                 f"{item['val99_successes']}/{item['n']}"]
        cells += [format_stat(item[key]) for key in ["train99_step", "val95_step", "val99_step",
                  "generalization_delay", "val99_wall_seconds", "mean_step_ms"]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "均值 ± 样本标准差（ddof=1）；每列 CSV 同时给出有效 n。达标时间与步数仅对成功运行统计。",
              "连续三次评估达标后记第一检查点；停止发生在第三检查点。ΔT 保留原始符号。", "",
              "## 逐次运行", "", "| 方法 | seed | lr | ηλ | Tval99 | 最终 val acc | 训练步数 |",
              "|---|---|---|---|---|---|---|"]
    for row in flat:
        hit = row["val99_step"] if row["val99_step"] is not None else f">{row['max_steps']}"
        lines.append(f"| {row['optimizer']} | {row['seed']} | {row['lr']:.0e} | {row['eta_lambda']:.4g} | "
                     f"{hit} | {row['final_val_acc']:.4%} | {row['steps']} |")
    lines += ["", "## 资源", "", "| 方法 | 峰值分配显存/MiB | 优化器持久状态/MiB |", "|---|---|---|"]
    for item in comparison:
        mem = item["peak_memory_bytes"]["mean"]
        state = item["optimizer_state_bytes"]["mean"]
        lines.append(f"| {item['optimizer']} | {mem / 2**20:.2f} | {state / 2**20:.2f} |")
    lines += ["", "显存为 torch.cuda.max_memory_allocated，包含模型、梯度、状态和前向/反向临时分配；不等于 nvidia-smi 占用。",
        "每步耗时包含取样、前向、反向、优化器和同步；不含完整集合评估。wall time 包含评估和运行期间日志/检查点开销，不含模型初始化。",
        "断点续跑的离线停机时间不计入 wall time，segments>1 的运行需单独解释计时。", "",
        "## 图表", "", "![准确率随步数](accuracy_steps.png)", "",
        "![验证准确率随时间](validation_time.png)", "", "![学习率敏感性](lr_sensitivity.png)", "",
        "## 限制", "", "三个 seed 只反映本模型、任务和配置下的波动，不能推出普适优势。",
        "Lion 有三次单 seed 学习率探索；AdamW 使用预先指定的组内起点（1e-3），未执行等量学习率搜索。",
        "因此这里是固定 AdamW 基线比较，不能据此声称两种优化器在相同调参预算下的最优性能差异。",
        "预算和权重衰减采用计划中的试跑起点（100000 步、AdamW=1、Lion=5），已冻结于本次本地实验，尚需组内确认。"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Analysis saved: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/lion_study")
    analyze(parser.parse_args().root)


if __name__ == "__main__":
    main()
