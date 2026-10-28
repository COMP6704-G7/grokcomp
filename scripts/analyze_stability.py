"""Audit fixed-horizon continuations without redefining original threshold metrics."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "results/diagnostics/formal_stability"
if not (target / "complete.json").exists():
    raise RuntimeError("Wait for all six continuations to finish")
rows_out, histories = [], []
for path in sorted(target.glob("*/summary.json")):
    result = json.loads(path.read_text())
    rows = [json.loads(s) for s in (path.parent / "metrics.jsonl").read_text().splitlines()]
    original = ROOT / json.loads((path.parent / "continuation_origin.json").read_text())["original_run"]
    original_rows = [json.loads(s) for s in (original / "metrics.jsonl").read_text().splitlines()]
    assert rows[:len(original_rows)] == original_rows, "Original prefix changed"
    current_metadata = json.loads((path.parent / "metadata.json").read_text())
    original_metadata = json.loads((original / "metadata.json").read_text())
    for key in ["source_sha256", "initial_weights_sha256", "split_sha256", "gpu", "torch"]:
        assert current_metadata[key] == original_metadata[key], (path, key)
    current_config = json.loads((path.parent / "config.json").read_text())
    original_config = json.loads((original / "config.json").read_text())
    for key in ["lr", "seed", "optimizer", "device"]:
        assert current_config[key] == original_config[key]
    for key, value in original_config["protocol"].items():
        if key not in ["max_steps", "stop_at_99"]:
            assert current_config["protocol"][key] == value
    assert result["steps"] == 10000 and result["segments"] >= 2
    record = {"optimizer": result["optimizer"], "seed": result["seed"], "lr": result["lr"],
        "decay": result["weight_decay"], "original_stop_step": original_rows[-1]["step"],
        "first_val95": result["val95"]["step"] if result["val95"] else None,
        "first_val99": result["val99"]["step"] if result["val99"] else None,
        "final_val_acc": result["final_val_acc"]}
    after = [r for r in rows if result["val99"] and r["step"] >= result["val99"]["confirmed_step"]]
    worst = min(after, key=lambda r: r["val_acc"]) if after else None
    record["min_val_after_confirmed99"] = worst["val_acc"] if worst else None
    record["worst_later_step"] = worst["step"] if worst else None
    for threshold in [.95, .99]:
        # At least three observations and no later observed violations.
        record[f"stable{int(threshold*100)}_within_10000"] = next((r["step"] for i,r in enumerate(rows)
            if r["step"] > 0 and len(rows)-i >= 3
            and all(x["val_acc"] >= threshold for x in rows[i:])), None)
    rows_out.append(record)
    histories.append((result, rows))
with (target / "comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
    writer.writeheader()
    writer.writerows(rows_out)
aggregate = []
for opt in ["lion", "adamw"]:
    group = [r for r in rows_out if r["optimizer"] == opt]
    result = {"optimizer": opt, "n": len(group), "any_drop_below99": sum(r["min_val_after_confirmed99"] < .99 for r in group),
        "any_drop_below95": sum(r["min_val_after_confirmed99"] < .95 for r in group)}
    for name in ["first_val95", "first_val99", "stable95_within_10000", "stable99_within_10000"]:
        values = [r[name] for r in group if r[name] is not None]
        result[name] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                        "std": statistics.stdev(values) if len(values)>1 else None}
    aggregate.append(result)
(target / "comparison.json").write_text(json.dumps(aggregate, indent=2)+"\n")

plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
fig, axes = plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
colors = {"adamw":"#D86828", "lion":"#176B87"}
for result, rows in histories:
    label = f"{result['optimizer'].upper()} seed {result['seed']}"
    for ax in axes:
        ax.plot([r["step"] for r in rows],[r["val_acc"] for r in rows],
                color=colors[result["optimizer"]], alpha=.75, label=label)
        ax.axhline(.99,color="gray",linestyle=":",linewidth=.8)
        ax.grid(alpha=.2)
        ax.set(xlabel="Optimizer steps",ylabel="Validation accuracy")
axes[0].set(title="Full continuation", ylim=(0,1.025))
axes[1].set(title="After original stopping points",xlim=(1200,10000),ylim=(.85,1.005))
axes[1].legend(fontsize=8,loc="lower right",ncols=2)
fig.suptitle("Original formal checkpoints continued to 10k · no hyperparameter changes")
fig.savefig(target / "stability.png",dpi=180)
fig.savefig(target / "stability.pdf")
plt.close(fig)

lines = ["# 原六次正式运行的固定预算续跑", "",
    "原有六次固定基线比较的参数、优化器状态、随机数和取样状态完整延续到10000步，不重新调参。"
    "原始日志前缀逐行相同，已自动验证。此前的首次阈值指标保留，新增稳定性观察。", "",
    "| 方法 | seed | 首次val99 | 原停止步 | 达标后最低val | 最低点步数 | 预算内持续99%起点 | 最终val |",
    "|---|---|---|---|---|---|---|---|"]
for r in rows_out:
    lines.append(f"| {r['optimizer']} | {r['seed']} | {r['first_val99']} | {r['original_stop_step']} | "
        f"{r['min_val_after_confirmed99']:.2%} | {r['worst_later_step']} | "
        f"{r['stable99_within_10000'] or '—'} | {r['final_val_acc']:.2%} |")
lines += ["", "![续跑稳定性](stability.png)", "",
    "预算内持续99%起点是回顾性指标：从该点到10000步所有已观察准确率都≥99%，且至少三个检查点。"
    "它不是事先注册的主指标，也不保证10000步以后稳定；不能静默替换原有首次达标排名。", "",
    "续跑分为两个计算时段，离线停机不计时；这里比较步数和准确率，不据此重新宣称wall-time优势。"
    "本项只核对原固定AdamW/1与Lion/5比较，没有继续补充搜索选出的AdamW/2e-3那一组。"]
(target / "STABILITY.md").write_text("\n".join(lines)+"\n")
print(json.dumps(aggregate, indent=2))
