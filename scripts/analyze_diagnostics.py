"""Summarize controlled decay experiments and original-source reproduction."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.metrics import summarize

out = ROOT / "results/diagnostics"
runs = []
for path in sorted((out / "decay_sweep").glob("*/summary.json")):
    result = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    history = [json.loads(s) for s in (path.parent / "metrics.jsonl").read_text().splitlines()]
    runs.append((result, config, history))
if not runs:
    raise ValueError("No completed controlled diagnostics")
flat = []
for result, config, rows in runs:
    first = result["val99"]
    after = [r["val_acc"] for r in rows if first and r["step"] >= first["confirmed_step"]]
    # Retrospective within-horizon stability, with >=3 observed checkpoints.
    # This does not claim stability after the end of the observation window.
    stable99 = next((r["step"] for i,r in enumerate(rows)
        if r["step"] > 0 and len(rows)-i >= 3
        and all(later["val_acc"] >= .99 for later in rows[i:])), None)
    flat.append(dict(optimizer=result["optimizer"], lr=result["lr"],
        reference_decay=config["protocol"]["adamw_weight_decay"], decay=result["weight_decay"],
        eta_lambda=result["eta_lambda"], horizon=result["steps"],
        train99=result["train99"]["step"] if result["train99"] else None,
        val95=result["val95"]["step"] if result["val95"] else None,
        val99=first["step"] if first else None,
        delay=result["generalization_delay"], final_val_acc=result["final_val_acc"],
        min_val_after_confirmed99=min(after) if after else None,
        stable99_within_horizon=stable99))
with (out / "decay_comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for index, opt in enumerate(["adamw", "lion"]):
    for result, config, rows in sorted(runs, key=lambda r: -r[1]["protocol"]["adamw_weight_decay"]):
        if result["optimizer"] != opt:
            continue
        color = {1: "#D86828", .1: "#176B87", .01: "#549936", 0: "#775994"}[
            config["protocol"]["adamw_weight_decay"]]
        label = f"wd={result['weight_decay']:g}"
        for ax, column in [(axes[0,index], "val_acc"), (axes[1,index], "train_acc")]:
            ax.plot([r["step"] for r in rows], [r[column] for r in rows], color=color, label=label)
            ax.set(ylim=(0,1.025), xlabel="Optimizer steps", ylabel=column.replace('_',' '))
            ax.grid(alpha=.2)
    axes[0,index].set_title(f"{opt.upper()} · lr={.001 if opt == 'adamw' else .0002:g}")
    axes[0,index].legend(loc="lower right")
fig.suptitle("Controlled decay diagnostic · seed 0 · fixed 10,000-step horizon · no early stopping")
fig.savefig(out / "decay_diagnostic.png", dpi=180)
fig.savefig(out / "decay_diagnostic.pdf")
plt.close(fig)

original = sorted((ROOT / "results").glob("res_diagnostic_original_adam0*.pt"))
legacy_summary = None
if original:
    legacy = torch.load(original[0], map_location="cpu", weights_only=False)
    legacy_rows = [{"step": s, "wall_seconds": None, "train_acc": t, "val_acc": v}
        for s, t, v in zip(legacy["its"], legacy["train_acc"], legacy["val_acc"])]
    legacy_summary = {"optimizer": "Adam", "weight_decay": 0,
        "steps": legacy["its"][-1], "train_acc": legacy["train_acc"][-1],
        "val_acc": legacy["val_acc"][-1], "first_val95": next(
            (s for s,a in zip(legacy['its'],legacy['val_acc']) if a>=.95), None),
        "first_val99": next((s for s,a in zip(legacy['its'],legacy['val_acc']) if a>=.99), None),
        "note": "Original training accuracy is accumulated over changing weights during an epoch; not a synchronized train metric"}
    (out / "original_reproduction_summary.json").write_text(json.dumps(legacy_summary, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(8,4.5), layout="constrained")
    ax.plot(legacy["its"], legacy["train_acc"], "--", label="Original train (online epoch metric)")
    ax.plot(legacy["its"], legacy["val_acc"], label="Original validation")
    ax.set(xlabel="Optimizer steps", ylabel="Accuracy", ylim=(0,1.025),
        title="Unmodified group script · Adam · weight decay=0 · seed 0")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / "original_reproduction.png", dpi=180)
    fig.savefig(out / "original_reproduction.pdf")
    plt.close(fig)

lines = ["# 第二轮诊断与重跑", "", "## 已查明的配置差异", "",
    "原仓库入口实际默认 Adam、weight_decay=0；上一轮本地协议采用 AdamW、weight_decay=1，Lion=5。"
    "原文件的注释虽提到AdamW/1，但注释不是入口实际配置。此前把这一试跑值当作正式组内基线并过早宣布完成，是实验解释上的错误。", "",
    "上一轮日志是真实结果，仍只对其明确配置有效；不能拿它替代原仓库基线复现，也不能由此推断Lion对长延迟grokking的普遍影响。", "",
    "## 泄漏与评估核对", "",
    "六个已保存的正式检查点经独立推理复算，准确率与日志一致；训练验证索引不相交且覆盖全集。"
    "打乱验证标签后的准确率约0.67%–1.31%；改变未来答案token，对前一个位置的logits影响为0。"
    "这些检查没有发现答案泄漏或把训练集冒充验证集的证据。详情见checkpoint_audit.json。", "",
    "## 原始脚本复现", ""]
if legacy_summary:
    lines.append(f"原始代码不改训练循环，seed0、Adam/λ=0，运行{legacy_summary['steps']}步。"
        f"最终训练准确率{legacy_summary['train_acc']:.2%}，验证准确率{legacy_summary['val_acc']:.2%}；"
        f"首次单次val95={legacy_summary['first_val95']}，val99={legacy_summary['first_val99']}。"
        "这里报告原始逐epoch验证指标，不与新协议三次连续阈值混为同一种统计。")
lines += ["", "![原始脚本复现](original_reproduction.png)", "", "## 控制变量重跑", "",
    "保持原模型、seed0划分、batch、评估及训练取样方式不变；每个方法内部仅改变weight decay。"
    "AdamW学习率固定1e-3；Lion固定2e-4、衰减为AdamW的5倍，保证两方法的峰值ηλ匹配。"
    "所有运行固定10000步，关闭达标停止，以检查后续稳定性。此为探索性诊断，不能当成新的三seed正式比较。", "",
    "| 优化器 | λ | ηλ | Ttrain99 | Tval95 | Tval99 | 最终val | 首次确认99%后最低val | 预算内持续99%起点 |",
    "|---|---|---|---|---|---|---|---|---|"]
for row in flat:
    def threshold(key):
        return row[key] if row[key] is not None else f">{row['horizon']}"
    minimum = f"{row['min_val_after_confirmed99']:.2%}" if row['min_val_after_confirmed99'] is not None else "—"
    lines.append(f"| {row['optimizer']} | {row['decay']:g} | {row['eta_lambda']:g} | "
        f"{threshold('train99')} | {threshold('val95')} | {threshold('val99')} | "
        f"{row['final_val_acc']:.2%} | {minimum} | {row['stable99_within_horizon'] or '—'} |")
lines += ["", "![固定预算衰减对照](decay_diagnostic.png)", "",
    "预算内持续99%起点是回顾性指标：从该检查点起到10000步的所有观测均≥99%，且至少有三个观测。"
    "它仅表示已观察区间内的稳定性，不保证预算结束之后也保持；不替换原计划中的三次连续阈值。", "",
    "诊断seed0的高衰减AdamW在首次确认99%之后出现过明显回落；原停止规则会掩盖这种后续不稳定。"
    "不能据此断言原三个正式seed都出现同样回落；六个原检查点的续跑另列于formal_stability。"
    "这是停止规则与结论范围的限制，不能把首次达标等同于稳定泛化。", "",
    "## 如何使用", "", "不要为了让Lion赢而挑设置。先区分原仓库复现、固定配置比较、控制变量诊断三类证据；"
    "只有在基线任务和参数冻结后，才能将选定设置用于新的多seed确认。"
    "任何未达标的诊断均按10000步右删失报告，不能据此说永远不会grok。"]
(out / "DIAGNOSIS.md").write_text("\n".join(lines)+"\n")
print('Diagnostics saved:', out)
