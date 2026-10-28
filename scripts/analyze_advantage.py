"""Analyze the locked-design study without selecting on confirmation outcomes."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.metrics import mean_std
from lion_experiment.train import atomic_json
from scripts.confirm_advantage import tail_accuracy


def load_runs(root, phase):
    result = []
    for p in sorted((root / phase).glob("*/summary.json")):
        result.append((json.loads(p.read_text()),
            [json.loads(s) for s in (p.parent / "metrics.jsonl").read_text().splitlines()],
            json.loads((p.parent / "config.json").read_text()),
            json.loads((p.parent / "metadata.json").read_text())))
    return result


def stat_text(stats, digits=1):
    if stats["mean"] is None:
        return "—"
    return f"{stats['mean']:.{digits}f}" + (
        f" ± {stats['std']:.{digits}f}" if stats["std"] is not None else " (n=1)")


def analyze(root):
    root = Path(root)
    if not (root / "complete.json").exists():
        raise RuntimeError("All confirmation runs must finish before final analysis")
    locked = json.loads((root / "design_locked.json").read_text())
    design = locked["design"]
    frozen = json.loads((root / "frozen_selection.json").read_text())
    tuning, confirmation = load_runs(root, "tuning"), load_runs(root, "confirmation")
    assert len(tuning) == len(confirmation) == 6
    for phase, runs in [("tuning", tuning), ("confirmation", confirmation)]:
        expected_seeds = [design["tuning_seed"]] if phase == "tuning" else design["confirmation_seeds"]
        assert {r[0]["seed"] for r in runs} == set(expected_seeds)
        assert {opt:sum(r[0]["optimizer"]==opt for r in runs) for opt in ["lion","adamw"]} == {"lion":3,"adamw":3}
        for summary, rows, cfg, metadata in runs:
            assert summary["steps"] == design["protocol"]["max_steps"]
            assert summary["segments"] == 1, "Timing comparison requires uninterrupted runs"
            assert cfg["protocol"]["stop_at_99"] is False
            assert math.isclose(summary["eta_lambda"], design["target_eta_lambda"], abs_tol=1e-12)
            assert [r["step"] for r in rows] == list(range(0,10001,100))
            for name, col, threshold in [("train99","train_acc",.99),("val95","val_acc",.95),("val99","val_acc",.99)]:
                candidates = [rows[i]["step"] for i in range(1,len(rows)-2)
                    if all(r[col]>=threshold for r in rows[i:i+3])]
                assert (summary[name]["step"] if summary[name] else None) == (candidates[0] if candidates else None)
    for seed in design["confirmation_seeds"]:
        pair = [r for r in confirmation if r[0]["seed"] == seed]
        assert len(pair) == 2
        assert pair[0][2]["protocol"] == pair[1][2]["protocol"]
        for key in ["source_sha256","initial_weights_sha256","split_sha256","torch","cuda","gpu"]:
            assert pair[0][3][key] == pair[1][3][key], (seed,key)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    flat = []
    for summary, rows, cfg, metadata in confirmation:
        after99 = [r["val_acc"] for r in rows
            if summary["val99"] and r["step"] >= summary["val99"]["confirmed_step"]]
        stable99 = next((r["step"] for i,r in enumerate(rows)
            if r["step"] > 0 and len(rows)-i>=3 and all(x["val_acc"]>=.99 for x in rows[i:])), None)
        row = {"optimizer":summary["optimizer"],"seed":summary["seed"],"lr":summary["lr"],
            "weight_decay":summary["weight_decay"],"eta_lambda":summary["eta_lambda"],
            "primary_val95_success":summary["val95"] is not None,
            "Ttrain99":summary["train99"]["step"] if summary["train99"] else None,
            "Tval95":summary["val95"]["step"] if summary["val95"] else None,
            "Tval95_wall_seconds":summary["val95"]["wall_seconds"] if summary["val95"] else None,
            "Tval99":summary["val99"]["step"] if summary["val99"] else None,
            "delay":summary["generalization_delay"], "stable99_within_10000":stable99,
            "final_val_acc":summary["final_val_acc"],"tail_mean_val_acc":tail_accuracy(rows),
            "min_val_after_confirmed99":min(after99) if after99 else None,
            "optimizer_state_bytes":summary["optimizer_state_bytes"],
            "peak_memory_bytes":summary["peak_memory_bytes"],"mean_step_ms":summary["mean_step_ms"],
            "final_wall_seconds":summary["wall_seconds"]}
        flat.append(row)
    with (out / "per_run.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    aggregates = []
    for opt in ["lion","adamw"]:
        group = [r for r in flat if r["optimizer"]==opt]
        item = {"optimizer":opt,"n":len(group),
            "val95_successes":sum(r["primary_val95_success"] for r in group),
            "val99_successes":sum(r["Tval99"] is not None for r in group)}
        for key in ["Tval95","Tval95_wall_seconds","Tval99","Ttrain99","delay","stable99_within_10000",
                    "final_val_acc","tail_mean_val_acc","mean_step_ms","optimizer_state_bytes","peak_memory_bytes"]:
            item[key] = mean_std([r[key] for r in group if r[key] is not None])
        aggregates.append(item)
    atomic_json(out / "comparison.json",aggregates)
    paired = []
    for seed in design["confirmation_seeds"]:
        lion = next(r for r in flat if r["optimizer"]=="lion" and r["seed"]==seed)
        adam = next(r for r in flat if r["optimizer"]=="adamw" and r["seed"]==seed)
        paired.append({"seed":seed,"tail_accuracy_lion_minus_adamw":lion["tail_mean_val_acc"]-adam["tail_mean_val_acc"],
            "final_accuracy_lion_minus_adamw":lion["final_val_acc"]-adam["final_val_acc"],
            "val95_step_lion_minus_adamw":lion["Tval95"]-adam["Tval95"]
                if lion["Tval95"] is not None and adam["Tval95"] is not None else None})
    atomic_json(out / "paired_differences.json",paired)

    colors = {"lion":"#176B87","adamw":"#D86828"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for name, xcol, title in [("accuracy_steps","step","Fresh-seed confirmation · equal shrinkage · full 10k horizon"),
                             ("validation_time","wall_seconds","Validation against elapsed time · one A100 · FP32")]:
        fig, ax = plt.subplots(figsize=(8,4.8),layout="constrained")
        seen = set()
        for summary, rows, _, _ in confirmation:
            opt = summary["optimizer"]
            columns = [("val_acc","-","validation")]
            if xcol=="step": columns.append(("train_acc","--","train"))
            for col,style,label in columns:
                key=(opt,label)
                ax.plot([r[xcol] for r in rows],[r[col] for r in rows],style,color=colors[opt],alpha=.7,
                    label=f"{opt.upper()} {label}" if key not in seen else None)
                seen.add(key)
        ax.axhline(.95,color="gray",linestyle=":",linewidth=.9)
        ax.set(xlabel="Optimizer steps" if xcol=="step" else "Wall time including evaluation (s)",
               ylabel="Accuracy",ylim=(0,1.025),title=title)
        ax.grid(alpha=.2); ax.legend(loc="lower right",ncols=2 if xcol=="step" else 1)
        for ext in ["png","pdf"]: fig.savefig(out / f"{name}.{ext}",dpi=180)
        plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,opt in zip(axes,["adamw","lion"]):
        for summary,rows,_,_ in sorted(tuning,key=lambda r:r[0]["lr"]):
            if summary["optimizer"]==opt:
                ax.plot([r["step"] for r in rows],[r["val_acc"] for r in rows],label=f"lr={summary['lr']:.0e}")
        ax.axhline(.95,color="gray",linestyle=":",linewidth=.9)
        ax.set(xlabel="Optimizer steps",ylabel="Validation accuracy",ylim=(0,1.025),title=opt.upper())
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Seed 7 only · three candidates each · eta × decay fixed at 1e-5")
    for ext in ["png","pdf"]: fig.savefig(out / f"lr_sensitivity.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    for seed in design["confirmation_seeds"]:
        a=next(r for r in flat if r["seed"]==seed and r["optimizer"]=="adamw")
        l=next(r for r in flat if r["seed"]==seed and r["optimizer"]=="lion")
        axes[0].plot([0,1],[a["tail_mean_val_acc"],l["tail_mean_val_acc"]],"o-",alpha=.8,label=f"seed {seed}")
    axes[0].set(xticks=[0,1],xticklabels=["AdamW","Lion"],ylabel="Mean validation accuracy, steps 8100–10000",ylim=(0,1.025))
    axes[0].legend(); axes[0].grid(alpha=.2)
    for index,opt in enumerate(["adamw","lion"]):
        value=next(s for s in aggregates if s["optimizer"]==opt)["optimizer_state_bytes"]["mean"]/2**20
        axes[1].bar(index,value,color=colors[opt],width=.55)
        axes[1].text(index,value,f"{value:.2f}",ha="center",va="bottom")
    axes[1].set(xticks=[0,1],xticklabels=["AdamW","Lion"],ylabel="Persistent optimizer state (MiB)",ylim=(0,3.9))
    fig.suptitle("Paired held-out-seed accuracy and measured optimizer-state storage")
    for ext in ["png","pdf"]: fig.savefig(out / f"paired_accuracy_memory.{ext}",dpi=180)
    plt.close(fig)

    a={r["optimizer"]:r for r in aggregates}
    primary_lion_win = a["lion"]["val95_successes"] > a["adamw"]["val95_successes"] or (
        a["lion"]["val95_successes"]==a["adamw"]["val95_successes"]==3 and
        a["lion"]["Tval95"]["mean"] < a["adamw"]["Tval95"]["mean"])
    conclusion={"lion_primary_advantage_observed":primary_lion_win,
        "tail_mean_accuracy_difference":a["lion"]["tail_mean_val_acc"]["mean"]-a["adamw"]["tail_mean_val_acc"]["mean"],
        "all_three_paired_tail_differences_positive":all(r["tail_accuracy_lion_minus_adamw"]>0 for r in paired),
        "selected_at_grid_edge":{opt:frozen["selected"][opt]["lr"] in
            [min(design["candidates"][opt]),max(design["candidates"][opt])] for opt in ["lion","adamw"]},
        "scope":"Three fresh seeds in the preselected low-shrinkage regime; not a universal or significance claim."}
    atomic_json(out / "conclusion.json",conclusion)
    lines=["# Lion 优势验证：新 seed、等次数调参与相同收缩强度", "",
        "本轮先冻结设计，再在seed7上各筛选三个候选，选定后用未参与选择的seed4/5/6完成确认。"
        "低衰减场景来自之前探索，属于对已发现条件的独立seed验证，不能推广到所有任务或参数。", "",
        "## 已冻结协议", "", f"设计锁定时间：{locked['locked_at']}；学习率冻结时间：{frozen['frozen_at']}。",
        "原组库模型、mod97模乘和50%划分保持一致；同seed相同初始化、划分和取样。"
        "所有候选和确认运行都满10000步，FP32，每100步完整评估，阈值要求连续三次。"
        "每候选及两方法间ηλ都固定1e-5，weight_decay=1e-5/lr。Lion由本项目自写，AdamW为内置对照。", "",
        f"选择结果：Lion lr={frozen['selected']['lion']['lr']:.0e}, λ={frozen['selected']['lion']['weight_decay']:.6g}；"
        f"AdamW lr={frozen['selected']['adamw']['lr']:.0e}, λ={frozen['selected']['adamw']['weight_decay']:.6g}。",
        "选择规则对双方相同：95%达标优先、达标起点更早、末2000步均值更高、最终准确率更高、lr更低。", "",
        "## 新seed确认结果", "", "| 指标 | Lion | AdamW |", "|---|---|---|"]
    lines += [f"| val95 达标数/3（主指标） | {a['lion']['val95_successes']}/3 | {a['adamw']['val95_successes']}/3 |",
        f"| Tval95 步数，成功样本均值±SD | {stat_text(a['lion']['Tval95'])} | {stat_text(a['adamw']['Tval95'])} |",
        f"| val99 达标数/3 | {a['lion']['val99_successes']}/3 | {a['adamw']['val99_successes']}/3 |",
        f"| 末2000步平均val准确率 | {100*a['lion']['tail_mean_val_acc']['mean']:.2f}% | {100*a['adamw']['tail_mean_val_acc']['mean']:.2f}% |",
        f"| 最终val准确率 | {100*a['lion']['final_val_acc']['mean']:.2f}% | {100*a['adamw']['final_val_acc']['mean']:.2f}% |",
        f"| 优化器状态 / MiB | {a['lion']['optimizer_state_bytes']['mean']/2**20:.2f} | {a['adamw']['optimizer_state_bytes']['mean']/2**20:.2f} |", "",
        "达标均值只纳入成功样本，不能把两边不同成功数的条件均值直接当总体速度比。未达标标记>10000。"
        "标准差为样本SD；每列有效n见comparison.json。末窗口取8100至10000步共20个检查点。", "",
        "| 方法 | seed | Tval95 | Tval99 | 最终val | 末窗口val均值 |", "|---|---|---|---|---|---|"]
    for r in flat:
        lines.append(f"| {r['optimizer']} | {r['seed']} | {r['Tval95'] or '>10000'} | {r['Tval99'] or '>10000'} | "
            f"{r['final_val_acc']:.2%} | {r['tail_mean_val_acc']:.2%} |")
    lines += ["", "## 结论范围", "",
        ("这三个新seed在预定主指标上观察到Lion优势。" if primary_lion_win else
         "这三个新seed未在预定主指标上确认Lion优势，应保留这一结果。"),
        f"末窗口验证准确率的平均差（Lion−AdamW）为{100*conclusion['tail_mean_accuracy_difference']:.2f}个百分点。"
        "内存优势针对持久优化器状态，不是整个训练显存减半。"
        "三个seed不能支持普适或显著性结论；没有根据确认结果重选lr、λ、seed或预算。"
        "本轮选中值位于候选边界，结果限定于已列明的候选范围，不称为全局最优比较。", "",
        "![步数曲线](accuracy_steps.png)", "", "![时间曲线](validation_time.png)", "",
        "![seed7敏感性](lr_sensitivity.png)", "", "![成对准确率与状态内存](paired_accuracy_memory.png)"]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"selection":frozen["selected"],"comparison":aggregates,"conclusion":conclusion},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"results/advantage_validation")
    analyze(parser.parse_args().root)
