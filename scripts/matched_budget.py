"""Supplement fixed-baseline results with an equal-count AdamW LR search.

Designed after inspecting the initial fixed-baseline comparison, not preregistered.
Original Lion tuning/formal results are reused unchanged, with provenance.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lion_experiment.train import atomic_json, run
from lion_experiment.suite import run_name, selection_key
from lion_experiment.analyze import analyze

source = ROOT / "results/lion_study"
target = ROOT / "results/matched_budget_study"
config = json.loads((ROOT / "configs/lion_study.json").read_text())
candidates = [0.0005, 0.001, 0.002]
target.mkdir(parents=True, exist_ok=True)
atomic_json(target / "design.json", {
    "purpose": "Supplementary equal-count LR search after the fixed-baseline comparison",
    "adamw_lrs": candidates, "lion_lrs": config["lion_lrs"],
    "tuning_seed": 0, "formal_seeds": [1, 2, 3], "max_steps_per_trial": 100000,
    "note": "Design added after seeing fixed-baseline results; no formal seed used in LR ranking",
    "reused_lion_source": str(source), "protocol": config})
for phase in ["tuning", "formal"]:
    (target / phase).mkdir(exist_ok=True)
    for path in (source / phase).glob("lion_*"):
        link = target / phase / path.name
        if not link.exists():
            link.symlink_to(Path("../../lion_study") / phase / path.name, target_is_directory=True)

results = []
for lr in candidates:
    results.append(run(config, "adamw", lr, config["tuning_seed"],
        target / "baseline_tuning" / run_name("adamw", 0, lr), "cuda", True, "tuning"))
winner = min(results, key=selection_key)
frozen_lion = json.loads((source / "frozen_selection.json").read_text())
atomic_json(target / "frozen_selection.json", {**frozen_lion, "selected_adamw_lr": winner["lr"],
    "adamw_tuning_results": results, "supplementary_design": True})
for seed in config["formal_seeds"]:
    name = run_name("adamw", seed, winner["lr"])
    if winner["lr"] == config["adamw_lr"]:
        link = target / "formal" / name
        if not link.exists():
            link.symlink_to(Path("../../lion_study/formal") / name, target_is_directory=True)
    else:
        run(config, "adamw", winner["lr"], seed, target / "formal" / name,
            "cuda", True, "formal")
analyze(target)
# The shared analyzer describes the initial fixed-baseline study by default.
path = target / "analysis/results.md"
text = path.read_text().replace("# Lion / AdamW 实验结果", "# Lion / AdamW 等次数调参补充实验")
text = text.replace(
    "Lion 有三次单 seed 学习率探索；AdamW 使用预先指定的组内起点（1e-3），未执行等量学习率搜索。\n因此这里是固定 AdamW 基线比较，不能据此声称两种优化器在相同调参预算下的最优性能差异。",
    "Lion 与 AdamW 各有三次 seed0 学习率筛选，每次上限100000步，采用相同阈值、停止和选择规则。"
    "AdamW 候选为5e-4、1e-3、2e-3；选出的学习率为" + f"{winner['lr']:.0e}。\n"
    "这项补充设计在首次固定基线比较后加入，不能视为事前注册的确认性实验。"
    "两种算法使用各自学习率范围和固定衰减，没有声称搜索到全局最优；实际搜索计算量因提前达标而不同。"
    "正式Lion结果来自原始运行，通过相对符号链接复用并保留来源，未重复计算或伪造。")
path.write_text(text)
print(f"Equal-count supplementary search complete: AdamW lr={winner['lr']}", flush=True)
