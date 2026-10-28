> Historical study notebook. See [current project README](README.md), [requirements audit](docs/REQUIREMENTS_AUDIT.md), and [consolidated contribution](docs/LION_CONTRIBUTION.md) for the release.

# COMP6704 Group 7 · Lion 实验

依据 `docs/COMP6704_Group7_Lion_实验计划.docx` 完成。组库已通过本机 SSH key 克隆到本目录；上游原模型和其他任务入口保留。原README是Grokfast项目说明，本文件是本次Lion实验的使用入口。

<!-- ADVANTAGE_VALIDATION_START -->
## 最新：独立seed优势验证

三个新seed在预先固定的95%验证达标指标上观察到Lion优势。末2000步平均验证准确率，Lion与AdamW的均值分别为98.76%和41.84%，差值为+56.92个百分点。优化器持久状态分别约1.61 MiB和3.23 MiB。

本轮双方各3次seed7筛选，冻结后用新seed4/5/6确认；所有运行固定10000步、ηλ相同。见[新seed结果与完整图表](results/advantage_validation/analysis/RESULTS.md)、[优势验证PDF](deliverables/Lion_优势验证说明.pdf)、[优势验证展示](deliverables/Lion_优势验证展示.pptx)、[验证协议](docs/ADVANTAGE_VALIDATION_PROTOCOL.md)。

该优势验证针对此前探索发现的低衰减场景。所有新seed运行前已锁定选择规则与预算；seed7只用于筛选，seed4/5/6只用于确认，没有根据确认结果重选参数。三个seed不足以支持普适或统计显著性结论，结果应表述为该任务、模型和协议下的实测优势。选中学习率在本轮候选边界，不能声称覆盖任意超参数或找到全局最优。
<!-- ADVANTAGE_VALIDATION_END -->

## 自写优化器的范围

Lion的核心更新由本项目自行实现于 `lion_experiment/optim.py`：方向计算、参数衰减、参数更新和动量更新都在本地 `Lion.step()` 中显式完成，训练入口直接导入这个类。没有调用第三方Lion实现，也没有把官方Lion包装为本地优化器。

继承的 `torch.optim.Optimizer` 仅提供参数组、清理梯度和状态保存等通用框架；Lion更新算法本身由本地代码执行。`vendor/lion_official.py` 仅供单独的正确性对照测试，训练代码不导入它。另有不依赖官方实现的两步手算用例，专门验证旧动量的使用顺序、sign方向和解耦衰减。AdamW对照当前使用PyTorch内置实现，需要与自写Lion区分。

## 当前结论：第二轮诊断后修订

上一轮把初版运行和材料过早称为全部完成。复核确认两点：① 原入口默认Adam/weight_decay=0，初版本地采用的是AdamW/1与Lion/5；② 连续三次达到99%不能保证之后稳定。代码、旧日志和原始阈值统计保留，结论范围已修订。

第二轮已完成原始默认脚本50000步复现、8组固定10000步衰减对照，以及原6个正式检查点续跑到10000步（保持参数和随机状态不变）。六个原日志前缀逐行一致。独立检查未发现标签泄漏或训练验证重叠；11项原有测试通过。旧入口仅指定Lion时的beta2默认值也已修正为0.99；此前新入口一直使用0.99。

| 原固定基线比较 | Lion | AdamW |
|---|---|---|
| 首次连续三次达到99%的平均起点 | 1566.7步 | 1366.7步 |
| 观察到10000步的持续99%平均起点（回顾性） | 1566.7步 | 7100步 |
| 首次确认后曾跌破95%的seed数 | 0/3 | 3/3 |

Lion三个seed确认后均保持99%以上；AdamW三个seed后续最低约58.10%、46.52%、94.27%。因此“Lion没有优势”过于笼统：它在该本地高衰减配置下的后续稳定性更好。回顾性稳定指标不替换原主指标，也不保证10000步之后或其他配置的表现。

直接查看[中文诊断与重跑说明PDF](deliverables/Lion_诊断与重跑说明.pdf)、[诊断细节与8组曲线](results/diagnostics/DIAGNOSIS.md)、[原六检查点续跑结果](results/diagnostics/formal_stability/STABILITY.md)。报告和展示已纳入复核结果，仍区分本地初步配置与未确认的组内最终协议。

## 已完成的结果（2026-10-27）

已完成2次1000步pilot、3次seed0的Lion学习率筛选，以及seed1/2/3的6次正式成对运行。按预设共同停止规则提前结束，共24700次更新；所有运行的累计测量wall time约171秒。11项测试通过，另复核了日志、阈值起点和正式运行的源文件校验值。

Lion选择lr=3e−4、λ=5。两种方法的val95和val99成功率均为3/3。

| 正式指标（均值±样本标准差） | Lion | AdamW |
|---|---|---|
| Tval99 / steps | 1566.7 ± 321.5 | 1366.7 ± 152.8 |
| 泛化延迟 ΔT / steps | 366.7 ± 230.9 | 866.7 ± 152.8 |
| val99 wall time / s | 10.5 ± 2.5 | 9.6 ± 1.3 |
| 优化器持久状态 / MiB | 1.61 | 3.23 |

Lion的泛化延迟较短，但训练拟合较晚，平均验证达标步数和总时间没有胜过固定AdamW基线。此配置没有出现很长的grokking平台，报告据实呈现这一点。

可直接查看[结果与三张图](results/lion_study/analysis/results.md)、[报告PDF](deliverables/Lion_Report.pdf)、[可编辑Word](deliverables/Lion_Report.docx)、[展示PDF](deliverables/Lion_Presentation.pdf)、[可编辑PowerPoint](deliverables/Lion_Presentation.pptx)和[中文讲稿/问答](deliverables/Lion_Speaker_Notes_zh.md)。初版展示为8页主讲加1页备用；现已加入补充分析、基线诊断和稳定性续跑。

公平性复核后另完成AdamW的三档seed0学习率筛选（5e−4、1e−3、2e−3），使两种方法都有3次筛选、每次相同上限及停止规则；正式Lion日志原样复用，另完成选中AdamW的seed1/2/3。详见[等次数调参补充结果](results/matched_budget_study/analysis/results.md)。该补充设计是在首次固定基线结果之后加入，属于探索性分析；实际调参耗时因提前停止而不同。报告第9节和展示补充页已纳入它，完整展示现包含13页（含诊断、续跑与备用问答），可按汇报时间选择主讲页。

补充筛选选出AdamW lr=2e−3，其正式Tval99为933.3 ± 251.7步（3/3成功），对比Lion的1566.7 ± 321.5步。累计共17次独立真实运行、34000次更新（复用的Lion日志不重复计数）。

## 开始

```bash
cd /data2/huangyuqi/polyu/comp6704
source .venv/bin/activate
python -m pytest -q
```

当前 `.venv` 复用本机现有 PyTorch 2.5.1+cu124，并隔离安装绘图和测试依赖。独立机器可用 Python3.10+ 新建环境，再按对应CUDA安装PyTorch，然后 `pip install -r requirements-lion.txt`。重建时不要照搬本机venv的绝对路径。

## 执行

```bash
# 短程检查；不会混入正式统计
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage pilot --resume

# 完整流程：三个Lion候选 → 冻结学习率 → 新seed1/2/3的Lion和AdamW → 分析
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage all --resume

# 或分阶段完成
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage tune --resume
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.suite --stage formal --resume

# 重新生成图表和结果表
python -m lion_experiment.analyze

# 补充等次数AdamW调参及选中基线（复用Lion原始结果）
CUDA_VISIBLE_DEVICES=0 python scripts/matched_budget.py

# 独立运行示例
CUDA_VISIBLE_DEVICES=0 python -m lion_experiment.train \
  --optimizer lion --lr 2e-4 --seed 4 --phase smoke \
  --max-steps 1000 --output results/manual_lion_seed4
```

`--resume`跳过已完成且配置相同的运行，未完成运行从最后安全检查点恢复。配置不同时拒绝覆盖；新协议应使用新输出目录。原入口也支持 `python main.py --optimizer Lion --beta1 .9 --beta2 .99 --lr 2e-4 --weight_decay 5 --budget 1000 --label lion`，但正式比较使用新入口，原入口的按epoch日志不符合本实验评估协议。

在Codex沙箱内可能看不到GPU；本次GPU训练通过允许访问GPU的执行环境运行。终端直接执行不受这一沙箱限制。

## 产物

| 路径 | 内容 |
|---|---|
| `configs/lion_study.json` | 冻结前的公共配置与预定候选 |
| `lion_experiment/optim.py` | 自写Lion核心更新 |
| `lion_experiment/model.py` | 从组库提取的共享模型和数据 |
| `vendor/` | 官方对照代码、SHA-256来源记录和许可证 |
| `results/lion_study/pilot/` | 两种优化器的1000步试跑 |
| `results/lion_study/tuning/` | seed0三个学习率的日志与检查点 |
| `results/lion_study/frozen_selection.json` | 学习率选择、依据和冻结配置 |
| `results/lion_study/formal/` | seed1/2/3成对运行 |
| `results/lion_study/analysis/` | 三张PNG/PDF图、结果表和结果说明 |
| `docs/ALGORITHM.md` | 一页算法和复杂度笔记 |
| `docs/PROTOCOL.md` | 数据、评估、停止、计时和公平性约定 |

每次运行有config.json、metadata.json、metrics.jsonl、summary.json、checkpoint.pt。metadata含实际划分索引和初始权重哈希；summary包含阈值的起始及确认检查点、泛化延迟、实际时间、峰值显存及优化器状态字节。CSV中空字段是缺失或预算内未达标，不能当成0。

本地采用计划试跑起点：100000步上限、AdamW lr=1e−3/λ=1、Lion λ=5。该配置及固定AdamW基线仍需组内确认；报告明确限制了调参预算上的结论。详见实验协议。

代码和结果保存在本地，尚未提交或推送到远程。

## 重建报告与展示

```bash
pip install -r requirements-docs.txt
python scripts/build_deliverables.py
libreoffice --headless --convert-to pdf --outdir deliverables \
  deliverables/Lion_Report.docx deliverables/Lion_Presentation.pptx
```

报告、演示和讲稿由真实结果表生成；图表需先由分析入口生成。`docs/TESTED_ENVIRONMENT.json`记录实测软件版本，`docs/SOURCE_MANIFEST.json`对应正式实验时的实验模块源文件。

## 第二轮复现入口

```bash
python scripts/audit_training.py
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_original.py --budget 50000
CUDA_VISIBLE_DEVICES=0 python scripts/decay_diagnostics.py
CUDA_VISIBLE_DEVICES=0 python scripts/continue_stability.py
python scripts/analyze_diagnostics.py
python scripts/analyze_stability.py
python scripts/build_deliverables.py
python scripts/build_diagnosis_note.py
```

原始脚本复现输出覆盖同名label的原始曲线，需保留旧结果时指定新label。新入口与续跑入口使用独立输出目录并支持从安全检查点恢复。续跑有两个计时分段，此项不重新声称wall-time优势。
