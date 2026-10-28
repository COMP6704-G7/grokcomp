# 新seed优势验证协议

本项验证源自此前seed0的探索：低衰减下，两个方法都能拟合训练集，但验证准确率的上升明显晚于训练拟合；Lion在固定预算内的验证表现更好。为检验这个现象是否能重复，本轮在确认结果可见之前锁定规则，并将筛选seed与确认seed分开。

## 对比条件

| 项目 | 设定 |
|---|---|
| 模型和任务 | 原组库两层128宽4头Transformer，mod97模乘、y不为0 |
| 划分和初值 | 同seed相同权重、4656/4656划分与取样；记录SHA-256 |
| 筛选seed | 7；候选学习率排序仅使用seed7，低衰减场景由此前探索提出 |
| 确认seed | 4、5、6，不参与参数选择 |
| Lion候选lr | 1e-4、2e-4、3e-4 |
| AdamW候选lr | 5e-4、1e-3、2e-3 |
| 共同收缩强度 | 每个候选ηλ=1e-5，λ=1e-5/η；warmup比例同源，故逐步收缩也相同 |
| 每次预算 | 固定10000次更新；筛选和确认都关闭提前停止 |
| 评估 | 每100步完整训练/验证集；阈值连续三次达到才确认 |
| 精度、batch | FP32、TF32关；batch512，保留原末尾48条 |
| 算力条件 | GPU0单张A100、串行；确认阶段交替两个方法的运行次序 |
| 实现 | 本地自写Lion.step；AdamW为PyTorch对照；不启用Grokfast |

两方法各3次筛选、各3次确认，总12次、120000步，实际消耗相同的更新步数。候选lr范围不同，符合比较两个不同更新尺度算法的目的，但不能称为覆盖任意超参数或全局最优。

## 预先固定的选择和评价

候选排序依次是：val95达标优先、val95连续三次达标段的起点更早、末2000步验证均值更高、最终验证准确率更高、较低lr。两方法使用相同的排序规则，完成全部筛选后写冻结文件，确认阶段不改参数。

保留原计划的主指标：10000步内val95达标数及成功运行的Tval95。未达标记为“>10000”，不填成10000步成功，成功数不同的条件均值不直接转换成总体加速倍数。

补充指标为最终验证准确率、8100到10000步共20个检查点的平均验证准确率、val99、达标后回落、观察期内持续99%的起点、实际时间和优化器状态字节。稳定性起点为明确的回顾性指标，不替代原95%主指标。

只用3个确认seed，结论限定在这个任务、模型、候选范围和低衰减条件；不宣称统计显著性或所有场景的优势。若95%主指标不支持Lion优势，仍如实保存；内存优势和其他补充观察独立说明。

## 可复核产物

- `configs/advantage_validation.json`：设计。
- `results/advantage_validation/design_locked.json`：首次运行前的设计、时间戳和SHA-256。
- `results/advantage_validation/frozen_selection.json`：筛选证据、冻结参数及时间戳。
- `results/advantage_validation/tuning/`：全部6次筛选。
- `results/advantage_validation/confirmation/`：全部6次确认。
- `results/advantage_validation/analysis/`：逐次CSV、配对差异、汇总JSON和完整曲线。

复现顺序：`python scripts/confirm_advantage.py` → `python scripts/analyze_advantage.py` → `python scripts/build_advantage_deliverables.py`。原始数据与既有实验保留，不用新结果覆盖历史。
