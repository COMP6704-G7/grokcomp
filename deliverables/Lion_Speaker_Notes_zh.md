# Lion 展示讲稿与问答

初版配置比较加第二轮基线诊断及六检查点稳定性续跑，13页（含备用问答）；仍需区分已测本地配置和未确认的组内最终协议。

## 1. Lion: first crossing and sustained generalization

开场约30秒：我们实现了Lion，并在小组的模97模乘Transformer任务上完成真实实验。主要发现是：Lion减少了训练拟合后到泛化之间的间隔，也减少了优化器状态内存，但本配置下平均达到99%验证准确率并没有比AdamW更快。

## 2. One shared task, one paired protocol

约40秒：数据共9312条，训练验证各4656条。使用仓库原始两层128宽4头因果Transformer，预测答案token。相同seed两种方法有相同初始权重与数据划分，并保存哈希核对。每100步在同一检查点评估完整训练和验证集，连续三次达标，记录起点，第三次确认后停止。

## 3. Lion uses one momentum buffer

约40秒：方向是旧动量和当前梯度的加权和取sign，参数先做解耦衰减，再减去学习率乘方向，最后用beta2更新动量。注意方向使用旧动量。两个算法每步都是O(P)，Lion一个动量，AdamW两个。官方FP32和FP64逐步对照完全一致，总计11项检查通过。

## 4. Learning rate frozen before the formal seeds

约35秒：seed0上三档Lion学习率的99%验证达标步数分别为6200、4300、1600，按预设规则选3e-4，再冻结后运行seed1到3。注意这张是单seed探索，而且weight decay固定5，eta lambda也随学习率变化，不能说是隔离学习率作用的消融。AdamW是固定基线，没有等量搜索。

## 5. Lion memorizes later; both generalize early

约45秒：虚线是完整训练集准确率，实线是验证准确率，每种颜色有三个seed。AdamW训练99%起点都在500步，Lion约1200步。Lion验证99%平均1567步，AdamW1367步。观察不到很长的几万步grokking平台，因此按当前任务如实报告，不把曲线解释成复现了长延迟grokking。

## 6. First-crossing wall time slightly favors AdamW

约35秒：横轴是包含评估的真实同步wall time，Lion达到验证99%平均约10.5秒，AdamW9.6秒。单训练步Lion约6.3毫秒，AdamW6.7毫秒，但训练整体还取决于需要多少步，不能把单步小优势等同总训练快。计时来自很短的同卡串行运行，不声称硬件无关的加速。

## 7. Three paired seeds: tradeoffs, not a universal win

约45秒：两个方法成功率都是3/3。Lion平均Tval99多200步，泛化间隔约367步而AdamW867步。逐seed看Lion仅一个seed更快。优化器状态1.61对3.23MiB确实接近减半，但总峰值130.61对132.22MiB只降约1.2%。因此要区分算法持久状态和整个训练显存。

## 8. What we can conclude—and what remains open

结尾约35秒：实现已验证、实验可复现；本配置Lion省优化器状态内存且间隔短，但验证收敛均值没有胜过固定AdamW。三个seed和不等调参预算不能支持普遍优越结论。预算和weight decay需组内确认。如果统一设置本来希望长延迟grokking，应先统一协议再新建输出重跑，不能改当前结果定义。

## 9. Supplement: equal-count learning-rate search

补充实验：在首次固定基线比较后，给AdamW也做了seed0三档学习率筛选，和Lion相同次数、每次相同步数上限与停止规则。候选为5e-4、1e-3、2e-3。正式Lion日志原样复用，选中的AdamW在三个新seed比较。应注明这是后补的探索性设计，不能当作事前注册实验。

## 10. Baseline audit: the original default is different

第二轮复核发现，我把注释中的AdamW和衰减1当成基线，但原始入口实际默认是Adam和衰减0。原脚本不改循环跑五万步，训练接近满分，验证仍只有约88%，说明原设置存在明显的泛化延迟。独立保存模型核对、打乱标签和因果性检查未发现泄漏。旧结果只能解释高衰减本地设置，不代表基线已经复现完成。

## 11. Controlled reruns isolate the decay setting

固定seed0、原模型和新入口采样方式，分别只改变衰减，所有运行关闭提前停止并跑一万步。AdamW学习率千分之一，Lion万分之二、衰减五倍，使eta lambda匹配。图中可比较同一算法内部不同衰减，不能挑出Lion占优的设置当新主结果。这轮是诊断，后续多seed确认必须单独设计。

## 12. Original seeds: first crossing versus stability

这页延续原六个正式检查点到一万步，学习率、衰减、随机状态均不变。首次连续三次达到99%的原指标保留，同时展示后续曲线，不能把首次达标等同稳定泛化。预算内稳定起点是后验诊断，不静默替换原主指标；续跑计时分段，不重新宣称wall time优势。

## 13. Backup: likely questions

备用问答：为什么不跑满十万步？共同停止规则是验证99%连续三次达标，所有运行提前成功。为什么原文献要几万步？配置、正则、划分和任务条件可改变动力学，本次只报告统一配置的实际现象。如何验证公平？同seed初始权重和数据划分哈希、同GPU顺序执行、同精度评估和停止规则。为什么较短gap不等于较快？gap从训练拟合时刻起算，Lion训练拟合较晚。
