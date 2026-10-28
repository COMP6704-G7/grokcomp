"""Create a short Chinese diagnosis note from the completed diagnostic artifacts."""
import csv
import json
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/diagnostics"
OUT = ROOT / "deliverables"
with (SOURCE / "decay_comparison.csv").open() as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 8:
    raise ValueError("Wait for all eight diagnostics before issuing the note")
original = json.loads((SOURCE / "original_reproduction_summary.json").read_text())
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21), Cm(29.7)
section.top_margin = section.bottom_margin = Cm(1.7)
section.left_margin = section.right_margin = Cm(1.8)
for name in ["Normal", "Title", "Heading 1", "Heading 2"]:
    style = doc.styles[name]
    style.font.name = "Noto Sans CJK SC"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
doc.styles["Normal"].font.size = Pt(10.5)
doc.styles["Normal"].paragraph_format.space_after = Pt(6)
doc.styles["Normal"].paragraph_format.line_spacing = 1.12
for name in ["Title", "Heading 1", "Heading 2"]:
    doc.styles[name].font.color.rgb = RGBColor.from_string("143342")
doc.styles["Heading 1"].font.size = Pt(16)
doc.styles["Heading 2"].font.size = Pt(12)

doc.add_heading("Lion 实验：问题定位与第二轮重跑", 0)
doc.add_paragraph("COMP6704 Group 7 · 2026 年 10 月 27 日")
doc.add_heading("结论状态", 1)
doc.add_paragraph("上一轮完成了实现、初版运行和材料，但过早把它称为全部完成。"
    "基线配置未统一，而且提前停止没有检查后续稳定性。旧数据仍保留，但结论只对那套本地高衰减配置有效。"
    "本轮已完成原始脚本 50000 步复现、八组固定10000步的衰减控制实验，以及原六个正式检查点的稳定性续跑。")
doc.add_heading("问题一：把代码注释当成了实际默认配置", 2)
doc.add_paragraph("原仓库入口真正的默认值是 Adam、weight decay=0；注释中另有 AdamW、weight decay=1 的描述。"
    "上一轮采用后者，并给 Lion 设为 5。正则化设置发生了明显变化，不能直接用它判断原始基线是否复现。"
    "控制实验保持同一算法内部其他设置不变，显示衰减强度会显著改变验证准确率的上升速度。")
doc.add_heading("问题二：首次连续达标不等于持续稳定", 2)
doc.add_paragraph("固定跑满 10000 步后，AdamW（衰减1）在第1300步开始连续达标、1500步确认，"
    "但之后验证准确率最低回落到51.31%。原来的停止规则符合计划字面要求，却掩盖了后续回落。"
    "因此新增达标后最低准确率和预算内持续99%的回顾性指标，不偷偷改写原有阈值定义。")
doc.add_heading("未发现的数据与实现问题", 2)
doc.add_paragraph("六个保存模型独立复算后与原日志一致；训练和验证索引不重叠；"
    "打乱验证标签后的准确率约0.67%–1.31%；改变未来答案token不会影响前一个位置的预测。"
    "Lion核心更新仍与官方实现逐步一致，11项原有检查通过。另修正了旧入口只指定Lion时仍继承β₂=0.98的问题，"
    "现在默认选0.99；此前新实验入口一直使用0.99，已有结果不受此修正影响。")
doc.add_heading("原始默认脚本重跑", 2)
doc.add_paragraph(f"保持原始训练循环，seed0，Adam/衰减0，运行{original['steps']}步；最终训练准确率"
    f"{original['train_acc']:.2%}，验证准确率{original['val_acc']:.2%}，未达95%。"
    "这显示长期泛化延迟，但没有完整复现README所报的95%达标步数；该差异明确保留为未解决的复现限制。")
doc.add_picture(str(SOURCE / "original_reproduction.png"), width=Cm(13.6))
doc.paragraphs[-1].alignment = 1

heading = doc.add_heading("控制变量重跑结果", 1)
heading.paragraph_format.page_break_before = True
doc.add_paragraph("固定seed0、原模型和划分，关闭提前停止。AdamW学习率固定1e-3；Lion固定2e-4，"
    "衰减取AdamW的5倍，使两方法峰值ηλ匹配。每个算法内部仅改变衰减。未达标用“>10000”标记。")
table = doc.add_table(rows=1, cols=5)
table.style = "Light Shading Accent 1"
for cell, value in zip(table.rows[0].cells, ["方法", "衰减", "首次val99", "最终val", "达标后最低val"]):
    cell.text = value
for row in rows:
    vals = [row["optimizer"], row["decay"], row["val99"] or ">10000",
        f"{float(row['final_val_acc']):.2%}",
        f"{float(row['min_val_after_confirmed99']):.2%}" if row['min_val_after_confirmed99'] else "—"]
    for cell, value in zip(table.add_row().cells, vals):
        cell.text = value
doc.add_picture(str(SOURCE / "decay_diagnostic.png"), width=Cm(16.9))
doc.add_paragraph("这些是单seed诊断，不替代三个新seed的正式确认，也不能为了让Lion赢而挑一档衰减。"
    "下一版正式比较需要明确采用哪个基线设置，以及评价首次达标还是持续稳定。旧报告已增加本轮诊断及初步状态说明。")
doc.add_paragraph("详细日志、独立核查、CSV和图表：results/diagnostics/。复现入口："
    "scripts/reproduce_original.py、scripts/audit_training.py、scripts/decay_diagnostics.py、scripts/analyze_diagnostics.py。")
stability = SOURCE / "formal_stability"
if (stability / "comparison.csv").exists():
    with (stability / "comparison.csv").open() as stream:
        continued = list(csv.DictReader(stream))
    heading = doc.add_heading("原三个seed的六个检查点续跑", 1)
    heading.paragraph_format.page_break_before = True
    doc.add_paragraph("为核验初版比较，原有六个检查点恢复完整模型、优化器、调度器、随机数和取样状态，"
        "继续到统一10000步；没有重新选择学习率或衰减。初版日志前缀逐行一致。这里分别列出首次达标和观察期内持续稳定，"
        "不把后验稳定性指标偷偷替换为原始主指标。")
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Shading Accent 1"
    for cell, text in zip(table.rows[0].cells, ["方法", "seed", "首次val99", "后续最低val", "持续99%起点"]):
        cell.text = text
    for row in continued:
        values = [row['optimizer'], row['seed'], row['first_val99'],
            f"{float(row['min_val_after_confirmed99']):.2%}", row['stable99_within_10000'] or "—"]
        for cell, text in zip(table.add_row().cells, values):
            cell.text = text
    doc.add_picture(str(stability / "stability.png"), width=Cm(16.9))
    doc.add_paragraph("“持续99%起点”仅表示从该点到10000步所有已观察的准确率均≥99%，且至少有三个观测；"
        "不保证以后不回落。原来的首次达标统计仍然成立，但对稳定收敛的解释必须结合后续曲线。"
        "续跑分为两个计算时段，这里不重新比较wall time；补充调参后选出的AdamW/2e-3那组三seed未在本项中延长。")
    doc.add_heading("修订后的结论", 2)
    doc.add_paragraph("首次连续达到99%的均值仍为Lion 1566.7步、AdamW 1366.7步；"
        "但从此保持到10000步的回顾性起点为Lion 1566.7步、AdamW 7100步。"
        "Lion的三个seed确认后均没有跌破99%，AdamW三个seed都曾跌破95%。"
        "因此，初版简单地说Lion没有优势过于笼统：它在这套配置下的后续稳定性更好。"
        "这仍不是普适优越性或已完成组内最终协议的证明。")
doc.save(OUT / "Lion_诊断与重跑说明.docx")
print('Chinese diagnosis note saved.')
