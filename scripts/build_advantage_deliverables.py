"""Build a focused advantage addendum and slides from actual new-seed results."""
import csv
import json
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt as DocPt, RGBColor as DocRGB
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"results/advantage_validation"
ANALYSIS=STUDY/"analysis"
OUT=ROOT/"deliverables"
OUT.mkdir(exist_ok=True)
stats={r["optimizer"]:r for r in json.loads((ANALYSIS/"comparison.json").read_text())}
frozen=json.loads((STUDY/"frozen_selection.json").read_text())
locked=json.loads((STUDY/"design_locked.json").read_text())
conclusion=json.loads((ANALYSIS/"conclusion.json").read_text())
with (ANALYSIS/"per_run.csv").open() as stream: per_run=list(csv.DictReader(stream))
paired=json.loads((ANALYSIS/"paired_differences.json").read_text())


def metric(opt,key,digits=1,scale=1):
    s=stats[opt][key]
    if s["mean"] is None: return "— (n=0)"
    result=f"{scale*s['mean']:.{digits}f}"
    if s["std"] is not None: result+=f" ± {scale*s['std']:.{digits}f}"
    return result+f" (n={s['n']})"


def outcome():
    if conclusion["lion_primary_advantage_observed"]:
        return "三个新seed在预先固定的95%验证达标指标上观察到Lion优势。"
    return "三个新seed未在预先固定的95%验证达标指标上确认Lion优势；其他收益需单独说明。"


delta=100*conclusion["tail_mean_accuracy_difference"]
summary=(outcome()+f"末2000步平均验证准确率，Lion与AdamW的均值分别为"
    f"{100*stats['lion']['tail_mean_val_acc']['mean']:.2f}%和{100*stats['adamw']['tail_mean_val_acc']['mean']:.2f}%，"
    f"差值为{delta:+.2f}个百分点。优化器持久状态分别约1.61 MiB和3.23 MiB。")
scope=("该优势验证针对此前探索发现的低衰减场景。所有新seed运行前已锁定选择规则与预算；"
    "seed7只用于筛选，seed4/5/6只用于确认，没有根据确认结果重选参数。"
    "三个seed不足以支持普适或统计显著性结论，结果应表述为该任务、模型和协议下的实测优势。"
    "选中学习率在本轮候选边界，不能声称覆盖任意超参数或找到全局最优。")
protocol=("mod97模乘，9312条等式、50%训练/50%验证；沿用组库两层、128宽、4头Transformer。"
    "同seed相同初始权重、划分和取样；batch512并保留原末尾48条，FP32。"
    "每100步在同一模型上完整评估训练和验证集；连续三次达到95%才计为主指标成功，记录起点。"
    "所有筛选和确认运行固定10000步，不在首次达标时停止。")
selection=(f"两方法各筛选三档学习率，选择规则完全相同。每次固定ηλ=1e-5，λ=1e-5/η。"
    f"最终Lion η={frozen['selected']['lion']['lr']:.0e}、λ={frozen['selected']['lion']['weight_decay']:.6g}；"
    f"AdamW η={frozen['selected']['adamw']['lr']:.0e}、λ={frozen['selected']['adamw']['weight_decay']:.6g}。"
    "学习率和衰减沿等收缩强度路径共同选择，未声称分别优化了任意超参数的全局最优。")

doc=Document(); sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin=sec.bottom_margin=Cm(1.65)
sec.left_margin=sec.right_margin=Cm(1.8)
for name in ["Normal","Title","Heading 1","Heading 2"]:
    style=doc.styles[name]; style.font.name="Noto Sans CJK SC"
    style.element.rPr.rFonts.set(qn("w:eastAsia"),"Noto Sans CJK SC")
doc.styles["Normal"].font.size=DocPt(10.5)
doc.styles["Normal"].paragraph_format.space_after=DocPt(6)
doc.styles["Normal"].paragraph_format.line_spacing=1.1
for name in ["Title","Heading 1","Heading 2"]: doc.styles[name].font.color.rgb=DocRGB.from_string("143342")
doc.styles["Heading 1"].font.size=DocPt(16)
doc.styles["Heading 2"].font.size=DocPt(12)


def table(headers,values):
    t=doc.add_table(rows=1,cols=len(headers));t.style="Light Shading Accent 1"
    for c,v in zip(t.rows[0].cells,headers):c.text=v
    for row in values:
        for c,v in zip(t.add_row().cells,row):c.text=str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:run.font.size=DocPt(9)


doc.add_heading("Lion 优势验证：独立新seed确认",0)
doc.add_paragraph("COMP6704 Group 7 · 2026年10月27日 · 低衰减场景")
doc.add_heading("实测结论",1);doc.add_paragraph(summary)
table(["预定指标","Lion","AdamW"],[
    ["val95达标数/3（主指标）",f"{stats['lion']['val95_successes']}/3",f"{stats['adamw']['val95_successes']}/3"],
    ["Tval95 / 步（仅成功运行）",metric('lion','Tval95'),metric('adamw','Tval95')],
    ["val99达标数/3",f"{stats['lion']['val99_successes']}/3",f"{stats['adamw']['val99_successes']}/3"],
    ["末2000步验证准确率 / %",metric('lion','tail_mean_val_acc',2,100),metric('adamw','tail_mean_val_acc',2,100)],
    ["最终验证准确率 / %",metric('lion','final_val_acc',2,100),metric('adamw','final_val_acc',2,100)],
    ["持久优化器状态 / MiB",f"{stats['lion']['optimizer_state_bytes']['mean']/2**20:.2f}",f"{stats['adamw']['optimizer_state_bytes']['mean']/2**20:.2f}"]])
doc.add_paragraph("均值±样本标准差；n为该列有效运行数。未达标为右删失，不能用10000步冒充成功达标。"
    "双方成功数不同时，不把仅成功样本的均值直接换算为总体加速倍数。")
doc.add_heading("为什么这个比较能说明问题",2);doc.add_paragraph(protocol);doc.add_paragraph(selection)
doc.add_paragraph(scope)
heading=doc.add_heading("三个独立seed的完整记录",1);heading.paragraph_format.page_break_before=True
table(["方法","seed","Tval95","Tval99","末窗口准确率"],[
    [r['optimizer'],r['seed'],r['Tval95'] or '>10000',r['Tval99'] or '>10000',f"{100*float(r['tail_mean_val_acc']):.2f}%"]
    for r in per_run])
doc.add_picture(str(ANALYSIS/"accuracy_steps.png"),width=Cm(16.9))
doc.add_paragraph("实线为完整验证集准确率，虚线为完整训练集准确率，每种颜色含三个新seed。"
    "横线为95%阈值；曲线均显示完整10000步，保留未达标和回落。")
doc.add_heading("配对差异",2)
for r in paired:
    doc.add_paragraph(f"seed{r['seed']}：末窗口准确率差（Lion−AdamW）"
        f"{100*r['tail_accuracy_lion_minus_adamw']:+.2f}个百分点。")
heading=doc.add_heading("筛选、时间与复现",1);heading.paragraph_format.page_break_before=True
doc.add_picture(str(ANALYSIS/"lr_sensitivity.png"),width=Cm(16.9))
doc.add_paragraph("筛选只使用seed7；每种方法三个候选、相同步数上限和选择规则。")
doc.add_picture(str(ANALYSIS/"validation_time.png"),width=Cm(16.9))
doc.add_paragraph("时间来自同一A100、串行FP32运行，包含评估与运行内日志开销。"
    "内存收益仅指优化器状态，不能写成整个训练显存减半。")
doc.add_paragraph("复现：python scripts/confirm_advantage.py；python scripts/analyze_advantage.py；"
    "python scripts/build_advantage_deliverables.py。原始设计、时间戳、冻结参数和全部日志在results/advantage_validation/。")
doc.save(OUT/"Lion_优势验证说明.docx")

prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)


def text(s,x,y,w,h,lines,size=23,color="203440",bold=False):
    shape=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
    for i,line in enumerate(lines if isinstance(lines,list) else [lines]):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.space_after=Pt(17)
        for r in p.runs:r.font.name="DejaVu Sans";r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)


def slide(title,note):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string("FAFBFC")
    bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.12));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string("176B87");bar.line.fill.background()
    text(s,.45,.35,12.3,.7,title,30,"143342",True)
    text(s,.55,7.05,11.5,.3,"COMP6704 · Group 7 · fresh-seed low-decay validation",10,"526671")
    text(s,12.3,7.05,.5,.3,str(len(prs.slides)),10,"526671")
    s.notes_slide.notes_text_frame.text=note
    return s


s=slide("Lion: fresh-seed advantage validation",summary+scope)
text(s,.75,1.5,11.8,1.0,
    f"val95 success: Lion {stats['lion']['val95_successes']}/3 · AdamW {stats['adamw']['val95_successes']}/3",31,"176B87",True)
text(s,.75,3.0,11.8,3.0,[
    f"Tail-window validation accuracy: {100*stats['lion']['tail_mean_val_acc']['mean']:.1f}% vs {100*stats['adamw']['tail_mean_val_acc']['mean']:.1f}%",
    f"Paired mean difference: {delta:+.1f} percentage points",
    "Persistent optimizer state: 1.61 vs 3.23 MiB",
    "Observed in this regime; three fresh seeds, no universal ranking"],25)
s=slide("Lock the comparison before confirmation",protocol+selection)
text(s,.75,1.4,11.9,5.2,[
    "Tune on seed 7; confirm on unseen seeds 4, 5 and 6",
    "Three learning-rate candidates per optimizer · 10k steps per trial",
    "Same per-step shrinkage: learning rate × decay = 1e−5",
    "Original group model, paired initialization, split and sampler",
    "Primary: 3-checkpoint validation ≥95%; no early stopping",
    "Custom Lion update; PyTorch AdamW baseline"],24)
s=slide("Learning-rate selection uses seed 7 only",selection)
s.shapes.add_picture(str(ANALYSIS/"lr_sensitivity.png"),Inches(.45),Inches(1.35),width=Inches(12.4))
text(s,.75,6.5,11.8,.4,
    f"Selected: Lion lr={frozen['selected']['lion']['lr']:.0e}; AdamW lr={frozen['selected']['adamw']['lr']:.0e}",18,"526671")
s=slide("Generalization on the three fresh seeds",summary)
s.shapes.add_picture(str(ANALYSIS/"accuracy_steps.png"),Inches(.45),Inches(1.3),width=Inches(9.0))
text(s,9.65,1.8,3.0,4.8,[
    "Full 10k horizon", "Same model/split", "No hidden failed runs", "", "Solid: validation", "Dashed: train"],20,"176B87")
s=slide("Accuracy differences and optimizer-state storage",summary)
s.shapes.add_picture(str(ANALYSIS/"paired_accuracy_memory.png"),Inches(.45),Inches(1.35),width=Inches(12.4))
s=slide("What this result supports",scope)
text(s,.75,1.4,11.9,5.1,[
    "The primary advantage is observed on fresh seeds" if conclusion['lion_primary_advantage_observed'] else
    "The prespecified primary advantage was not confirmed",
    "Equal candidate count, horizon, selection rule and shrinkage strength",
    "Generalization dynamics depend on the regularization regime",
    "Half the moment storage; total training memory remains separate",
    "Report the chosen regime and all seeds; avoid universal claims"],24)
prs.save(OUT/"Lion_优势验证展示.pptx")
notes=["# Lion优势验证：中文讲稿", "", summary, ""]
for i,s in enumerate(prs.slides,1):notes += [f"## {i}. {s.shapes[1].text}","",s.notes_slide.notes_text_frame.text,""]
(OUT/"Lion_优势验证讲稿.md").write_text("\n".join(notes))
section=("## Lion under matched low per-step shrinkage\n\n"
    f"A protocol locked before confirmation compared custom Lion with PyTorch AdamW on three fresh seeds (4–6). "
    "Each optimizer received three 10,000-step learning-rate trials on seed 7. All candidates used identical "
    "per-step shrinkage ηλ=1e−5, paired group-model initialization and splits, and full-set evaluation every "
    "100 steps. Configuration selection prioritized the first three-checkpoint 95% validation crossing, with "
    "identical tie breakers. No early stopping was used.\n\n"
    f"Lion reached 95% validation accuracy in {stats['lion']['val95_successes']}/3 confirmation runs, compared "
    f"with {stats['adamw']['val95_successes']}/3 for AdamW. Mean validation accuracy over steps 8100–10000 "
    f"was {100*stats['lion']['tail_mean_val_acc']['mean']:.2f}% versus "
    f"{100*stats['adamw']['tail_mean_val_acc']['mean']:.2f}%, a difference of {delta:+.2f} percentage points. "
    "Persistent optimizer state was approximately 1.61 versus 3.23 MiB. Failed threshold crossings are "
    "right-censored and excluded from successful-crossing means.\n\n"
    "The low-shrinkage regime was identified in earlier exploratory work; these results confirm behavior "
    "within that regime on unused seeds, not universal superiority or a statistically significant effect. "
    "No confirmation outcomes were used to revise the locked configuration.\n")
(OUT/"Lion_Advantage_Report_Section.md").write_text(section)
# Keep the newest result easy to find without erasing the prior studies.
readme=ROOT/"README_LION.md"
begin,end="<!-- ADVANTAGE_VALIDATION_START -->","<!-- ADVANTAGE_VALIDATION_END -->"
new=(f"{begin}\n## 最新：独立seed优势验证\n\n"+summary+"\n\n"
    "本轮双方各3次seed7筛选，冻结后用新seed4/5/6确认；所有运行固定10000步、ηλ相同。"
    "见[新seed结果与完整图表](results/advantage_validation/analysis/RESULTS.md)、"
    "[优势验证PDF](deliverables/Lion_优势验证说明.pdf)、"
    "[优势验证展示](deliverables/Lion_优势验证展示.pptx)、"
    "[验证协议](docs/ADVANTAGE_VALIDATION_PROTOCOL.md)。\n\n"+scope+f"\n{end}\n\n")
content=readme.read_text()
if begin in content:
    start=content.index(begin);stop=content.index(end,start)+len(end)
    content=content[:start]+new+content[stop:].lstrip('\n')
else:
    index=content.index("## ")
    content=content[:index]+new+content[index:]
readme.write_text(content)
print('Focused advantage report, six-slide presentation and report section saved.')
