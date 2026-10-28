"""Build editable report and presentation directly from completed run artifacts."""
import csv
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt, RGBColor as DocRGB
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/lion_study"
ANALYSIS = STUDY / "analysis"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)
stats = {row["optimizer"]: row for row in json.loads((ANALYSIS / "comparison.json").read_text())}
frozen = json.loads((STUDY / "frozen_selection.json").read_text())
SUPPLEMENT = ROOT / "results/matched_budget_study"
supplement = None
if (SUPPLEMENT / "analysis/comparison.json").exists():
    supplement = {row["optimizer"]: row for row in json.loads(
        (SUPPLEMENT / "analysis/comparison.json").read_text())}
    supplement_selection = json.loads((SUPPLEMENT / "frozen_selection.json").read_text())
config = frozen["protocol"]
DIAGNOSTICS = ROOT / "results/diagnostics"
diagnostic_rows = None
if (DIAGNOSTICS / "decay_comparison.csv").exists():
    with (DIAGNOSTICS / "decay_comparison.csv").open() as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    original_reproduction = json.loads((DIAGNOSTICS / "original_reproduction_summary.json").read_text())
STABILITY = DIAGNOSTICS / "formal_stability"
stability_rows = None
if (STABILITY / "comparison.csv").exists():
    with (STABILITY / "comparison.csv").open() as stream:
        stability_rows = list(csv.DictReader(stream))
runs = [json.loads(p.read_text()) for p in sorted((STUDY / "formal").glob("*/summary.json"))]
if len(runs) != 6 or any(r["segments"] != 1 for r in runs):
    raise ValueError("Deliverables require all six uninterrupted formal runs")


def stat(opt, key, digits=1):
    s = stats[opt][key]
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


selected = frozen["selected_lr"]
intro = (
    f"We implemented Lion and evaluated it against a fixed AdamW baseline on mod-97 multiplication, "
    f"using the group's two-layer Transformer. Three learning-rate trials on seed 0 selected Lion lr={selected:.0e}. "
    "The selected configuration was evaluated on three new, paired seeds (1, 2, 3). Both methods reached "
    "99% validation accuracy in all formal runs. Lion reduced the measured memorization-to-generalization "
    "gap and optimizer-state memory, but did not improve mean steps or mean wall time to the first "
    "three-checkpoint 99% validation crossing."
)
interpretation = (
    f"Lion reached train99 at {stat('lion', 'train99_step')} steps, compared with "
    f"{stat('adamw', 'train99_step')} for AdamW. Its smaller delay therefore partly reflects later "
    "memorization. The paired Lion-minus-AdamW differences in val99 steps were −300, +400 and +500 "
    "for seeds 1, 2 and 3. Lion won on one of three seeds and required 200 more steps on average. "
    "The results do not establish a general convergence advantage for Lion."
)
limitations = [
    "Three formal seeds provide limited precision; mean ± sample SD is descriptive, not a significance claim.",
    "Lion received three single-seed learning-rate trials. AdamW used a fixed reference setting; this is not an equal-search-budget comparison of tuned optima.",
    "Lion lr=3e−4 and weight decay=5 give ηλ=0.0015; AdamW gives 0.001. Learning rate and effective shrinkage both vary in the sensitivity study.",
    "The budget and weight decays are local frozen starting values from the plan and still require group confirmation before replacing a shared baseline.",
    "All initial runs reached the stopping rule early. This high-decay setting does not show a long grokking plateau; the later fixed-horizon follow-up separately checks post-stop stability.",
    "The group architecture and split RNG are preserved. The new independent batch sampler and checkpoint evaluation differ from the original epoch-logging script; old logs cannot be compared directly.",
    "Step and wall times are short-run measurements on one A100 with synchronized FP32 execution. They do not isolate optimizer-kernel latency or establish a hardware-independent speedup."
]
resources = (
    f"Mean training-step time was {stat('lion', 'mean_step_ms', 2)} ms for Lion and "
    f"{stat('adamw', 'mean_step_ms', 2)} ms for AdamW. This includes sampling, forward/backward passes, "
    "the optimizer, scheduler and synchronization, but excludes full-set evaluation. Peak allocated GPU memory "
    f"averaged {stats['lion']['peak_memory_bytes']['mean']/2**20:.2f} MiB and "
    f"{stats['adamw']['peak_memory_bytes']['mean']/2**20:.2f} MiB respectively. "
    "Persistent optimizer state was 1.61 MiB versus 3.23 MiB. Halving moment storage reduced total peak "
    "allocation by only about 1.2%, because model, gradients, activations and temporary tensors still dominate."
)
if stability_rows:
    stability_stats = {row["optimizer"]: row for row in json.loads((STABILITY / "comparison.json").read_text())}
    intro += (" Continuing the original six checkpoints to 10000 steps found that all three Lion runs "
              "stayed above 99% after confirmation, whereas all three AdamW runs later fell below 95%. "
              "The retrospective within-horizon stable99 starting point averaged 1566.7 steps for Lion "
              "and 7100 steps for AdamW. Thus first crossing and observed sustained generalization give "
              "different rankings; the latter is a follow-up diagnostic, not a replacement primary endpoint.")

report = Document()
section = report.sections[0]
section.page_width, section.page_height = DocInches(8.27), DocInches(11.69)
section.top_margin = section.bottom_margin = DocInches(.65)
section.left_margin = section.right_margin = DocInches(.75)
normal = report.styles["Normal"]
normal.font.name = "DejaVu Sans"
normal.font.size = DocPt(10.5)
normal.paragraph_format.space_after = DocPt(6)
normal.paragraph_format.line_spacing = 1.12
for name in ["Title", "Heading 1", "Heading 2"]:
    report.styles[name].font.name = "DejaVu Sans"
    report.styles[name].font.color.rgb = DocRGB.from_string("143342")
report.styles["Heading 1"].font.size = DocPt(17)
report.styles["Heading 2"].font.size = DocPt(12)
header = section.header.paragraphs[0]
header.text = "COMP6704 · GROUP 7                                      LION OPTIMIZER STUDY"
header.runs[0].font.size = DocPt(8)
footer = section.footer.paragraphs[0]
footer.text = "Local experiment · 27 October 2026                                           "
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def paragraph(text, style=None):
    return report.add_paragraph(text, style)


def table(headers, values):
    tbl = report.add_table(rows=1, cols=len(headers))
    tbl.style = "Light Shading Accent 1"
    for cell, text in zip(tbl.rows[0].cells, headers):
        cell.text = text
    for values_row in values:
        for cell, text in zip(tbl.add_row().cells, values_row):
            cell.text = str(text)
    for row in tbl.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                for run in par.runs:
                    run.font.size = DocPt(9)


report.add_heading("Lion on Modular Multiplication", 0)
paragraph("Algorithm, reproducible experiments and comparison with AdamW")
paragraph("HUANG Yuqi · COMP6704 Group 7 · 27 October 2026")
if diagnostic_rows:
    paragraph("STATUS: PRELIMINARY CONFIGURATION STUDY, WITH SECOND-ROUND BASELINE DIAGNOSIS. "
              "The original CLI defaults to Adam / weight decay 0. Sections 1–9 describe local "
              "AdamW / weight decay 1 and Lion / weight decay 5 experiments; they do not reproduce "
              "that default baseline. Section 10 investigates this mismatch. These materials are "
              "not a final, group-confirmed long-grokking comparison.")
report.add_heading("1. Main finding", 1)
paragraph(intro)
report.add_heading("2. Algorithm and implementation", 1)
paragraph("For parameter θ, gradient g and previous momentum m (initially zero), Lion performs:")
paragraph("u = sign(β₁m + (1 − β₁)g)\nθ ← (1 − ηλ)θ − ηu\nm ← β₂m + (1 − β₂)g")
paragraph("The direction uses the old momentum; sign(0)=0. Weight decay is decoupled from the gradient. "
          "The implementation keeps one momentum tensor, supports parameter groups, closures and checkpoint "
          "restoration, and skips parameters without gradients. Lion uses β=(0.9,0.99); the fixed AdamW "
          "baseline uses the group reference β=(0.9,0.98), ε=1e−8, foreach=False.")
paragraph("Both optimizers have O(P) update complexity for P parameters. Lion stores about 4P bytes of FP32 "
          "momentum, versus about 8P for AdamW's two moments, plus AdamW step scalars. This concerns "
          "persistent optimizer state rather than the entire training memory footprint.")
paragraph("Correctness: 40-step, identical-gradient comparisons against Google's downloaded official implementation "
          "matched parameters and momentum exactly (zero tolerance) in FP32 and FP64, including zero/missing "
          "gradients and parameter groups. Eleven tests also check original-model and split equivalence, "
          "consecutive thresholds and interrupted-run restoration.")

report.add_page_break()
report.add_heading("3. Experimental protocol", 1)
table(["Item", "Fixed setting"], [
    ["Task / data", "x·y mod 97; 0≤x<97, 1≤y<97; 9312 equations"],
    ["Split", "4656 train / 4656 validation; paired within each seed"],
    ["Model", "Original group decoder; 2 layers, width 128, 4 heads; 422784 parameters"],
    ["Input / loss", "[x, op, y, eq]; next answer-token cross-entropy over original 99-class head"],
    ["Batch / precision", "512, final batch 48; FP32; TF32 disabled"],
    ["Schedule", "Original 10-step linear warmup; first update lr=0, then constant"],
    ["Evaluation", "Full train and validation at same weights every 100 optimizer steps"],
    ["Thresholds", "train99, val95, val99; each requires 3 consecutive evaluations"],
    ["Stop / budget", "Stop on confirmed val99 streak, else 100000 steps"],
    ["Hardware", "One NVIDIA A100-SXM4-80GB; sequential runs; PyTorch 2.5.1+cu124"],
    ["Baseline", "AdamW lr=1e−3, weight decay=1"],
    ["Lion", "Selected lr=3e−4, weight decay=5"]])
paragraph("The group model is initialized before randperm, preserving the original seed-to-model and seed-to-split "
          "mapping. Same-seed comparisons have matching initial-weight and split SHA-256 values. Batch shuffling "
          "uses an independent generator shared by paired methods. No Grokfast filter is used.")
paragraph("For each threshold, T denotes the first checkpoint in the first confirmed three-checkpoint streak. "
          "Confirmation and stopping occur 200 steps later. ΔT=Tval99−Ttrain99 is kept signed. A failed run "
          "would be reported as >100000 and excluded from successful-time means. All six formal runs were "
          "uninterrupted and successful, so no censored values occur in this result table.")
report.add_heading("4. Learning-rate selection", 1)
paragraph("Three seed-0 Lion trials used learning rates 1e−4, 2e−4 and 3e−4 with fixed weight decay 5. "
          "Their Tval99 values were 6200, 4300 and 1600 steps. The prespecified ranking prioritizes val99 "
          "success, then earliest val99, earliest val95, final validation accuracy and lower learning rate. "
          "The selected value was frozen before seeds 1, 2 and 3 were evaluated. ηλ equals 0.0005, 0.001 "
          "and 0.0015 across the candidates; this sensitivity plot also changes effective shrinkage.")

report.add_page_break()
report.add_heading("5. Formal results", 1)
table(["Metric", "Lion", "AdamW"], [
    ["val95 / val99 successes", "3/3 · 3/3", "3/3 · 3/3"],
    ["Ttrain99 (steps)", stat("lion", "train99_step"), stat("adamw", "train99_step")],
    ["Tval95 (steps)", stat("lion", "val95_step"), stat("adamw", "val95_step")],
    ["Tval99 (steps)", stat("lion", "val99_step"), stat("adamw", "val99_step")],
    ["ΔT (steps)", stat("lion", "generalization_delay"), stat("adamw", "generalization_delay")],
    ["Wall time to val99 (s)", stat("lion", "val99_wall_seconds", 2), stat("adamw", "val99_wall_seconds", 2)],
    ["Training step time (ms)", stat("lion", "mean_step_ms", 2), stat("adamw", "mean_step_ms", 2)]])
paragraph("Values are mean ± sample standard deviation (ddof=1), n=3. Wall time includes full-set evaluation, "
          "logging and runtime checkpoint overhead after initialization. The threshold time is the first "
          "checkpoint's time, not the later confirmation time. Step-zero evaluation is diagnostic only.")
table(["Seed", "Lion Tval99", "AdamW Tval99", "Difference"], [
    ["1", "1200", "1500", "−300"], ["2", "1800", "1400", "+400"], ["3", "1700", "1200", "+500"]])
report.add_heading("Interpretation", 2)
paragraph(interpretation)
paragraph("At the primary 95% validation threshold, Lion also required slightly more steps on average "
          "(1433 versus 1333). The gap is noisy across three seeds. Shorter generalization delay should "
          "therefore be reported together with absolute validation threshold steps and times.")
report.add_heading("Computation and storage", 2)
paragraph(resources)

report.add_page_break()
report.add_heading("6. Training curves", 1)
report.add_picture(str(ANALYSIS / "accuracy_steps.png"), width=DocInches(6.4))
paragraph("Figure 1. Full-set train (dashed) and validation (solid) accuracy for all three formal seeds. "
          "Colors identify methods; curves end at their own common-rule stopping point.")
report.add_picture(str(ANALYSIS / "validation_time.png"), width=DocInches(6.4))
paragraph("Figure 2. Validation accuracy against measured wall time, including checkpoint evaluation. "
          "Same hardware and precision; one run at a time.")

report.add_page_break()
report.add_heading("7. Sensitivity and scope", 1)
report.add_picture(str(ANALYSIS / "lr_sensitivity.png"), width=DocInches(6.7))
paragraph("Figure 3. Single-seed exploratory learning-rate sweep, with weight decay fixed at 5. "
          "This figure does not estimate across-seed uncertainty or isolate learning rate from ηλ.")
for text in limitations:
    paragraph(text, "List Bullet")
report.add_heading("8. Reproduction and references", 1)
paragraph("From the project directory: python -m pytest -q; then CUDA_VISIBLE_DEVICES=0 python -m "
          "lion_experiment.suite --stage all --resume. The suite writes raw logs, configuration and source "
          "hashes, split indices, checkpoints, frozen selection and analysis outputs. "
          "See README_LION.md and docs/PROTOCOL.md for the exact protocol.")
paragraph("[1] Chen et al. Symbolic Discovery of Optimization Algorithms (2023). https://arxiv.org/abs/2302.06675\n"
          "[2] Google AutoML Lion implementation and tuning guidance. https://github.com/google/automl/tree/master/lion\n"
          "[3] Lee et al. Grokfast: Accelerated Grokking by Amplifying Slow Gradients (2024). https://arxiv.org/abs/2405.20233\n"
          "[4] Group repository (private): https://github.com/COMP6704-G7/grokcomp — source revision 93b2248.")
if supplement:
    supplemental_heading = report.add_heading("9. Supplementary equal-count baseline search", 1)
    supplemental_heading.paragraph_format.page_break_before = True
    paragraph("After inspecting the initial fixed-baseline comparison, we added an exploratory AdamW "
              "learning-rate search to address the plan's matching tuning-budget requirement. AdamW received "
              "three seed-0 trials (5e−4, 1e−3, 2e−3), each with the same 100000-step cap, early-stop rule "
              "and selection ranking as Lion. Both algorithms therefore received three trials; actual GPU "
              "time differs because successful trials stop early. Lion's original tuning and formal runs "
              "are reused unchanged via relative links with recorded provenance.")
    paragraph(f"The supplementary search selected AdamW lr={supplement_selection['selected_adamw_lr']:.0e}, "
              "with weight decay still fixed at 1. Formal comparison uses seeds 1, 2 and 3; the selection "
              "ranking uses seed 0 only. If the selected AdamW setting matches the initial baseline, its "
              "original formal logs are reused; otherwise the selected baseline is run on all three seeds.")
    def supplement_stat(opt, key):
        row = supplement[opt][key]
        return f"{row['mean']:.1f} ± {row['std']:.1f}"
    table(["Metric", "Lion", "Selected AdamW"], [
        ["Tval95 (steps)", supplement_stat("lion", "val95_step"), supplement_stat("adamw", "val95_step")],
        ["Tval99 (steps)", supplement_stat("lion", "val99_step"), supplement_stat("adamw", "val99_step")],
        ["ΔT (steps)", supplement_stat("lion", "generalization_delay"), supplement_stat("adamw", "generalization_delay")],
        ["Wall time to val99 (s)", supplement_stat("lion", "val99_wall_seconds"), supplement_stat("adamw", "val99_wall_seconds")],
        ["val99 successes", f"{supplement['lion']['val99_successes']}/3", f"{supplement['adamw']['val99_successes']}/3"]])
    paragraph("This supplemental design was introduced after seeing the initial comparison and should be "
              "reported as exploratory, not preregistered confirmation. Equal trial count and caps do not "
              "imply a global optimum or equal actual search compute. Each method uses its own learning-rate "
              "range and fixed weight decay. The original fixed-baseline results in Sections 1–8 remain intact.")
    paragraph("Supplementary logs, frozen selections, result tables and the three updated figures are under "
              "results/matched_budget_study/. Reproduce with: python scripts/matched_budget.py. "
              "The original config's adamw_lr=1e−3 remains the nominal reference; selected_adamw_lr in the "
              "supplement's frozen_selection.json and each run's explicit lr field define the actual baseline.")
if diagnostic_rows:
    heading = report.add_heading("10. Baseline audit and controlled reruns", 1)
    heading.paragraph_format.page_break_before = True
    paragraph("The first round treated the comment describing AdamW / weight decay 1 as a settled "
              "baseline. The actual group CLI defaults are Adam / weight decay 0. This unconfirmed "
              "configuration change was a methodological error in declaring the experiment complete, "
              "rather than evidence of a failed Lion implementation. The original local measurements "
              "remain valid for their own settings and are preserved.")
    paragraph("Independent audit of all six original formal checkpoints reproduced their validation "
              "accuracies. Train and validation indices are disjoint; shuffled labels give about "
              "0.67%–1.31% accuracy. Changing a future answer token changes the preceding prediction "
              "logits by exactly zero. These checks found no evidence of target leakage or logged "
              "training accuracy being substituted for validation accuracy.")
    paragraph(f"The unmodified group script was rerun with its original Adam / zero-decay defaults "
              f"for {original_reproduction['steps']} steps, seed 0. Final online train accuracy was "
              f"{original_reproduction['train_acc']:.2%}; validation accuracy was "
              f"{original_reproduction['val_acc']:.2%}. It had not reached even one 95% validation "
              "checkpoint by that horizon. This demonstrates a long generalization delay, but does "
              "not claim an exact reproduction of the README's reported threshold step.")
    report.add_picture(str(DIAGNOSTICS / "original_reproduction.png"), width=DocInches(6.4))
    paragraph("Figure 4. Exact original training source, with its own epoch-level metrics. Train "
              "accuracy is accumulated during updates and is not a synchronized full-set metric.")
    heading = report.add_heading("Controlled decay diagnostic", 1)
    heading.paragraph_format.page_break_before = True
    paragraph("Within each optimizer, only weight decay changes. The model, seed-0 split, batch "
              "sampler and evaluation rule stay fixed. AdamW lr=1e−3; Lion lr=2e−4 with 5× the "
              "decay, matching peak ηλ between the two methods. All eight runs continue to 10000 "
              "steps without early stopping; this tests stability beyond first confirmation.")
    report.add_picture(str(DIAGNOSTICS / "decay_diagnostic.png"), width=DocInches(6.4))
    table(["Method", "Decay", "Tval99", "Final val"], [
        [r["optimizer"], r["decay"], r["val99"] or ">10000", f"{float(r['final_val_acc']):.2%}"]
        for r in diagnostic_rows])
    paragraph("Interpretation is restricted to these controlled seed-0 runs. Small decay and no "
              "decay must not be pooled with the high-decay trials as one optimizer result. "
              "A 10000-step non-crossing is right-censored, not proof that grokking never occurs. "
              "The first-round report should remain preliminary until the desired baseline regime "
              "is specified and a separate multi-seed confirmation is completed. See "
              "results/diagnostics/DIAGNOSIS.md for raw outcomes and the reproduction commands.")
if stability_rows:
    heading = report.add_heading("11. Original formal seeds: stability continuation", 1)
    heading.paragraph_format.page_break_before = True
    paragraph("The six original fixed-baseline checkpoints were continued to 10000 steps each. "
              "Weights, optimizer, scheduler, RNG and sampler states were restored. Learning rate, "
              "decay and all other model/training settings were unchanged; the original metric "
              "prefix was verified exactly. This extends observation, not tuning against formal seeds.")
    table(["Method", "Seed", "First val99", "Min later val", "Stable99 to 10k"], [
        [r["optimizer"], r["seed"], r["first_val99"],
         f"{float(r['min_val_after_confirmed99']):.2%}", r["stable99_within_10000"] or "—"]
        for r in stability_rows])
    report.add_picture(str(STABILITY / "stability.png"), width=DocInches(6.4))
    paragraph("First val99 retains the original three-consecutive-checkpoint definition. Stable99 "
              "is a retrospective within-horizon diagnostic: all observed checkpoints from that "
              "point to step 10000 exceed 99%, with at least three observations remaining. It "
              "does not replace the prespecified metric and does not guarantee future stability.")
    paragraph("The continuation has two timing segments, so no new wall-time ranking is claimed. "
              "Only the original fixed AdamW lr=1e−3 / decay=1 versus Lion lr=3e−4 / decay=5 "
              "comparison was extended; the supplementary selected AdamW lr=2e−3 runs were not.")
report.save(OUT / "Lion_Report.docx")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
NAVY, TEAL, ORANGE, INK, MUTED = "143342", "176B87", "D86828", "203440", "526671"


def text(slide, x, y, w, h, lines, size=22, color=INK, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(.03)
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.space_after = Pt(15)
        for run in p.runs:
            run.font.name = "DejaVu Sans"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)
    return shape


def slide(title, note):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string("FAFBFC")
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(.12))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = RGBColor.from_string(TEAL)
    stripe.line.fill.background()
    text(s, .45, .35, 12.3, .65, title, 30, NAVY, True)
    text(s, .5, 7.07, 11.5, .24, "COMP6704 · Group 7 · HUANG Yuqi · 27 Oct 2026", 10, MUTED)
    text(s, 12.35, 7.03, .5, .3, str(len(prs.slides)), 11, MUTED)
    s.notes_slide.notes_text_frame.text = note
    return s


s = slide("Lion: first crossing and sustained generalization",
    "开场约30秒：我们实现了Lion，并在小组的模97模乘Transformer任务上完成真实实验。主要发现是：Lion减少了训练拟合后到泛化之间的间隔，也减少了优化器状态内存，但本配置下平均达到99%验证准确率并没有比AdamW更快。")
text(s, .7, 1.5, 11.9, 1.3, "3 new seeds · both methods succeed 3/3", 32, TEAL, True)
text(s, .75, 3.0, 11.4, 2.8, [
    "Lion: less optimizer state and a shorter train→validation gap",
    "AdamW: earlier first crossing; continuation reveals later regressions",
    "Lion remains above 99% after confirmation through step 10k" if stability_rows else
    "Fixed baseline comparison; results are specific to this setting"], 25)
if diagnostic_rows:
    text(s, .75, 6.25, 11.8, .5,
         "PRELIMINARY: original CLI uses Adam / decay 0; baseline audit follows.", 18, ORANGE, True)

s = slide("One shared task, one paired protocol",
    "约40秒：数据共9312条，训练验证各4656条。使用仓库原始两层128宽4头因果Transformer，预测答案token。相同seed两种方法有相同初始权重与数据划分，并保存哈希核对。每100步在同一检查点评估完整训练和验证集，连续三次达标，记录起点，第三次确认后停止。")
text(s, .7, 1.4, 6.2, 4.8, [
    "x·y mod 97; y ≠ 0",
    "9,312 equations; 50% / 50% split",
    "2 layers · width 128 · 4 heads",
    "Batch 512 · FP32 · no Grokfast filter"], 24)
text(s, 7.25, 1.4, 5.3, 4.9, [
    "Same-seed weights and split verified",
    "Full-set evaluation every 100 steps",
    "3-checkpoint threshold confirmation",
    "One A100, sequential runs"], 23, TEAL)
text(s, .75, 6.2, 11.9, .6, "Primary: Tval95   |   Supplemental: Tval99   |   Delay: Tval99 − Ttrain99", 19, MUTED)

s = slide("Lion uses one momentum buffer",
    "约40秒：方向是旧动量和当前梯度的加权和取sign，参数先做解耦衰减，再减去学习率乘方向，最后用beta2更新动量。注意方向使用旧动量。两个算法每步都是O(P)，Lion一个动量，AdamW两个。官方FP32和FP64逐步对照完全一致，总计11项检查通过。")
text(s, .7, 1.45, 12, 2.8, [
    "u = sign(β₁m + (1 − β₁)g)",
    "θ ← (1 − ηλ)θ − ηu",
    "m ← β₂m + (1 − β₂)g"], 29, TEAL)
text(s, .75, 4.4, 11.8, 2.2, [
    "β₁=0.9, β₂=0.99 · O(P) updates · one persistent FP32 momentum",
    "Official reference: exact parameter/momentum match over 40 steps",
    "11 checks cover optimizer, shared model/split, thresholds and resume"], 21)

s = slide("Learning rate frozen before the formal seeds",
    "约35秒：seed0上三档Lion学习率的99%验证达标步数分别为6200、4300、1600，按预设规则选3e-4，再冻结后运行seed1到3。注意这张是单seed探索，而且weight decay固定5，eta lambda也随学习率变化，不能说是隔离学习率作用的消融。AdamW是固定基线，没有等量搜索。")
s.shapes.add_picture(str(ANALYSIS / "lr_sensitivity.png"), Inches(.45), Inches(1.3), width=Inches(12.4))
text(s, .7, 6.6, 12, .4, "Selected lr=3e−4, λ=5; ηλ=0.0015 vs AdamW 0.001. Single-seed exploration.", 17, MUTED)

s = slide("Lion memorizes later; both generalize early",
    "约45秒：虚线是完整训练集准确率，实线是验证准确率，每种颜色有三个seed。AdamW训练99%起点都在500步，Lion约1200步。Lion验证99%平均1567步，AdamW1367步。观察不到很长的几万步grokking平台，因此按当前任务如实报告，不把曲线解释成复现了长延迟grokking。")
s.shapes.add_picture(str(ANALYSIS / "accuracy_steps.png"), Inches(.45), Inches(1.25), width=Inches(9.15))
text(s, 9.9, 1.8, 2.9, 4.6, [
    "Train99", "Lion ≈1,200", "AdamW ≈500", "", "Smaller ΔT partly reflects later fitting"], 20, TEAL)

s = slide("First-crossing wall time slightly favors AdamW",
    "约35秒：横轴是包含评估的真实同步wall time，Lion达到验证99%平均约10.5秒，AdamW9.6秒。单训练步Lion约6.3毫秒，AdamW6.7毫秒，但训练整体还取决于需要多少步，不能把单步小优势等同总训练快。计时来自很短的同卡串行运行，不声称硬件无关的加速。")
s.shapes.add_picture(str(ANALYSIS / "validation_time.png"), Inches(.45), Inches(1.25), width=Inches(9.15))
text(s, 9.9, 1.8, 2.9, 4.6, [
    "Time to val99", f"Lion {stats['lion']['val99_wall_seconds']['mean']:.1f} s",
    f"AdamW {stats['adamw']['val99_wall_seconds']['mean']:.1f} s", "", "Includes evaluation"], 20, TEAL)

s = slide("Three paired seeds: tradeoffs, not a universal win",
    "约45秒：两个方法成功率都是3/3。Lion平均Tval99多200步，泛化间隔约367步而AdamW867步。逐seed看Lion仅一个seed更快。优化器状态1.61对3.23MiB确实接近减半，但总峰值130.61对132.22MiB只降约1.2%。因此要区分算法持久状态和整个训练显存。")
slide_table = s.shapes.add_table(7, 3, Inches(.7), Inches(1.4), Inches(11.9), Inches(3.9)).table
values = [["Metric", "Lion", "AdamW"], ["val99 success", "3/3", "3/3"],
    ["Tval99 (steps)", stat("lion", "val99_step"), stat("adamw", "val99_step")],
    ["Delay ΔT (steps)", stat("lion", "generalization_delay"), stat("adamw", "generalization_delay")],
    ["Step time (ms)", stat("lion", "mean_step_ms", 2), stat("adamw", "mean_step_ms", 2)],
    ["Optimizer state (MiB)", "1.61", "3.23"], ["Total peak allocation (MiB)", "130.61", "132.22"]]
for row, texts in zip(slide_table.rows, values):
    for cell, value in zip(row.cells, texts):
        cell.text = value
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.name = "DejaVu Sans"
                run.font.size = Pt(19)
text(s, .75, 5.7, 11.8, 1.0, ["Mean ± sample SD, n=3; paired val99 differences: −300, +400, +500 steps.",
    "Half the moment storage ≠ half the total training memory."], 20, MUTED)

s = slide("What we can conclude—and what remains open",
    "结尾约35秒：实现已验证、实验可复现；本配置Lion省优化器状态内存且间隔短，但验证收敛均值没有胜过固定AdamW。三个seed和不等调参预算不能支持普遍优越结论。预算和weight decay需组内确认。如果统一设置本来希望长延迟grokking，应先统一协议再新建输出重跑，不能改当前结果定义。")
text(s, .75, 1.4, 11.9, 4.8, [
    "Verified implementation and reproducible logs, figures and checkpoints",
    "First crossing favors AdamW; continuation stability favors Lion" if stability_rows else
    "Shorter delay and less optimizer state; no first-crossing step advantage",
    "Only 3 formal seeds; equal-count search is an exploratory supplement",
    "Local budget / weight decay still need group confirmation",
    "Original zero-decay baseline differs from this high-decay study"], 24)

if supplement:
    s = slide("Supplement: equal-count learning-rate search",
        "补充实验：在首次固定基线比较后，给AdamW也做了seed0三档学习率筛选，和Lion相同次数、每次相同步数上限与停止规则。候选为5e-4、1e-3、2e-3。正式Lion日志原样复用，选中的AdamW在三个新seed比较。应注明这是后补的探索性设计，不能当作事前注册实验。")
    text(s, .75, 1.35, 11.8, 1.2,
         f"3 LR trials per optimizer · selected AdamW lr={supplement_selection['selected_adamw_lr']:.0e}", 25, TEAL, True)
    text(s, .75, 2.8, 11.8, 3.4, [
        f"Mean Tval99: Lion {supplement['lion']['val99_step']['mean']:.0f}; selected AdamW {supplement['adamw']['val99_step']['mean']:.0f} steps",
        f"val99 successes: Lion {supplement['lion']['val99_successes']}/3; selected AdamW {supplement['adamw']['val99_successes']}/3",
        "Same seed-0 ranking, cap and stopping rule; actual search compute varies",
        "Added after the first comparison; supplementary exploratory evidence"], 23)

if diagnostic_rows:
    s = slide("Baseline audit: the original default is different",
        "第二轮复核发现，我把注释中的AdamW和衰减1当成基线，但原始入口实际默认是Adam和衰减0。原脚本不改循环跑五万步，训练接近满分，验证仍只有约88%，说明原设置存在明显的泛化延迟。独立保存模型核对、打乱标签和因果性检查未发现泄漏。旧结果只能解释高衰减本地设置，不代表基线已经复现完成。")
    s.shapes.add_picture(str(DIAGNOSTICS / "original_reproduction.png"), Inches(.45), Inches(1.25), width=Inches(9.15))
    text(s, 9.9, 1.6, 2.9, 5.0, [
        "Original CLI", "Adam / wd=0", "", "Local first round", "AdamW / wd=1", "Lion / wd=5"], 20, TEAL)
    s = slide("Controlled reruns isolate the decay setting",
        "固定seed0、原模型和新入口采样方式，分别只改变衰减，所有运行关闭提前停止并跑一万步。AdamW学习率千分之一，Lion万分之二、衰减五倍，使eta lambda匹配。图中可比较同一算法内部不同衰减，不能挑出Lion占优的设置当新主结果。这轮是诊断，后续多seed确认必须单独设计。")
    s.shapes.add_picture(str(DIAGNOSTICS / "decay_diagnostic.png"), Inches(.7), Inches(1.15), height=Inches(5.55))
    text(s, 9.35, 1.65, 3.3, 4.8, ["Fixed 10k steps", "No early stopping", "Matched ηλ", "", "Diagnostic seed 0; not final confirmation"], 20, TEAL)

if stability_rows:
    s = slide("Original seeds: first crossing versus stability",
        "这页延续原六个正式检查点到一万步，学习率、衰减、随机状态均不变。首次连续三次达到99%的原指标保留，同时展示后续曲线，不能把首次达标等同稳定泛化。预算内稳定起点是后验诊断，不静默替换原主指标；续跑计时分段，不重新宣称wall time优势。")
    s.shapes.add_picture(str(STABILITY / "stability.png"), Inches(.45), Inches(1.35), width=Inches(12.4))
    text(s, .7, 6.4, 12, .45,
         "Same hyperparameters and checkpoint state; retrospective stability is a separate diagnostic.", 17, MUTED)

s = slide("Backup: likely questions",
    "备用问答：为什么不跑满十万步？共同停止规则是验证99%连续三次达标，所有运行提前成功。为什么原文献要几万步？配置、正则、划分和任务条件可改变动力学，本次只报告统一配置的实际现象。如何验证公平？同seed初始权重和数据划分哈希、同GPU顺序执行、同精度评估和停止规则。为什么较短gap不等于较快？gap从训练拟合时刻起算，Lion训练拟合较晚。")
text(s, .75, 1.4, 11.8, 5.2, [
    "Why stop early? The initial common rule; follow-up extends observation to 10k.",
    "Where is long-delay grokking? The original zero-decay control shows a delay.",
    "Why not compare legacy logs? They use epoch-level, changing-weight train metrics.",
    "Equal trial count? Initial baseline is fixed; the supplement matches search count.",
    "What is reproducible? Model/split hashes, raw checkpoint metrics and run state."], 22)
prs.save(OUT / "Lion_Presentation.pptx")

# Keep the writing inspectable and easy to merge into the group report.
md = ["# Lion on Modular Multiplication", "", intro, "", "## Results", "",
      (ANALYSIS / "results.md").read_text().split("## 图表")[0], "", "## Interpretation", "", interpretation,
      "", "## Computation and storage", "", resources, "", "## Limitations", ""]
md += ["- " + item for item in limitations]
(OUT / "Lion_Report.md").write_text("\n".join(md) + "\n")
notes = ["# Lion 展示讲稿与问答", "", "英文幻灯片，共8页主讲和1页备用问答，建议约5分钟。", ""]
if supplement:
    notes[2] = "英文幻灯片，共8页主讲、1页补充分析和1页备用问答，建议约5–6分钟。"
if diagnostic_rows:
    notes[2] = "初版配置比较加第二轮基线诊断，12页（含补充分析与备用问答）；不能作为组内已确认的最终结果。"
if stability_rows:
    notes[2] = "初版配置比较加第二轮基线诊断及六检查点稳定性续跑，13页（含备用问答）；仍需区分已测本地配置和未确认的组内最终协议。"
for index, sl in enumerate(prs.slides, 1):
    title = sl.shapes[1].text
    notes.extend([f"## {index}. {title}", "", sl.notes_slide.notes_text_frame.text, ""])
(OUT / "Lion_Speaker_Notes_zh.md").write_text("\n".join(notes))
if supplement:
    with (OUT / "Lion_Report.md").open("a") as file:
        file.write("\n## 等次数调参补充分析\n\n" +
                   (SUPPLEMENT / "analysis/results.md").read_text().split("## 图表")[0] +
                   "\n此设计在首次固定基线结果之后加入，是补充探索性分析；两种方法各3次seed0调参、相同步数上限和停止规则，实际消耗计算量可不同。\n")
if diagnostic_rows:
    p = OUT / "Lion_Report.md"
    p.write_text("> 当前状态：初版配置比较，第二轮正在/已完成基线诊断；不代表组内最终实验完成。\n\n" +
                 p.read_text() + "\n## 第二轮基线诊断\n\n" +
                 (DIAGNOSTICS / "DIAGNOSIS.md").read_text().split("## 如何使用")[0])
if stability_rows:
    with (OUT / "Lion_Report.md").open("a") as file:
        file.write("\n## 原六正式检查点稳定性续跑\n\n" + (STABILITY / "STABILITY.md").read_text())
print(f"Editable report, presentation and speaker notes saved in {OUT}")
