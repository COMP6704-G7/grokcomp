"""Export the consolidated member chapter to editable Word without retraining."""
import re
from pathlib import Path

from docx import Document
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/LION_CONTRIBUTION.md"


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.replace("**", "").replace("`", "")


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.7)
    section.left_margin = section.right_margin = Cm(1.8)
    for name in ["Normal", "Title", "Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.name = "DejaVu Sans"
        style.font.color.rgb = RGBColor.from_string("173445" if name != "Normal" else "222222")
    normal = doc.styles["Normal"]
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    doc.styles["Title"].font.size = Pt(21)
    doc.styles["Heading 1"].font.size = Pt(14)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(10)
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("COMP6704 · Lion contribution · ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    lines = SOURCE.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\-\s]+", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            table = doc.add_table(rows=1, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for cell, value in zip(table.rows[0].cells, rows[0]):
                cell.text = plain(value)
            header_repeat = OxmlElement("w:tblHeader")
            table.rows[0]._tr.get_or_add_trPr().append(header_repeat)
            for row in rows[1:]:
                for cell, value in zip(table.add_row().cells, row):
                    cell.text = plain(value)
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after = Pt(3)
                        for run in p.runs:
                            run.font.size = Pt(8.3)
            continue
        picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if picture:
            doc.add_picture(str((SOURCE.parent / picture[2]).resolve()), width=Inches(6.6))
            caption = doc.add_paragraph(picture[1])
            caption.alignment = 1
            caption.paragraph_format.space_after = Pt(8)
            for run in caption.runs:
                run.font.size = Pt(8)
        elif line.startswith("# "):
            doc.add_heading(plain(line[2:]), 0)
        elif line.startswith("## "):
            doc.add_heading(plain(line[3:]), 1)
        else:
            paragraph = doc.add_paragraph(plain(line))
            if re.match(r"\[\d+\] ", line):
                paragraph.paragraph_format.space_after = Pt(3)
                for run in paragraph.runs:
                    run.font.size = Pt(8)
        index += 1
    out = ROOT / "deliverables"
    out.mkdir(exist_ok=True)
    target = out / "Lion_Contribution.docx"
    doc.save(target)
    print(target)


if __name__ == "__main__":
    main()
