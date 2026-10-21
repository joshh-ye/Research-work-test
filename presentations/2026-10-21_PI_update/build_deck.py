"""Build the PI update deck (reports/PI_UPDATE_2026-10-21.pptx) and the speaker-notes Markdown from one content spec.

Formatting contract (user spec): white background, black Arial text, titles 28 pt bold, body 18 pt, sub-bullets 16 pt,
standard Title Slide / Title and Content / Title Only layouts, plain tables with thin black borders and a bold header row.
Run: <venv with python-pptx>/bin/python build_deck.py <out_dir>
"""
import copy
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from deck_content import SLIDES  # noqa: E402

BLACK = RGBColor(0, 0, 0)
FONT = "Arial"
TITLE_PT, BODY_PT, SUB_PT, TABLE_PT = 28, 18, 16, 14
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN_X = Inches(0.6)
CONTENT_W = SLIDE_W - 2 * MARGIN_X
NO_STYLE_TABLE = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # built-in "No Style, Table Grid"


def style_runs(paragraph, size, bold=False):
    for run in paragraph.runs:
        f = run.font
        f.name = FONT
        f.size = Pt(size)
        f.bold = bold
        f.color.rgb = BLACK
    # Paragraph-end properties too, so empty/added text inherits the same look.
    end = paragraph._p.get_or_add_endParaRPr()
    end.set("sz", str(size * 100))
    end.set("b", "1" if bold else "0")


def set_text(shape, lines, title=False):
    """lines: list of str | (str, level) | dict(text, level, numbered)."""
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            item = {"text": item}
        elif isinstance(item, tuple):
            item = {"text": item[0], "level": item[1]}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = item.get("level", 0)
        run = p.add_run()
        run.text = item["text"]
        if title:
            style_runs(p, TITLE_PT, bold=True)
        else:
            style_runs(p, BODY_PT if p.level == 0 else SUB_PT)
            p.space_after = Pt(6)
            if item.get("numbered"):
                pPr = p._p.get_or_add_pPr()
                for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
                    for el in pPr.findall(qn(tag)):
                        pPr.remove(el)
                num = etree.SubElement(pPr, qn("a:buAutoNum"))
                num.set("type", "arabicPeriod")


def place(ph, left, top, width, height):
    ph.left, ph.top, ph.width, ph.height = left, top, width, height


def fit_layouts(prs):
    """Resize the default 4:3 template placeholders to the 16:9 canvas (master and the three layouts used)."""
    title_box = (MARGIN_X, Inches(0.35), CONTENT_W, Inches(1.0))
    body_box = (MARGIN_X, Inches(1.5), CONTENT_W, Inches(5.6))
    for ph in prs.slide_master.placeholders:
        t = ph.placeholder_format.type
        if t is not None and "TITLE" in str(t):
            place(ph, *title_box)
        elif t is not None and "BODY" in str(t):
            place(ph, *body_box)
    layouts = prs.slide_layouts
    tl = layouts[0]  # Title Slide
    for ph in tl.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            place(ph, Inches(0.9), Inches(2.0), SLIDE_W - Inches(1.8), Inches(1.9))
        elif idx == 1:
            place(ph, Inches(0.9), Inches(4.2), SLIDE_W - Inches(1.8), Inches(1.4))
    for li in (1, 5):  # Title and Content, Title Only
        for ph in layouts[li].placeholders:
            idx = ph.placeholder_format.idx
            if idx == 0:
                place(ph, *title_box)
            elif idx == 1:
                place(ph, *body_box)
    # Footer/date/slide-number placeholders: keep them inside the 16:9 frame.
    for layout in [prs.slide_master, *layouts]:
        for ph in layout.placeholders:
            idx = ph.placeholder_format.idx
            if idx in (10, 11, 12):
                ph.top = SLIDE_H - Inches(0.6)
                ph.height = Inches(0.4)
                if idx == 10:
                    ph.left, ph.width = MARGIN_X, Inches(3)
                elif idx == 11:
                    ph.left, ph.width = Inches(4.67), Inches(4)
                else:
                    ph.left, ph.width = SLIDE_W - MARGIN_X - Inches(3), Inches(3)


def border(tcPr, tag):
    ln = etree.SubElement(tcPr, qn(tag))
    ln.set("w", "9525")  # 0.75 pt
    ln.set("cap", "flat")
    ln.set("cmpd", "sng")
    fill = etree.SubElement(ln, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr")).set("val", "000000")
    etree.SubElement(ln, qn("a:prstDash")).set("val", "solid")


def add_table(slide, spec):
    header, rows = spec["header"], spec["rows"]
    widths = spec["col_widths"]  # inches; must sum to <= CONTENT_W
    assert abs(sum(widths) - CONTENT_W / 914400) < 0.05, sum(widths)
    n_rows, n_cols = len(rows) + 1, len(header)
    top = Inches(1.5)
    row_h = Inches(spec.get("row_h", 0.5))
    gf = slide.shapes.add_table(n_rows, n_cols, MARGIN_X, top, CONTENT_W, row_h * n_rows)
    gf.name = spec.get("name", "Results table")
    table = gf.table
    tbl = gf._element.graphic.graphicData.tbl
    tblPr = tbl.tblPr
    for attr in ("bandRow", "bandCol", "firstCol", "lastRow", "lastCol"):
        if tblPr.get(attr) is not None:
            del tblPr.attrib[attr]
    tblPr.set("firstRow", "1")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = NO_STYLE_TABLE
    for j, w in enumerate(widths):
        table.columns[j].width = Inches(w)
    for i in range(n_rows):
        table.rows[i].height = row_h
        values = header if i == 0 else rows[i - 1]
        for j in range(n_cols):
            cell = table.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            tf = cell.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            run = tf.paragraphs[0].add_run()
            run.text = values[j]
            style_runs(tf.paragraphs[0], TABLE_PT, bold=(i == 0))
            tcPr = cell._tc.get_or_add_tcPr()
            for child in list(tcPr):
                tcPr.remove(child)
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                border(tcPr, tag)
            etree.SubElement(tcPr, qn("a:noFill"))
    return gf


def main(out_dir):
    out_dir = Path(out_dir)
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    fit_layouts(prs)
    prs.core_properties.title = "PI update 2026-10-21: CRC sequence-model evaluation"
    prs.core_properties.author = ""
    for s in SLIDES:
        kind = s["kind"]
        layout = prs.slide_layouts[{"title": 0, "bullets": 1, "table": 5}[kind]]
        slide = prs.slides.add_slide(layout)
        set_text(slide.shapes.title, [s["title"]], title=True)
        if kind == "title":
            set_text(slide.placeholders[1], s["subtitle"])
            for p in slide.placeholders[1].text_frame.paragraphs:
                style_runs(p, BODY_PT)
        elif kind == "bullets":
            set_text(slide.placeholders[1], s["bullets"])
        else:
            add_table(slide, s["table"])
        slide.notes_slide.notes_text_frame.text = " ".join(s["notes"])
    pptx_path = out_dir / "PI_UPDATE_2026-10-21.pptx"
    prs.save(pptx_path)

    md = ["# PI update, 21 October 2026: speaker script", "",
          "Generated from `reports/pi_update_2026-10-21/deck_content.py` (same text as the PowerPoint speaker notes).",
          f"{len(SLIDES)} slides; target about 10 minutes.", ""]
    section = None
    for n, s in enumerate(SLIDES, 1):
        if s["section"] != section:
            section = s["section"]
            md += [f"## {section}", ""]
        md += [f"### {s['start']} — Slide {n}: {s['title']}", "", " ".join(s["notes"]), ""]
    (out_dir / "PI_UPDATE_SPEAKER_NOTES.md").write_text("\n".join(md))
    print(pptx_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
