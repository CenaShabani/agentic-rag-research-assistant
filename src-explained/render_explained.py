"""Render a line-by-line explanation PDF for one source file.

The design keeps the walkthrough honest: it reads the REAL source file off disk
and walks it top to bottom. The per-file annotation scripts supply an ordered list
of `(line_count, explanation)` segments; this renderer consumes exactly that many
lines per segment and shows them with their true line numbers, then the
explanation. A hard assertion fails the build if the segment line-counts do not
sum to the file length — so no line can be silently skipped and the code shown can
never drift from the code that runs.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Preformatted,
    Spacer, Table, TableStyle,
)

INK = colors.HexColor("#1f2a37")
ACCENT = colors.HexColor("#2f6f4f")
ACCENT_DK = colors.HexColor("#20503a")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d1d5db")
CODE_BG = colors.HexColor("#f6f8fa")
LN = colors.HexColor("#9aa5b1")

_ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=_ss["Normal"], fontName="Helvetica-Bold",
                       fontSize=22, textColor=colors.white, leading=27, alignment=TA_CENTER)
SUB = ParagraphStyle("S", parent=_ss["Normal"], fontSize=12, textColor=colors.white,
                     leading=16, alignment=TA_CENTER)
INTRO = ParagraphStyle("I", parent=_ss["Normal"], fontSize=10.5, leading=15, textColor=INK,
                       spaceAfter=8)
EXPL = ParagraphStyle("E", parent=_ss["Normal"], fontSize=10, leading=14.5, textColor=INK,
                      spaceBefore=3, spaceAfter=12, leftIndent=2)
CODE = ParagraphStyle("C", parent=_ss["Normal"], fontName="Courier", fontSize=8.3,
                      leading=11, textColor=INK)
SMALL = ParagraphStyle("Sm", parent=_ss["Normal"], fontSize=8.5, leading=11, textColor=MUTED)


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(src_path: str, out_pdf: str, subtitle: str, intro: str,
          segments: list[tuple[int, str]]) -> None:
    lines = Path(src_path).read_text(encoding="utf-8").splitlines()
    total = len(lines)
    # Allow exactly one segment to use count=None ("the rest of the file"), so the
    # annotations don't have to hit the exact total by hand.
    none_count = sum(1 for c, _ in segments if c is None)
    assert none_count <= 1, "at most one segment may use count=None"
    if none_count == 1:
        fixed = sum(c for c, _ in segments if c is not None)
        segments = [((total - fixed) if c is None else c, e) for c, e in segments]
    covered = sum(c for c, _ in segments)
    assert covered == total, (
        f"{src_path}: segments cover {covered} lines but file has {total}. "
        f"Fix the annotation line-counts."
    )

    story = []
    fname = Path(src_path).name

    # Header band
    story.append(Table(
        [[[Spacer(1, 8), Paragraph(f"src/{fname}", TITLE),
           Paragraph(subtitle, SUB), Spacer(1, 8)]]],
        colWidths=[6.7 * inch],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT_DK),
            ("LEFTPADDING", (0, 0), (-1, -1), 20), ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            ("TOPPADDING", (0, 0), (-1, -1), 16), ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ]),
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(intro, INTRO))
    story.append(Paragraph(f"Walkthrough of all {total} lines, in order. "
                           "Left numbers are the real line numbers in the file.", SMALL))
    story.append(Spacer(1, 10))

    idx = 0
    for count, expl in segments:
        block = lines[idx: idx + count]
        numbered = "\n".join(f"{idx + j + 1:>3}  {block[j]}" for j in range(len(block)))
        idx += count
        # Code block (keep the code and its explanation together where possible).
        story.append(Table(
            [[Preformatted(numbered if numbered else " ", CODE)]],
            colWidths=[6.7 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("LINEBEFORE", (0, 0), (0, 0), 2, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ))
        story.append(Spacer(1, 3))
        story.append(Paragraph(expl, EXPL))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.9 * inch, 0.55 * inch, f"Line-by-line explanation — src/{fname}")
        canvas.drawRightString(7.6 * inch, 0.55 * inch, str(doc.page))
        canvas.setStrokeColor(RULE)
        canvas.line(0.9 * inch, 0.72 * inch, 7.6 * inch, 0.72 * inch)
        canvas.restoreState()

    doc = BaseDocTemplate(out_pdf, pagesize=LETTER,
                          leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                          topMargin=0.7 * inch, bottomMargin=0.85 * inch,
                          title=f"src/{fname} explained", author="Cena Shabani")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="m")
    doc.addPageTemplates([PageTemplate(id="body", frames=[frame], onPage=footer)])
    doc.build(story)
    print(f"wrote {out_pdf}  ({total} lines, {len(segments)} segments)")
