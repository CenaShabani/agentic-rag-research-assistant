"""Build the project deck from saved figures and the published numbers.

Standard: ../../../05-communication/context/project-deck.md

Every image on every slide is a file read from outputs/figures/. Nothing is drawn here
and nothing is drawn in PowerPoint - if a chart is wrong, fix the code that exports it
and re-run this. A missing figure raises rather than silently leaving a slide blank.

The slide *content* lives in `slides.py`; this file is the layout engine. Keeping
them apart means a number can be corrected without touching geometry, and the
argument can be read top to bottom in one file without wading through Emu maths.

Run:  python deck/build_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

PROJECT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT / "outputs" / "figures"
OUT = PROJECT / "deck" / f"{PROJECT.name}-deck.pptx"

# 16:9
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.9)

INK = RGBColor(0x14, 0x1A, 0x21)
MUTED = RGBColor(0x5A, 0x66, 0x75)
ACCENT = RGBColor(0x2F, 0x6F, 0x4F)   # same green as the figures' agent colour
WARN = RGBColor(0xA6, 0x46, 0x2F)


def figure(name: str) -> Path:
    """Resolve a figure by filename anywhere under outputs/figures/, or fail loudly."""
    hits = [p for p in FIGURES.rglob(name) if p.is_file()]
    if not hits:
        raise SystemExit(
            f"figure '{name}' not found under {FIGURES}. "
            "Deck figures must be exported by src/figures.py - never drawn in PowerPoint."
        )
    return hits[0]


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, text, *, left, top, width, height, size, bold=False,
            color=INK, align=PP_ALIGN.LEFT, spacing=1.15):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.line_spacing = spacing
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = "Calibri"
    return box


def title_slide(prs, claim: str, sub: str, notes: str):
    s = blank(prs)
    textbox(s, claim, left=MARGIN, top=Inches(2.2), width=W - 2 * MARGIN,
            height=Inches(2.4), size=40, bold=True)
    textbox(s, sub, left=MARGIN, top=Inches(4.7), width=W - 2 * MARGIN,
            height=Inches(1.2), size=18, color=MUTED)
    s.notes_slide.notes_text_frame.text = notes
    return s


def content_slide(prs, title: str, body: str, notes: str, fig: str | None = None):
    """Title states the claim. Body is short. Figure, if any, sits right."""
    s = blank(prs)
    textbox(s, title, left=MARGIN, top=Inches(0.6), width=W - 2 * MARGIN,
            height=Inches(1.1), size=28, bold=True)
    body_w = Inches(5.6) if fig else W - 2 * MARGIN
    textbox(s, body, left=MARGIN, top=Inches(1.9), width=body_w,
            height=Inches(4.4), size=17, spacing=1.35)
    if fig:
        path = figure(fig)
        s.shapes.add_picture(str(path), Inches(7.0), Inches(1.9), width=Inches(5.6))
    s.notes_slide.notes_text_frame.text = notes
    return s


def build() -> Path:
    """Assemble every slide in `slides.SLIDES` and write the .pptx."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import slides

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    title_slide(prs, slides.TITLE_CLAIM, slides.TITLE_SUB, slides.TITLE_NOTES)
    for title, body, notes, fig in slides.SLIDES:
        # Empty speaker notes are a failed deliverable, not a style choice - the
        # standard says a slide without them is unfinished, so this refuses to
        # build one rather than shipping a slide nobody can present cold.
        if not notes.strip():
            raise SystemExit(f"slide '{title}' has no speaker notes")
        content_slide(prs, title, body, notes, fig=fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")


