#!/usr/bin/env python3
"""Rebuild the augmentation talk as a native, fully-editable PowerPoint.
Paper format: Background -> Methods -> Results -> Discussion -> Conclusion.
Design: white backgrounds throughout, restrained palette (navy / teal / light gray),
and deliberately varied layouts per slide."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# ---- palette (2 brand colors + neutrals, white canvas) ----
NAVY   = RGBColor(0x1E, 0x3A, 0x5F)   # primary brand
TEAL   = RGBColor(0x0F, 0x76, 0x6E)   # secondary brand
INK    = RGBColor(0x22, 0x2A, 0x33)   # body text
MUTED  = RGBColor(0x5B, 0x66, 0x72)   # secondary text
GRAY   = RGBColor(0xF2, 0xF4, 0xF6)   # light-gray fill
GRAYLN = RGBColor(0xD8, 0xDD, 0xE3)   # hairline
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    r.shadow.inherit = False
    r.fill.solid(); r.fill.fore_color.rgb = bg
    r.line.fill.background()
    return s


def rect(s, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.RECTANGLE):
    sp = s.shapes.add_shape(shape, x, y, w, h)
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
    return sp


def textbox(s, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tb, tf


def setpar(p, text, size, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT,
           space_after=6, level=0, font="Calibri"):
    p.text = text
    p.alignment = align
    p.space_after = Pt(space_after)
    p.level = level
    r = p.runs[0]
    r.font.size = Pt(size); r.font.color.rgb = color
    r.font.bold = bold; r.font.italic = italic
    r.font.name = font
    return p


def bullets(tf, items, size=18, lead_color=INK):
    """items: (text, level, bold) tuples or plain strings. Uses a subtle
    en-dash marker rather than a heavy round bullet."""
    for i, it in enumerate(items):
        text, level, bold = it if isinstance(it, tuple) else (it, 0, False)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        marker = "•  " if level == 0 else "◦  "
        setpar(p, marker + text, size, lead_color if level == 0 else MUTED,
               bold=bold, space_after=9, level=level)


def kicker_title(s, kicker, title, kcolor=TEAL):
    """Left-aligned section kicker + title, with a thin navy rule.
    Deliberately NOT a full-width color band (that reads as AI-default)."""
    kb, kf = textbox(s, Inches(0.75), Inches(0.55), Inches(11.8), Inches(0.4))
    setpar(kf.paragraphs[0], kicker.upper(), 13.5, kcolor, bold=True, space_after=0)
    tb, tf = textbox(s, Inches(0.75), Inches(0.92), Inches(11.8), Inches(0.95))
    setpar(tf.paragraphs[0], title, 29, NAVY, bold=True, space_after=0)


def pic_fit(s, path, cx, cy, max_w, max_h):
    iw, ih = Image.open(path).size
    ar = iw / ih
    w = max_w; h = int(w / ar)
    if h > max_h:
        h = max_h; w = int(h * ar)
    return s.shapes.add_picture(path, cx - w // 2, cy - h // 2, w, h)


A = "assets/"

# =====================================================================
# SLIDE 1 — TITLE  (layout: white canvas, navy side rail)
# =====================================================================
s = slide()
rect(s, 0, 0, Inches(0.35), SH, fill=NAVY)          # slim navy rail
rect(s, Inches(0.75), Inches(2.35), Inches(1.9), Inches(0.09), fill=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(2.55), Inches(11.4), Inches(2.2))
setpar(tf.paragraphs[0], "Data Augmentation for Genomics", 48, NAVY, bold=True, space_after=14)
p = tf.add_paragraph()
setpar(p, "Bringing a standard machine learning technique into a genomics model (Borzoi)",
       22, MUTED, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.55), Inches(11.4), Inches(1.0))
setpar(tf.paragraphs[0], "Joshua Ye", 19, INK, bold=True, space_after=2)
p = tf.add_paragraph(); setpar(p, "2026", 15, MUTED, space_after=0)

# =====================================================================
# SLIDE 2 — BACKGROUND  (layout: statement + intuition side card)
# =====================================================================
s = slide()
kicker_title(s, "Background", "Data augmentation is everywhere in ML")
tb, tf = textbox(s, Inches(0.75), Inches(2.15), Inches(6.4), Inches(4.4))
bullets(tf, [
    ("In vision & speech, augmentation is standard practice:", 0, True),
    ("flip, crop, shift, rotate inputs so a model learns many views of the same data", 1, False),
    ("One of the most popular, low cost ways to improve deep models when data is limited", 0, False),
    ("This project: apply the same idea inside a genomics model (Borzoi), using transforms valid for DNA", 0, True),
], size=18)
# intuition card — light gray, teal top border only
cx, cw = Inches(7.55), Inches(5.05)
rect(s, cx, Inches(2.15), cw, Inches(4.3), fill=GRAY)
rect(s, cx, Inches(2.15), cw, Inches(0.09), fill=TEAL)
tb, tf = textbox(s, cx + Inches(0.32), Inches(2.42), Inches(4.4), Inches(3.9))
setpar(tf.paragraphs[0], "Intuition: how a baby learns", 17, TEAL, bold=True, space_after=12)
for t in ["We learn by interacting with objects in 3D",
          "Pick up a cup, shift it, rotate it, turn it over",
          "Every angle is a different image of the same object",
          "The brain learns the object is invariant to viewpoint"]:
    p = tf.add_paragraph(); setpar(p, "•  " + t, 15.5, INK, space_after=9)
p = tf.add_paragraph()
setpar(p, "Robust understanding comes from many views of one thing, in any domain.",
       14, MUTED, italic=True, space_after=0)

# =====================================================================
# SLIDE 3 — WHAT IS BORZOI  (layout: text + horizontal pipeline diagram)
# =====================================================================
s = slide()
kicker_title(s, "Background", "What is Borzoi?")
tb, tf = textbox(s, Inches(0.75), Inches(2.15), Inches(11.8), Inches(2.2))
bullets(tf, [
    ("A deep neural network that predicts genomic tracks (expression, accessibility, TF binding) directly from DNA sequence", 0, False),
    ("Pretrained on large scale genomic data", 0, False),
    ("We transfer learn it onto our own dataset (CRC transcription factors)", 0, True),
], size=19)
# horizontal pipeline
stages = [("DNA sequence", GRAY, INK), ("Borzoi\n(deep net)", NAVY, WHITE), ("Genomic tracks", GRAY, INK)]
bx, by, bw, bh, gap = Inches(1.35), Inches(4.85), Inches(3.2), Inches(1.25), Inches(1.15)
for i, (label, fill, fg) in enumerate(stages):
    x = bx + i * (bw + gap)
    box = rect(s, x, by, bw, bh, fill=fill, line=GRAYLN, lw=1)
    tf = box.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = label
    for para in tf.paragraphs:
        para.alignment = PP_ALIGN.CENTER
        for r in para.runs:
            r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = fg; r.font.name = "Calibri"
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    if i < 2:
        ar = rect(s, x + bw + Inches(0.18), by + Inches(0.42), Inches(0.8), Inches(0.4),
                  fill=TEAL, shape=MSO_SHAPE.RIGHT_ARROW)
tb, tf = textbox(s, Inches(1.35), Inches(6.4), Inches(10.6), Inches(0.6))
setpar(tf.paragraphs[0], "Goal: adapt a powerful pretrained model to our biology.",
       18, TEAL, bold=True, italic=True, space_after=0)

# =====================================================================
# SLIDE 4 — THE PROBLEM  (layout: big framing question hero)
# =====================================================================
s = slide()
kicker_title(s, "Background · The Problem", "Higher accuracy, but data is limited")
tb, tf = textbox(s, Inches(0.75), Inches(2.2), Inches(11.8), Inches(3.0))
bullets(tf, [
    ("Deep models are data hungry", 0, True),
    ("With too little training data, the model overfits:", 0, False),
    ("memorizes the training set", 1, False),
    ("generalizes poorly to unseen sequences", 1, False),
    ("We can't simply collect more labeled genomic data", 0, False),
], size=20)
# framing question — outlined navy band, not filled full-width
rect(s, Inches(0.75), Inches(5.75), Inches(11.8), Inches(1.05), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.0), Inches(5.75), Inches(11.3), Inches(1.05), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0], "How do we get more from the data we already have?",
       23, NAVY, bold=True, align=PP_ALIGN.CENTER, space_after=0)

# =====================================================================
# SLIDE 5 — THE PAPER  (layout: full-height figure right, text left)
# =====================================================================
s = slide()
kicker_title(s, "Background · The Paper", "Grounded in the literature: EvoAug")
tb, tf = textbox(s, Inches(0.75), Inches(2.1), Inches(6.7), Inches(4.7))
setpar(tf.paragraphs[0], "Data augmentation for genomic DNNs is well established.",
       18, NAVY, bold=True, space_after=8)
p = tf.add_paragraph()
setpar(p, "EvoAug (Lee, Tang, Toneyan & Koo, Genome Biology 2023): evolution inspired augmentations that improve generalization and interpretability when data is limited.",
       16, INK, space_after=10)
p = tf.add_paragraph()
setpar(p, "We apply the Borzoi standard pair on the train split only, keeping sequence and targets aligned:",
       15, MUTED, space_after=8)
for t in ["Random genomic shift: jitter the window center (±128 bp)",
          "Reverse complement: with probability 0.5"]:
    p = tf.add_paragraph(); setpar(p, "•  " + t, 15.5, INK, space_after=6)
pic = pic_fit(s, A + "evoaug_paper.png", Inches(10.15), Inches(3.9), Inches(5.3), Inches(3.9))
tb, tf = textbox(s, Inches(7.7), Inches(6.55), Inches(4.9), Inches(0.6))
setpar(tf.paragraphs[0], "Lee et al., Genome Biology 2023;24:105. doi:10.1186/s13059-023-02941-w",
       10.5, MUTED, italic=True, align=PP_ALIGN.CENTER, space_after=0)

# =====================================================================
# SLIDE 5b — THE PARADOX  (layout: puzzle banner + 3 resolution cards)
# Explains why random mutation augmentation works (EvoAug core idea).
# =====================================================================
s = slide()
kicker_title(s, "Background · The Paper", "Wait, doesn't mutating bases break the pattern?")
# the puzzle banner
rect(s, Inches(0.75), Inches(2.0), Inches(11.8), Inches(0.95), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(2.0), Inches(11.2), Inches(0.95), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "The puzzle: EvoAug also randomly changes bases. Changing DNA should change the signal, yet it improves the model. Why?",
       17, NAVY, bold=True, italic=True, space_after=0)
# three resolution cards
cards = [
    ("Only a light sprinkle", NAVY,
     ["EvoAug mutates ~5 to 15% of bases (plus small ≤30 bp indels & shifts)",
      "A motif is only ~6 to 12 key bp inside a long window",
      "On average the motif survives; the pattern is perturbed, not erased"]),
    ("Same label = learn invariance", TEAL,
     ["The mutated sequence keeps the original target",
      "Forces the model to ignore small random changes",
      "Mirrors real genetic variation; noncoding DNA is functionally robust"]),
    ("Two stage training", NAVY,
     ["Augment only in the first training stage",
      "Then fine tune on clean, unperturbed data",
      "Reanchors to true biology & removes augmentation bias"]),
]
cw, gap0 = Inches(3.78), Inches(0.23)
for i, (title, col, pts) in enumerate(cards):
    x = Inches(0.75) + i * (cw + gap0)
    rect(s, x, Inches(3.35), cw, Inches(3.15), fill=WHITE, line=GRAYLN, lw=1.25)
    rect(s, x, Inches(3.35), cw, Inches(0.55), fill=col)
    tb, tf = textbox(s, x + Inches(0.22), Inches(3.35), cw - Inches(0.4), Inches(0.55), MSO_ANCHOR.MIDDLE)
    setpar(tf.paragraphs[0], f"{i+1}.  {title}", 15.5, WHITE, bold=True, space_after=0)
    tb, tf = textbox(s, x + Inches(0.25), Inches(4.05), cw - Inches(0.45), Inches(2.35))
    for j, t in enumerate(pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        setpar(p, "•  " + t, 13.5, INK, space_after=8)
tb, tf = textbox(s, Inches(0.75), Inches(6.65), Inches(11.8), Inches(0.6))
setpar(tf.paragraphs[0],
       "It's inspired by mutation, exactly how evolution generates variation while conserving function. (Our project used only shift + reverse complement; mutations are a future direction.)",
       12, MUTED, italic=True, space_after=0)

# =====================================================================
# SLIDE 6 — METHODS  (layout: two-column comparison cards)
# =====================================================================
s = slide()
kicker_title(s, "Methods", "Generate more views of the same data")
tb, tf = textbox(s, Inches(0.75), Inches(1.95), Inches(11.8), Inches(0.7))
setpar(tf.paragraphs[0],
       "Not fake data, but real symmetries in nature. Same biology, same target; the model learns what to be invariant to.",
       16.5, MUTED, italic=True, space_after=0)
cards = [
    ("Random genomic shift", NAVY, ["Jitter the window center by ±128 bp",
                                     "Classic Borzoi / Enformer stochastic shift",
                                     "Snaps to whole 32 bp bins so targets stay aligned to the grid",
                                     "Analogy: sliding the object"]),
    ("Reverse complement", TEAL, ["Reverse complement the sequence with p = 0.5",
                                   "Reverse the target bins to match",
                                   "DNA is read on both strands, a real symmetry",
                                   "Analogy: seeing the object flipped"]),
]
for i, (title, col, pts) in enumerate(cards):
    x = Inches(0.75 + i * 6.15)
    rect(s, x, Inches(2.75), Inches(5.85), Inches(3.65), fill=WHITE, line=GRAYLN, lw=1.25)
    rect(s, x, Inches(2.75), Inches(0.12), Inches(3.65), fill=col)   # left color spine
    tb, tf = textbox(s, x + Inches(0.35), Inches(2.95), Inches(5.3), Inches(0.6))
    setpar(tf.paragraphs[0], title, 20, col, bold=True, space_after=0)
    tb, tf = textbox(s, x + Inches(0.35), Inches(3.6), Inches(5.3), Inches(2.7))
    for j, t in enumerate(pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        setpar(p, "•  " + t, 15.5, INK, space_after=9)
tb, tf = textbox(s, Inches(0.75), Inches(6.6), Inches(11.8), Inches(0.6))
setpar(tf.paragraphs[0],
       "Caveat: true Borzoi shifts at single base resolution; we quantize to 32 bp bins to avoid rebinning targets. Same idea, coarser shift.",
       11.5, MUTED, italic=True, space_after=0)

# =====================================================================
# SLIDE 7 — METHODS  (layout: clean summary table)
# =====================================================================
s = slide()
kicker_title(s, "Methods", "Same idea, applied to DNA")
tb, tf = textbox(s, Inches(0.75), Inches(2.0), Inches(11.8), Inches(0.6))
setpar(tf.paragraphs[0], "We show the model multiple valid views of each sequence:", 18, INK, space_after=0)
rows = [("Augmentation", "Analogy", "Setting"),
        ("Random genomic shift", "sliding / shifting the object", "±128 bp"),
        ("Reverse complement", "seeing the object flipped", "p = 0.5")]
tbl = s.shapes.add_table(3, 3, Inches(0.9), Inches(2.7), Inches(11.5), Inches(1.85)).table
tbl.columns[0].width = Inches(3.9); tbl.columns[1].width = Inches(5.2); tbl.columns[2].width = Inches(2.4)
for r in range(3):
    for c in range(3):
        cell = tbl.cell(r, c); cell.text = rows[r][c]
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.15)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c < 2 else PP_ALIGN.CENTER
        run = para.runs[0]; run.font.size = Pt(16.5); run.font.name = "Calibri"
        if r == 0:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        else:
            run.font.color.rgb = INK
            cell.fill.solid(); cell.fill.fore_color.rgb = GRAY if r % 2 else WHITE
tb, tf = textbox(s, Inches(0.75), Inches(4.85), Inches(11.8), Inches(2.2))
bullets(tf, [
    ("DNA is genuinely read on both strands → reverse complement is real", 0, False),
    ("Small shifts reflect that the signal isn't tied to one exact position", 0, False),
    ("The model learns the biology is invariant to these transforms", 0, True),
], size=18)

# =====================================================================
# SLIDE 7a — IMPLEMENTATION: BUILDING A COMPARABLE BASELINE
# (layout: problem statement + 4-step pipeline)
# =====================================================================
s = slide()
kicker_title(s, "Methods · Implementation", "Building a baseline we can actually compare to")
rect(s, Inches(0.75), Inches(1.85), Inches(11.8), Inches(0.95), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(1.85), Inches(11.2), Inches(0.95), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Problem: original Borzoi outputs ~7,611 generic tracks; our transfer head outputs 183 "
       "CRC-specific tracks. The heads don't line up — there's no 1:1 track to compare against.",
       16, NAVY, bold=True, italic=True, space_after=0)

steps = [
    ("1. Narrow the search space", NAVY,
     ["Restrict to Borzoi output channels 2186–6069 (targets_human.csv track_index)",
      "This is the block of ChIP-relevant assay tracks → 3,884 candidate tracks"]),
    ("2. Keep ChIP, drop histone marks", TEAL,
     ["Keep descriptions containing “chip” (e.g. CHIP:POLR2A:...)",
      "Drop histone modification marks (H3K27ac, H3K4me1, H3K4me3, H3K9me3, ... down to generic H3/H4/H2A/H2B)",
      "TF/DNA-binding ChIP is comparable to our CRC TF targets; histone marks are not → 1,882 eligible tracks"]),
    ("3. Best-match by correlation", NAVY,
     ["For each of our 183 CRC targets, compute Pearson r vs. every one of the 1,882 eligible tracks",
      "Streamed over all val bins/intervals (memory-safe, no giant matrix in RAM)"]),
    ("4. Take the best track per target", TEAL,
     ["argmax over the 1,882 correlations → one best-matching Borzoi track per CRC target",
      "That best-match prediction becomes the baseline for that target",
      "Saved to borzoi_filtered_best_match_val.csv (track id, description, factor, r)"]),
]
cw, ch, gapx, gapy = Inches(5.75), Inches(2.15), Inches(0.3), Inches(0.22)
for i, (title, col, pts) in enumerate(steps):
    x = Inches(0.75) + (i % 2) * (cw + gapx)
    y = Inches(3.05) + (i // 2) * (ch + gapy)
    rect(s, x, y, cw, ch, fill=WHITE, line=GRAYLN, lw=1.25)
    rect(s, x, y, Inches(0.1), ch, fill=col)
    tb, tf = textbox(s, x + Inches(0.3), y + Inches(0.12), cw - Inches(0.5), Inches(0.45))
    setpar(tf.paragraphs[0], title, 15.5, col, bold=True, space_after=0)
    tb, tf = textbox(s, x + Inches(0.3), y + Inches(0.62), cw - Inches(0.5), ch - Inches(0.75))
    for j, t in enumerate(pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        setpar(p, "•  " + t, 12.5, INK, space_after=6)

# =====================================================================
# SLIDE 7b — EVALUATION DESIGN  (layout: statement + Q&A resolution)
# Explains why the split matters and why the baseline is evaluated on it too.
# =====================================================================
s = slide()
kicker_title(s, "Methods · Evaluation Design", "Why evaluate on val AND test?")
tb, tf = textbox(s, Inches(0.75), Inches(2.0), Inches(11.8), Inches(1.5))
bullets(tf, [
    ("Train / val / test exists to prevent our model from leaking information about held-out data", 0, True),
    ("Val: used during training for early stopping / model selection → some indirect influence", 0, False),
    ("Test: touched only once, at the very end → the unbiased, final number", 0, False),
], size=17)
rect(s, Inches(0.75), Inches(3.85), Inches(11.8), Inches(1.15), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(3.85), Inches(11.2), Inches(1.15), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "“But the split is only for training — why evaluate the frozen Borzoi baseline on it too?”",
       16.5, NAVY, bold=True, italic=True, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.2), Inches(11.8), Inches(1.9))
bullets(tf, [
    ("Borzoi baseline never trains — the split can't leak into it", 0, False),
    ("But a fair comparison needs both models scored on the exact same genomic intervals", 0, True),
    ("So the baseline is run on val (for iteration) and must also be run on test, to match our model's final, untouched test number — apples to apples", 0, False),
], size=16)

# =====================================================================
# SLIDE 8 — RESULTS  (layout: metrics dashboard — stat tiles + table)
# =====================================================================
s = slide()
kicker_title(s, "Results", "Accuracy improvement (internal ablation)", kcolor=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(1.78), Inches(11.8), Inches(0.4))
setpar(tf.paragraphs[0], "Our model, without vs. with the augmentation + training recipe (mean Pearson R, 183 tracks)", 15, MUTED, space_after=0)
# stat tiles (dashboard feel)
tiles = [("Validation gain", "+0.14", "~28% relative", TEAL),
         ("Test gain", "+0.05", "~9% relative", NAVY),
         ("New val R", "0.64", "was 0.50", TEAL),
         ("New test R", "0.66", "was 0.61", NAVY)]
tw, th, gap0 = Inches(2.78), Inches(1.75), Inches(0.24)
for i, (lab, big, sub, col) in enumerate(tiles):
    x = Inches(0.75) + i * (tw + gap0)
    rect(s, x, Inches(2.35), tw, th, fill=GRAY)
    rect(s, x, Inches(2.35), tw, Inches(0.08), fill=col)
    tb, tf = textbox(s, x, Inches(2.5), tw, th - Inches(0.2), MSO_ANCHOR.MIDDLE)
    setpar(tf.paragraphs[0], lab.upper(), 12, MUTED, bold=True, align=PP_ALIGN.CENTER, space_after=4)
    p = tf.add_paragraph(); setpar(p, big, 40, col, bold=True, align=PP_ALIGN.CENTER, space_after=2)
    p = tf.add_paragraph(); setpar(p, sub, 13, MUTED, align=PP_ALIGN.CENTER, space_after=0)
# supporting table
rows = [("Split", "Baseline", "+ Augmentation", "Gain"),
        ("Validation", "0.5022", "0.6434", "+0.1412"),
        ("Test", "0.6104", "0.6634", "+0.0530")]
tbl = s.shapes.add_table(3, 4, Inches(1.6), Inches(4.5), Inches(10.1), Inches(1.9)).table
for c, wv in enumerate([2.6, 2.4, 2.9, 2.2]):
    tbl.columns[c].width = Inches(wv)
for r in range(3):
    for c in range(4):
        cell = tbl.cell(r, c); cell.text = rows[r][c]
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE; cell.margin_left = Inches(0.15)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = para.runs[0]; run.font.size = Pt(16); run.font.name = "Calibri"
        if r == 0:
            run.font.bold = True; run.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        else:
            run.font.color.rgb = INK
            if c == 3:
                run.font.bold = True; run.font.color.rgb = TEAL
            cell.fill.solid(); cell.fill.fore_color.rgb = GRAY if r % 2 else WHITE

# =====================================================================
# SLIDE 9 — RESULTS  (layout: two-figure comparison)
# =====================================================================
s = slide()
kicker_title(s, "Results", "Validation scatter: predicted vs. actual", kcolor=TEAL)
for i, (lbl, img, col) in enumerate([("Baseline", "baseline_val_scatter.png", MUTED),
                                       ("+ Augmentation", "aug_val_scatter.png", TEAL)]):
    cx = Inches(3.6 + i * 6.1)
    tb, tf = textbox(s, cx - Inches(2.4), Inches(1.95), Inches(4.8), Inches(0.5))
    setpar(tf.paragraphs[0], lbl, 19, col, bold=True, align=PP_ALIGN.CENTER, space_after=0)
    pic_fit(s, A + img, cx, Inches(4.55), Inches(4.25), Inches(3.95))
tb, tf = textbox(s, Inches(0.75), Inches(6.85), Inches(11.8), Inches(0.5))
setpar(tf.paragraphs[0], "Subtle but consistent improvement across the validation split.",
       13.5, MUTED, italic=True, align=PP_ALIGN.CENTER, space_after=0)

# =====================================================================
# SLIDE 10 — RESULTS  (layout: stacked wide figures)
# =====================================================================
s = slide()
kicker_title(s, "Results", "Per track Pearson R (validation)", kcolor=TEAL)
for i, (lbl, img, col) in enumerate([("Baseline", "baseline_val_pearson.png", MUTED),
                                       ("+ Augmentation", "aug_val_pearson.png", TEAL)]):
    cy = Inches(2.7 + i * 2.15)
    tb, tf = textbox(s, Inches(0.75), cy - Inches(0.35), Inches(2.2), Inches(0.7), MSO_ANCHOR.MIDDLE)
    setpar(tf.paragraphs[0], lbl, 16, col, bold=True, space_after=0)
    pic_fit(s, A + img, Inches(7.95), cy, Inches(9.0), Inches(1.9))

# =====================================================================
# SLIDE 11 — RESULTS  (layout: interpretation list)
# =====================================================================
s = slide()
kicker_title(s, "Results", "What it means", kcolor=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(2.05), Inches(11.8), Inches(4.2))
bullets(tf, [
    ("Validation jumped +0.14 (~28% relative), the larger gain", 0, True),
    ("Test improved +0.05 (~9% relative), smaller but same direction", 0, False),
    ("The val/test gap flipped: val (0.50) sat below test (0.61); now val (0.64) is just under test (0.66)", 0, False),
    ("That earlier low val was unusual; the new run is more internally consistent, a sign the gain is real, not noise", 0, False),
    ("Improvement comes from better data use, not more data or a bigger model", 0, True),
], size=18)
tb, tf = textbox(s, Inches(0.75), Inches(6.55), Inches(11.8), Inches(0.7))
setpar(tf.paragraphs[0],
       "Caveat: both runs emit a ConstantInputWarning (constant signal tracks → undefined R, dropped from the mean); worth confirming the same track count feeds each mean.",
       11.5, MUTED, italic=True, space_after=0)

# =====================================================================
# SLIDE 11b — RESULTS  (layout: final benchmark stat tiles + scatter figure)
# The real headline comparison: our final trained model vs the original,
# untrained Borzoi baseline — same held-out genomic intervals (val split).
# =====================================================================
s = slide()
kicker_title(s, "Results", "Final benchmark: our model vs. original Borzoi", kcolor=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(1.7), Inches(11.8), Inches(0.4))
setpar(tf.paragraphs[0], "Val split, 183 tracks, paired by genomic interval", 15, MUTED, space_after=0)
tiles = [("Borzoi baseline R", "0.625", "mean, filtered tracks", MUTED),
         ("Our model R", "0.643", "mean  (+0.018)", TEAL),
         ("Tracks we win", "119/183", "65% · 33 lost, 31 tied", NAVY),
         ("High-conf. tracks", "46% vs 34%", "R ≥ 0.7", TEAL)]
tw, th, gap0 = Inches(2.78), Inches(1.55), Inches(0.24)
for i, (lab, big, sub, col) in enumerate(tiles):
    x = Inches(0.75) + i * (tw + gap0)
    rect(s, x, Inches(2.25), tw, th, fill=GRAY)
    rect(s, x, Inches(2.25), tw, Inches(0.08), fill=col)
    tb, tf = textbox(s, x, Inches(2.38), tw, th - Inches(0.2), MSO_ANCHOR.MIDDLE)
    setpar(tf.paragraphs[0], lab.upper(), 11.5, MUTED, bold=True, align=PP_ALIGN.CENTER, space_after=4)
    p = tf.add_paragraph(); setpar(p, big, 30, col, bold=True, align=PP_ALIGN.CENTER, space_after=2)
    p = tf.add_paragraph(); setpar(p, sub, 12, MUTED, align=PP_ALIGN.CENTER, space_after=0)
pic_fit(s, A + "final_paired_performance_val.png", Inches(6.65), Inches(5.4), Inches(10.8), Inches(3.1))

# =====================================================================
# SLIDE 11c — RESULTS  (layout: honest statistics slide)
# =====================================================================
s = slide()
kicker_title(s, "Results", "Is the gain statistically significant?", kcolor=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(1.95), Inches(11.8), Inches(1.7))
bullets(tf, [
    ("Naive paired test across 183 tracks: paired t p = 0.00017, Wilcoxon p = 1.1e-9", 0, False),
    ("Effect size (Cohen's d) = 0.28 — small-to-moderate, not a dramatic shift", 0, False),
], size=17)
rect(s, Inches(0.75), Inches(3.75), Inches(11.8), Inches(1.35), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(3.75), Inches(11.2), Inches(1.35), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Caveat: the 183 tracks are not independent — many are the same assay (e.g. CTCF ChIP-seq) "
       "repeated across cell lines, so errors are correlated. Treating them as 183 independent samples "
       "is pseudoreplication and inflates significance.",
       15.5, NAVY, bold=True, italic=True, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.35), Inches(11.8), Inches(1.6))
bullets(tf, [
    ("Honest read: the p-value is real arithmetic but overstates confidence given non-independent tracks", 0, True),
    ("The direction and consistency of the gain (65% win rate, concentrated in high-confidence tracks) is the more trustworthy signal than the raw p-value", 0, False),
], size=15.5)

# =====================================================================
# SLIDE 12 — DISCUSSION  (layout: two-column limitations / future)
# =====================================================================
s = slide()
kicker_title(s, "Discussion", "Limitations & future directions")
# limitations column
tb, tf = textbox(s, Inches(0.75), Inches(2.1), Inches(5.85), Inches(0.5))
setpar(tf.paragraphs[0], "Limitations", 17, NAVY, bold=True, space_after=8)
tb, tf = textbox(s, Inches(0.75), Inches(2.6), Inches(5.85), Inches(4.3))
bullets(tf, [
    ("Test-split baseline not yet run — Borzoi has only been benchmarked on val so far; test is the number that will actually count", 0, True),
    ("Track-level significance testing treats 183 tracks as independent when many share an assay across cell lines (pseudoreplication)", 0, False),
    ("Shift is quantized to 32 bp bins, not true single base resolution", 0, False),
    ("Only two augmentations tested (shift + reverse complement)", 0, False),
    ("Single dataset (CRC TFs); gains may not transfer to other biology", 0, False),
], size=15.5)
# divider
rect(s, Inches(6.83), Inches(2.15), Inches(0.02), Inches(4.4), fill=GRAYLN)
# future column
tb, tf = textbox(s, Inches(7.05), Inches(2.1), Inches(5.55), Inches(0.5))
setpar(tf.paragraphs[0], "Future work", 17, TEAL, bold=True, space_after=8)
tb, tf = textbox(s, Inches(7.05), Inches(2.6), Inches(5.55), Inches(4.3))
bullets(tf, [
    ("Run baseline_borzoi.py --split test to get the final, apples-to-apples test benchmark", 0, True),
    ("Use a track-cluster-aware or assay-grouped significance test instead of per-track pseudoreplication", 0, False),
    ("Single base shifts with on the fly target rebinning", 0, False),
    ("Add EvoAug style augmentations (mutation, insertion, deletion)", 0, False),
    ("Validate across additional datasets and model backbones", 0, False),
], size=15.5, lead_color=INK)

# =====================================================================
# SLIDE 13 — CONCLUSION  (layout: large quote on white)
# =====================================================================
s = slide()
rect(s, Inches(0.75), Inches(1.55), Inches(0.14), Inches(4.4), fill=TEAL)  # quote bar
tb, tf = textbox(s, Inches(1.25), Inches(0.7), Inches(11.0), Inches(0.5))
setpar(tf.paragraphs[0], "CONCLUSION", 14, TEAL, bold=True, space_after=0)
tb, tf = textbox(s, Inches(1.25), Inches(1.75), Inches(11.0), Inches(4.0), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Data augmentation, a mainstream ML technique, transfers cleanly to genomics.",
       32, NAVY, bold=True, space_after=16)
p = tf.add_paragraph()
setpar(p, "Using biologically valid transforms (shift + reverse complement) plus a redesigned head and training recipe, our model beats the original Borzoi baseline on 65% of tracks (val, mean R +0.018, high-confidence tracks 46% vs 34%) — a real, if modest, gain.",
       19, INK, space_after=10)
p = tf.add_paragraph()
setpar(p, "The test-split benchmark is the number that will actually decide this — that run is still pending.",
       19, INK, space_after=0)
tb, tf = textbox(s, Inches(1.25), Inches(6.35), Inches(11.0), Inches(0.7))
setpar(tf.paragraphs[0], "A popular ML method, working in a genomics model — final verdict pending the test split.",
       17, TEAL, italic=True, space_after=0)

prs.save("presentation.pptx")
print("Saved presentation.pptx —", len(prs.slides._sldIdLst), "slides")
