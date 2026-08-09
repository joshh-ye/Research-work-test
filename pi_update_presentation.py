#!/usr/bin/env python3
"""PI update: transfer-learned model vs. original Borzoi baseline.
Standalone from the augmentation deck — just the baseline-comparison story:
how the baseline was built, the results, and an honest significance caveat."""

from pathlib import Path
import csv

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

NAVY   = RGBColor(0x1E, 0x3A, 0x5F)
TEAL   = RGBColor(0x0F, 0x76, 0x6E)
INK    = RGBColor(0x22, 0x2A, 0x33)
MUTED  = RGBColor(0x5B, 0x66, 0x72)
GRAY   = RGBColor(0xF2, 0xF4, 0xF6)
GRAYLN = RGBColor(0xD8, 0xDD, 0xE3)
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
    for i, it in enumerate(items):
        text, level, bold = it if isinstance(it, tuple) else (it, 0, False)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        marker = "•  " if level == 0 else "◦  "
        setpar(p, marker + text, size, lead_color if level == 0 else MUTED,
               bold=bold, space_after=9, level=level)


def kicker_title(s, kicker, title, kcolor=TEAL):
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

# ---------------------------------------------------------------------
# Results are read from the comparison CSVs, never hard-coded. Val lives in
# results_comparison/, test in results_comparison_test/ (written by
# run_test_analysis.sh once the ACCRE baseline job's outputs are pulled down).
# A split with no CSVs yet renders as "pending" instead of a stale number.
# ---------------------------------------------------------------------
SPLIT_DIRS = {"val": Path("results_comparison"), "test": Path("results_comparison_test")}


def read_row(path):
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def load_split(split):
    d = SPLIT_DIRS[split]
    summary = read_row(d / f"summary_stats_{split}.csv")
    paired = read_row(d / f"paired_stats_{split}.csv")
    if summary is None or paired is None:
        return None
    st = {r["model"]: r for r in summary}
    p = paired[0]
    return {
        "transfer_mean": float(st["transfer"]["mean"]),
        "borzoi_mean": float(st["filtered_borzoi"]["mean"]),
        "transfer_hi": float(st["transfer"]["pct_r_ge_0.7"]),
        "borzoi_hi": float(st["filtered_borzoi"]["pct_r_ge_0.7"]),
        "n": int(p["n"]),
        "mean_diff": float(p["mean_diff"]),
        "wins": int(p["n_transfer_wins"]),
        "losses": int(p["n_borzoi_wins"]),
        "ties": int(p["n_tie"]),
        "pct_wins": float(p["pct_transfer_wins"]),
        "t_p": float(p["paired_t_p"]),
        "w_p": float(p["wilcoxon_p"]),
        "d": float(p["cohens_d_paired"]),
        "dir": d,
    }


def fig(split, name):
    """Curated asset if one was hand-tuned for this split, else the run's own figure."""
    for cand in (Path(A) / f"final_{name}_{split}.png", SPLIT_DIRS[split] / f"{name}_{split}.png"):
        if cand.exists():
            return str(cand)
    return None


def pfmt(p):
    return f"{p:.5f}".rstrip("0") if p >= 1e-4 else f"{p:.1e}"


VAL = load_split("val")
TEST = load_split("test")
if VAL is None:
    raise SystemExit("results_comparison/summary_stats_val.csv missing — run compare_transfer_borzoi.py --split val")


def result_slide(split, stats):
    """Tiles + paired-performance figure for one split."""
    s = slide()
    label = "val split" if split == "val" else "TEST split — held out, scored once"
    kicker_title(s, "Results", f"Our model vs. original Borzoi ({label})", kcolor=TEAL)
    tb, tf = textbox(s, Inches(0.75), Inches(1.7), Inches(11.8), Inches(0.4))
    setpar(tf.paragraphs[0], f"{stats['n']} tracks, paired by genomic interval", 15, MUTED, space_after=0)
    tiles = [
        ("Borzoi baseline R", f"{stats['borzoi_mean']:.3f}", "mean, filtered tracks", MUTED),
        ("Our model R", f"{stats['transfer_mean']:.3f}", f"mean  ({stats['mean_diff']:+.3f})", TEAL),
        ("Tracks we win", f"{stats['wins']}/{stats['n']}",
         f"{stats['pct_wins']:.0f}% · {stats['losses']} lost, {stats['ties']} tied", NAVY),
        ("High-conf. tracks", f"{stats['transfer_hi']:.0f}% vs {stats['borzoi_hi']:.0f}%", "R ≥ 0.7", TEAL),
    ]
    tw, th, gap0 = Inches(2.78), Inches(1.55), Inches(0.24)
    for i, (lab, big, sub, col) in enumerate(tiles):
        x = Inches(0.75) + i * (tw + gap0)
        rect(s, x, Inches(2.25), tw, th, fill=GRAY)
        rect(s, x, Inches(2.25), tw, Inches(0.08), fill=col)
        tb, tf = textbox(s, x, Inches(2.38), tw, th - Inches(0.2), MSO_ANCHOR.MIDDLE)
        setpar(tf.paragraphs[0], lab.upper(), 11.5, MUTED, bold=True, align=PP_ALIGN.CENTER, space_after=4)
        p = tf.add_paragraph(); setpar(p, big, 30, col, bold=True, align=PP_ALIGN.CENTER, space_after=2)
        p = tf.add_paragraph(); setpar(p, sub, 12, MUTED, align=PP_ALIGN.CENTER, space_after=0)
    path = fig(split, "paired_performance")
    if path:
        pic_fit(s, path, Inches(6.65), Inches(5.4), Inches(10.8), Inches(3.1))


def pending_slide():
    s = slide()
    kicker_title(s, "Results", "Test split — run submitted, results not yet in", kcolor=TEAL)
    tb, tf = textbox(s, Inches(0.75), Inches(2.1), Inches(11.8), Inches(2.2))
    bullets(tf, [
        ("Our model's test predictions are done (results_full_2/test_preds.npy, test_targets.npy)", 0, False),
        ("Still needed: original-Borzoi inference over the test intervals — GPU job on ACCRE "
         "(baseline_borzoi_filtered.slurm), which writes results_baseline_filtered_test/", 0, True),
        ("Copy those two files down and run ./run_test_analysis.sh — this slide then fills in "
         "automatically with the real test numbers", 0, False),
    ], size=17)
    rect(s, Inches(0.75), Inches(4.9), Inches(11.8), Inches(1.1), fill=GRAY, line=NAVY, lw=1.5)
    tb, tf = textbox(s, Inches(1.05), Inches(4.9), Inches(11.2), Inches(1.1), MSO_ANCHOR.MIDDLE)
    setpar(tf.paragraphs[0],
           "No test numbers are shown in this deck until they exist — the val result below is "
           "explicitly the non-final read.", 16.5, NAVY, bold=True, italic=True, space_after=0)

# =====================================================================
# SLIDE 1 — TITLE
# =====================================================================
s = slide()
rect(s, 0, 0, Inches(0.35), SH, fill=NAVY)
rect(s, Inches(0.75), Inches(2.35), Inches(1.9), Inches(0.09), fill=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(2.55), Inches(11.4), Inches(2.2))
setpar(tf.paragraphs[0], "Transfer Model vs. Original Borzoi", 42, NAVY, bold=True, space_after=14)
p = tf.add_paragraph()
setpar(p, "PI update: how the baseline was built, where we stand, what's next",
       21, MUTED, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.55), Inches(11.4), Inches(1.0))
setpar(tf.paragraphs[0], "Joshua Ye", 19, INK, bold=True, space_after=2)
p = tf.add_paragraph(); setpar(p, "2026", 15, MUTED, space_after=0)

# =====================================================================
# SLIDE 2 — THE PROBLEM: HEADS DON'T MATCH
# =====================================================================
s = slide()
kicker_title(s, "Baseline Construction", "Problem: the two models' outputs don't line up")
tb, tf = textbox(s, Inches(0.75), Inches(2.1), Inches(11.8), Inches(2.0))
bullets(tf, [
    ("Our transfer-learned model predicts 183 CRC transcription-factor tracks (our own targets)", 0, False),
    ("Original Borzoi predicts ~7,611 generic genomic tracks — a completely different label set", 0, False),
    ("There is no 1:1 track to compare against — we need to construct a fair baseline first", 0, True),
], size=19)
rect(s, Inches(0.75), Inches(4.55), Inches(11.8), Inches(1.1), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(4.55), Inches(11.2), Inches(1.1), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Approach: for each of our 183 targets, find the single best-correlated track "
       "among Borzoi's outputs and use that as the baseline prediction for that target.",
       17.5, NAVY, bold=True, italic=True, space_after=0)

# =====================================================================
# SLIDE 3 — HOW THE BASELINE WAS BUILT (4-step pipeline)
# =====================================================================
s = slide()
kicker_title(s, "Baseline Construction", "How the baseline was built")
steps = [
    ("1. Narrow the search space", NAVY,
     ["Restrict to Borzoi output channels 2186–6069 (targets_human.csv track_index)",
      "This is the block of ChIP-relevant assay tracks",
      "→ 3,884 candidate tracks"]),
    ("2. Keep ChIP, drop histone marks", TEAL,
     ["Keep descriptions containing “chip” (e.g. CHIP:POLR2A:...)",
      "Drop histone modification marks (H3K27ac, H3K4me1/2/3, H3K9me3, H3K27me3, generic H3/H4/H2A/H2B, ...)",
      "TF/DNA-binding ChIP is comparable to our targets; histone marks are not",
      "→ 1,882 eligible tracks"]),
    ("3. Best-match by correlation", NAVY,
     ["For each of our 183 targets, compute Pearson r vs. every one of the 1,882 eligible tracks",
      "Streamed over all val bins/intervals — memory-safe, no giant matrix in RAM"]),
    ("4. Take the best track per target", TEAL,
     ["argmax over the 1,882 correlations → one best-matching Borzoi track per target",
      "That prediction becomes the baseline for that target",
      "Saved with full metadata (track id, description, factor, r) for auditability"]),
]
cw, ch, gapx, gapy = Inches(5.75), Inches(2.15), Inches(0.3), Inches(0.22)
for i, (title, col, pts) in enumerate(steps):
    x = Inches(0.75) + (i % 2) * (cw + gapx)
    y = Inches(2.0) + (i // 2) * (ch + gapy)
    rect(s, x, y, cw, ch, fill=WHITE, line=GRAYLN, lw=1.25)
    rect(s, x, y, Inches(0.1), ch, fill=col)
    tb, tf = textbox(s, x + Inches(0.3), y + Inches(0.12), cw - Inches(0.5), Inches(0.45))
    setpar(tf.paragraphs[0], title, 15.5, col, bold=True, space_after=0)
    tb, tf = textbox(s, x + Inches(0.3), y + Inches(0.62), cw - Inches(0.5), ch - Inches(0.75))
    for j, t in enumerate(pts):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        setpar(p, "•  " + t, 12.5, INK, space_after=6)

# =====================================================================
# SLIDE 4 — WHAT THE MATCHING ACTUALLY LOOKS LIKE (correlation matrix)
# =====================================================================
_heat = Path("results_comparison_test/pearson_matrix_heatmap_test.png")
if _heat.exists():
    s = slide()
    kicker_title(s, "Baseline Construction", "What the best-match step actually sees")
    tb, tf = textbox(s, Inches(0.72), Inches(1.72), Inches(4.6), Inches(4.6))
    bullets(tf, [
        ("Every one of our 183 targets scored against all 1,882 eligible Borzoi tracks "
         "— 344k correlations", 0, False),
        ("Cyan dot = the winning track (argmax), i.e. the baseline for that target", 0, True),
        ("Bright blocks = same-TF neighbourhoods (CTCF, POLR2A): Borzoi's assay identity, "
         "not cell type, drives the match", 0, False),
        ("Only 66/183 best matches are the same TF — the rest are stand-ins", 0, False),
        ("Both axes sorted by factor for display only; the apparent diagonal is that sorting, "
         "not a result", 0, True),
    ], size=13.5)
    pic_fit(s, str(_heat), Inches(9.1), Inches(4.0), Inches(7.6), Inches(4.6))
    tb, tf = textbox(s, Inches(0.72), Inches(6.45), Inches(11.9), Inches(0.6))
    setpar(tf.paragraphs[0],
           "Caveat: best vs. 2nd-best differs by only 0.011 median r — the winning track's "
           "identity is not stable, though its value is.",
           12.5, MUTED, italic=True, space_after=0)

# =====================================================================
# SLIDE 5 — WHY VAL AND TEST BOTH MATTER
# =====================================================================
s = slide()
kicker_title(s, "Evaluation Design", "Why the baseline is scored on val AND test")
tb, tf = textbox(s, Inches(0.75), Inches(2.0), Inches(11.8), Inches(1.3))
bullets(tf, [
    ("Val is used during our model's training for early stopping / model selection", 0, False),
    ("Test is touched only once, at the end — the unbiased, final number", 0, False),
], size=18)
rect(s, Inches(0.75), Inches(3.55), Inches(11.8), Inches(1.15), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(3.55), Inches(11.2), Inches(1.15), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Borzoi baseline never trains, so the split can't leak into it directly — but a fair "
       "comparison still needs both models scored on the exact same genomic intervals.",
       16.5, NAVY, bold=True, italic=True, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.0), Inches(11.8), Inches(1.6))
bullets(tf, [
    ("Val comparison is our iterable read on performance", 0, True),
    (("Test-split comparison is in — that is the number that counts"
      if TEST else
      "Test-split baseline run is still pending — that will be the number that counts"), 0, False),
], size=17)

# =====================================================================
# SLIDES 5–6 — RESULTS (val, then test if available)
# =====================================================================
result_slide("val", VAL)
if TEST:
    result_slide("test", TEST)
else:
    pending_slide()

# =====================================================================
# SLIDE 6 — IS IT SIGNIFICANT? (honest caveat)
# =====================================================================
s = slide()
kicker_title(s, "Results", "Is the gain statistically significant?", kcolor=TEAL)
tb, tf = textbox(s, Inches(0.75), Inches(1.95), Inches(11.8), Inches(1.4))
SIG = TEST or VAL
SIG_LABEL = "test" if TEST else "val"
bullets(tf, [
    (f"Naive paired test across {SIG['n']} tracks ({SIG_LABEL}): paired t p = {pfmt(SIG['t_p'])}, "
     f"Wilcoxon p = {pfmt(SIG['w_p'])}", 0, False),
    (f"Effect size (Cohen's d) = {SIG['d']:.2f} — small-to-moderate, not dramatic", 0, False),
], size=17)
rect(s, Inches(0.75), Inches(3.6), Inches(11.8), Inches(1.4), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.05), Inches(3.6), Inches(11.2), Inches(1.4), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0],
       "Caveat: the 183 tracks are not independent — many are the same assay (e.g. CTCF ChIP-seq) "
       "repeated across cell lines, so errors are correlated. Treating them as 183 independent "
       "samples is pseudoreplication and inflates significance.",
       15.5, NAVY, bold=True, italic=True, space_after=0)
tb, tf = textbox(s, Inches(0.75), Inches(5.3), Inches(11.8), Inches(1.5))
bullets(tf, [
    ("Honest read: the p-value is real arithmetic but overstates confidence given non-independent tracks", 0, True),
    (f"The direction and consistency of the gain — {SIG['pct_wins']:.0f}% win rate, concentrated in "
     "high-confidence tracks — is the more trustworthy signal", 0, False),
], size=15.5)

# =====================================================================
# SLIDE 7 — NEXT STEPS
# =====================================================================
s = slide()
kicker_title(s, "Next Steps", "What's needed before this is a final result")
next_steps = [
    ("Use a track-cluster-aware or assay-grouped significance test instead of per-track pseudoreplication", 0, False),
    ("Audit for any remaining mismatched or constant-signal tracks in the best-match selection", 0, False),
]
if TEST:
    next_steps.insert(0, ("Test-split baseline is done — write up val and test side by side", 0, True))
    bottom = (f"Bottom line: the gain holds on the held-out test split "
              f"({TEST['transfer_mean']:.3f} vs {TEST['borzoi_mean']:.3f}, {TEST['pct_wins']:.0f}% of tracks).")
else:
    next_steps.insert(0, ("Pull the ACCRE Borzoi test-split baseline down and run ./run_test_analysis.sh", 0, True))
    bottom = "Bottom line: a real, modest improvement on val — test split will decide it."
tb, tf = textbox(s, Inches(0.75), Inches(2.1), Inches(11.8), Inches(4.2))
bullets(tf, next_steps, size=19)
rect(s, Inches(0.75), Inches(5.6), Inches(11.8), Inches(1.05), fill=GRAY, line=NAVY, lw=1.5)
tb, tf = textbox(s, Inches(1.0), Inches(5.6), Inches(11.3), Inches(1.05), MSO_ANCHOR.MIDDLE)
setpar(tf.paragraphs[0], bottom, 18, NAVY, bold=True, align=PP_ALIGN.CENTER, space_after=0)

prs.save("pi_update.pptx")
print("Saved pi_update.pptx —", len(prs.slides._sldIdLst), "slides")
