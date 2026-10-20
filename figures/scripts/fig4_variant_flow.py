"""Fig 4: variant provenance and label flow (hg38). All counts computed from saved files and asserted against the
documented values in variant_provenance/PROVENANCE.md.
Sources: variant_provenance/{exploratory_all_rows,allele_matched_variants,unmatched_or_excluded}.tsv,
         results_phase2/e2_validation/labels/label_counts.tsv (mpra_primary row).
Run: ./torch-venv/bin/python figures/scripts/fig4_variant_flow.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style as S

S.apply()
P = S.ROOT / "variant_provenance"
allr = pd.read_csv(P / "exploratory_all_rows.tsv", sep="\t")
am = pd.read_csv(P / "allele_matched_variants.tsv", sep="\t")
un = pd.read_csv(P / "unmatched_or_excluded.tsv", sep="\t")
lc = pd.read_csv(S.ROOT / "results_phase2/e2_validation/labels/label_counts.tsv", sep="\t").set_index("label").loc["mpra_primary"]
vc = allr.match_status.value_counts()
n_rows, n_unique = len(allr), allr.rsid.nunique()
n_absent = int((un.match_status == "S6_CCV_absent_from_input").sum())
n_s6 = n_unique + n_absent
n_match, n_nonrep, n_na = int(vc["matched"]), int(vc["alt_not_in_reported_pair"]), int(vc["no_allele_info"])
n_pos, n_neg, n_excl, n_reg = int(lc.n_pos), int(lc.n_neg), int(lc.n_excluded), int(lc.n_informative_regions)
n_tested = n_pos + n_neg + n_excl
assert (n_s6, n_unique, n_rows, n_match, n_nonrep, n_na) == (5741, 5739, 10148, 5729, 4396, 23) and len(am) == n_match
assert (n_tested, n_pos, n_neg, n_excl, n_reg) == (4323, 134, 4098, 91, 58) and n_match - n_tested == int(lc.n_untested)
rows = [
    ("S6", "Chen 2024 Supp. Data 6 CCVs (hg19)", n_s6, "source"),
    ("input_unique", "S6_CCV input: unique rsIDs/positions (hg38)", n_unique, "S6 minus 2 rsID-drift losses"),
    ("input_rows", "S6_CCV input rows (one per REF>ALT)", n_rows, "1-3 ALTs per position"),
    ("matched", "ALT = GWAS-reported allele (allele-matched, 1 per position)", n_match, "primary set"),
    ("nonreported", "ALT not the reported counter-allele (exploratory only)", n_nonrep, "excluded"),
    ("na", "no allele information", n_na, "excluded (10 rsIDs)"),
    ("mpra_tested", "tested in MPRA (Law 2024 ST2)", n_tested, "label_counts: pos+neg+excluded"),
    ("untested", "not tested in MPRA", n_match - n_tested, "no label"),
    ("pos", "MPRA positives (primary)", n_pos, ""), ("neg", "MPRA negatives", n_neg, ""), ("excl", "excluded (ambiguous)", n_excl, ""),
    ("regions", "informative regions (>=1 pos and >=1 neg)", n_reg, "130 pos / 2518 neg in these"),
]
df = pd.DataFrame(rows, columns=["node", "label", "count", "note"])

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(-9, 100); ax.axis("off")
def box(x, y, w, h, title, n, col, sub=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=col, ec=S.INK2, lw=0.7))
    ax.text(x + w / 2, y + h * 0.68, f"{n:,}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, title, ha="center", va="center", fontsize=6.6, color=S.INK, linespacing=1.15)
def arrow(a, b, txt=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=S.INK2, lw=0.8))
    if txt: ax.text((a[0] + b[0]) / 2 + 1, (a[1] + b[1]) / 2, txt, fontsize=6.3, color=S.INK2, va="center")
W, H = 24, 15
c1, c2, c3 = "#dbeaf5", "#e8f3ea", "#fbe9cf"
box(2, 82, W, H, "Chen 2024 S6 CCVs\n(hg19, 142 regions)", n_s6, c1)
box(36, 82, W, H, "S6_CCV input rsIDs\n(hg38 positions)", n_unique, c1)
box(70, 82, W, H, "input rows\n(1-3 ALT / position)", n_rows, c1)
arrow((26.5, 89.5), (35.5, 89.5)); ax.text(31, 92, "-2", fontsize=6.3, color=S.INK2, ha="center"); arrow((60.5, 89.5), (69.5, 89.5))
ax.text(36 + W / 2, 79.5, "", fontsize=6)
box(2, 52, W, H, "ALT = GWAS-reported allele\n(allele-matched, primary set)", n_match, c2)
box(36, 52, W, H, "other ALT at matched position\n(exploratory only)", n_nonrep, "#eeeeee")
box(70, 52, W, H, "no allele information\n(10 rsIDs)", n_na, "#eeeeee")
for xx in (14, 48, 82): pass
arrow((82, 81.5), (82, 67.8)); arrow((82, 81.5), (48, 67.8)); arrow((82, 81.5), (14, 67.8))
ax.text(62, 40, "REF = hg38 FASTA in all input rows\n(0 mismatches); \"-2\" = rsID merge/reassignment", fontsize=6.2, color=S.INK2, ha="left")
box(2, 24, W, H, "tested in MPRA\n(Law 2024 ST2)", n_tested, c3)
box(36, 24, W, H, "not tested\n(no label)", n_match - n_tested, "#eeeeee")
arrow((14, 51.5), (14, 39.8)); arrow((14, 51.5), (48, 39.8))
box(2, -8, 15, 14, "primary positives", n_pos, "#f4c9b3")
box(19, -8, 15, 14, "negatives", n_neg, "#dbeaf5")
box(36, -8, 15, 14, "excluded", n_excl, "#eeeeee")
arrow((9, 23.5), (9.5, 6.8)); arrow((14, 23.5), (26.5, 6.8)); arrow((14, 23.5), (43.5, 6.8))
ax.add_patch(FancyBboxPatch((58, -8), 40, 20, boxstyle="round,pad=0.2,rounding_size=1.2", fc="white", ec=S.INK2, lw=0.7))
ax.text(78, 7.5, f"{n_reg}", ha="center", fontsize=11, fontweight="bold")
ax.text(78, 1.5, "informative regions for within-region P1 AUROC\n(130 pos, 2,518 neg scored; 4,232 variants)", ha="center", va="center", fontsize=6.4)
fig.suptitle("Variant provenance: CRC fine-mapped CCVs to MPRA-labelled, allele-matched set (hg38)", fontsize=9, y=0.97)
S.save_figure(fig, "fig4_variant_flow", df)
