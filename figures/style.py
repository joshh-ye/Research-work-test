"""Shared figure style for Phase 3 publication figures (static matplotlib, colour-blind-safe).
Palette: Okabe-Ito (Okabe & Ito 2008), fixed entity -> colour mapping so colour follows the entity, not rank.
Every figure script calls save_figure(), which writes PDF + PNG (300 dpi) to figures/out/ and the plotted
values to figures/data/<name>.csv.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT, DATA, TABLES = FIG / "out", FIG / "data", FIG / "tables"
for _d in (OUT, DATA, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID = "#1a1a1a", "#555555", "#8a8a8a", "#e3e3e3"
# fixed model colours
MODEL_COL = {"Borzoi-TL": "#0072B2", "Borzoi (pretrained)": "#E69F00", "Enformer": "#009E73",
             "AlphaGenome": "#CC79A7", "baseline": "#999999"}
MODEL_MARK = {"Borzoi-TL": "s", "Borzoi (pretrained)": "o", "Enformer": "^", "AlphaGenome": "D"}
# exposure classes ordered from most to least exposure of the pretrained model
EXPO_ORDER = ["identical_experiment", "same_cellline_same_target", "same_cellline_other_target",
              "same_target_other_cellline", "unexposed"]
EXPO_COL = {"identical_experiment": "#D55E00", "same_cellline_same_target": "#E69F00",
            "same_cellline_other_target": "#56B4E9", "same_target_other_cellline": "#009E73",
            "unexposed": "#CC79A7"}
EXPO_LABEL = {"identical_experiment": "identical experiment", "same_cellline_same_target": "same cell line, same TF",
              "same_cellline_other_target": "same cell line, other TF",
              "same_target_other_cellline": "same TF, other cell line", "unexposed": "unexposed"}
OTHER_COL = "#b0b0b0"
CELL_COLS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.grid": False,
        "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def panel_label(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def hgrid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, name, df):
    """Write plotted values CSV + PDF + PNG(300 dpi). Returns the CSV path."""
    csv = DATA / f"{name}.csv"
    df.to_csv(csv, index=False)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", csv, OUT / f"{name}.pdf", OUT / f"{name}.png")
    return csv
