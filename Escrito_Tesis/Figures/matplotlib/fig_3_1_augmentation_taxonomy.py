"""
fig_3_1_augmentation_taxonomy.py
Taxonomy tree of data augmentation methods for NLP.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_TOKEN   = "#BBDEFB"
C_PARA    = "#C8E6C9"
C_FEAT    = "#E1BEE7"
C_GEN     = "#FFE0B2"
C_HYBRID  = "#FFCDD2"

B_TOKEN   = "#1565C0"
B_PARA    = "#2E7D32"
B_FEAT    = "#6A1B9A"
B_GEN     = "#E65100"
B_HYBRID  = "#C62828"

LEAF_BASELINE_FILL  = "#E3F2FD"
LEAF_BASELINE_EDGE  = "#1565C0"
LEAF_GRAY_FILL      = "#ECEFF1"
LEAF_GRAY_EDGE      = "#78909C"
LEAF_GOLD_FILL      = "#FFF8E1"
LEAF_GOLD_EDGE      = "#F9A825"

ROOT_FILL = "#0D47A1"
ROOT_TEXT = "white"
LINE_COLOR = "#546E7A"

# ---------------------------------------------------------------------------
# Figure & axes
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 20, 13
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_box(cx, cy, w, h, facecolor, edgecolor, lw=1.5, zorder=3,
             linestyle="solid"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=zorder, linestyle=linestyle,
    )
    ax.add_patch(box)
    return box


def draw_line(x1, y1, x2, y2, color=LINE_COLOR, lw=1.8, zorder=1,
              linestyle="solid"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=zorder,
            linestyle=linestyle, solid_capstyle="round")


def leaf_style(style):
    if style == "baseline":
        return LEAF_BASELINE_FILL, LEAF_BASELINE_EDGE, 2.0, "normal"
    elif style == "baseline_bold":
        return LEAF_BASELINE_FILL, LEAF_BASELINE_EDGE, 2.8, "bold"
    else:
        return LEAF_GRAY_FILL, LEAF_GRAY_EDGE, 1.4, "normal"

# ---------------------------------------------------------------------------
# Layout coordinates
# ---------------------------------------------------------------------------
Y_TITLE  = 12.5
Y_ROOT   = 11.5
Y_RAIL1  = 10.55
Y_LVL1   = 9.55
Y_RAIL2  = 8.5
Y_LVL2   = 7.45
Y_HYBRID = 5.75
Y_RAIL3  = 4.85
Y_LVL3   = 3.95

ROOT_W, ROOT_H = 6.5, 0.75
LVL1_W, LVL1_H = 3.4, 0.82
LEAF_W, LEAF_H = 2.1, 0.60
HYBRID_W = 2.6

# ---------------------------------------------------------------------------
# Leaf X positions (carefully spaced to avoid overlaps)
#
# LEAF_W = 2.1 => half-width = 1.05
# Minimum gap between boxes = 0.25
# Minimum center-to-center within same group = 2.35
# Minimum center-to-center across groups = 2.6 (bigger gap)
#
# 9 leaves total across FIG_W=20:
#   Token(2): positions 1.4, 3.75           => spans [0.35,2.45] [2.70,4.80]
#   Para(2):  positions 5.65, 8.00          => spans [4.60,6.70] [6.95,9.05]
#   Feat(3):  positions 9.65, 11.75, 13.85  => spans [8.60,10.70] [10.70,12.80] [12.80,14.90]
#   Gen(2):   positions 15.50, 17.85        => spans [14.45,16.55] [16.80,18.90]
#
# Gaps: Token-Para: 4.60-4.80=-0.20 OVERLAP. Need more space.
# Adjusted:
#   Token(2): 1.20, 3.55
#   Para(2):  5.45, 7.80
#   Feat(3):  9.70, 11.90, 14.10
#   Gen(2):   16.00, 18.35
# Gaps: 3.55+1.05=4.60; 5.45-1.05=4.40 => gap=0 needed is 4.40>4.60? No: 4.40<4.60 OK
# Actually: Token last right edge = 3.55+1.05=4.60
#           Para first left edge = 5.45-1.05=4.40 => overlap of 0.20!
# Fix: push para right or token left.
#   Token(2): 1.10, 3.35
#   Para(2):  5.60, 7.90
# Token right = 3.35+1.05=4.40; Para left = 5.60-1.05=4.55 => gap 0.15 OK
# Para right = 7.90+1.05=8.95; Feat left = 9.70-1.05=8.65 => overlap 0.30!
# Fix feat:
#   Feat(3):  10.05, 12.15, 14.25
# Para right = 8.95; Feat left = 10.05-1.05=9.00 => gap 0.05 tight but OK
# Feat right = 14.25+1.05=15.30; Gen left = 16.00-1.05=14.95 => gap -0.35 OVERLAP
# Fix gen:
#   Gen(2):   16.40, 18.65
# Feat right = 15.30; Gen left = 16.40-1.05=15.35 => gap 0.05 OK
# Gen right = 18.65+1.05=19.70 => fits in 20.
# ---------------------------------------------------------------------------
LEAF_X = {
    "token": [1.10, 3.35],
    "para":  [5.60, 7.90],
    "feat":  [10.05, 12.15, 14.25],
    "gen":   [16.40, 18.65],
}

X_TOKEN = sum(LEAF_X["token"]) / len(LEAF_X["token"])   # 2.225
X_PARA  = sum(LEAF_X["para"])  / len(LEAF_X["para"])    # 6.75
X_FEAT  = sum(LEAF_X["feat"])  / len(LEAF_X["feat"])    # 12.15
X_GEN   = sum(LEAF_X["gen"])   / len(LEAF_X["gen"])     # 17.525

X_HYBRID = (X_FEAT + X_GEN) / 2  # ~14.84

# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
ax.text(FIG_W / 2, Y_TITLE,
        u"Taxonom\u00eda de M\u00e9todos de Aumentaci\u00f3n de Datos para NLP",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#263238", zorder=10)

# ---------------------------------------------------------------------------
# ROOT NODE
# ---------------------------------------------------------------------------
draw_box(FIG_W / 2, Y_ROOT, ROOT_W, ROOT_H, ROOT_FILL, "#0D47A1", lw=2.5)
ax.text(FIG_W / 2, Y_ROOT,
        u"Aumentaci\u00f3n de Datos para NLP",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color=ROOT_TEXT, zorder=5)

# ---------------------------------------------------------------------------
# Trunk -> rail -> categories
# ---------------------------------------------------------------------------
draw_line(FIG_W / 2, Y_ROOT - ROOT_H / 2, FIG_W / 2, Y_RAIL1, lw=2.2)
draw_line(X_TOKEN, Y_RAIL1, X_GEN, Y_RAIL1, lw=2.2)
for xc in [X_TOKEN, X_PARA, X_FEAT, X_GEN]:
    draw_line(xc, Y_RAIL1, xc, Y_LVL1 + LVL1_H / 2, lw=2.2)

# ---------------------------------------------------------------------------
# LEVEL 1 CATEGORY BOXES
# ---------------------------------------------------------------------------
categories = [
    (X_TOKEN, u"Transformaciones\na Nivel de Token", C_TOKEN, B_TOKEN),
    (X_PARA,  u"Par\u00e1frasis",                    C_PARA,  B_PARA),
    (X_FEAT,  u"Perturbaciones en\nEspacio de Caract.", C_FEAT, B_FEAT),
    (X_GEN,   u"Generaci\u00f3n\nCompleta",          C_GEN,   B_GEN),
]

for cx, label, fill, border in categories:
    draw_box(cx, Y_LVL1, LVL1_W, LVL1_H, fill, border, lw=2.0)
    ax.text(cx, Y_LVL1, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=border, zorder=5,
            linespacing=1.25)

# ---------------------------------------------------------------------------
# LEAF NODES
# ---------------------------------------------------------------------------
leaves_data = [
    ("token", X_TOKEN, [
        ("EDA", "baseline"),
        (u"Sustituci\u00f3n de\nSin\u00f3nimos", "gray"),
    ]),
    ("para", X_PARA, [
        (u"Traducci\u00f3n\nInversa", "baseline"),
        (u"Par\u00e1frasis T5", "baseline"),
    ]),
    ("feat", X_FEAT, [
        ("SMOTE", "baseline_bold"),
        ("ADASYN", "gray"),
        ("Mixup", "baseline"),
    ]),
    ("gen", X_GEN, [
        (u"Generaci\u00f3n\ncon LLM", "baseline"),
        (u"Aumentaci\u00f3n\nContextual BERT", "baseline"),
    ]),
]

for key, cat_x, children in leaves_data:
    positions = LEAF_X[key]

    # Drop from category to rail
    draw_line(cat_x, Y_LVL1 - LVL1_H / 2, cat_x, Y_RAIL2, lw=1.6)

    # Horizontal rail
    if len(positions) > 1:
        draw_line(positions[0], Y_RAIL2, positions[-1], Y_RAIL2, lw=1.6)

    for i, (label, style) in enumerate(children):
        cx = positions[i]
        fill, edge, lw_val, fw = leaf_style(style)
        draw_box(cx, Y_LVL2, LEAF_W, LEAF_H, fill, edge, lw=lw_val)
        ax.text(cx, Y_LVL2, label, ha="center", va="center",
                fontsize=8.8, fontweight=fw, color="#263238", zorder=5,
                linespacing=1.1)
        draw_line(cx, Y_RAIL2, cx, Y_LVL2 + LEAF_H / 2, lw=1.6)

# ---------------------------------------------------------------------------
# HYBRID CATEGORY BOX
# ---------------------------------------------------------------------------
draw_box(X_HYBRID, Y_HYBRID, HYBRID_W, LVL1_H, C_HYBRID, B_HYBRID, lw=2.0)
ax.text(X_HYBRID, Y_HYBRID,
        u"M\u00e9todos\nH\u00edbridos",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=B_HYBRID, zorder=5,
        linespacing=1.25)

# Dashed connections from Feature and Generation leaf rows to Hybrid box
feat_bottom = Y_LVL2 - LEAF_H / 2
gen_bottom  = Y_LVL2 - LEAF_H / 2
hybrid_top  = Y_HYBRID + LVL1_H / 2

# From the Feature leaves area (middle leaf = X_FEAT)
draw_line(X_FEAT, feat_bottom, X_HYBRID - 0.5, hybrid_top,
          color=B_HYBRID, lw=1.5, linestyle=(0, (5, 3)))

# From the Generation leaves area (middle = X_GEN)
draw_line(X_GEN, gen_bottom, X_HYBRID + 0.5, hybrid_top,
          color=B_HYBRID, lw=1.5, linestyle=(0, (5, 3)))

# ---------------------------------------------------------------------------
# HYBRID LEAF NODES
# ---------------------------------------------------------------------------
hybrid_children = [
    ("SMOTExT", "gray"),
    ("ImbLLM", "gray"),
    ("Esta Tesis", "gold"),
]

h_n = len(hybrid_children)
h_spacing = 2.8
h_total = h_spacing * (h_n - 1)
h_x_start = X_HYBRID - h_total / 2

draw_line(X_HYBRID, Y_HYBRID - LVL1_H / 2, X_HYBRID, Y_RAIL3, lw=1.6)

h_xs = [h_x_start + i * h_spacing for i in range(h_n)]

if h_n > 1:
    draw_line(h_xs[0], Y_RAIL3, h_xs[-1], Y_RAIL3, lw=1.6)

for i, (label, style) in enumerate(hybrid_children):
    cx = h_xs[i]
    draw_line(cx, Y_RAIL3, cx, Y_LVL3 + LEAF_H / 2, lw=1.6)

    if style == "gold":
        # Outer glow
        draw_box(cx, Y_LVL3, LEAF_W + 0.22, LEAF_H + 0.22,
                 "#FFF9C4", LEAF_GOLD_EDGE, lw=1.0, zorder=2)
        # Main box
        draw_box(cx, Y_LVL3, LEAF_W, LEAF_H, LEAF_GOLD_FILL,
                 LEAF_GOLD_EDGE, lw=3.5)
        # Star
        ax.text(cx - LEAF_W / 2 + 0.20, Y_LVL3 + 0.01,
                "\u2605", ha="center", va="center", fontsize=13,
                color="#F57F17", zorder=6)
        ax.text(cx + 0.10, Y_LVL3, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#E65100", zorder=5)
    else:
        fill, edge, lw_val, fw = leaf_style(style)
        draw_box(cx, Y_LVL3, LEAF_W, LEAF_H, fill, edge, lw=lw_val)
        ax.text(cx, Y_LVL3, label, ha="center", va="center",
                fontsize=8.8, fontweight=fw, color="#263238", zorder=5)

# ---------------------------------------------------------------------------
# LEGEND
# ---------------------------------------------------------------------------
legend_y = 2.0
legend_x0 = 3.2
legend_sp = 5.8

legend_items = [
    (u"Evaluado como l\u00ednea base\nen esta investigaci\u00f3n",
     LEAF_BASELINE_FILL, LEAF_BASELINE_EDGE, 2.0),
    (u"Referenciado en\nla literatura",
     LEAF_GRAY_FILL, LEAF_GRAY_EDGE, 1.4),
    (u"M\u00e9todo propuesto",
     LEAF_GOLD_FILL, LEAF_GOLD_EDGE, 3.0),
]

legend_border = FancyBboxPatch(
    (0.6, legend_y - 0.7), FIG_W - 1.2, 1.4,
    boxstyle="round,pad=0.15",
    facecolor="white", edgecolor="#B0BEC5", linewidth=1.0,
    zorder=1, alpha=0.5,
)
ax.add_patch(legend_border)

ax.text(legend_x0 - 1.7, legend_y, "Leyenda:", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#263238", zorder=5)

for i, (label, fill, edge, lw_item) in enumerate(legend_items):
    cx = legend_x0 + i * legend_sp
    sw, sh = 1.0, 0.45
    draw_box(cx, legend_y, sw, sh, fill, edge, lw=lw_item)
    ax.text(cx + sw / 2 + 0.20, legend_y, label, ha="left", va="center",
            fontsize=9, color="#37474F", zorder=5, linespacing=1.15)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_suffix(".png")
fig.savefig(out_path, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved  {out_path}")
