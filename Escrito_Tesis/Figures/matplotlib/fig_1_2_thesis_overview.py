"""
fig_1_2_thesis_overview.py
Hero figure for thesis introduction: horizontal pipeline overview.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
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
C_BLUE   = "#E3F2FD"
C_GREEN  = "#E8F5E9"
C_PURPLE = "#F3E5F5"
C_ORANGE = "#FFF3E0"
C_PINK   = "#FCE4EC"
C_RESULT = "#E8F5E9"

BORDER_BLUE   = "#1565C0"
BORDER_GREEN  = "#2E7D32"
BORDER_PURPLE = "#6A1B9A"
BORDER_ORANGE = "#E65100"
BORDER_PINK   = "#AD1457"
BORDER_RESULT = "#1B5E20"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 16, 5.4
BOX_W, BOX_H = 2.2, 2.6
Y_MID = 2.7          # vertical centre of the pipeline boxes
GAP   = 0.95         # horizontal gap between boxes

# x-centres of the five blocks
xs = []
x = 1.4
for _ in range(5):
    xs.append(x)
    x += BOX_W + GAP

# ---------------------------------------------------------------------------
# Figure & axes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
# remove the seaborn grid for this diagram
ax.grid(False)

# ---------------------------------------------------------------------------
# Helper: draw a rounded box
# ---------------------------------------------------------------------------
def draw_box(cx, cy, w, h, facecolor, edgecolor, lw=1.5, zorder=2):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box

# ---------------------------------------------------------------------------
# Helper: thick connecting arrow
# ---------------------------------------------------------------------------
def connect(x1, y1, x2, y2, color="#455A64", lw=2.2, style="-|>",
            mutation=25, zorder=1, linestyle="solid", shrinkA=0, shrinkB=0):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        lw=lw, color=color, zorder=zorder,
        connectionstyle="arc3,rad=0",
        linestyle=linestyle,
        shrinkA=shrinkA, shrinkB=shrinkB,
    )
    ax.add_patch(arrow)
    return arrow

# ---------------------------------------------------------------------------
# Block 1 - Datos Escasos
# ---------------------------------------------------------------------------
cx = xs[0]
draw_box(cx, Y_MID, BOX_W, BOX_H, C_BLUE, BORDER_BLUE)
ax.text(cx, Y_MID + 0.85, "Datos Escasos", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#0D47A1", zorder=5)

# 3 small circles representing few samples
circle_colors = ["#42A5F5", "#1E88E5", "#0D47A1"]
offsets_x = [-0.45, 0.0, 0.45]
offsets_y = [0.05, 0.20, -0.10]
for dx, dy, cc in zip(offsets_x, offsets_y, circle_colors):
    circ = plt.Circle((cx + dx, Y_MID + dy), 0.16, color=cc,
                       ec="white", lw=1.2, zorder=5)
    ax.add_patch(circ)

ax.text(cx, Y_MID - 0.75, "10\u201350 / clase", ha="center", va="center",
        fontsize=10, color="#37474F", style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Block 2 - LLM
# ---------------------------------------------------------------------------
cx = xs[1]
draw_box(cx, Y_MID, BOX_W, BOX_H, C_GREEN, BORDER_GREEN)
ax.text(cx, Y_MID + 0.85, "Generaci\u00f3n", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#1B5E20", zorder=5)
ax.text(cx, Y_MID + 0.50, "con LLM", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#1B5E20", zorder=5)

# LLM icon: stacked rectangle (stylised document)
doc_w, doc_h = 0.7, 0.55
doc_cx, doc_cy = cx, Y_MID - 0.05
for i, (ddx, ddy) in enumerate([(0.12, 0.12), (0.0, 0.0)]):
    rect = FancyBboxPatch(
        (doc_cx - doc_w / 2 + ddx, doc_cy - doc_h / 2 + ddy),
        doc_w, doc_h, boxstyle="round,pad=0.04",
        facecolor="#A5D6A7" if i == 0 else "#66BB6A",
        edgecolor="#2E7D32", linewidth=1.0, zorder=4 + i,
    )
    ax.add_patch(rect)
# lines on top document
for ly in [0.15, 0.0, -0.15]:
    ax.plot([doc_cx - 0.22, doc_cx + 0.22],
            [doc_cy + ly, doc_cy + ly],
            lw=1.0, color="#1B5E20", zorder=6, alpha=0.5)

ax.text(cx, Y_MID - 0.75, "3\u00d7 candidatos", ha="center", va="center",
        fontsize=10, color="#37474F", style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Block 3 - Embeddings
# ---------------------------------------------------------------------------
cx = xs[2]
draw_box(cx, Y_MID, BOX_W, BOX_H, C_PURPLE, BORDER_PURPLE)
ax.text(cx, Y_MID + 0.85, "Espacio de", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#4A148C", zorder=5)
ax.text(cx, Y_MID + 0.50, "Embeddings", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#4A148C", zorder=5)

# mini scatter plot
rng = np.random.default_rng(42)
n_pts = 18
sx = rng.normal(cx, 0.35, n_pts)
sy = rng.normal(Y_MID - 0.05, 0.22, n_pts)
scatter_colors = ["#AB47BC", "#7B1FA2", "#CE93D8"]
for i in range(n_pts):
    ax.plot(sx[i], sy[i], "o", color=scatter_colors[i % 3],
            markersize=4.5, zorder=5, markeredgecolor="white",
            markeredgewidth=0.4)

ax.text(cx, Y_MID - 0.75, "\u211D\u2077\u2076\u2078", ha="center",
        va="center", fontsize=11, color="#37474F", style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Block 4 - Filtrado Geom\u00e9trico (KEY block)
# ---------------------------------------------------------------------------
cx = xs[3]
draw_box(cx, Y_MID, BOX_W, BOX_H, C_ORANGE, BORDER_ORANGE, lw=2.8)
ax.text(cx, Y_MID + 0.85, "Filtrado", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#BF360C", zorder=5)
ax.text(cx, Y_MID + 0.50, "Geom\u00e9trico", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#BF360C", zorder=5)

# Funnel shape
funnel_top_half = 0.55
funnel_bot_half = 0.18
funnel_top_y = Y_MID + 0.18
funnel_bot_y = Y_MID - 0.38
funnel_verts = [
    (cx - funnel_top_half, funnel_top_y),
    (cx + funnel_top_half, funnel_top_y),
    (cx + funnel_bot_half, funnel_bot_y),
    (cx - funnel_bot_half, funnel_bot_y),
]
funnel = plt.Polygon(funnel_verts, closed=True,
                      facecolor="#FFE0B2", edgecolor=BORDER_ORANGE,
                      linewidth=1.5, zorder=4)
ax.add_patch(funnel)

# dots entering funnel (many)
for ddx in np.linspace(-0.40, 0.40, 7):
    ax.plot(cx + ddx, funnel_top_y + 0.08, "o", color="#EF6C00",
            markersize=3.5, zorder=6)

# dots exiting funnel (few)
for ddx in [-0.07, 0.07]:
    ax.plot(cx + ddx, funnel_bot_y - 0.07, "o", color="#2E7D32",
            markersize=4.5, zorder=6)

ax.text(cx, Y_MID - 0.68, "Dist, LOF, Pond.", ha="center",
        va="center", fontsize=9.2, color="#37474F", style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Block 5 - Clasificador
# ---------------------------------------------------------------------------
cx = xs[4]
draw_box(cx, Y_MID, BOX_W, BOX_H, C_PINK, BORDER_PINK)
ax.text(cx, Y_MID + 0.85, "Clasificador", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#880E4F", zorder=5)
ax.text(cx, Y_MID + 0.50, "Supervisado", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#880E4F", zorder=5)

# bar-chart icon
bar_heights = [0.30, 0.50, 0.42, 0.55]
bar_w = 0.22
bar_x0 = cx - (len(bar_heights) * bar_w + (len(bar_heights) - 1) * 0.06) / 2
bar_base = Y_MID - 0.38
bar_colors = ["#F48FB1", "#EC407A", "#F06292", "#E91E63"]
for i, bh in enumerate(bar_heights):
    bx = bar_x0 + i * (bar_w + 0.06)
    rect = FancyBboxPatch(
        (bx, bar_base), bar_w, bh,
        boxstyle="round,pad=0.02",
        facecolor=bar_colors[i], edgecolor="#AD1457",
        linewidth=0.8, zorder=5,
    )
    ax.add_patch(rect)

ax.text(cx, Y_MID - 0.75, "macro F1", ha="center", va="center",
        fontsize=10, color="#37474F", style="italic", zorder=5)

# ---------------------------------------------------------------------------
# Connecting arrows between blocks
# ---------------------------------------------------------------------------
arrow_y = Y_MID

for i in range(4):
    x_start = xs[i] + BOX_W / 2 + 0.12
    x_end   = xs[i + 1] - BOX_W / 2 - 0.12

    if i == 0:
        # expansion arrow: thin start, wider head
        connect(x_start, arrow_y, x_end, arrow_y,
                color="#546E7A", lw=2.5, mutation=28)
    elif i == 2:
        # arrow into the filter block (slightly thicker to emphasise)
        connect(x_start, arrow_y, x_end, arrow_y,
                color="#546E7A", lw=3.0, mutation=30)
    else:
        connect(x_start, arrow_y, x_end, arrow_y,
                color="#546E7A", lw=2.5, mutation=28)

# ---------------------------------------------------------------------------
# Accepted / rejected paths from Block 4
# ---------------------------------------------------------------------------
b4_cx = xs[3]

# Accepted (green check) - going right toward classifier (already covered
# by the main arrow). Add a small label above that arrow.
ax.text((xs[3] + xs[4]) / 2, arrow_y + 0.35, "\u2713 aceptados",
        ha="center", va="center", fontsize=9.5, color="#2E7D32",
        fontweight="bold", zorder=5)

# Rejected (red cross) - downward dashed arrow
rej_y = Y_MID - BOX_H / 2 - 0.12
connect(b4_cx, rej_y, b4_cx, rej_y - 0.70,
        color="#C62828", lw=1.8, mutation=20, linestyle="dashed")
ax.text(b4_cx, rej_y - 0.95, "\u2717 rechazados", ha="center", va="center",
        fontsize=9.5, color="#C62828", fontweight="bold", zorder=5)

# ---------------------------------------------------------------------------
# Step numbers (small circled numbers above each box)
# ---------------------------------------------------------------------------
for i, cx_i in enumerate(xs, start=1):
    circ = plt.Circle((cx_i, Y_MID + BOX_H / 2 + 0.32), 0.18,
                        facecolor="#ECEFF1", edgecolor="#607D8B",
                        linewidth=1.0, zorder=5)
    ax.add_patch(circ)
    ax.text(cx_i, Y_MID + BOX_H / 2 + 0.32, str(i), ha="center",
            va="center", fontsize=10, fontweight="bold", color="#37474F",
            zorder=6)

# ---------------------------------------------------------------------------
# Result callout
# ---------------------------------------------------------------------------
res_w, res_h = 8.5, 0.62
res_cx = FIG_W / 2
res_cy = 0.52
res_box = FancyBboxPatch(
    (res_cx - res_w / 2, res_cy - res_h / 2), res_w, res_h,
    boxstyle="round,pad=0.15",
    facecolor=C_RESULT, edgecolor=BORDER_RESULT,
    linewidth=1.8, zorder=5,
)
ax.add_patch(res_box)
ax.text(res_cx, res_cy, "Resultado:  +2.25 pp sobre SMOTE   (p < 0.0001,  d = 0.74)",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color="#1B5E20", zorder=6)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_suffix(".png")
fig.savefig(out_path, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved  {out_path}")
