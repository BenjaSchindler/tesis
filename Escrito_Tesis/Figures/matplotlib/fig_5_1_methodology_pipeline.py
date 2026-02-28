#!/usr/bin/env python3
"""
Figure 5.1 - Pipeline Metodologico Completo
Publication-quality methodology pipeline diagram for thesis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style setup  (matches other thesis figures)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "blue":   {"bg": "#E3F2FD", "border": "#1565C0", "accent": "#1E88E5"},
    "green":  {"bg": "#E8F5E9", "border": "#2E7D32", "accent": "#43A047"},
    "purple": {"bg": "#F3E5F5", "border": "#6A1B9A", "accent": "#8E24AA"},
    "orange": {"bg": "#FFF3E0", "border": "#E65100", "accent": "#FB8C00"},
    "yellow": {"bg": "#FFFDE7", "border": "#F9A825", "accent": "#FDD835"},
    "pink":   {"bg": "#FCE4EC", "border": "#AD1457", "accent": "#D81B60"},
}
TEXT_COLOR = "#333333"
ARROW_COLOR = "#555555"

# ---------------------------------------------------------------------------
# Figure & axes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-2.5, 8.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------------------
# Helper: draw a rounded box
# ---------------------------------------------------------------------------
def draw_box(x, y, w, h, color_key, linewidth=1.8, zorder=2):
    c = COLORS[color_key]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=c["bg"],
        edgecolor=c["border"],
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box

# ---------------------------------------------------------------------------
# Helper: draw an arrow between two (x, y) points
# ---------------------------------------------------------------------------
def draw_arrow(xy_a, xy_b, label="", color=ARROW_COLOR, style="-|>",
               linestyle="-", linewidth=1.8, label_offset=(0, 0.18),
               fontsize=8.5, zorder=3, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        xy_a, xy_b,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=16,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        mx = (xy_a[0] + xy_b[0]) / 2 + label_offset[0]
        my = (xy_a[1] + xy_b[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=fontsize, color=TEXT_COLOR, style="italic",
                zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.85))

# ---------------------------------------------------------------------------
# Stage geometry  (x, y_bottom, width, height)
# ---------------------------------------------------------------------------
BOX_H = 4.0
BOX_W = 2.3
GAP = 0.55          # horizontal gap between boxes
Y_BASE = 2.0        # baseline y for all boxes

stage_specs = [
    # (x,          y,      w,     h,     colour_key)
    (0.0,         Y_BASE, BOX_W, BOX_H, "blue"),     # Stage 1
    (3.05,        Y_BASE, BOX_W, BOX_H, "green"),    # Stage 2
    (6.10,        Y_BASE, BOX_W, BOX_H, "purple"),   # Stage 3
    (9.15,        Y_BASE, 2.7,   BOX_H + 0.5, "orange"),  # Stage 4 (bigger)
    (12.55,       Y_BASE, BOX_W, BOX_H, "yellow"),   # Stage 5
    (15.60,       Y_BASE, BOX_W, BOX_H, "pink"),     # Stage 6
]

boxes = []
for spec in stage_specs:
    boxes.append(draw_box(*spec[:4], spec[4],
                           linewidth=2.4 if spec[4] == "orange" else 1.8))

# ---------------------------------------------------------------------------
# Text inside each box
# ---------------------------------------------------------------------------
def txt(x, y, s, **kw):
    defaults = dict(ha="center", va="center", color=TEXT_COLOR,
                    fontsize=9, zorder=5)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)

# ---- Stage 1: Datos Etiquetados -------------------------------------------
cx1 = 0.0 + BOX_W / 2
txt(cx1, Y_BASE + BOX_H - 0.45, "Datos Etiquetados",
    fontsize=11, fontweight="bold")
txt(cx1, Y_BASE + BOX_H - 0.95, "n = 10, 25, 50\npor clase",
    fontsize=8.5, color="#555")
# Mini data-table icon
tbl_x, tbl_y = cx1 - 0.45, Y_BASE + 0.75
for row in range(4):
    for col in range(3):
        rect = plt.Rectangle(
            (tbl_x + col * 0.30, tbl_y + row * 0.25), 0.28, 0.22,
            linewidth=0.6, edgecolor="#90CAF9",
            facecolor="#BBDEFB" if row == 0 else "#E3F2FD", zorder=5)
        ax.add_patch(rect)
txt(cx1, Y_BASE + 0.25, "7 datasets + 3 NER", fontsize=7.5,
    color=COLORS["blue"]["border"], style="italic")

# ---- Stage 2: Generacion LLM ----------------------------------------------
cx2 = 3.05 + BOX_W / 2
txt(cx2, Y_BASE + BOX_H - 0.45, "Generacion LLM",
    fontsize=11, fontweight="bold")
txt(cx2, Y_BASE + BOX_H - 0.95, "Gemini 2.0 Flash", fontsize=9.5,
    fontweight="semibold", color=COLORS["green"]["border"])
txt(cx2, Y_BASE + BOX_H - 1.35, "In-context learning", fontsize=8.5,
    color="#555")
# Stylised prompt icon
prompt_lines = ["|>>>  prompt  |", "|  + ejemplos |", "|  -> salida  |"]
for i, line in enumerate(prompt_lines):
    txt(cx2, Y_BASE + 1.55 - i * 0.32, line, fontsize=7,
        fontfamily="monospace", color=COLORS["green"]["accent"])
txt(cx2, Y_BASE + 0.25, "3x candidatos", fontsize=7.5,
    color=COLORS["green"]["border"], style="italic")

# ---- Stage 3: Codificacion ------------------------------------------------
cx3 = 6.10 + BOX_W / 2
txt(cx3, Y_BASE + BOX_H - 0.45, "Codificacion",
    fontsize=11, fontweight="bold")
txt(cx3, Y_BASE + BOX_H - 0.95, "all-mpnet-base-v2", fontsize=8.5,
    fontweight="semibold", color=COLORS["purple"]["border"])
txt(cx3, Y_BASE + BOX_H - 1.35, "SentenceTransformers", fontsize=8,
    color="#555")
# Embedding vector visualisation
np.random.seed(42)
bar_x = np.linspace(cx3 - 0.7, cx3 + 0.7, 16)
bar_h = np.random.uniform(0.15, 0.65, 16)
for bx, bh in zip(bar_x, bar_h):
    ax.bar(bx, bh, width=0.07, bottom=Y_BASE + 0.8, color="#CE93D8",
           edgecolor="#AB47BC", linewidth=0.4, zorder=5)
txt(cx3, Y_BASE + 0.25, "768 dimensiones", fontsize=7.5,
    color=COLORS["purple"]["border"], style="italic")

# ---- Stage 4: Filtrado Geometrico (core contribution) ---------------------
cx4 = 9.15 + 2.7 / 2
y4_top = Y_BASE + BOX_H + 0.5
txt(cx4, y4_top - 0.40, "Filtrado Geometrico",
    fontsize=12, fontweight="bold")
txt(cx4, y4_top - 0.80, "Contribucion principal",
    fontsize=8, color=COLORS["orange"]["border"], style="italic")

methods = [
    ("LOF (densidad local)", False),
    ("Cascada (distancia)", True),     # best performer
    ("Combinado (LOF + coseno)", False),
    ("Muestreador guiado", False),
    ("Sin filtro (control)", False),
]
for i, (name, star) in enumerate(methods):
    yy = y4_top - 1.35 - i * 0.55
    label = name + ("  \u2605" if star else "")
    fw = "bold" if star else "normal"
    fc = COLORS["orange"]["accent"] if star else TEXT_COLOR
    bg = "#FFE0B2" if star else COLORS["orange"]["bg"]
    # small rounded pill for each method
    pill = FancyBboxPatch(
        (cx4 - 1.15, yy - 0.18), 2.30, 0.36,
        boxstyle="round,pad=0.08", facecolor=bg,
        edgecolor=COLORS["orange"]["border"] if star else "#FFCC80",
        linewidth=1.2 if star else 0.7, zorder=5)
    ax.add_patch(pill)
    txt(cx4, yy, label, fontsize=8, fontweight=fw, color=fc)

# ---- Stage 5: Ponderacion Suave -------------------------------------------
cx5 = 12.55 + BOX_W / 2
txt(cx5, Y_BASE + BOX_H - 0.45, "Ponderacion Suave",
    fontsize=11, fontweight="bold")
txt(cx5, Y_BASE + BOX_H - 1.0,
    r"$w(x) = s^{\,1/T}$", fontsize=12, color=TEXT_COLOR)
txt(cx5, Y_BASE + BOX_H - 1.40, "T = 0.5", fontsize=9,
    color=COLORS["yellow"]["border"], fontweight="semibold")
txt(cx5, Y_BASE + BOX_H - 1.75, "Pesos de muestra", fontsize=8, color="#555")

# Gradient bar visualisation
grad_y = Y_BASE + 1.0
n_seg = 20
for j in range(n_seg):
    frac = j / (n_seg - 1)
    c_val = plt.cm.YlOrRd(0.15 + 0.65 * frac)
    seg_h = 0.15 + 0.45 * frac
    ax.bar(cx5 - 0.65 + j * 0.065, seg_h, width=0.058, bottom=grad_y,
           color=c_val, edgecolor="none", zorder=5)
txt(cx5 - 0.65, grad_y - 0.20, "bajo", fontsize=6.5, ha="left", color="#999")
txt(cx5 + 0.65, grad_y - 0.20, "alto", fontsize=6.5, ha="right", color="#999")
txt(cx5, Y_BASE + 0.25, "Pesos de muestra", fontsize=7.5,
    color=COLORS["yellow"]["border"], style="italic")

# ---- Stage 6: Clasificador ------------------------------------------------
cx6 = 15.60 + BOX_W / 2
txt(cx6, Y_BASE + BOX_H - 0.45, "Clasificador",
    fontsize=11, fontweight="bold")

classifiers = [
    "Regresion Logistica",
    "SVM Lineal",
    "Ridge",
    "Random Forest",
    "MLP",
]
for i, clf in enumerate(classifiers):
    yy = Y_BASE + BOX_H - 1.15 - i * 0.50
    pill = FancyBboxPatch(
        (cx6 - 0.95, yy - 0.17), 1.90, 0.34,
        boxstyle="round,pad=0.06", facecolor="#F8BBD0",
        edgecolor="#F06292", linewidth=0.6, zorder=5)
    ax.add_patch(pill)
    txt(cx6, yy, clf, fontsize=7.5)

txt(cx6, Y_BASE + 0.25, "5 clasificadores", fontsize=7.5,
    color=COLORS["pink"]["border"], style="italic")

# ---------------------------------------------------------------------------
# Arrows between stages
# ---------------------------------------------------------------------------
arr_y = Y_BASE + BOX_H / 2  # vertical centre of standard boxes

# S1 -> S2
draw_arrow((0.0 + BOX_W + 0.05, arr_y), (3.05 - 0.05, arr_y),
           label="Prompt +\nejemplos", fontsize=7.5, label_offset=(0, 0.25))
# S2 -> S3
draw_arrow((3.05 + BOX_W + 0.05, arr_y), (6.10 - 0.05, arr_y),
           label="Textos\ngenerados", fontsize=7.5, label_offset=(0, 0.25))
# S3 -> S4
draw_arrow((6.10 + BOX_W + 0.05, arr_y + 0.25), (9.15 - 0.05, arr_y + 0.25),
           label="Embeddings", fontsize=7.5, label_offset=(0, 0.25))
# S4 -> S5  (accepted, green)
draw_arrow((9.15 + 2.7 + 0.05, arr_y + 0.25), (12.55 - 0.05, arr_y + 0.25),
           label="Aceptados", fontsize=7.5, color="#2E7D32",
           label_offset=(0, 0.25))
# S5 -> S6
draw_arrow((12.55 + BOX_W + 0.05, arr_y), (15.60 - 0.05, arr_y),
           label="Datos\nponderados", fontsize=7.5, label_offset=(0, 0.25))

# S4 rejected arrow (downward, red dashed)
draw_arrow((cx4, Y_BASE - 0.05), (cx4, Y_BASE - 1.2),
           label="Rechazados", color="#C62828", linestyle="--",
           label_offset=(0.75, 0.15), fontsize=7.5)
# Small "waste" icon at the bottom of the rejected arrow
ax.text(cx4, Y_BASE - 1.6, "\u2718", fontsize=14, ha="center", va="center",
        color="#C62828", zorder=5)

# ---------------------------------------------------------------------------
# Output arrow from Stage 6
# ---------------------------------------------------------------------------
draw_arrow((15.60 + BOX_W + 0.05, arr_y), (18.0, arr_y),
           label="", color=COLORS["pink"]["border"], linewidth=2.0)
# macro-F1 label
ax.text(18.15, arr_y, "macro\nF1", fontsize=10, ha="left", va="center",
        fontweight="bold", color=COLORS["pink"]["border"], zorder=5)

# ---------------------------------------------------------------------------
# Bracket below stages 4-6: baseline comparison
# ---------------------------------------------------------------------------
bkt_y = Y_BASE - 0.55
bkt_x_left = 9.15
bkt_x_right = 15.60 + BOX_W

# Horizontal line
ax.plot([bkt_x_left, bkt_x_right], [bkt_y, bkt_y],
        color="#666", linewidth=1.2, zorder=4)
# Vertical ticks
for xx in [bkt_x_left, bkt_x_right]:
    ax.plot([xx, xx], [bkt_y, bkt_y + 0.25], color="#666",
            linewidth=1.2, zorder=4)
# Label
ax.text((bkt_x_left + bkt_x_right) / 2, bkt_y - 0.30,
        "Comparacion con 9 lineas base (SMOTE, EDA, BackTranslation, ...)",
        ha="center", va="top", fontsize=8.5, color="#444", style="italic",
        zorder=5)

# ---------------------------------------------------------------------------
# Results callout box
# ---------------------------------------------------------------------------
res_x, res_y = 16.2, Y_BASE + BOX_H + 1.2
res_box = FancyBboxPatch(
    (res_x - 1.15, res_y - 0.55), 2.80, 1.15,
    boxstyle="round,pad=0.18",
    facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2.0, zorder=6)
ax.add_patch(res_box)
ax.text(res_x + 0.25, res_y + 0.28, "+2.25 pp vs SMOTE",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#1B5E20", zorder=7)
ax.text(res_x + 0.25, res_y - 0.15, "(p < 0.0001)",
        ha="center", va="center", fontsize=8.5, color="#2E7D32", zorder=7)
# Small arrow from callout to stage 6
draw_arrow((res_x - 1.15, res_y - 0.10), (cx6 + 0.6, Y_BASE + BOX_H + 0.15),
           color="#2E7D32", linewidth=1.0, style="-|>",
           connectionstyle="arc3,rad=-0.2")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(9.0, 7.85, "Pipeline Metodologico Completo",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=TEXT_COLOR, zorder=10,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib/fig_5_1_methodology_pipeline.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
