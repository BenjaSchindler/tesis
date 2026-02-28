#!/usr/bin/env python3
"""
Figure 6.g - Resumen de Validacion de Hipotesis
Traffic-light style hypothesis validation summary diagram for thesis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
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
# Data
# ---------------------------------------------------------------------------
hypotheses = [
    {
        "id": "H1",
        "desc": u"Hip\u00f3tesis Principal: Filtrado geom\u00e9trico supera a SMOTE",
        "evidence": "+2.25pp, p<0.0001, d=0.74, WR=83.8%",
        "status": "CONFIRMADA",
        "color": "green",
    },
    {
        "id": "H1a",
        "desc": u"Filtrado binario mejora calidad",
        "evidence": "+2.08pp, p<0.0001, d=0.65, WR=80.0%",
        "status": "CONFIRMADA",
        "color": "green",
    },
    {
        "id": "H1b",
        "desc": u"Ponderaci\u00f3n suave \u2265 filtrado binario",
        "evidence": u"+0.13pp (principal) / +3.66pp (optimizado)",
        "status": "PARCIAL",
        "color": "yellow",
    },
    {
        "id": "H1c",
        "desc": u"Generalizaci\u00f3n a NER sin modificaci\u00f3n",
        "evidence": "+9.26pp, WR=100%",
        "status": "CONFIRMADA",
        "color": "green",
    },
    {
        "id": "H1d",
        "desc": u"Clasificador modula beneficio",
        "evidence": "Lineales > Forest, p<0.05 para 4/5",
        "status": "CONFIRMADA",
        "color": "green",
    },
]

# ---------------------------------------------------------------------------
# Colour definitions
# ---------------------------------------------------------------------------
GREEN_LIGHT = "#4CAF50"
GREEN_DARK  = "#2E7D32"
GREEN_BG    = "#E8F5E9"

YELLOW_LIGHT = "#FFC107"
YELLOW_DARK  = "#F9A825"
YELLOW_BG    = "#FFFDE7"

ROW_ALT_A = "#FAFAFA"
ROW_ALT_B = "#FFFFFF"

TEXT_COLOR = "#333333"
TEXT_SECONDARY = "#555555"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 7.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
ROW_HEIGHT = 1.10
ROW_GAP = 0.12
TOTAL_ROWS = len(hypotheses)
TOP_Y = 5.8                       # top of the first row
ROW_X = 0.3                       # left margin
ROW_W = 13.4                      # row width

# Column positions (x coordinates)
COL_LIGHT_X   = 1.05              # traffic light circle centre
COL_ID_X      = 2.0               # hypothesis ID text
COL_DESC_X    = 2.7               # description text
COL_EVID_X    = 9.0               # evidence text
COL_STATUS_X  = 12.8              # status label

# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------
header_y = TOP_Y + ROW_HEIGHT + ROW_GAP + 0.05
header_box = FancyBboxPatch(
    (ROW_X, header_y - 0.10), ROW_W, 0.65,
    boxstyle="round,pad=0.12",
    facecolor="#37474F", edgecolor="#263238",
    linewidth=1.5, zorder=3,
)
ax.add_patch(header_box)

header_style = dict(ha="left", va="center", fontsize=10.5,
                    fontweight="bold", color="white", zorder=5)
ax.text(COL_LIGHT_X - 0.35, header_y + 0.22, "Estado", **header_style)
ax.text(COL_ID_X,           header_y + 0.22, "ID",     **header_style)
ax.text(COL_DESC_X,         header_y + 0.22, u"Descripci\u00f3n", **header_style)
ax.text(COL_EVID_X,         header_y + 0.22, "Evidencia",  **header_style)
ax.text(COL_STATUS_X - 0.3, header_y + 0.22, "Resultado",  **header_style)

# ---------------------------------------------------------------------------
# Draw each hypothesis row
# ---------------------------------------------------------------------------
for i, h in enumerate(hypotheses):
    y_bottom = TOP_Y - i * (ROW_HEIGHT + ROW_GAP)
    y_center = y_bottom + ROW_HEIGHT / 2

    # Alternating background
    bg_color = ROW_ALT_A if i % 2 == 0 else ROW_ALT_B
    row_box = FancyBboxPatch(
        (ROW_X, y_bottom), ROW_W, ROW_HEIGHT,
        boxstyle="round,pad=0.10",
        facecolor=bg_color, edgecolor="#E0E0E0",
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(row_box)

    # --- Traffic light circle ---
    if h["color"] == "green":
        circle_face = GREEN_LIGHT
        circle_edge = GREEN_DARK
    else:
        circle_face = YELLOW_LIGHT
        circle_edge = YELLOW_DARK

    circle = plt.Circle(
        (COL_LIGHT_X, y_center), 0.28,
        facecolor=circle_face, edgecolor=circle_edge,
        linewidth=2.2, zorder=4,
    )
    ax.add_patch(circle)

    # Inner highlight (gloss effect)
    highlight = plt.Circle(
        (COL_LIGHT_X - 0.07, y_center + 0.08), 0.10,
        facecolor="white", edgecolor="none", alpha=0.35, zorder=5,
    )
    ax.add_patch(highlight)

    # --- Hypothesis ID ---
    ax.text(COL_ID_X, y_center, h["id"],
            ha="left", va="center", fontsize=12, fontweight="bold",
            color=TEXT_COLOR, zorder=5)

    # --- Description ---
    ax.text(COL_DESC_X, y_center, h["desc"],
            ha="left", va="center", fontsize=10,
            color=TEXT_COLOR, zorder=5)

    # --- Evidence metric ---
    ax.text(COL_EVID_X, y_center, h["evidence"],
            ha="left", va="center", fontsize=9.5,
            fontfamily="monospace", fontweight="bold",
            color=TEXT_SECONDARY, zorder=5)

    # --- Status text ---
    if h["color"] == "green":
        status_color = GREEN_DARK
        status_bg    = GREEN_BG
    else:
        status_color = YELLOW_DARK
        status_bg    = YELLOW_BG

    # Status pill background
    status_text = h["status"]
    pill_w = 2.0
    pill_h = 0.42
    pill = FancyBboxPatch(
        (COL_STATUS_X - pill_w / 2, y_center - pill_h / 2),
        pill_w, pill_h,
        boxstyle="round,pad=0.10",
        facecolor=status_bg, edgecolor=status_color,
        linewidth=1.4, zorder=4,
    )
    ax.add_patch(pill)

    ax.text(COL_STATUS_X, y_center, status_text,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=status_color, zorder=5)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(7.0, 7.15, u"Resumen de Validaci\u00f3n de Hip\u00f3tesis",
        ha="center", va="center", fontsize=17, fontweight="bold",
        color=TEXT_COLOR, zorder=10)

# Subtitle / decorative line
ax.plot([1.5, 12.5], [6.90, 6.90], color="#BDBDBD", linewidth=1.0, zorder=3)

# ---------------------------------------------------------------------------
# Legend at bottom
# ---------------------------------------------------------------------------
legend_y = -0.15
ax.add_patch(plt.Circle((3.5, legend_y), 0.18,
             facecolor=GREEN_LIGHT, edgecolor=GREEN_DARK,
             linewidth=1.5, zorder=4))
ax.text(3.85, legend_y, "= Confirmada", ha="left", va="center",
        fontsize=9.5, color=TEXT_SECONDARY, zorder=5)

ax.add_patch(plt.Circle((6.5, legend_y), 0.18,
             facecolor=YELLOW_LIGHT, edgecolor=YELLOW_DARK,
             linewidth=1.5, zorder=4))
ax.text(6.85, legend_y, "= Parcialmente confirmada", ha="left", va="center",
        fontsize=9.5, color=TEXT_SECONDARY, zorder=5)

# ---------------------------------------------------------------------------
# Summary box (bottom right)
# ---------------------------------------------------------------------------
summary_x = 11.5
summary_y = -0.15
summary_box = FancyBboxPatch(
    (summary_x - 0.15, summary_y - 0.35), 2.55, 0.70,
    boxstyle="round,pad=0.12",
    facecolor=GREEN_BG, edgecolor=GREEN_DARK,
    linewidth=1.2, zorder=4,
)
ax.add_patch(summary_box)
ax.text(summary_x + 1.12, summary_y, "4/5 confirmadas, 1 parcial",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color=GREEN_DARK, zorder=5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib/fig_6_g_hypothesis_validation.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
