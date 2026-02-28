"""
fig_7_1_contributions_future.py
Mind map diagram: thesis contributions radiating above centre,
future work directions below centre.
"""

import matplotlib
matplotlib.use("Agg")
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
# Layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 18, 14
CX, CY = FIG_W / 2, FIG_H / 2 + 0.3   # centre of the radial layout

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(False)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
# Centre
C_CENTER_BG   = "#1A237E"
C_CENTER_EDGE = "#0D1642"

# Contributions - methodology (C1-C4, green/teal)
C_METHOD_BG   = "#E0F2F1"
C_METHOD_EDGE = "#00695C"
C_METHOD_TXT  = "#004D40"

# Contributions - findings (C5-C8, blue)
C_FIND_BG     = "#E3F2FD"
C_FIND_EDGE   = "#1565C0"
C_FIND_TXT    = "#0D47A1"

# Future directions
C_FUTURE_BG   = "#FFF3E0"
C_FUTURE_EDGE = "#E65100"
C_FUTURE_TXT  = "#BF360C"

# Lines
C_LINE_SOLID  = "#37474F"
C_LINE_DASH   = "#8D6E63"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_box(cx, cy, w, h, facecolor, edgecolor, lw=1.8, zorder=3,
             alpha=1.0, pad=0.15):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(box)
    return box


def draw_line(x1, y1, x2, y2, color=C_LINE_SOLID, lw=2.0,
              linestyle="solid", zorder=2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
            linestyle=linestyle, zorder=zorder, solid_capstyle="round")


def draw_arrow_dashed(x1, y1, x2, y2, color=C_LINE_DASH, lw=1.8, zorder=2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        lw=lw, color=color, zorder=zorder,
        linestyle="dashed",
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)

# ---------------------------------------------------------------------------
# Centre node
# ---------------------------------------------------------------------------
center_w, center_h = 4.8, 1.6
draw_box(CX, CY, center_w, center_h, C_CENTER_BG, C_CENTER_EDGE,
         lw=3.0, zorder=5, pad=0.25)
ax.text(CX, CY + 0.22, "Filtrado Geom\u00e9trico", ha="center", va="center",
        fontsize=17, fontweight="bold", color="white", zorder=6)
ax.text(CX, CY - 0.32, "de Muestras LLM", ha="center", va="center",
        fontsize=17, fontweight="bold", color="white", zorder=6)

# ---------------------------------------------------------------------------
# Contributions data
# ---------------------------------------------------------------------------
contributions = [
    ("C1", "Sistema de filtrado\ngeom\u00e9trico",       "+2.25pp vs SMOTE",        "method"),
    ("C2", "Ponderaci\u00f3n suave",                     "Scores \u2192 pesos\nde muestra", "method"),
    ("C3", "Validaci\u00f3n\nmulti-dataset",             "7 dominios,\n10+ datasets",      "method"),
    ("C4", "Generalizaci\u00f3n\na NER",                 "+9.26pp, WR=100%",        "method"),
    ("C5", "Simplicidad >\ncomplejidad",                 "Nivel 1 > Nivel 4",       "finding"),
    ("C6", "Beneficio en\nclases dif\u00edciles",        "+10.44pp (F1<30%)",       "finding"),
    ("C7", "Curriculum\nlearning",                       "50% candidatos\nes \u00f3ptimo", "finding"),
    ("C8", "Filtrado uniforme\n> adaptativo",            "Uniforme +2.33\nvs h\u00edbrido +0.77", "finding"),
]

# Semicircle above centre: angles from 160 to 20 deg (left to right)
n_contrib = len(contributions)
angles_deg = np.linspace(162, 18, n_contrib)
radius = 4.6
contrib_box_w, contrib_box_h = 2.75, 2.0

for i, (label, title, metric, ctype) in enumerate(contributions):
    angle = np.radians(angles_deg[i])
    bx = CX + radius * np.cos(angle)
    by = CY + radius * np.sin(angle)

    # Pick colours
    if ctype == "method":
        bg, edge, txt = C_METHOD_BG, C_METHOD_EDGE, C_METHOD_TXT
        label_bg = "#00897B"
    else:
        bg, edge, txt = C_FIND_BG, C_FIND_EDGE, C_FIND_TXT
        label_bg = "#1976D2"

    # Line from centre to box
    # Compute attachment points on center box edge and contribution box edge
    dx = bx - CX
    dy = by - CY
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist

    # Start point: from center box edge
    sx = CX + ux * (center_w / 2 + 0.15)
    sy = CY + uy * (center_h / 2 + 0.15)

    # End point: towards contribution box centre, stop at box edge
    # Use a simple approximation
    ex = bx - ux * (contrib_box_w / 2 + 0.15)
    ey = by - uy * (contrib_box_h / 2 + 0.10)

    draw_line(sx, sy, ex, ey, color=edge, lw=2.0)

    # Draw contribution box
    draw_box(bx, by, contrib_box_w, contrib_box_h, bg, edge, lw=2.0)

    # Label badge (small pill at top-left of box)
    badge_x = bx - contrib_box_w / 2 + 0.45
    badge_y = by + contrib_box_h / 2 - 0.22
    badge = FancyBboxPatch(
        (badge_x - 0.32, badge_y - 0.17), 0.64, 0.34,
        boxstyle="round,pad=0.06",
        facecolor=label_bg, edgecolor="none", zorder=6,
    )
    ax.add_patch(badge)
    ax.text(badge_x, badge_y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=7)

    # Title text
    ax.text(bx, by + 0.18, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=txt,
            zorder=6, linespacing=1.15)

    # Metric text (smaller, italic)
    ax.text(bx, by - contrib_box_h / 2 + 0.38, metric, ha="center",
            va="center", fontsize=9, color="#546E7A", style="italic",
            zorder=6, linespacing=1.15)

# ---------------------------------------------------------------------------
# Section label for contributions
# ---------------------------------------------------------------------------
ax.text(CX, CY + radius + contrib_box_h / 2 + 0.75,
        "Contribuciones", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238",
        zorder=6, style="italic")

# small legend for the two contribution types
legend_y = CY + radius + contrib_box_h / 2 + 0.25
# Method
draw_box(CX - 2.2, legend_y, 1.6, 0.38, C_METHOD_BG, C_METHOD_EDGE,
         lw=1.2, zorder=5, pad=0.06)
ax.text(CX - 2.2, legend_y, "Metodol\u00f3gicas", ha="center", va="center",
        fontsize=9, color=C_METHOD_TXT, fontweight="bold", zorder=6)
# Finding
draw_box(CX + 2.2, legend_y, 1.6, 0.38, C_FIND_BG, C_FIND_EDGE,
         lw=1.2, zorder=5, pad=0.06)
ax.text(CX + 2.2, legend_y, "Hallazgos", ha="center", va="center",
        fontsize=9, color=C_FIND_TXT, fontweight="bold", zorder=6)

# ---------------------------------------------------------------------------
# Future directions data
# ---------------------------------------------------------------------------
futures = [
    ("F1", "Extensi\u00f3n\nmultilingue"),
    ("F2", "Comparaci\u00f3n\nde LLMs"),
    ("F3", "Optimizaci\u00f3n\nbayesiana"),
    ("F4", "Temperatura\nadaptativa\npor clase"),
    ("F5", "Presupuesto\nadaptativo"),
    ("F6", "Otras tareas\nNLP"),
]

n_future = len(futures)
future_box_w, future_box_h = 2.2, 1.55
future_y = CY - 4.6
future_spacing = (FIG_W - 2.0) / n_future
future_x0 = 1.0 + future_spacing / 2

for i, (label, title) in enumerate(futures):
    fx = future_x0 + i * future_spacing
    fy = future_y

    # Dashed arrow from centre down to box
    # Start from bottom of centre box
    s_x = CX + (fx - CX) * 0.3
    s_y = CY - center_h / 2 - 0.15
    e_y = fy + future_box_h / 2 + 0.20

    draw_arrow_dashed(s_x, s_y, fx, e_y, color=C_FUTURE_EDGE, lw=1.6)

    # Draw box
    draw_box(fx, fy, future_box_w, future_box_h, C_FUTURE_BG, C_FUTURE_EDGE,
             lw=1.8)

    # Label badge
    badge_x = fx
    badge_y = fy + future_box_h / 2 - 0.22
    badge = FancyBboxPatch(
        (badge_x - 0.28, badge_y - 0.16), 0.56, 0.32,
        boxstyle="round,pad=0.05",
        facecolor="#F4511E", edgecolor="none", zorder=6,
    )
    ax.add_patch(badge)
    ax.text(badge_x, badge_y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=7)

    # Title
    ax.text(fx, fy - 0.10, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_FUTURE_TXT,
            zorder=6, linespacing=1.2)

# ---------------------------------------------------------------------------
# Section label for future work
# ---------------------------------------------------------------------------
ax.text(CX, future_y - future_box_h / 2 - 0.65,
        "Direcciones de Trabajo Futuro", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238",
        zorder=6, style="italic")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(CX, FIG_H - 0.55,
        "Contribuciones y Direcciones de Trabajo Futuro",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="#1A237E", zorder=6)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).with_suffix(".png")
fig.savefig(out_path, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved  {out_path}")
