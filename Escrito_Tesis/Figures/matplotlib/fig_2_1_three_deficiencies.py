#!/usr/bin/env python3
"""
Figure 2.1 -- Tres Deficiencias de los Enfoques Existentes
===========================================================
Publication-quality 1x3 figure illustrating three key deficiencies
in existing LLM-based data augmentation approaches:
  (a) Ausencia de Validacion Post-Generacion
  (b) Ponderacion Uniforme
  (c) Validacion Limitada
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

# -- reproducibility ---------------------------------------------------------
np.random.seed(42)

# -- style -------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

# -- colour palette ----------------------------------------------------------
C_BLUE = "#3B7DD8"       # class A (blue cluster)
C_ORANGE = "#E07B39"     # class B (orange cluster)
C_SYNTH = "#888888"      # synthetic neutral
C_GOOD = "#2CA02C"       # green check / good
C_BAD = "#D62728"        # red / problem
C_WARN = "#D4AC0D"       # warning yellow
C_GRAY = "#AAAAAA"       # disabled / missing
C_LLM_BG = "#F5E6CC"     # LLM box background
C_DATASET = "#D5E8D4"    # dataset box fill
C_DATASET_Q = "#F8D7DA"  # dataset with question

# ============================================================================
#  FIGURE
# ============================================================================
fig = plt.figure(figsize=(18, 6.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

# ============================================================================
#  Panel (a): Ausencia de Validacion Post-Generacion
# ============================================================================
ax = ax_a
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a)  Ausencia de Validaci\u00f3n Post-Generaci\u00f3n",
             fontsize=12, fontweight="bold", pad=12)

# -- Class clusters ----------------------------------------------------------
# Blue cluster (class A) - lower left
blue_cx, blue_cy = 2.8, 3.2
blue_pts = np.random.multivariate_normal(
    [blue_cx, blue_cy], [[0.6, 0.1], [0.1, 0.5]], size=18
)
ax.scatter(blue_pts[:, 0], blue_pts[:, 1], c=C_BLUE, s=45, alpha=0.7,
           edgecolors="white", linewidths=0.4, zorder=3)
# Cluster ellipse
ell_a = mpatches.Ellipse((blue_cx, blue_cy), 4.2, 3.6, angle=-10,
                          fill=False, edgecolor=C_BLUE, linewidth=1.4,
                          linestyle="--", alpha=0.5)
ax.add_patch(ell_a)
ax.text(blue_cx, blue_cy - 2.3, "Clase A", fontsize=9, ha="center",
        color=C_BLUE, fontweight="bold")

# Orange cluster (class B) - right
orange_cx, orange_cy = 7.8, 3.5
orange_pts = np.random.multivariate_normal(
    [orange_cx, orange_cy], [[0.5, -0.1], [-0.1, 0.6]], size=18
)
ax.scatter(orange_pts[:, 0], orange_pts[:, 1], c=C_ORANGE, s=45, alpha=0.7,
           edgecolors="white", linewidths=0.4, zorder=3)
ell_b = mpatches.Ellipse((orange_cx, orange_cy), 3.8, 3.6, angle=10,
                          fill=False, edgecolor=C_ORANGE, linewidth=1.4,
                          linestyle="--", alpha=0.5)
ax.add_patch(ell_b)
ax.text(orange_cx, orange_cy - 2.3, "Clase B", fontsize=9, ha="center",
        color=C_ORANGE, fontweight="bold")

# -- LLM box at the top -----------------------------------------------------
llm_box = mpatches.FancyBboxPatch(
    (3.0, 9.0), 4.5, 1.8,
    boxstyle="round,pad=0.3", facecolor=C_LLM_BG,
    edgecolor="#8B7355", linewidth=1.5, zorder=5
)
ax.add_patch(llm_box)
ax.text(5.25, 10.15, "LLM", fontsize=13, ha="center", va="center",
        fontweight="bold", color="#5C4033", zorder=6)
ax.text(5.25, 9.45, "Generador", fontsize=9, ha="center", va="center",
        color="#5C4033", zorder=6)

# -- Synthetic samples (triangles) coming from LLM --------------------------
# Good ones: land in correct cluster
synth_good_A = np.array([[2.5, 3.8], [3.3, 3.0], [2.2, 2.8]])
synth_good_B = np.array([[8.0, 4.0], [7.5, 3.2]])

# Bad ones: in the gap or wrong cluster
synth_bad = np.array([
    [5.2, 3.5],    # between clusters
    [5.5, 2.2],    # between clusters
    [7.2, 3.8],    # wrong cluster for A
    [3.5, 4.5],    # wrong cluster for B (if it was meant for B)
])

all_synth = np.vstack([synth_good_A, synth_good_B, synth_bad])

# Draw arrows from LLM to synthetic samples
for pt in all_synth:
    ax.annotate("", xy=(pt[0], pt[1] + 0.35), xytext=(5.25, 9.0),
                arrowprops=dict(arrowstyle="-|>", color="#999999",
                                lw=0.7, alpha=0.35,
                                connectionstyle="arc3,rad=0.1"),
                zorder=2)

# Plot all synthetic as triangles
ax.scatter(all_synth[:, 0], all_synth[:, 1], marker="^", c=C_SYNTH,
           s=100, edgecolors="black", linewidths=0.7, zorder=5)

# -- All get green checks (the problem: no validation) ----------------------
for pt in all_synth:
    ax.text(pt[0] + 0.35, pt[1] + 0.35, "\u2713", fontsize=12,
            color=C_GOOD, fontweight="bold", ha="center", va="center",
            zorder=6)

# -- Red warning indicators on the bad ones ----------------------------------
for pt in synth_bad:
    # Red circle around the bad sample
    circ = plt.Circle((pt[0], pt[1]), 0.55, fill=False,
                       edgecolor=C_BAD, linewidth=1.8, linestyle="-",
                       alpha=0.8, zorder=4)
    ax.add_patch(circ)

# -- Large warning triangle --------------------------------------------------
warn_x, warn_y = 9.2, 8.5
warn_tri = mpatches.RegularPolygon(
    (warn_x, warn_y), numVertices=3, radius=0.9,
    orientation=0, facecolor=C_WARN, edgecolor="#8B6914",
    linewidth=1.5, alpha=0.9, zorder=5
)
ax.add_patch(warn_tri)
ax.text(warn_x, warn_y - 0.12, "!", fontsize=18, ha="center",
        va="center", fontweight="bold", color="#5C3300", zorder=6)
ax.text(warn_x, warn_y - 1.3, "Sin\nvalidaci\u00f3n", fontsize=8,
        ha="center", va="center", color=C_BAD, fontweight="bold")

# -- Caption below -----------------------------------------------------------
ax.text(5.0, -1.6,
        "Todas las muestras se incorporan\nsin verificar su calidad geom\u00e9trica",
        fontsize=9.5, ha="center", va="center", fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                  alpha=0.9))

# ============================================================================
#  Panel (b): Ponderacion Uniforme
# ============================================================================
ax = ax_b
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(b)  Ponderaci\u00f3n Uniforme",
             fontsize=12, fontweight="bold", pad=12)

# -- Class cluster (single class) -------------------------------------------
cluster_cx, cluster_cy = 5.0, 5.0
cluster_pts = np.random.multivariate_normal(
    [cluster_cx, cluster_cy], [[0.7, 0.15], [0.15, 0.7]], size=22
)
ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], c=C_BLUE, s=45,
           alpha=0.6, edgecolors="white", linewidths=0.4, zorder=3)
ell_main = mpatches.Ellipse((cluster_cx, cluster_cy), 4.5, 4.0, angle=0,
                             fill=False, edgecolor=C_BLUE, linewidth=1.4,
                             linestyle="--", alpha=0.5)
ax.add_patch(ell_main)
ax.text(cluster_cx, cluster_cy - 0.1, "Datos\nreales", fontsize=8.5,
        ha="center", va="center", color=C_BLUE, fontweight="bold",
        alpha=0.7)

# -- Centroid ----------------------------------------------------------------
ax.scatter(cluster_cx, cluster_cy, marker="*", c="#FF8C00", s=200,
           edgecolors="k", linewidths=0.5, zorder=6)

# -- Synthetic samples at various distances ----------------------------------
# High quality (close to center)
synth_high = np.array([[5.3, 5.5], [4.6, 5.3], [5.2, 4.5]])
# Medium quality (edge of cluster)
synth_med = np.array([[3.2, 6.5], [6.8, 3.5], [3.0, 3.8]])
# Low quality (far away)
synth_low = np.array([[1.0, 8.5], [9.0, 1.5], [0.8, 1.2], [9.2, 8.8]])

all_synth_b = np.vstack([synth_high, synth_med, synth_low])
qualities = (["alta"] * len(synth_high) +
             ["media"] * len(synth_med) +
             ["baja"] * len(synth_low))
q_colors = ([C_GOOD] * len(synth_high) +
            [C_WARN] * len(synth_med) +
            [C_BAD] * len(synth_low))

# Draw synthetic triangles with quality-based colors
for i, (pt, qc) in enumerate(zip(all_synth_b, q_colors)):
    ax.scatter(pt[0], pt[1], marker="^", c=qc, s=110,
               edgecolors="black", linewidths=0.7, zorder=5)

# -- Dashed lines from centroid to each synthetic ----------------------------
for pt in all_synth_b:
    ax.plot([cluster_cx, pt[0]], [cluster_cy, pt[1]], color="#AAAAAA",
            linewidth=0.7, linestyle=":", alpha=0.6, zorder=2)

# -- Weight labels (all w=1.0) and weight bars ------------------------------
for i, pt in enumerate(all_synth_b):
    # Offset the label based on position relative to center
    dx = pt[0] - cluster_cx
    dy = pt[1] - cluster_cy
    norm = max(np.sqrt(dx**2 + dy**2), 0.1)
    off_x = 0.55 * dx / norm
    off_y = 0.55 * dy / norm

    lx = pt[0] + off_x
    ly = pt[1] + off_y

    ax.text(lx, ly, "w=1.0", fontsize=7.5, ha="center", va="center",
            fontweight="bold", color="#333333", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFDD",
                      ec="#999966", alpha=0.9, linewidth=0.6))

# -- Small equal-sized weight bars at the bottom -----------------------------
bar_y = 0.2
bar_h = 0.7
bar_w = 0.6
n_bars = len(all_synth_b)
bar_start_x = 5.0 - (n_bars * (bar_w + 0.15)) / 2

ax.text(5.0, bar_y + bar_h + 0.5, "Pesos asignados:",
        fontsize=8.5, ha="center", va="center", color="#555555",
        fontweight="bold")

for i in range(n_bars):
    bx = bar_start_x + i * (bar_w + 0.15)
    rect = mpatches.FancyBboxPatch(
        (bx, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0.05", facecolor=C_GRAY,
        edgecolor="#666666", linewidth=0.7, alpha=0.7
    )
    ax.add_patch(rect)
    ax.text(bx + bar_w / 2, bar_y + bar_h / 2, "1.0",
            fontsize=6.5, ha="center", va="center", fontweight="bold",
            color="#333333")

# -- "=" sign to emphasize uniformity ----------------------------------------
ax.text(5.0, bar_y - 0.55, "= = = = = = = = = =", fontsize=9,
        ha="center", va="center", color=C_BAD, fontweight="bold", alpha=0.7)

# -- Caption -----------------------------------------------------------------
ax.text(5.0, -1.6,
        "Todas las muestras reciben peso id\u00e9ntico\nsin importar su posici\u00f3n",
        fontsize=9.5, ha="center", va="center", fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                  alpha=0.9))

# ============================================================================
#  Panel (c): Validacion Limitada
# ============================================================================
ax = ax_c
ax.set_xlim(-1, 11)
ax.set_ylim(-2, 11.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(c)  Validaci\u00f3n Limitada",
             fontsize=12, fontweight="bold", pad=12)


def draw_dataset_box(ax, cx, cy, label, icon_color, border_color,
                     mark=None, mark_color=None, width=2.8, height=1.6):
    """Draw a dataset/document icon as a rounded rectangle."""
    box = mpatches.FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.2", facecolor=icon_color,
        edgecolor=border_color, linewidth=1.5, zorder=4
    )
    ax.add_patch(box)

    # Table lines inside the box
    for frac in [0.3, 0.5, 0.7]:
        y_line = cy - height / 2 + frac * height
        ax.plot([cx - width / 2 + 0.25, cx + width / 2 - 0.25],
                [y_line, y_line], color=border_color, linewidth=0.5,
                alpha=0.4, zorder=5)

    ax.text(cx, cy - height / 2 - 0.35, label, fontsize=8,
            ha="center", va="top", color="#444444", fontweight="bold")

    if mark:
        ax.text(cx + width / 2 + 0.3, cy + height / 2 - 0.15,
                mark, fontsize=18, ha="center", va="center",
                color=mark_color, fontweight="bold", zorder=6)


# -- Main evaluated dataset (top center, with checkmark) --------------------
draw_dataset_box(ax, 5.0, 9.0, "Dataset evaluado",
                 C_DATASET, "#6B8E6B", mark="\u2713", mark_color=C_GOOD)

# Highlight box around it
highlight = mpatches.FancyBboxPatch(
    (3.2, 7.85), 3.6, 2.3,
    boxstyle="round,pad=0.15", facecolor="none",
    edgecolor=C_GOOD, linewidth=2.0, linestyle="-", alpha=0.6, zorder=3
)
ax.add_patch(highlight)
ax.text(5.0, 10.5, "Un solo dominio evaluado", fontsize=8.5,
        ha="center", va="center", color=C_GOOD, fontweight="bold")

# -- Arrow from main dataset down to a result box ---------------------------
ax.annotate("", xy=(5.0, 7.0), xytext=(5.0, 8.0),
            arrowprops=dict(arrowstyle="-|>", color=C_GOOD, lw=2.0),
            zorder=5)
# Result box
result_box = mpatches.FancyBboxPatch(
    (3.8, 6.0), 2.4, 0.9,
    boxstyle="round,pad=0.15", facecolor="#E8F5E9",
    edgecolor=C_GOOD, linewidth=1.3, zorder=4
)
ax.add_patch(result_box)
ax.text(5.0, 6.45, "F1 = 0.85", fontsize=9, ha="center",
        va="center", fontweight="bold", color=C_GOOD, zorder=5)

# -- Separator line ----------------------------------------------------------
ax.plot([0.5, 9.5], [5.2, 5.2], color="#BBBBBB", linewidth=1.2,
        linestyle="--", alpha=0.6)
ax.text(5.0, 5.5, "\u00bfGeneraliza a otros dominios/tareas?",
        fontsize=9, ha="center", va="center", color=C_BAD,
        fontweight="bold", fontstyle="italic")

# -- Other datasets with question marks --------------------------------------
other_datasets = [
    (1.8, 3.5, "Dominio\nCl\u00ednico"),
    (5.0, 3.5, "Dominio\nFinanciero"),
    (8.2, 3.5, "Dominio\nLegal"),
]

for cx, cy, label in other_datasets:
    draw_dataset_box(ax, cx, cy, label, C_DATASET_Q, "#C0857A",
                     mark="?", mark_color=C_BAD, width=2.4, height=1.3)

# -- NER task with question mark ---------------------------------------------
ner_box = mpatches.FancyBboxPatch(
    (1.5, 0.2), 7.0, 1.4,
    boxstyle="round,pad=0.2", facecolor="#FFF3CD",
    edgecolor="#C9A825", linewidth=1.3, zorder=4
)
ax.add_patch(ner_box)

# NER example text
ax.text(5.0, 1.15, "Otra tarea (NER):", fontsize=8.5,
        ha="center", va="center", fontweight="bold", color="#7A6A00",
        zorder=5)
ax.text(5.0, 0.55, '"[Juan]$_{\\mathsf{PER}}$ trabaja en [Santiago]$_{\\mathsf{LOC}}$"',
        fontsize=8, ha="center", va="center", color="#555555",
        zorder=5)

# Question mark for NER
ax.text(9.0, 1.0, "?", fontsize=20, ha="center", va="center",
        color=C_BAD, fontweight="bold", zorder=6)

# -- Arrows from separator to other datasets ---------------------------------
for cx, cy, _ in other_datasets:
    ax.annotate("", xy=(cx, cy + 0.75), xytext=(cx, 5.1),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.2,
                                linestyle="--"),
                zorder=3)

# Arrow to NER box
ax.annotate("", xy=(5.0, 1.7), xytext=(5.0, 2.5),
            arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.2,
                            linestyle="--"),
            zorder=3)

# -- Caption -----------------------------------------------------------------
ax.text(5.0, -1.6,
        "Evaluaci\u00f3n en un solo dominio\nimpide confirmar generalidad",
        fontsize=9.5, ha="center", va="center", fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                  alpha=0.9))

# ============================================================================
#  Suptitle
# ============================================================================
fig.suptitle(
    "Tres Deficiencias de los Enfoques Existentes",
    fontsize=16, fontweight="bold", y=1.02,
)

# ============================================================================
#  Shared legend
# ============================================================================
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BLUE,
           markersize=8, label="Datos reales", markeredgecolor="white",
           markeredgewidth=0.5),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_GRAY,
           markersize=9, label="Muestra sint\u00e9tica",
           markeredgecolor="black", markeredgewidth=0.5),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_GOOD,
           markersize=8, label="Validado / Correcto",
           markeredgecolor="white", markeredgewidth=0.5),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BAD,
           markersize=8, label="Problema / Sin verificar",
           markeredgecolor="white", markeredgewidth=0.5),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=False,
    bbox_to_anchor=(0.5, -0.04),
)

# ============================================================================
#  Save
# ============================================================================
out_dir = pathlib.Path(__file__).resolve().parent
out_path = out_dir / "fig_2_1_three_deficiencies.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {out_path}")
