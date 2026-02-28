#!/usr/bin/env python3
"""
fig_batch_conceptual.py
=======================
Generates 6 conceptual thesis diagrams:
  1. fig_1_1_low_resource_problem.png   - Abundant vs scarce data
  2. fig_2_2_smote_vs_llm.png          - SMOTE vs LLM augmentation
  3. fig_2_3_research_opportunity.png   - Venn diagram positioning
  4. fig_3_2_evolution_timeline.png     - Augmentation evolution timeline
  5. fig_4_1_hypothesis_map.png         - Hypothesis-to-experiment mapping
  6. fig_5_6_ner_adaptation.png         - Filter adaptation for NER
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

OUT_DIR = pathlib.Path("/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib")

np.random.seed(42)

# ── Colour palette ──────────────────────────────────────────────────
C_BLUE    = "#3B7DD8"
C_ORANGE  = "#E8802A"
C_GREEN   = "#2CA02C"
C_RED     = "#D62728"
C_PURPLE  = "#9467BD"
C_GOLD    = "#DAA520"

# Light fills
LF_BLUE   = "#D6E8FA"
LF_GREEN  = "#D5F0D5"
LF_PURPLE = "#E8D5F5"
LF_ORANGE = "#FDE8D0"

# Border colours (material design inspired)
B_BLUE    = "#1565C0"
B_GREEN   = "#2E7D32"
B_PURPLE  = "#6A1B9A"
B_ORANGE  = "#E65100"
B_GOLD    = "#B8860B"


# =====================================================================
#  HELPER: draw a rounded box on an axes
# =====================================================================
def draw_box(ax, cx, cy, w, h, facecolor, edgecolor, lw=1.5, zorder=2,
             alpha=1.0, linestyle="solid"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder, alpha=alpha, linestyle=linestyle,
    )
    ax.add_patch(box)
    return box


def connect(ax, x1, y1, x2, y2, color="#455A64", lw=2.0, style="-|>",
            mutation=22, zorder=1, linestyle="solid", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        lw=lw, color=color, zorder=zorder,
        connectionstyle=connectionstyle, linestyle=linestyle,
    )
    ax.add_patch(arrow)
    return arrow


# =====================================================================
#  FIGURE 1: Low-resource problem  (14x6)
# =====================================================================
def make_fig1():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: Abundant data ---
    ax = ax_left
    ax.set_title("Datos Abundantes", fontsize=14, fontweight="bold",
                 color="#0D47A1", pad=12)

    # 3 classes, 35 points each
    rng = np.random.default_rng(10)
    centers = [(-2.0, 2.0), (2.5, 2.5), (0.5, -1.8)]
    colors_cls = [C_BLUE, C_ORANGE, C_GREEN]
    names = ["Clase A", "Clase B", "Clase C"]

    all_pts_left = []
    labels_left = []
    for i, (cx, cy) in enumerate(centers):
        pts = rng.multivariate_normal(
            [cx, cy], [[0.45, 0.05], [0.05, 0.45]], size=35
        )
        ax.scatter(pts[:, 0], pts[:, 1], c=colors_cls[i], s=30, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=4, label=names[i])
        all_pts_left.append(pts)
        labels_left.extend([i] * len(pts))

    # Decision boundaries (approximate linear separators)
    # Between A and B: roughly x = 0.25
    ax.plot([0.25, 0.25], [-4.5, 5.5], 'k-', lw=1.8, alpha=0.6, zorder=3)
    # Between A and C: roughly y = 0.0
    ax.plot([-4.5, 1.0], [0.1, 0.1], 'k-', lw=1.8, alpha=0.6, zorder=3)
    # Between B and C: roughly y = 0.4x - 0.5
    xline = np.linspace(0.25, 5.0, 50)
    yline = 0.5 * xline - 0.2
    ax.plot(xline, yline, 'k-', lw=1.8, alpha=0.6, zorder=3)

    ax.set_xlim(-4.5, 5.5)
    ax.set_ylim(-4.5, 5.5)
    ax.set_xlabel("Dim 1", fontsize=11)
    ax.set_ylabel("Dim 2", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.5, -0.10, "n >> 50 por clase", transform=ax.transAxes,
            ha="center", fontsize=12, fontstyle="italic", color="#37474F")

    # --- Right panel: Scarce data ---
    ax = ax_right
    ax.set_title("Datos Escasos", fontsize=14, fontweight="bold",
                 color="#B71C1C", pad=12)

    rng2 = np.random.default_rng(20)
    for i, (cx, cy) in enumerate(centers):
        pts = rng2.multivariate_normal(
            [cx, cy], [[0.7, 0.1], [0.1, 0.7]], size=rng2.integers(5, 9)
        )
        ax.scatter(pts[:, 0], pts[:, 1], c=colors_cls[i], s=55, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=4, label=names[i])

    # Uncertain decision boundaries (dashed)
    ax.plot([0.25, 0.25], [-4.5, 5.5], 'k--', lw=1.3, alpha=0.4, zorder=3)
    ax.plot([-4.5, 1.0], [0.1, 0.1], 'k--', lw=1.3, alpha=0.4, zorder=3)
    xline2 = np.linspace(0.25, 5.0, 50)
    yline2 = 0.5 * xline2 - 0.2
    ax.plot(xline2, yline2, 'k--', lw=1.3, alpha=0.4, zorder=3)

    # Overlap shaded regions
    overlap_regions = [
        (0.25, 0.1, 1.8, 1.5),   # between A-B-C
        (-1.0, -0.5, 1.5, 1.2),  # between A-C
        (1.5, 0.5, 2.0, 1.5),    # between B-C
    ]
    for (ox, oy, ow, oh) in overlap_regions:
        rect = mpatches.FancyBboxPatch(
            (ox, oy), ow, oh, boxstyle="round,pad=0.1",
            facecolor="#CCCCCC", alpha=0.25, edgecolor="none", zorder=2
        )
        ax.add_patch(rect)

    # Red question marks in overlap regions
    qm_positions = [(0.8, 0.8), (-0.2, 0.0), (2.2, 1.0), (1.2, 0.3)]
    for qx, qy in qm_positions:
        ax.text(qx, qy, "?", fontsize=20, fontweight="bold", color=C_RED,
                ha="center", va="center", zorder=6, alpha=0.8)

    ax.set_xlim(-4.5, 5.5)
    ax.set_ylim(-4.5, 5.5)
    ax.set_xlabel("Dim 1", fontsize=11)
    ax.set_ylabel("Dim 2", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.5, -0.10, "n = 10 por clase", transform=ax.transAxes,
            ha="center", fontsize=12, fontstyle="italic", color="#B71C1C")

    fig.suptitle("Impacto de la Escasez de Datos en las Fronteras de Decisi\u00f3n",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "fig_1_1_low_resource_problem.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  FIGURE 2: SMOTE vs LLM  (14x6)
# =====================================================================
def make_fig2():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    rng = np.random.default_rng(55)

    # Real points for 2 classes
    real_blue = rng.multivariate_normal([-1.5, 1.5], [[0.3, 0.05], [0.05, 0.3]], size=8)
    real_orange = rng.multivariate_normal([2.0, -1.0], [[0.35, 0.05], [0.05, 0.35]], size=8)

    # ---- Left panel: SMOTE ----
    ax = ax_left
    ax.set_title("SMOTE", fontsize=14, fontweight="bold", color="#0D47A1", pad=12)

    # Draw real points
    ax.scatter(real_blue[:, 0], real_blue[:, 1], c=C_BLUE, s=70,
               edgecolors="white", linewidths=0.6, zorder=5, label="Clase 1 (real)")
    ax.scatter(real_orange[:, 0], real_orange[:, 1], c=C_ORANGE, s=70,
               edgecolors="white", linewidths=0.6, zorder=5, label="Clase 2 (real)")

    # SMOTE-generated: interpolation between pairs
    smote_blue = []
    for _ in range(6):
        i1, i2 = rng.choice(len(real_blue), 2, replace=False)
        lam = rng.uniform(0.2, 0.8)
        pt = real_blue[i1] + lam * (real_blue[i2] - real_blue[i1])
        smote_blue.append(pt)
        ax.plot([real_blue[i1, 0], real_blue[i2, 0]],
                [real_blue[i1, 1], real_blue[i2, 1]],
                ':', color=C_BLUE, alpha=0.3, lw=0.8, zorder=2)
    smote_blue = np.array(smote_blue)

    smote_orange = []
    for _ in range(6):
        i1, i2 = rng.choice(len(real_orange), 2, replace=False)
        lam = rng.uniform(0.2, 0.8)
        pt = real_orange[i1] + lam * (real_orange[i2] - real_orange[i1])
        smote_orange.append(pt)
        ax.plot([real_orange[i1, 0], real_orange[i2, 0]],
                [real_orange[i1, 1], real_orange[i2, 1]],
                ':', color=C_ORANGE, alpha=0.3, lw=0.8, zorder=2)
    smote_orange = np.array(smote_orange)

    # Plot SMOTE points as squares (lighter colours)
    ax.scatter(smote_blue[:, 0], smote_blue[:, 1], marker='s', c="#8FBCE8",
               s=55, edgecolors=C_BLUE, linewidths=0.8, zorder=4,
               label="SMOTE (Clase 1)")
    ax.scatter(smote_orange[:, 0], smote_orange[:, 1], marker='s', c="#F4C191",
               s=55, edgecolors=C_ORANGE, linewidths=0.8, zorder=4,
               label="SMOTE (Clase 2)")

    # Convex hulls
    for pts, color in [(real_blue, C_BLUE), (real_orange, C_ORANGE)]:
        hull = ConvexHull(pts)
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        ax.plot(pts[hull_pts, 0], pts[hull_pts, 1], '--', color=color,
                lw=1.2, alpha=0.5, zorder=3)

    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-3.5, 4.0)
    ax.set_xlabel("Dim 1", fontsize=11)
    ax.set_ylabel("Dim 2", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.text(0.5, -0.10, "Interpolaci\u00f3n lineal dentro del casco convexo",
            transform=ax.transAxes, ha="center", fontsize=10, fontstyle="italic",
            color="#37474F")

    # ---- Right panel: LLM Generation ----
    ax = ax_right
    ax.set_title("Generaci\u00f3n con LLM", fontsize=14, fontweight="bold",
                 color="#1B5E20", pad=12)

    # Same real points
    ax.scatter(real_blue[:, 0], real_blue[:, 1], c=C_BLUE, s=70,
               edgecolors="white", linewidths=0.6, zorder=5, label="Clase 1 (real)")
    ax.scatter(real_orange[:, 0], real_orange[:, 1], c=C_ORANGE, s=70,
               edgecolors="white", linewidths=0.6, zorder=5, label="Clase 2 (real)")

    # LLM-generated points: diverse positions
    # Good: inside class clusters
    llm_good_blue = rng.multivariate_normal([-1.5, 1.5], [[0.4, 0.0], [0.0, 0.4]], size=3)
    llm_good_orange = rng.multivariate_normal([2.0, -1.0], [[0.4, 0.0], [0.0, 0.4]], size=3)

    # Bad: between clusters
    llm_bad_between = np.array([[0.2, 0.3], [0.5, -0.1], [-0.2, 0.1]])

    # Very bad: outside both clusters
    llm_bad_outside = np.array([[-3.0, -2.5], [4.0, 3.0], [3.5, 2.5]])

    # Plot LLM-generated as triangles
    ax.scatter(llm_good_blue[:, 0], llm_good_blue[:, 1], marker='^', c="#8FBCE8",
               s=70, edgecolors=C_GREEN, linewidths=1.5, zorder=4,
               label="LLM bueno")
    ax.scatter(llm_good_orange[:, 0], llm_good_orange[:, 1], marker='^', c="#F4C191",
               s=70, edgecolors=C_GREEN, linewidths=1.5, zorder=4)

    ax.scatter(llm_bad_between[:, 0], llm_bad_between[:, 1], marker='^', c="#F5F5F5",
               s=70, edgecolors=C_RED, linewidths=1.5, zorder=4,
               label="LLM malo (entre clases)")
    ax.scatter(llm_bad_outside[:, 0], llm_bad_outside[:, 1], marker='^', c="#F5F5F5",
               s=80, edgecolors=C_RED, linewidths=2.0, zorder=4,
               label="LLM malo (fuera)")

    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-3.5, 4.0)
    ax.set_xlabel("Dim 1", fontsize=11)
    ax.set_ylabel("Dim 2", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.text(0.5, -0.10, "Generaci\u00f3n libre: diversa pero sin control geom\u00e9trico",
            transform=ax.transAxes, ha="center", fontsize=10, fontstyle="italic",
            color="#37474F")

    fig.suptitle("SMOTE vs LLM: Distribuci\u00f3n Espacial de Muestras Sint\u00e9ticas",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "fig_2_2_smote_vs_llm.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  FIGURE 3: Research opportunity Venn  (10x8)
# =====================================================================
def make_fig3():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    fig.patch.set_facecolor("white")

    # Three overlapping circles
    r = 2.2
    circles_data = [
        # (cx, cy, color, label, sublabel)
        (-1.3, 0.8, "#BBDEFB", B_BLUE, "Validaci\u00f3n\nPost-Generaci\u00f3n",
         "Filtros geom\u00e9tricos"),
        (1.3, 0.8, "#C8E6C9", B_GREEN, "Ponderaci\u00f3n\nAdaptativa",
         "Pesos por calidad"),
        (0.0, -1.0, "#E1BEE7", B_PURPLE, "Generalizaci\u00f3n\nMulti-Tarea",
         "Clasificaci\u00f3n + NER"),
    ]

    for cx, cy, fc, ec, label, sublabel in circles_data:
        circle = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                            linewidth=2.0, alpha=0.3, zorder=2)
        ax.add_patch(circle)
        # Main label
        ax.text(cx + (cx * 0.55), cy + (cy * 0.45), label,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=ec, zorder=5)
        # Sub-label
        ax.text(cx + (cx * 0.55), cy + (cy * 0.45) - 0.65, sublabel,
                ha="center", va="center", fontsize=10, fontstyle="italic",
                color="#555555", zorder=5)

    # Center intersection
    # Draw a small highlighted region
    center_circle = plt.Circle((0.0, 0.15), 0.65, facecolor="#FFD54F",
                                edgecolor=B_GOLD, linewidth=2.5, alpha=0.7, zorder=4)
    ax.add_patch(center_circle)
    ax.text(0.0, 0.15, "ESTA\nTESIS", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#5D4037", zorder=6)

    fig.suptitle("Posicionamiento de la Investigaci\u00f3n",
                 fontsize=16, fontweight="bold", y=0.95)

    out = OUT_DIR / "fig_2_3_research_opportunity.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  FIGURE 4: Evolution timeline  (16x5)
# =====================================================================
def make_fig4():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(-1.5, 5.5)
    ax.axis("off")
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Main horizontal arrow
    connect(ax, 0.5, 2.0, 15.5, 2.0, color="#78909C", lw=3.0, mutation=28,
            style="-|>")

    # Era definitions: (x_center, width, color_fill, color_border, title, items, description)
    eras = [
        (3.0, 4.0, "#BBDEFB", B_BLUE, "M\u00e9todos Cl\u00e1sicos",
         ["SMOTE (2002)", "ADASYN (2008)", "EDA (2019)"],
         "Gu\u00eda geom\u00e9trica sin\ncoherencia ling\u00fc\u00edstica"),
        (8.0, 3.5, "#C8E6C9", B_GREEN, "Generaci\u00f3n con LLMs",
         ["GPT-3.5/4 (2023)", "Gemini (2024)"],
         "Coherencia ling\u00fc\u00edstica sin\ncontrol geom\u00e9trico"),
        (12.8, 3.5, LF_ORANGE, B_ORANGE, "Enfoques H\u00edbridos",
         ["SMOTExT (2025)", "Esta Tesis (2025) \u2605"],
         "Combina ambas\npropiedades"),
    ]

    for cx, w, fc, ec, title, items, desc in eras:
        # Box
        box_h = 2.8
        box_y = 2.6
        draw_box(ax, cx, box_y + box_h / 2, w, box_h, fc, ec, lw=2.0, zorder=3)

        # Title
        ax.text(cx, box_y + box_h - 0.35, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec, zorder=5)

        # Items
        for j, item in enumerate(items):
            fw = "bold" if "Esta Tesis" in item else "normal"
            fs = 10.5 if "Esta Tesis" in item else 9.5
            ax.text(cx, box_y + box_h - 0.85 - j * 0.45, item,
                    ha="center", va="center", fontsize=fs, fontweight=fw,
                    color="#37474F", zorder=5)

        # Description below the arrow
        ax.text(cx, 0.6, desc, ha="center", va="center", fontsize=9,
                fontstyle="italic", color="#546E7A", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDBDBD",
                          alpha=0.85))

    # Year markers on arrow
    year_markers = [(1.5, "2002"), (5.0, "2019"), (7.0, "2023"),
                    (9.0, "2024"), (11.5, "2025")]
    for mx, label in year_markers:
        ax.plot(mx, 2.0, '|', color="#455A64", markersize=10, zorder=4)
        ax.text(mx, 1.6, label, ha="center", va="center", fontsize=8,
                color="#455A64", zorder=5)

    fig.suptitle("Evoluci\u00f3n de los Enfoques de Aumentaci\u00f3n de Datos",
                 fontsize=16, fontweight="bold", y=0.98)

    out = OUT_DIR / "fig_3_2_evolution_timeline.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  FIGURE 5: Hypothesis map  (16x10)
# =====================================================================
def make_fig5():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Top: Main hypothesis box ---
    h1_cx, h1_cy = 8.0, 9.0
    h1_w, h1_h = 12.0, 1.2
    draw_box(ax, h1_cx, h1_cy, h1_w, h1_h, "#1565C0", "#0D47A1", lw=2.5, zorder=3)
    ax.text(h1_cx, h1_cy, "H1: El filtrado geom\u00e9trico supera a SMOTE",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white", zorder=5)

    # --- 4 branches ---
    hypotheses = [
        {
            "label": "H1a",
            "text": "Filtrado\nbinario",
            "exp": "Exp. principal\n(3,675 configs)",
            "metric": "Delta F1,\np-valor",
            "x": 2.5,
        },
        {
            "label": "H1b",
            "text": "Ponderaci\u00f3n\nsuave",
            "exp": "Ablaci\u00f3n\n(540 configs)",
            "metric": "Delta vs\nbinario",
            "x": 6.0,
        },
        {
            "label": "H1c",
            "text": "Generalizaci\u00f3n\nNER",
            "exp": "Extensi\u00f3n NER\n(63 configs)",
            "metric": "Delta,\nWR 100%",
            "x": 10.0,
        },
        {
            "label": "H1d",
            "text": "Efecto\nclasificador",
            "exp": "An\u00e1lisis por\nclasificador",
            "metric": "Lineal >\nForest",
            "x": 13.5,
        },
    ]

    # Row y-positions
    y_hyp = 6.8
    y_exp = 4.5
    y_met = 2.2

    # Colour definitions for rows
    hyp_colors = ["#1976D2", "#1E88E5", "#2196F3", "#42A5F5"]
    hyp_fills  = ["#BBDEFB", "#C3E1FA", "#D0E8FB", "#DBF0FD"]
    exp_fill   = "#C8E6C9"
    exp_border = "#388E3C"
    met_fill   = "#F5F5F5"
    met_border = "#9E9E9E"

    box_w = 2.8
    box_h_hyp = 1.3
    box_h_exp = 1.3
    box_h_met = 1.1

    for i, h in enumerate(hypotheses):
        x = h["x"]

        # Hypothesis box
        draw_box(ax, x, y_hyp, box_w, box_h_hyp, hyp_fills[i], hyp_colors[i],
                 lw=1.8, zorder=3)
        ax.text(x, y_hyp + 0.18, h["label"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=hyp_colors[i], zorder=5)
        ax.text(x, y_hyp - 0.28, h["text"], ha="center", va="center",
                fontsize=9.5, color="#37474F", zorder=5)

        # Experiment box
        draw_box(ax, x, y_exp, box_w, box_h_exp, exp_fill, exp_border,
                 lw=1.5, zorder=3)
        ax.text(x, y_exp, h["exp"], ha="center", va="center",
                fontsize=9.5, color="#1B5E20", zorder=5)

        # Metric box (dashed border)
        draw_box(ax, x, y_met, box_w, box_h_met, met_fill, met_border,
                 lw=1.3, zorder=3, linestyle="dashed")
        ax.text(x, y_met, h["metric"], ha="center", va="center",
                fontsize=9.5, color="#424242", zorder=5)

        # Arrows: H1 -> sub-hypothesis
        connect(ax, x, h1_cy - h1_h / 2, x, y_hyp + box_h_hyp / 2,
                color=hyp_colors[i], lw=1.8, mutation=18, zorder=2)

        # Arrows: sub-hypothesis -> experiment
        connect(ax, x, y_hyp - box_h_hyp / 2, x, y_exp + box_h_exp / 2,
                color=exp_border, lw=1.5, mutation=16, zorder=2)

        # Arrows: experiment -> metric
        connect(ax, x, y_exp - box_h_exp / 2, x, y_met + box_h_met / 2,
                color=met_border, lw=1.3, mutation=14, zorder=2,
                linestyle="dashed")

    # Row labels on the left
    row_labels = [
        (y_hyp, "Hip\u00f3tesis", "#1565C0"),
        (y_exp, "Experimento", "#2E7D32"),
        (y_met, "M\u00e9trica", "#616161"),
    ]
    for ry, rl, rc in row_labels:
        ax.text(0.25, ry, rl, ha="left", va="center", fontsize=10,
                fontweight="bold", color=rc, rotation=90, zorder=5)

    fig.suptitle("Estructura de Hip\u00f3tesis y Correspondencia Experimental",
                 fontsize=16, fontweight="bold", y=0.98)

    out = OUT_DIR / "fig_4_1_hypothesis_map.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  FIGURE 6: NER Adaptation  (16x6)
# =====================================================================
def make_fig6():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Level 1: Representation (left) ----
    # Background box
    draw_box(ax, 3.0, 3.0, 5.2, 4.8, "#F5F5F5", "#BDBDBD", lw=1.2, zorder=1)
    ax.text(3.0, 5.2, "Nivel 1: Representaci\u00f3n", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#37474F", zorder=5)

    # Example sentence with color-coded entities
    # "[Juan] trabaja en [Google] en [Santiago]"
    sentence_y = 4.2
    parts = [
        ("\"", "#37474F", "normal"),
        ("Juan", "#C62828", "bold"),      # PER - red
        (" trabaja en ", "#37474F", "normal"),
        ("Google", "#1565C0", "bold"),     # ORG - blue
        (" en ", "#37474F", "normal"),
        ("Santiago", "#2E7D32", "bold"),   # LOC - green
        ("\"", "#37474F", "normal"),
    ]

    # Render sentence pieces side by side
    x_cursor = 1.0
    for text, color, weight in parts:
        ax.text(x_cursor, sentence_y, text, ha="left", va="center",
                fontsize=10, fontweight=weight, color=color, zorder=5,
                fontfamily="monospace")
        x_cursor += len(text) * 0.18

    # Entity labels
    entity_labels = [
        (1.25, 3.7, "PER", "#C62828"),
        (3.2, 3.7, "ORG", "#1565C0"),
        (4.35, 3.7, "LOC", "#2E7D32"),
    ]
    for ex, ey, elabel, ecolor in entity_labels:
        ax.text(ex, ey, elabel, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=ecolor, ec=ecolor))

    # Embedding model box
    draw_box(ax, 3.0, 2.2, 3.8, 0.8, "#E8EAF6", "#3F51B5", lw=1.5, zorder=3)
    ax.text(3.0, 2.2, "all-mpnet-base-v2", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#283593", zorder=5)

    # Embedding output
    draw_box(ax, 3.0, 1.2, 2.5, 0.6, "#E3F2FD", "#1565C0", lw=1.2, zorder=3)
    ax.text(3.0, 1.2, "embedding 768-d", ha="center", va="center",
            fontsize=9.5, color="#0D47A1", zorder=5)

    # Arrow from sentence to model
    connect(ax, 3.0, 3.3, 3.0, 2.65, color="#546E7A", lw=1.5, mutation=16)
    # Arrow from model to embedding
    connect(ax, 3.0, 1.8, 3.0, 1.55, color="#546E7A", lw=1.5, mutation=16)

    # ---- Arrow from Level 1 to Level 2 ----
    connect(ax, 5.6, 3.0, 7.0, 3.0, color="#546E7A", lw=2.0, mutation=20)

    # ---- Level 2: Class Assignment (center) ----
    draw_box(ax, 9.0, 3.0, 3.5, 4.8, "#F5F5F5", "#BDBDBD", lw=1.2, zorder=1)
    ax.text(9.0, 5.2, "Nivel 2: Asignaci\u00f3n de Clase", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#37474F", zorder=5)

    # Mini table
    table_data = [("PER", "1"), ("ORG", "1"), ("LOC", "1")]
    table_y_start = 4.2
    for j, (etype, count) in enumerate(table_data):
        ty = table_y_start - j * 0.45
        color_map = {"PER": "#C62828", "ORG": "#1565C0", "LOC": "#2E7D32"}
        ax.text(8.3, ty, etype, ha="center", va="center", fontsize=10,
                fontweight="bold", color=color_map[etype], zorder=5)
        ax.text(8.8, ty, ":", ha="center", va="center", fontsize=10,
                color="#37474F", zorder=5)
        ax.text(9.2, ty, count, ha="center", va="center", fontsize=10,
                color="#37474F", zorder=5)

    # Logic arrow
    connect(ax, 9.0, 2.8, 9.0, 2.2, color="#546E7A", lw=1.2, mutation=14)

    # Result box
    draw_box(ax, 9.0, 1.7, 3.0, 0.7, "#FFF3E0", B_ORANGE, lw=1.5, zorder=3)
    ax.text(9.0, 1.7, "max() -> clase dominante", ha="center", va="center",
            fontsize=9.5, color="#BF360C", zorder=5)

    # Label
    ax.text(9.0, 1.0, "Tipo de entidad dominante\n-> etiqueta de clase",
            ha="center", va="center", fontsize=8.5, fontstyle="italic",
            color="#546E7A", zorder=5)

    # ---- Arrow from Level 2 to Level 3 ----
    connect(ax, 10.75, 3.0, 12.0, 3.0, color="#546E7A", lw=2.0, mutation=20)

    # ---- Level 3: Filtering (right) ----
    draw_box(ax, 14.0, 3.0, 3.5, 4.8, "#F5F5F5", "#BDBDBD", lw=1.2, zorder=1)
    ax.text(14.0, 5.2, "Nivel 3: Filtrado", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#37474F", zorder=5)

    # Filter box (reuse methodology style: funnel-like)
    draw_box(ax, 14.0, 3.5, 2.8, 1.4, "#FFF3E0", B_ORANGE, lw=2.2, zorder=3)
    ax.text(14.0, 3.8, "Filtro Geom\u00e9trico", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#BF360C", zorder=5)
    ax.text(14.0, 3.2, "(sin modificaci\u00f3n)", ha="center", va="center",
            fontsize=9.5, fontstyle="italic", color="#795548", zorder=5)

    # Annotation
    ax.text(14.0, 2.2, "Mismos filtros que\nclasificaci\u00f3n de texto",
            ha="center", va="center", fontsize=9, fontstyle="italic",
            color="#546E7A", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50",
                      alpha=0.85))

    # Green check mark
    ax.text(14.0, 1.3, "\u2713", ha="center", va="center", fontsize=30,
            fontweight="bold", color="#2E7D32", zorder=5)

    fig.suptitle("Adaptaci\u00f3n del Sistema de Filtrado para NER",
                 fontsize=16, fontweight="bold", y=0.98)

    out = OUT_DIR / "fig_5_6_ner_adaptation.png"
    fig.savefig(out, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out}")


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating 6 conceptual thesis figures...")
    print("=" * 60)

    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()

    print("=" * 60)
    print("All 6 figures generated successfully.")
    print("=" * 60)
