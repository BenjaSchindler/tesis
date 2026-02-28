#!/usr/bin/env python3
"""
Thesis data visualization figures:
  - Figure 5.4: Dataset summary (horizontal bar chart)
  - Figure 5.5: Experimental matrix (styled table)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
OUT_DIR = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib"


# ============================================================================
# FIGURE 1: fig_5_4_dataset_summary.png
# ============================================================================
def fig_dataset_summary():
    # ----- data -----
    groups = [
        {
            "label": "Clasificación\nde Texto",
            "color": "#3B7DD8",
            "datasets": [
                ("sms_spam",              "Spam detection",      2,  1394),
                ("hate_speech_davidson",  "Hate speech",         3,  4948),
                ("20newsgroups",          "News (4 cat.)",       4,  7532),
                ("ag_news",              "News (4 cat.)",       4,  7600),
                ("emotion",              "Text emotions",       6,  2000),
                ("dbpedia14",            "Topic classif.",     14,  5000),
                ("20newsgroups_20class", "News (20 cat.)",     20,  7532),
            ],
        },
        {
            "label": "Escalabilidad",
            "color": "#2EAE6D",
            "datasets": [
                ("trec6",      "Questions",        6,   500),
                ("banking77",  "Banking intents", 77,  3080),
                ("clinc150",   "Dialog intents", 150,  4500),
            ],
        },
        {
            "label": "NER",
            "color": "#8E44AD",
            "datasets": [
                ("MultiNERD",  "General entities",   5, 10000),
                ("WikiANN",    "Wikipedia entities",  3, 10000),
                ("Few-NERD",   "Detailed entities",   8, 10000),
            ],
        },
    ]

    # Flatten (bottom-to-top ordering for horizontal bars)
    names = []
    domains = []
    classes = []
    test_sizes = []
    colors = []
    group_boundaries = []  # (start_idx, end_idx, label, color)

    idx = 0
    for g in reversed(groups):
        start = idx
        for ds in reversed(g["datasets"]):
            names.append(ds[0])
            domains.append(ds[1])
            classes.append(ds[2])
            test_sizes.append(ds[3])
            colors.append(g["color"])
            idx += 1
        group_boundaries.append((start, idx - 1, g["label"], g["color"]))

    n = len(names)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(y_pos, classes, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.65, alpha=0.88, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Número de Clases (escala log)", fontsize=12, fontweight="bold")
    ax.set_title("Conjuntos de Datos de la Evaluación Experimental",
                 fontsize=16, fontweight="bold", pad=18)

    # Annotate test-set size next to each bar
    x_max = ax.get_xlim()[1]
    for i, (c, ts) in enumerate(zip(classes, test_sizes)):
        # test size annotation
        ax.text(c * 1.18, y_pos[i], f"test={ts:,}",
                va="center", ha="left", fontsize=8.5, color="#333333",
                fontstyle="italic")

    # N-shot markers on far right
    nshot_x = x_max * 0.75
    for i in range(n):
        ax.text(nshot_x, y_pos[i], "10 · 25 · 50",
                va="center", ha="center", fontsize=7.5,
                color="#555555", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#F0F0F0",
                          ec="#CCCCCC", lw=0.6))

    # N-shot header label
    ax.text(nshot_x, n - 0.15, "N-shots", va="bottom", ha="center",
            fontsize=9, fontweight="bold", color="#444444")

    # Group separator lines and labels
    for start, end, label, clr in group_boundaries:
        # separator line above the group (between groups)
        if start > 0:
            sep_y = start - 0.5
            ax.axhline(y=sep_y, color="#999999", linewidth=1.2,
                       linestyle="--", zorder=2)
        # group label on left margin
        mid_y = (start + end) / 2.0
        ax.text(-0.02, mid_y, label, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=10, fontweight="bold",
                color=clr,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=clr, lw=1.5, alpha=0.9))

    # Legend patches
    legend_patches = []
    for g in groups:
        legend_patches.append(
            mpatches.Patch(color=g["color"], label=g["label"].replace("\n", " "))
        )
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#CCCCCC")

    ax.set_xlim(left=1)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.grid(axis="y", alpha=0.0)
    sns.despine(left=True, bottom=False)

    path = f"{OUT_DIR}/fig_5_4_dataset_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# FIGURE 2: fig_5_5_experimental_matrix.png
# ============================================================================
def fig_experimental_matrix():
    # ----- data -----
    headers = ["Experimento", "Datasets", "N-shots", "Clasificadores",
               "Métodos", "Seeds", "Total"]

    rows = [
        ["Principal",       "7", "3", "5",  "7", "5", "3,675"],
        ["Mod. baselines",  "7", "3", "3", "10", "3", "1,890"],
        ["Escalabilidad",   "3", "3", "3",  "7", "3", "567"],
        ["NER",             "3", "3", "1",  "3", "1", "27+"],
        ["Curriculum",      "7", "3", "3",  "1", "3", "1,890"],
        ["Emb. ablation",   "4", "3", "3",  "7", "1", "504+"],
    ]
    total_row = ["TOTAL", "—", "—", "—", "—", "—", "7,600+"]

    n_cols = len(headers)
    n_rows = len(rows) + 1  # +1 for total

    # Relative column widths
    col_widths = [2.0, 1.1, 1.1, 1.4, 1.1, 0.9, 1.4]
    total_w = sum(col_widths)
    col_widths_norm = [w / total_w for w in col_widths]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout parameters
    table_left = 0.05
    table_right = 0.95
    table_width = table_right - table_left
    header_height = 0.075
    row_height = 0.065
    table_top = 0.78
    header_top = table_top
    header_bottom = header_top - header_height

    # Compute column x positions
    col_x = [table_left]
    for i, w in enumerate(col_widths_norm):
        col_x.append(col_x[-1] + w * table_width)

    # ----- title & subtitle -----
    ax.text(0.5, 0.95, "Escala del Diseño Experimental",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="#1A1A2E", transform=ax.transAxes)
    ax.text(0.5, 0.88,
            "Más de 7,600 configuraciones evaluadas con rigor estadístico",
            ha="center", va="center", fontsize=13, fontstyle="italic",
            color="#555555", transform=ax.transAxes)

    # ----- header row -----
    header_color = "#1B2A4A"
    for j in range(n_cols):
        rect = FancyBboxPatch(
            (col_x[j], header_bottom), col_x[j + 1] - col_x[j], header_height,
            boxstyle="square,pad=0",
            facecolor=header_color, edgecolor="white", linewidth=1.5,
            transform=ax.transAxes, zorder=3,
        )
        ax.add_patch(rect)
        cx = (col_x[j] + col_x[j + 1]) / 2
        cy = header_bottom + header_height / 2
        fw = "bold"
        ax.text(cx, cy, headers[j], ha="center", va="center",
                fontsize=12, fontweight=fw, color="white",
                transform=ax.transAxes, zorder=4)

    # ----- data rows -----
    alt_colors = ["#F7F9FC", "#FFFFFF"]
    all_rows = rows + [total_row]

    for i, row_data in enumerate(all_rows):
        is_total = (i == len(all_rows) - 1)
        y_top = header_bottom - i * row_height
        y_bot = y_top - row_height

        if is_total:
            bg = "#FFF3CD"  # gold/yellow highlight
            text_color = "#1A1A2E"
        else:
            bg = alt_colors[i % 2]
            text_color = "#333333"

        for j in range(n_cols):
            rect = FancyBboxPatch(
                (col_x[j], y_bot), col_x[j + 1] - col_x[j], row_height,
                boxstyle="square,pad=0",
                facecolor=bg,
                edgecolor="#D0D0D0", linewidth=0.8,
                transform=ax.transAxes, zorder=2,
            )
            ax.add_patch(rect)

            cx = (col_x[j] + col_x[j + 1]) / 2
            cy = y_bot + row_height / 2
            val = row_data[j]

            # Styling
            fw = "normal"
            fs = 11.5
            tc = text_color

            if is_total:
                fw = "bold"
                fs = 13
                tc = "#8B6914"
            if j == n_cols - 1:  # Total column
                fw = "bold"
                fs = 13 if is_total else 12.5
                if not is_total:
                    tc = "#1B2A4A"
            if j == 0:  # Experiment name
                fw = "bold"
                if is_total:
                    tc = "#8B6914"

            ax.text(cx, cy, val, ha="center", va="center",
                    fontsize=fs, fontweight=fw, color=tc,
                    transform=ax.transAxes, zorder=4)

    # ----- outer border -----
    total_table_height = header_height + len(all_rows) * row_height
    table_bottom = header_bottom - len(all_rows) * row_height
    border = FancyBboxPatch(
        (table_left, table_bottom), table_width, header_height + len(all_rows) * row_height,
        boxstyle="round,pad=0.008",
        facecolor="none", edgecolor="#1B2A4A", linewidth=2.5,
        transform=ax.transAxes, zorder=5,
    )
    ax.add_patch(border)

    # ----- multiplication signs between numeric columns -----
    # Add small "x" signs between columns 1-5 for data rows (not total)
    for i, row_data in enumerate(rows):
        y_top = header_bottom - i * row_height
        y_bot = y_top - row_height
        cy = y_bot + row_height / 2
        for j in range(1, 5):  # between cols 1-2, 2-3, 3-4, 4-5
            bx = col_x[j + 1]  # right edge of column j
            ax.text(bx, cy, "×", ha="center", va="center",
                    fontsize=10, color="#AAAAAA", fontweight="bold",
                    transform=ax.transAxes, zorder=4)

    # ----- "=" sign before Total column -----
    for i, row_data in enumerate(rows):
        y_top = header_bottom - i * row_height
        y_bot = y_top - row_height
        cy = y_bot + row_height / 2
        bx = col_x[n_cols - 1]
        ax.text(bx, cy, "=", ha="center", va="center",
                fontsize=12, color="#AAAAAA", fontweight="bold",
                transform=ax.transAxes, zorder=4)

    # ----- footnote -----
    ax.text(0.5, table_bottom - 0.04,
            "Cada configuración se ejecuta de forma independiente con datos "
            "generados por LLM y clasificadores entrenados desde cero.",
            ha="center", va="top", fontsize=9.5, fontstyle="italic",
            color="#777777", transform=ax.transAxes)

    path = f"{OUT_DIR}/fig_5_5_experimental_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
if __name__ == "__main__":
    fig_dataset_summary()
    fig_experimental_matrix()
    print("Done.")
