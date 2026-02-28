"""
Batch generation of 4 thesis figures:
  1. fig_6_b_forest_plot.png       - Effect Size Forest Plot
  2. fig_6_e_nclasses_scatter.png  - Classes vs Effectiveness
  3. fig_6_f_geometric_vs_teacher.png - Geometric vs Teacher comparison
  4. fig_6_c_per_class_improvement.png - Per-class Difficulty Stratified
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
import os

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
OUT_DIR = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib"

saved_files = []

# =============================================================================
# FIGURE 1: Forest Plot - Effect Size (Cohen's d) with 95% CI
# =============================================================================
def make_forest_plot():
    # Data: (label, d, ci_low, ci_high, significant)
    # None entries are separators
    raw_data = [
        ("General (pond. suave vs SMOTE)", 0.95, 0.70, 1.20, True),
        ("10-shot", 1.48, 1.10, 1.86, True),
        ("25-shot", 1.57, 1.20, 1.94, True),
        ("50-shot", 0.80, 0.50, 1.10, True),
        ("---SEP---", None, None, None, None),
        ("SVC Lineal", 1.02, 0.70, 1.34, True),
        ("Ridge", 1.04, 0.72, 1.36, True),
        ("Reg. Logistica", 0.82, 0.52, 1.12, True),
        ("---SEP---", None, None, None, None),
        ("NER (vs no aum.)", 1.50, 1.10, 1.90, True),
    ]

    # Separate data rows from separators
    entries = []
    sep_positions = []
    y_pos = 0
    for item in reversed(raw_data):  # reverse so first item is at top
        if item[1] is None:
            sep_positions.append(y_pos - 0.5)
        else:
            entries.append((item[0], item[1], item[2], item[3], item[4], y_pos))
            y_pos += 1

    fig, ax = plt.subplots(figsize=(14, 8))

    # Threshold background shading
    ax.axvspan(-0.5, 0.2, alpha=0.04, color="gray")
    ax.axvspan(0.2, 0.5, alpha=0.07, color="#FFC107", zorder=0)
    ax.axvspan(0.5, 0.8, alpha=0.07, color="#FF9800", zorder=0)
    ax.axvspan(0.8, 2.2, alpha=0.07, color="#E53935", zorder=0)

    # Threshold labels at top
    label_y = len([e for e in raw_data if e[1] is not None]) - 0.2
    ax.text(0.1, label_y, "Pequeno", ha="center", va="bottom", fontsize=9,
            fontstyle="italic", color="#888888")
    ax.text(0.35, label_y, "Mediano", ha="center", va="bottom", fontsize=9,
            fontstyle="italic", color="#C68600")
    ax.text(0.65, label_y, "Grande", ha="center", va="bottom", fontsize=9,
            fontstyle="italic", color="#BF360C")

    # Vertical reference lines
    for xval, ls, lw, col in [
        (0.0, "--", 1.2, "#555555"),
        (0.2, ":", 1.0, "#999999"),
        (0.5, ":", 1.0, "#999999"),
        (0.8, ":", 1.0, "#999999"),
    ]:
        ax.axvline(xval, linestyle=ls, linewidth=lw, color=col, zorder=1)

    # Separator lines
    for sp in sep_positions:
        ax.axhline(sp, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)

    # Plot each entry
    y_labels = []
    y_ticks = []
    for label, d, ci_lo, ci_hi, sig, yp in entries:
        color = "#1565C0" if sig else "#BDBDBD"
        marker_size = 10 if sig else 7

        # CI line
        ax.plot([ci_lo, ci_hi], [yp, yp], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        # Point estimate
        ax.plot(d, yp, "o", color=color, markersize=marker_size, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)

        # Right-side annotation
        ann_text = f"d = {d:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]"
        ax.text(2.05, yp, ann_text, va="center", ha="left", fontsize=9.5,
                fontfamily="monospace", color="#333333")

        y_labels.append(label)
        y_ticks.append(yp)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("Cohen's d (tamano del efecto)", fontsize=12, fontweight="bold")
    ax.set_title("Tamanos del efecto (Cohen's d) con IC 95%",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(-0.15, 2.55)
    ax.set_ylim(-0.8, y_pos + 0.3)

    # Remove right/top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_6_b_forest_plot.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 2: Classes vs Effectiveness Scatter
# =============================================================================
def make_nclasses_scatter():
    # Data
    n_classes = np.array([2, 3, 4, 6, 14, 20, 77, 150])
    delta_pp = np.array([3.18, 1.86, 1.70, 5.42, 0.40, 2.36, -0.99, 0.01])
    datasets = ["sms_spam", "hate_speech", "20news/ag_news", "emotion/trec6",
                "dbpedia14", "20news_20class", "banking77", "clinc150"]
    n_pairs = np.array([15, 15, 30, 24, 15, 15, 9, 9])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Point colors: blue for positive, red for negative
    colors = ["#1565C0" if d >= 0 else "#E53935" for d in delta_pp]

    # Scatter with size proportional to n_pairs
    size_scale = 18
    scatter = ax.scatter(n_classes, delta_pp, s=n_pairs * size_scale,
                         c=colors, alpha=0.8, edgecolors="white",
                         linewidth=1.2, zorder=5)

    # Labels next to each point
    offsets = {
        "sms_spam": (10, 8),
        "hate_speech": (10, -12),
        "20news/ag_news": (10, 8),
        "emotion/trec6": (10, 8),
        "dbpedia14": (10, -12),
        "20news_20class": (10, 8),
        "banking77": (10, -14),
        "clinc150": (10, 8),
    }
    for i, ds in enumerate(datasets):
        ox, oy = offsets.get(ds, (10, 5))
        ax.annotate(ds, (n_classes[i], delta_pp[i]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5))

    # Trend line (linear fit on log scale)
    log_x = np.log10(n_classes)
    coeffs = np.polyfit(log_x, delta_pp, 1)
    x_fit = np.linspace(n_classes.min() * 0.8, n_classes.max() * 1.2, 200)
    y_fit = np.polyval(coeffs, np.log10(x_fit))
    ax.plot(x_fit, y_fit, "--", color="#1565C0", linewidth=1.8, alpha=0.6,
            label="Tendencia lineal (log)", zorder=3)

    # Horizontal reference at 0
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0, zorder=2)

    # Annotation box
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD",
                      edgecolor="#1565C0", alpha=0.9)
    ax.text(0.97, 0.97, "Spearman r = -0.260, p = 0.003",
            transform=ax.transAxes, fontsize=10.5, va="top", ha="right",
            bbox=bbox_props, fontweight="bold", color="#1565C0")

    ax.set_xscale("log")
    ax.set_xlabel("Numero de clases (escala log)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Delta vs SMOTE (pp)", fontsize=12, fontweight="bold")
    ax.set_title("Efectividad del filtro suave segun numero de clases",
                 fontsize=14, fontweight="bold", pad=15)

    # Size legend
    for np_val, lab in [(9, "9 pares"), (15, "15 pares"), (30, "30 pares")]:
        ax.scatter([], [], s=np_val * size_scale, c="#1565C0", alpha=0.5,
                   edgecolors="white", label=lab)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_6_e_nclasses_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 3: Geometric vs Teacher Bar Chart
# =============================================================================
def make_geometric_vs_teacher():
    methods = ["SMOTE\n(base)", "Geom.\nbinario", "Geom.\nponderado",
               "Teacher\nbinario", "Teacher\nponderado"]
    delta = [0.00, 2.78, 2.46, 2.43, 2.15]
    cohens_d = [None, 0.97, 1.05, 0.94, 0.98]
    victory_pct = [None, "88.1%", "96.0%", "88.9%", "89.7%"]
    bar_colors = ["#9E9E9E", "#1565C0", "#1E88E5", "#E65100", "#FB8C00"]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(methods))
    bars = ax.bar(x, delta, width=0.6, color=bar_colors, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # Annotate Cohen's d and victory% above each bar
    for i, (bar, d_val, v_pct) in enumerate(zip(bars, cohens_d, victory_pct)):
        if d_val is not None:
            y_top = bar.get_height() + 0.08
            ax.text(bar.get_x() + bar.get_width() / 2, y_top,
                    f"d = {d_val:.2f}\n{v_pct} victorias",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="#333333")

    # Significance bracket between Geom. ponderado (idx=2) and Teacher ponderado (idx=4)
    bracket_y = max(delta[2], delta[4]) + 0.75
    ax.plot([2, 2, 4, 4], [bracket_y - 0.08, bracket_y, bracket_y, bracket_y - 0.08],
            color="#333333", linewidth=1.5, zorder=5)
    ax.text(3, bracket_y + 0.05, "p = 0.0001", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#C62828")

    # Bottom annotation
    ax.text(0.5, -0.14,
            "Geometrico: solo embeddings  |  Teacher: requiere modelo adicional",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            fontstyle="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC"))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Delta F1 vs SMOTE (pp)", fontsize=12, fontweight="bold")
    ax.set_title("Comparacion: filtro geometrico vs. filtro teacher",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(-0.5, max(delta) + 1.5)
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_6_f_geometric_vs_teacher.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# FIGURE 4: Per-class Difficulty Stratified Bar Chart
# =============================================================================
def make_per_class_improvement():
    # Data: (dataset, class_name, smote_f1, soft_f1, delta)
    data = [
        # 20Newsgroups
        ("20Newsgroups", "alt.atheism",       50.9, 65.9, 15.0),
        ("20Newsgroups", "comp.graphics",     90.8, 91.3,  0.5),
        ("20Newsgroups", "sci.med",           79.6, 84.5,  4.9),
        ("20Newsgroups", "soc.religion.ch",   73.8, 81.2,  7.4),
        # Emotion
        ("Emotion", "anger",      27.8, 44.6, 16.9),
        ("Emotion", "fear",       48.8, 59.0, 10.2),
        ("Emotion", "joy",        29.4, 42.6, 13.3),
        ("Emotion", "love",       33.2, 40.8,  7.6),
        ("Emotion", "sadness",    41.3, 42.3,  1.0),
        ("Emotion", "surprise",   23.4, 30.3,  6.9),
        # Hate Speech
        ("Hate Speech", "hate_speech", 10.2, 14.9,  4.7),
        ("Hate Speech", "neither",     71.4, 67.7, -3.6),
        ("Hate Speech", "offensive",   73.3, 86.3, 13.0),
    ]

    # Assign difficulty tiers based on SMOTE F1
    def get_tier(smote_f1):
        if smote_f1 < 30:
            return "Dificil", "#EF5350"
        elif smote_f1 < 70:
            return "Media", "#FFA726"
        else:
            return "Facil", "#66BB6A"

    labels = [d[1] for d in data]
    deltas = [d[4] for d in data]
    datasets = [d[0] for d in data]
    smote_f1s = [d[2] for d in data]
    tiers = [get_tier(f)[0] for f in smote_f1s]
    colors = [get_tier(f)[1] for f in smote_f1s]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, width=0.7, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Value annotations on bars
    for i, (bar, delta_val) in enumerate(zip(bars, deltas)):
        y_pos = bar.get_height() if delta_val >= 0 else bar.get_height()
        va = "bottom" if delta_val >= 0 else "top"
        offset = 0.3 if delta_val >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
                f"{delta_val:+.1f}", ha="center", va=va, fontsize=8.5,
                fontweight="bold", color="#333333")

    # Horizontal reference at 0
    ax.axhline(0, color="#555555", linewidth=1.0, zorder=2)

    # Tier averages as horizontal lines
    tier_data = {}
    for t, d_val in zip(tiers, deltas):
        tier_data.setdefault(t, []).append(d_val)

    tier_colors_map = {"Dificil": "#EF5350", "Media": "#FFA726", "Facil": "#66BB6A"}
    for tier_name, tier_vals in tier_data.items():
        avg = np.mean(tier_vals)
        ax.axhline(avg, color=tier_colors_map[tier_name], linewidth=1.5,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.text(len(labels) - 0.3, avg + 0.3,
                f"Media {tier_name.lower()}: {avg:+.1f}",
                ha="right", va="bottom", fontsize=8.5,
                color=tier_colors_map[tier_name], fontstyle="italic")

    # Dataset separators and labels
    prev_ds = None
    ds_starts = {}
    for i, ds in enumerate(datasets):
        if ds != prev_ds:
            if prev_ds is not None:
                ax.axvline(i - 0.5, color="#CCCCCC", linewidth=1.0,
                           linestyle="--", zorder=1)
            ds_starts[ds] = i
            prev_ds = ds

    # Dataset labels at the top
    ds_ends = {}
    for ds in ds_starts:
        start = ds_starts[ds]
        indices = [j for j, d in enumerate(datasets) if d == ds]
        end = max(indices)
        ds_ends[ds] = end
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 18,
                ds, ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                          edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Delta F1 (pp) vs SMOTE", fontsize=12, fontweight="bold")
    ax.set_title("Mejora por clase segun dificultad (F1 SMOTE como referencia)",
                 fontsize=14, fontweight="bold", pad=25)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#EF5350", label="Dificil (F1 SMOTE < 30)"),
        mpatches.Patch(color="#FFA726", label="Media (30 <= F1 < 70)"),
        mpatches.Patch(color="#66BB6A", label="Facil (F1 >= 70)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10,
              framealpha=0.9)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Adjust y-limits to make room for dataset labels at top
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 1, ymax + 3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_6_c_per_class_improvement.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# Generate all figures
# =============================================================================
if __name__ == "__main__":
    print("Generating thesis figures...\n")

    p1 = make_forest_plot()
    print(f"  [1/4] Saved: {p1}")

    p2 = make_nclasses_scatter()
    print(f"  [2/4] Saved: {p2}")

    p3 = make_geometric_vs_teacher()
    print(f"  [3/4] Saved: {p3}")

    p4 = make_per_class_improvement()
    print(f"  [4/4] Saved: {p4}")

    print(f"\nAll 4 figures saved to {OUT_DIR}")
