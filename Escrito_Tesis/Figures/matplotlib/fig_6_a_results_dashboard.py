#!/usr/bin/env python3
"""
Figure 6.a - Resumen de Resultados Principales
2x2 Results Summary Dashboard for thesis.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
GREEN_DARK  = "#2E7D32"
GREEN_LIGHT = "#66BB6A"
RED_DARK    = "#C62828"
RED_LIGHT   = "#EF5350"
BASELINE_GREY = "#9E9E9E"
NER_BASE    = "#B0BEC5"
NER_NOFILT  = "#5C6BC0"
NER_CASCADE = "#26A69A"
NER_LOF     = "#AB47BC"
TEXT_COLOR  = "#333333"

# =========================================================================
# DATA
# =========================================================================

# --- Panel (a): Delta vs SMOTE por Metodo ---
# Sorted by delta descending
methods_a = [
    "Pond. suave",
    "Filtro binario",
    "Sin aumentacion",
    "Trad. inversa",
    "EDA",
    "BERT contextual",
    "Mixup",
    "Sobrem. aleatorio",
    "SMOTE (base)",
    "Parafr. T5",
]
f1_means_a = [0.7508, 0.7487, 0.7312, 0.7241, 0.7233,
              0.7222, 0.7247, 0.7232, 0.7248, 0.7213]
deltas_a   = [+2.61, +2.39, +0.64, -0.07, -0.15,
              -0.26, -0.01, -0.16, 0.0,  -0.35]
ci_lo_a    = [+1.98, +1.75, +0.42, -0.42, -0.57,
              -0.58, -0.13, -0.35, 0.0,  -0.62]
ci_hi_a    = [+3.25, +3.07, +0.89, +0.26, +0.27,
              +0.02, +0.11, +0.02, 0.0,  -0.11]

# Significance stars (*** for p<0.001)
sig_a = ["***", "***", "***", "", "", "", "", "", "", "***"]

# --- Panel (b): Ganancia por N-shot ---
nshots     = [10, 25, 50]
delta_pond = [+4.89, +2.17, +0.76]
delta_filt = [+4.44, +1.96, +0.77]
# Cohen's d annotations
cohens_d_pond = ["d=1.42", "d=0.71", "d=0.25"]
cohens_d_filt = ["d=1.28", "d=0.64", "d=0.26"]

# --- Panel (c): Ganancia por Clasificador ---
classifiers = ["Reg. logistica", "SVC (lineal)", "Ridge"]
delta_pond_c = [+2.09, +2.98, +2.74]
delta_filt_c = [+1.61, +2.86, +2.69]
sig_pond_c   = ["***", "***", "***"]
sig_filt_c   = ["***", "***", "***"]

# --- Panel (d): Extension a NER ---
corpora      = ["MultiNERD", "WikiANN", "Few-NERD"]
baseline_ner = [0.3158, 0.2194, 0.1645]
nofilt_ner   = [0.4401, 0.2872, 0.2417]
cascade_ner  = [0.4428, 0.2983, 0.2466]
lof_ner      = [0.4529, 0.2904, 0.2347]

# =========================================================================
# FIGURE
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("white")
fig.suptitle("Resumen de Resultados Principales",
             fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.97)

# =========================================================================
# Panel (a) - Delta vs SMOTE por Metodo  (horizontal bars)
# =========================================================================
ax = axes[0, 0]

# Reverse lists so highest delta is at the top of the horizontal bar chart
methods_rev = methods_a[::-1]
deltas_rev  = deltas_a[::-1]
ci_lo_rev   = ci_lo_a[::-1]
ci_hi_rev   = ci_hi_a[::-1]
sig_rev     = sig_a[::-1]

y_pos = np.arange(len(methods_rev))
colors = [GREEN_DARK if d > 0 else (BASELINE_GREY if d == 0 else RED_DARK)
          for d in deltas_rev]

# Error bar calculations: asymmetric [lower_err, upper_err]
xerr_lo = [d - lo for d, lo in zip(deltas_rev, ci_lo_rev)]
xerr_hi = [hi - d for d, hi in zip(deltas_rev, ci_hi_rev)]

ax.barh(y_pos, deltas_rev, height=0.62, color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5, zorder=3)
ax.errorbar(deltas_rev, y_pos, xerr=[xerr_lo, xerr_hi],
            fmt="none", ecolor="#555555", elinewidth=1.2, capsize=3, zorder=4)

# Vertical dashed line at 0
ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=2)

# Y-tick labels (bold for top two methods)
ax.set_yticks(y_pos)
labels_a = []
for m in methods_rev:
    if m in ("Pond. suave", "Filtro binario"):
        labels_a.append(m)
    else:
        labels_a.append(m)
ax.set_yticklabels(labels_a, fontsize=9.5)

# Bold the highlighted labels
for tick_label in ax.get_yticklabels():
    if tick_label.get_text() in ("Pond. suave", "Filtro binario"):
        tick_label.set_fontweight("bold")
        tick_label.set_color(GREEN_DARK)

# Add significance stars to the right of bars
for i, (d, s) in enumerate(zip(deltas_rev, sig_rev)):
    if s:
        x_offset = max(d, 0) + 0.15
        if d < 0:
            x_offset = d - 0.15
            ha = "right"
        else:
            ha = "left"
        ax.text(ci_hi_rev[i] + 0.12 if d >= 0 else ci_lo_rev[i] - 0.12,
                i, s, ha=ha, va="center", fontsize=9, fontweight="bold",
                color=GREEN_DARK if d > 0 else RED_DARK, zorder=5)

# Add delta values inside or beside bars
for i, d in enumerate(deltas_rev):
    if d != 0:
        sign = "+" if d > 0 else ""
        ax.text(d / 2 if abs(d) > 1.0 else (d + (0.25 if d > 0 else -0.25)),
                i, f"{sign}{d:.2f}", ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color="white" if abs(d) > 1.0 else TEXT_COLOR, zorder=5)

ax.set_xlabel("Delta F1 vs SMOTE (pp)", fontsize=10, fontweight="bold")
ax.set_title("Delta vs SMOTE por Metodo", fontsize=12, fontweight="bold",
             pad=10)
ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top")
ax.set_xlim(-1.0, 4.2)

# =========================================================================
# Panel (b) - Ganancia por N-shot  (grouped bars)
# =========================================================================
ax = axes[0, 1]

x_b = np.arange(len(nshots))
width = 0.32

bars_pond = ax.bar(x_b - width / 2, delta_pond, width, label="Pond. suave",
                   color=GREEN_DARK, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)
bars_filt = ax.bar(x_b + width / 2, delta_filt, width, label="Filtro binario",
                   color=GREEN_LIGHT, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)

# Baseline dashed line at 0
ax.axhline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=2)

# Cohen's d annotations
for i, (bp, bf) in enumerate(zip(bars_pond, bars_filt)):
    ax.text(bp.get_x() + bp.get_width() / 2, bp.get_height() + 0.12,
            cohens_d_pond[i], ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=GREEN_DARK, zorder=5)
    ax.text(bf.get_x() + bf.get_width() / 2, bf.get_height() + 0.12,
            cohens_d_filt[i], ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="#388E3C", zorder=5)

# Value labels on top of bars
for i, (dp, df) in enumerate(zip(delta_pond, delta_filt)):
    ax.text(x_b[i] - width / 2, dp / 2, f"+{dp:.2f}",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=5)
    ax.text(x_b[i] + width / 2, df / 2, f"+{df:.2f}",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=5)

ax.set_xticks(x_b)
ax.set_xticklabels([f"N={n}" for n in nshots], fontsize=10)
ax.set_ylabel("Delta F1 vs SMOTE (pp)", fontsize=10, fontweight="bold")
ax.set_title("Ganancia por N-shot", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.text(-0.08, 1.05, "(b)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top")
ax.set_ylim(-0.3, 6.2)

# =========================================================================
# Panel (c) - Ganancia por Clasificador  (grouped bars)
# =========================================================================
ax = axes[1, 0]

x_c = np.arange(len(classifiers))

bars_pond_c = ax.bar(x_c - width / 2, delta_pond_c, width,
                     label="Pond. suave", color=GREEN_DARK, alpha=0.9,
                     edgecolor="white", linewidth=0.5, zorder=3)
bars_filt_c = ax.bar(x_c + width / 2, delta_filt_c, width,
                     label="Filtro binario", color=GREEN_LIGHT, alpha=0.9,
                     edgecolor="white", linewidth=0.5, zorder=3)

# Baseline
ax.axhline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=2)

# Significance stars and value labels
for i in range(len(classifiers)):
    # Pond. suave
    dp = delta_pond_c[i]
    ax.text(x_c[i] - width / 2, dp + 0.08, sig_pond_c[i],
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=GREEN_DARK, zorder=5)
    ax.text(x_c[i] - width / 2, dp / 2, f"+{dp:.2f}",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=5)
    # Filtro binario
    df = delta_filt_c[i]
    ax.text(x_c[i] + width / 2, df + 0.08, sig_filt_c[i],
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#388E3C", zorder=5)
    ax.text(x_c[i] + width / 2, df / 2, f"+{df:.2f}",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=5)

ax.set_xticks(x_c)
ax.set_xticklabels(classifiers, fontsize=10)
ax.set_ylabel("Delta F1 vs SMOTE (pp)", fontsize=10, fontweight="bold")
ax.set_title("Ganancia por Clasificador", fontsize=12, fontweight="bold",
             pad=10)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.text(-0.08, 1.05, "(c)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top")
ax.set_ylim(-0.3, 4.0)

# =========================================================================
# Panel (d) - Extension a NER  (grouped bars, absolute F1)
# =========================================================================
ax = axes[1, 1]

x_d = np.arange(len(corpora))
w4 = 0.18  # narrower bars for 4 groups

bars_base = ax.bar(x_d - 1.5 * w4, baseline_ner, w4, label="Baseline",
                   color=NER_BASE, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)
bars_nf   = ax.bar(x_d - 0.5 * w4, nofilt_ner, w4, label="Sin filtro",
                   color=NER_NOFILT, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)
bars_cas  = ax.bar(x_d + 0.5 * w4, cascade_ner, w4, label="Cascade L1",
                   color=NER_CASCADE, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)
bars_lof  = ax.bar(x_d + 1.5 * w4, lof_ner, w4, label="LOF relajado",
                   color=NER_LOF, alpha=0.9, edgecolor="white",
                   linewidth=0.5, zorder=3)

# Value labels on top of each bar
for bars in [bars_base, bars_nf, bars_cas, bars_lof]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=TEXT_COLOR, rotation=45, zorder=5)

ax.set_xticks(x_d)
ax.set_xticklabels(corpora, fontsize=10)
ax.set_ylabel("F1 promedio", fontsize=10, fontweight="bold")
ax.set_title("Extension a NER", fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)
ax.text(-0.08, 1.05, "(d)", transform=ax.transAxes, fontsize=14,
        fontweight="bold", va="top")
ax.set_ylim(0, 0.55)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# =========================================================================
# Final layout
# =========================================================================
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_path = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib/fig_6_a_results_dashboard.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
