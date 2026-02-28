"""
Figure 5.3 - Soft Weighting Mechanism
Generates a three-panel publication-quality figure illustrating the soft
weighting process: filter scores, temperature transformation, and weighted
training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Figure & gridspec  (three panels with small gaps for arrows)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(
    1, 5,
    width_ratios=[0.30, 0.04, 0.36, 0.04, 0.26],
    wspace=0.05,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_arrow1 = fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[0, 2])
ax_arrow2 = fig.add_subplot(gs[0, 3])
ax_c = fig.add_subplot(gs[0, 4])

# Hide arrow axes
for ax_arr in (ax_arrow1, ax_arrow2):
    ax_arr.set_axis_off()

# =========================================================================
# Panel (a) - Filter Scores (bar chart)
# =========================================================================
n_samples = 12
scores = np.sort(rng.uniform(0.08, 0.97, n_samples))
# Ensure a nice spread: manually tweak extremes
scores[0] = 0.10
scores[1] = 0.18
scores[2] = 0.27
scores[3] = 0.35
scores[4] = 0.42
scores[5] = 0.48
scores[6] = 0.55
scores[7] = 0.63
scores[8] = 0.72
scores[9] = 0.80
scores[10] = 0.88
scores[11] = 0.95

# Colour map: red (low) -> yellow (mid) -> green (high)
cmap_rg = LinearSegmentedColormap.from_list(
    "rg", [(0.85, 0.15, 0.15), (1.0, 0.85, 0.2), (0.2, 0.7, 0.2)]
)
bar_colors = [cmap_rg(s) for s in scores]

bars = ax_a.bar(
    range(n_samples), scores,
    color=bar_colors, edgecolor="0.3", linewidth=0.6, width=0.72, zorder=3,
)

# Threshold line
threshold = 0.50
ax_a.axhline(threshold, color="0.25", ls="--", lw=1.3, zorder=4)
ax_a.text(
    n_samples - 0.5, threshold + 0.025,
    "Umbral binario",
    ha="right", va="bottom", fontsize=9, fontstyle="italic", color="0.25",
)

# Shade rejected region
ax_a.axhspan(0, threshold, color="red", alpha=0.04, zorder=1)
ax_a.axhspan(threshold, 1.0, color="green", alpha=0.04, zorder=1)
ax_a.text(0.3, 0.22, "Rechazado\n(modo binario)", fontsize=8, color="0.45",
          ha="left", transform=ax_a.transAxes)
ax_a.text(0.3, 0.78, "Aceptado\n(modo binario)", fontsize=8, color="0.45",
          ha="left", transform=ax_a.transAxes)

ax_a.set_xlabel("Muestra sintética", fontsize=11)
ax_a.set_ylabel("Puntuación del filtro", fontsize=11)
ax_a.set_title("Paso 1: Puntuaciones", fontsize=12, fontweight="bold", pad=10)
ax_a.set_ylim(0, 1.05)
ax_a.set_xticks(range(n_samples))
ax_a.set_xticklabels([f"$x_{{{i+1}}}$" for i in range(n_samples)], fontsize=8)
ax_a.tick_params(axis="y", labelsize=9)

# =========================================================================
# Arrow 1  (a -> b)
# =========================================================================
ax_arrow1.annotate(
    "",
    xy=(0.95, 0.5), xycoords="axes fraction",
    xytext=(0.05, 0.5), textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>", color="0.35", lw=2.2,
        mutation_scale=18,
    ),
)

# =========================================================================
# Panel (b) - Temperature Transformation
# =========================================================================
s = np.linspace(0, 1, 500)
w_min = 0.01

temperatures = {
    "T=1.0 (lineal)":      (1.0,  "0.50", "--",  1.5),
    "T=0.5 (amplifica diferencias)": (0.5, "#1f77b4", "-", 2.5),
    "T=0.25 (agresivo)":   (0.25, "#e67e22", ":",  1.8),
}

curves = {}
for label, (T, color, ls, lw) in temperatures.items():
    w = np.maximum(w_min, s ** (1.0 / T))
    curves[label] = w
    ax_b.plot(s, w, color=color, ls=ls, lw=lw, label=label, zorder=4)

# Shaded region between T=0.5 and T=1.0
w_t05 = curves["T=0.5 (amplifica diferencias)"]
w_t10 = curves["T=1.0 (lineal)"]
ax_b.fill_between(
    s, w_t10, w_t05,
    where=(w_t05 >= w_t10),
    color="#1f77b4", alpha=0.12, zorder=2,
    label="Efecto de amplificación",
)

# w_min line
ax_b.axhline(w_min, color="0.4", ls="--", lw=1.0, zorder=3)
ax_b.text(0.02, w_min + 0.025, r"$w_{\min}$", fontsize=10, color="0.4")

# Formula
ax_b.text(
    0.97, 0.15,
    r"$w(\mathbf{x}) = \max\!\left(w_{\min},\; s_{\mathrm{norm}}^{\,1/T}\right)$",
    transform=ax_b.transAxes,
    fontsize=12, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9),
)

ax_b.set_xlabel("Puntuación normalizada $s$", fontsize=11)
ax_b.set_ylabel("Peso $w(\\mathbf{x})$", fontsize=11)
ax_b.set_title(
    "Paso 2: Escalado por Temperatura",
    fontsize=12, fontweight="bold", pad=10,
)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1.05)
ax_b.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_b.tick_params(labelsize=9)

# =========================================================================
# Arrow 2  (b -> c)
# =========================================================================
ax_arrow2.annotate(
    "",
    xy=(0.95, 0.5), xycoords="axes fraction",
    xytext=(0.05, 0.5), textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>", color="0.35", lw=2.2,
        mutation_scale=18,
    ),
)

# =========================================================================
# Panel (c) - Weighted Training (scatter + decision boundary)
# =========================================================================
# Generate two-class data
n_real = 14
n_synth = 18

# Real data (class 0 and class 1)
real_c0 = rng.normal(loc=[1.8, 3.5], scale=0.55, size=(n_real // 2, 2))
real_c1 = rng.normal(loc=[4.2, 1.5], scale=0.55, size=(n_real // 2, 2))

# Synthetic data with varying quality
# High quality (w ~ 0.8)
synth_high = np.vstack([
    rng.normal(loc=[2.0, 3.2], scale=0.5, size=(3, 2)),
    rng.normal(loc=[4.0, 1.8], scale=0.5, size=(3, 2)),
])
# Medium quality (w ~ 0.4)
synth_med = np.vstack([
    rng.normal(loc=[2.5, 2.8], scale=0.7, size=(3, 2)),
    rng.normal(loc=[3.5, 2.2], scale=0.7, size=(3, 2)),
])
# Low quality (w ~ 0.1)
synth_low = np.vstack([
    rng.normal(loc=[3.0, 3.0], scale=0.8, size=(3, 2)),
    rng.normal(loc=[3.0, 2.0], scale=0.8, size=(3, 2)),
])

# Plot each group
ms_real = 110
ms_high = 75
ms_med = 45
ms_low = 22

ax_c.scatter(
    real_c0[:, 0], real_c0[:, 1],
    s=ms_real, c="#1f77b4", marker="o", edgecolors="0.2", linewidths=0.6,
    zorder=5, label=r"Real ($w = 1.0$)",
)
ax_c.scatter(
    real_c1[:, 0], real_c1[:, 1],
    s=ms_real, c="#1f77b4", marker="o", edgecolors="0.2", linewidths=0.6,
    zorder=5,
)

ax_c.scatter(
    synth_high[:, 0], synth_high[:, 1],
    s=ms_high, c="#2ca02c", marker="^", edgecolors="0.2", linewidths=0.6,
    zorder=4, label=r"Sintético ($w \approx 0.8$)",
)
ax_c.scatter(
    synth_med[:, 0], synth_med[:, 1],
    s=ms_med, c="#ff7f0e", marker="^", edgecolors="0.2", linewidths=0.6,
    zorder=4, label=r"Sintético ($w \approx 0.4$)",
)
ax_c.scatter(
    synth_low[:, 0], synth_low[:, 1],
    s=ms_low, c="#d62728", marker="^", edgecolors="0.2", linewidths=0.6,
    zorder=4, label=r"Sintético ($w \approx 0.1$)",
)

# Decision boundary (a plausible separating line)
xb = np.linspace(0.5, 5.5, 100)
yb = -0.85 * xb + 5.3
ax_c.plot(xb, yb, "k--", lw=1.6, zorder=3, label="Frontera de decisión")

ax_c.set_xlabel("Característica 1", fontsize=11)
ax_c.set_ylabel("Característica 2", fontsize=11)
ax_c.set_title("Paso 3: Entrenamiento", fontsize=12, fontweight="bold", pad=10)
ax_c.set_xlim(0.3, 5.7)
ax_c.set_ylim(0.0, 5.2)
ax_c.legend(
    loc="upper right", fontsize=7.8, framealpha=0.92,
    handletextpad=0.4, borderpad=0.5, labelspacing=0.45,
)
ax_c.tick_params(labelsize=9)

# =========================================================================
# Super-title
# =========================================================================
fig.suptitle(
    "Mecanismo de Ponderación Suave",
    fontsize=15, fontweight="bold", y=1.03,
)

# Subtitle letters
ax_a.text(
    0.5, -0.18, "(a)", transform=ax_a.transAxes,
    fontsize=11, fontweight="bold", ha="center",
)
ax_b.text(
    0.5, -0.18, "(b)", transform=ax_b.transAxes,
    fontsize=11, fontweight="bold", ha="center",
)
ax_c.text(
    0.5, -0.18, "(c)", transform=ax_c.transAxes,
    fontsize=11, fontweight="bold", ha="center",
)

# =========================================================================
# Save
# =========================================================================
out_path = (
    "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/"
    "Figures/matplotlib/fig_5_3_soft_weighting.png"
)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved to {out_path}")
