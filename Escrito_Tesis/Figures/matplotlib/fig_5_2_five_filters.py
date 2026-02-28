#!/usr/bin/env python3
"""
Figure 5.2 – Mecanismo de Seleccion de los 5 Filtros Geometricos
=================================================================
Publication-quality 2x3 grid (5 panels + 1 empty) illustrating how each
geometric filter accepts or rejects synthetic candidates in embedding space.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

# ── reproducibility ──────────────────────────────────────────────────
np.random.seed(42)

# ── style ────────────────────────────────────────────────────────────
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
    }
)

# ── colour palette ───────────────────────────────────────────────────
C_REAL = "#3B7DD8"  # blue for real data
C_ACCEPT = "#2CA02C"  # green accepted
C_REJECT = "#D62728"  # red rejected
C_NEUTRAL = "#7F7F7F"  # neutral gray
C_CENTROID = "#FF8C00"  # orange star
C_CONTOUR = "#6A5ACD"  # slate-blue contours
C_CONE = "#DAA520"  # golden-rod for cosine cone

# ── helper: generate shared data ─────────────────────────────────────
N_REAL = 10
N_SYNTH = 18

# Real data – compact 2-D Gaussian cluster
real_pts = np.random.multivariate_normal(
    mean=[3.0, 3.0], cov=[[0.35, 0.10], [0.10, 0.30]], size=N_REAL
)
centroid = real_pts.mean(axis=0)

# Synthetic candidates – broader scatter
synth_pts = np.vstack(
    [
        np.random.multivariate_normal(
            mean=[3.0, 3.0], cov=[[1.8, 0.2], [0.2, 1.8]], size=N_SYNTH - 4
        ),
        # a few deliberate outliers
        np.array([[0.2, 5.5], [5.8, 0.8], [6.0, 5.4], [0.5, 0.5]]),
    ]
)


# ── helper functions ─────────────────────────────────────────────────
def _draw_real(ax, pts, ctr):
    """Draw real data points and centroid star."""
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=C_REAL,
        s=70,
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
        label="Datos reales",
    )
    ax.scatter(
        ctr[0],
        ctr[1],
        marker="*",
        c=C_CENTROID,
        s=260,
        zorder=6,
        edgecolors="k",
        linewidths=0.5,
        label="Centroide",
    )


def _set_lims(ax, margin=0.8):
    all_pts = np.vstack([real_pts, synth_pts])
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.tick_params(labelsize=8)


def _annotate(ax, text):
    ax.annotate(
        text,
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=9,
        fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
    )


# ══════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(
    2,
    6,
    figure=fig,
    hspace=0.38,
    wspace=0.42,
    height_ratios=[1, 1],
)

# Top row: 3 panels spanning 2 columns each
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])

# Bottom row: 2 panels centred (columns 1-3 and 3-5)
ax_d = fig.add_subplot(gs[1, 0:2])
ax_e = fig.add_subplot(gs[1, 2:4])

# Hide the 6th cell
ax_empty = fig.add_subplot(gs[1, 4:6])
ax_empty.axis("off")

panels = [ax_a, ax_b, ax_c, ax_d, ax_e]
labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

# ── Panel (a): LOF ──────────────────────────────────────────────────
ax = ax_a
ax.set_title("(a)  LOF (Local Outlier Factor)")

# KDE-based density contours from real data
kde = gaussian_kde(real_pts.T, bw_method=0.45)
xg = np.linspace(-1, 7, 200)
yg = np.linspace(-1, 7, 200)
Xg, Yg = np.meshgrid(xg, yg)
Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
levels = np.percentile(Z[Z > 0], [30, 60, 85])
ax.contour(
    Xg, Yg, Z, levels=sorted(levels), colors=C_CONTOUR, linewidths=1.2, alpha=0.7
)
ax.contourf(Xg, Yg, Z, levels=[levels[0], Z.max()], colors=[C_CONTOUR], alpha=0.07)

# Classify synthetic by density
synth_density = kde(synth_pts.T)
lof_thresh = np.percentile(synth_density, 55)
accepted = synth_density >= lof_thresh
rejected = ~accepted

ax.scatter(
    synth_pts[accepted, 0],
    synth_pts[accepted, 1],
    marker="^",
    c=C_ACCEPT,
    s=80,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    label="Aceptado",
)
ax.scatter(
    synth_pts[rejected, 0],
    synth_pts[rejected, 1],
    marker="X",
    c=C_REJECT,
    s=70,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    alpha=0.7,
    label="Rechazado",
)
_draw_real(ax, real_pts, centroid)
_set_lims(ax)
_annotate(ax, "Densidad local vs. vecinos")

# ── Panel (b): Cascada Nivel 1 (Distancia) ──────────────────────────
ax = ax_b
ax.set_title("(b)  Cascada Nivel 1 (Distancia) \u2605")

# Distance circles
dists = np.linalg.norm(synth_pts - centroid, axis=1)
radii = np.percentile(dists, [33, 60, 90])
for i, r in enumerate(radii):
    circle = plt.Circle(
        centroid,
        r,
        fill=False,
        edgecolor="#888888",
        linestyle="--" if i > 0 else "-",
        linewidth=1.0 + (0.4 if i == 0 else 0),
        alpha=0.55,
    )
    ax.add_patch(circle)

# top-N by distance
n_accept_b = 7
ranking = np.argsort(dists)
acc_mask = np.zeros(len(synth_pts), dtype=bool)
acc_mask[ranking[:n_accept_b]] = True

ax.scatter(
    synth_pts[acc_mask, 0],
    synth_pts[acc_mask, 1],
    marker="^",
    c=C_ACCEPT,
    s=80,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    label="Aceptado (top-N)",
)
ax.scatter(
    synth_pts[~acc_mask, 0],
    synth_pts[~acc_mask, 1],
    marker="X",
    c=C_REJECT,
    s=70,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    alpha=0.7,
    label="Rechazado",
)
# Rank labels for accepted
for idx in ranking[:n_accept_b]:
    ax.annotate(
        f"{np.where(ranking == idx)[0][0]+1}",
        (synth_pts[idx, 0] + 0.12, synth_pts[idx, 1] + 0.12),
        fontsize=7,
        color=C_ACCEPT,
        fontweight="bold",
        zorder=7,
    )
_draw_real(ax, real_pts, centroid)
_set_lims(ax)
_annotate(ax, "Ranking por distancia euclidiana")

# ── Panel (c): Filtro Combinado ──────────────────────────────────────
ax = ax_c
ax.set_title("(c)  Filtro Combinado")

# LOF contour (reuse density)
ax.contour(
    Xg, Yg, Z, levels=sorted(levels), colors=C_CONTOUR, linewidths=1.0, alpha=0.5
)
ax.contourf(Xg, Yg, Z, levels=[levels[1], Z.max()], colors=[C_CONTOUR], alpha=0.06)

# Cosine similarity cone from origin
origin = np.array([0.0, 0.0])
ref_dir = centroid - origin
ref_angle = np.degrees(np.arctan2(ref_dir[1], ref_dir[0]))
cone_half = 15  # degrees

theta1 = np.radians(ref_angle - cone_half)
theta2 = np.radians(ref_angle + cone_half)
wedge_r = 9.0
wedge = mpatches.Wedge(
    origin,
    wedge_r,
    ref_angle - cone_half,
    ref_angle + cone_half,
    alpha=0.12,
    facecolor=C_CONE,
    edgecolor=C_CONE,
    linewidth=1.2,
    linestyle="--",
    label="Umbral coseno",
)
ax.add_patch(wedge)

# Classify: must pass BOTH LOF threshold (strict) and cosine angle
strict_lof = synth_density >= np.percentile(synth_density, 70)
angles = np.degrees(np.arctan2(synth_pts[:, 1], synth_pts[:, 0]))
in_cone = np.abs(angles - ref_angle) < cone_half
combined_acc = strict_lof & in_cone  # very few pass

ax.scatter(
    synth_pts[combined_acc, 0],
    synth_pts[combined_acc, 1],
    marker="^",
    c=C_ACCEPT,
    s=80,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    label="Aceptado",
)
ax.scatter(
    synth_pts[~combined_acc, 0],
    synth_pts[~combined_acc, 1],
    marker="X",
    c=C_REJECT,
    s=70,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    alpha=0.7,
    label="Rechazado",
)
_draw_real(ax, real_pts, centroid)
_set_lims(ax)
n_pass = combined_acc.sum()
pct = 100 * n_pass / len(synth_pts)
_annotate(ax, f"LOF \u2229 Coseno \u2192 {pct:.0f}% aceptaci\u00f3n")

# ── Panel (d): Muestreador Guiado ───────────────────────────────────
ax = ax_d
ax.set_title("(d)  Muestreador Guiado")

# Greedy diverse selection: coverage(0.6) + quality(0.4)
quality = 1.0 / (1.0 + np.linalg.norm(synth_pts - centroid, axis=1))
quality /= quality.max()

n_select = 6
selected_idx = []
remaining = list(range(len(synth_pts)))

# first pick: highest quality
first = remaining[np.argmax(quality[remaining])]
selected_idx.append(first)
remaining.remove(first)

for _ in range(n_select - 1):
    if not remaining:
        break
    sel_arr = synth_pts[selected_idx]
    scores = []
    for j in remaining:
        min_dist = cdist([synth_pts[j]], sel_arr).min()
        s = 0.6 * min_dist / 6.0 + 0.4 * quality[j]
        scores.append(s)
    best = remaining[np.argmax(scores)]
    selected_idx.append(best)
    remaining.remove(best)

guide_mask = np.zeros(len(synth_pts), dtype=bool)
guide_mask[selected_idx] = True

# Draw arrows showing selection order
for order, idx in enumerate(selected_idx[:-1]):
    nxt = selected_idx[order + 1]
    ax.annotate(
        "",
        xy=synth_pts[nxt],
        xytext=synth_pts[idx],
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555555",
            lw=0.9,
            connectionstyle="arc3,rad=0.15",
            alpha=0.45,
        ),
        zorder=3,
    )

ax.scatter(
    synth_pts[guide_mask, 0],
    synth_pts[guide_mask, 1],
    marker="^",
    c=C_ACCEPT,
    s=90,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    label="Seleccionado",
)
ax.scatter(
    synth_pts[~guide_mask, 0],
    synth_pts[~guide_mask, 1],
    marker="^",
    c=C_NEUTRAL,
    s=55,
    zorder=3,
    edgecolors="white",
    linewidths=0.5,
    alpha=0.45,
    label="Candidato",
)
# Number the selection order
for order, idx in enumerate(selected_idx):
    ax.annotate(
        f"{order+1}",
        (synth_pts[idx, 0] + 0.14, synth_pts[idx, 1] + 0.14),
        fontsize=7.5,
        color=C_ACCEPT,
        fontweight="bold",
        zorder=7,
    )

_draw_real(ax, real_pts, centroid)
_set_lims(ax)
_annotate(ax, "Cobertura (0.6) + Calidad (0.4)")

# ── Panel (e): Sin Filtro (Control) ─────────────────────────────────
ax = ax_e
ax.set_title("(e)  Sin Filtro (Control)")

ax.scatter(
    synth_pts[:, 0],
    synth_pts[:, 1],
    marker="^",
    c="#5D8AA8",
    s=75,
    zorder=4,
    edgecolors="white",
    linewidths=0.5,
    alpha=0.75,
    label="Todos aceptados",
)
_draw_real(ax, real_pts, centroid)
_set_lims(ax)
_annotate(ax, "Selecci\u00f3n aleatoria")

# ── Shared legend ────────────────────────────────────────────────────
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=C_REAL,
        markersize=9,
        label="Datos reales",
        markeredgecolor="white",
        markeredgewidth=0.5,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor=C_CENTROID,
        markersize=14,
        label="Centroide",
        markeredgecolor="k",
        markeredgewidth=0.4,
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor=C_ACCEPT,
        markersize=9,
        label="Candidato aceptado",
        markeredgecolor="white",
        markeredgewidth=0.5,
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor=C_REJECT,
        markersize=9,
        label="Candidato rechazado",
        markeredgecolor="white",
        markeredgewidth=0.5,
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower right",
    ncol=4,
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=False,
    bbox_to_anchor=(0.95, 0.02),
)

# ── suptitle ─────────────────────────────────────────────────────────
fig.suptitle(
    "Mecanismo de Selecci\u00f3n de los 5 Filtros Geom\u00e9tricos",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# ── save ─────────────────────────────────────────────────────────────
out_dir = pathlib.Path(__file__).resolve().parent
out_path = out_dir / "fig_5_2_five_filters.png"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved → {out_path}")
