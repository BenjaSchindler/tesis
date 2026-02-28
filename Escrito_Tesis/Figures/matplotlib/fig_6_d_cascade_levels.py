#!/usr/bin/env python3
"""Fig 6.D - Cascade Filter Level Comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Data from tab_filter_cascade.tex
configs = [
    ("Nivel 1\n(distancia)", 1, 0.2079, +0.34, 0.012),
    ("Nivel 2\n(+ similitud)", 2, 0.2074, +0.29, 0.040),
    ("Nivel 3\n(+ pureza KNN)", 3, 0.2074, +0.29, 0.002),
    ("Nivel 4\n(+ confianza)", 4, 0.2081, +0.36, 0.0009),
]

labels = [c[0] for c in configs]
n_filters = [c[1] for c in configs]
f1_scores = [c[2] for c in configs]
deltas = [c[3] for c in configs]
pvals = [c[4] for c in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Comparacion de Niveles de la Cascada de Filtros",
             fontsize=16, fontweight="bold", y=0.98)

# Panel (a): Delta F1 by level
colors = ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7"]
bars1 = ax1.bar(range(len(labels)), deltas, color=colors, edgecolor="#2E7D32",
                linewidth=1.5, width=0.6)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("$\\Delta$ F1 vs linea base (pp)", fontsize=12)
ax1.set_title("(a) Ganancia por Nivel de Cascada", fontsize=13, fontweight="bold")
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

# Annotate bars with delta values and significance
for i, (bar, delta, pval) in enumerate(zip(bars1, deltas, pvals)):
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"+{delta:.2f}{stars}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

# Highlight level 1 and level 4
ax1.annotate("Simplicidad\ncompetitiva", xy=(0, deltas[0]),
             xytext=(-0.5, deltas[0] + 0.12),
             fontsize=9, fontstyle="italic", color="#1B5E20",
             arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.2))

# Panel (b): Macro F1 absolute
baseline_f1 = 0.2045
bars2 = ax2.bar(range(len(labels)), f1_scores, color=colors, edgecolor="#2E7D32",
                linewidth=1.5, width=0.6)
ax2.axhline(y=baseline_f1, color="#D32F2F", linestyle="--", linewidth=1.5,
            label=f"Linea base (F1 = {baseline_f1:.4f})")
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Macro F1", fontsize=12)
ax2.set_title("(b) F1 Absoluto por Nivel", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="lower right")

# Annotate with F1 values
for bar, f1 in zip(bars2, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
             f"{f1:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Set y-limits for better visibility
ax2.set_ylim(0.203, 0.210)

# Add insight box
fig.text(0.5, 0.02,
         "Hallazgo: agregar criterios adicionales no mejora la seleccion. "
         "El nivel 1 (solo distancia) es competitivo con la cascada completa.",
         ha="center", fontsize=10, fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4",
                   edgecolor="#F9A825", alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])

out_path = "/home/benja/Desktop/Tesis/filters/Escrito_Tesis/Figures/matplotlib/fig_6_d_cascade_levels.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
