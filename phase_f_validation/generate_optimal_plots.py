#!/usr/bin/env python3
"""Generate plots for optimal combined configuration results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load data
results_dir = Path(__file__).parent / "results" / "optimal_combined"
plots_dir = Path(__file__).parent / "plots"

with open(results_dir / "optimal_fast_20251211_135415.json") as f:
    optimal = json.load(f)

with open(results_dir / "base_fast_20251211_143613.json") as f:
    base = json.load(f)

# Extract data
optimal_deltas = [r['delta'] * 100 for r in optimal['fold_results']]
base_deltas = [r['delta'] * 100 for r in base['results']]

# Figure 1: Por Defecto vs Óptima comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Bar comparison
ax = axes[0]
configs = ['Por Defecto', 'Óptima']
deltas = [base['delta_pct'], optimal['delta_pct']]
colors = ['#808080', '#2ecc71']
bars = ax.bar(configs, deltas, color=colors, edgecolor='black', linewidth=1.5)

# Add error bars for optimal (we have CI)
ci_lower = optimal['ci_95'][0] * 100
ci_upper = optimal['ci_95'][1] * 100
ax.errorbar(1, optimal['delta_pct'],
            yerr=[[optimal['delta_pct'] - ci_lower], [ci_upper - optimal['delta_pct']]],
            fmt='none', color='black', capsize=8, capthick=2)

# Annotations
ax.annotate(f"+{base['delta_pct']:.2f}%\np={base['p_value']:.3f}\n(no sig.)",
            xy=(0, base['delta_pct']), xytext=(0, base['delta_pct'] + 0.3),
            ha='center', fontsize=11, color='#666666')
ax.annotate(f"+{optimal['delta_pct']:.2f}%\np={optimal['p_value']:.6f}\n(***)",
            xy=(1, optimal['delta_pct']), xytext=(1, optimal['delta_pct'] + 0.3),
            ha='center', fontsize=11, fontweight='bold', color='#27ae60')

ax.set_ylabel(r'$\Delta$ Macro F1 (%)')
ax.set_title('(a) Comparación: Por Defecto vs Óptima')
ax.set_ylim(0, 2.0)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add improvement annotation
improvement = (optimal['delta_pct'] / base['delta_pct'] - 1) * 100
ax.annotate(f'3× mejor', xy=(0.5, 0.8), fontsize=14, ha='center',
            arrowprops=dict(arrowstyle='->', color='#e74c3c'),
            xytext=(0.5, 0.3), color='#e74c3c', fontweight='bold')

# Panel B: Per-fold results
ax = axes[1]
folds = range(1, 16)
width = 0.35
x = np.arange(len(folds))

bars1 = ax.bar(x - width/2, base_deltas, width, label='Por Defecto', color='#808080', alpha=0.7)
bars2 = ax.bar(x + width/2, optimal_deltas, width, label='Óptima', color='#2ecc71', alpha=0.7)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=np.mean(base_deltas), color='#808080', linestyle='--', linewidth=1.5, label=f'Media Por Defecto: {np.mean(base_deltas):.2f}%')
ax.axhline(y=np.mean(optimal_deltas), color='#27ae60', linestyle='--', linewidth=1.5, label=f'Media Óptima: {np.mean(optimal_deltas):.2f}%')

ax.set_xlabel('Iteración (Pliegue)')
ax.set_ylabel(r'$\Delta$ Macro F1 (%)')
ax.set_title('(b) Resultados por Iteración (15 pliegues)')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in folds])
ax.legend(loc='upper right', fontsize=9)

# Count wins
base_wins = sum(1 for d in base_deltas if d > 0)
optimal_wins = sum(1 for d in optimal_deltas if d > 0)
ax.text(0.02, 0.98, f'Victorias: Por Defecto {base_wins}/15, Óptima {optimal_wins}/15',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(plots_dir / 'optimal_base_vs_optimal.pdf', dpi=300, bbox_inches='tight')
plt.savefig(plots_dir / 'optimal_base_vs_optimal.png', dpi=150, bbox_inches='tight')
print(f"Saved: optimal_base_vs_optimal.pdf/png")

# Figure 2: Summary statistics
fig, ax = plt.subplots(figsize=(10, 6))

# Create summary comparison table as bar chart
metrics = ['Delta F1\n(%)', 'Win Rate\n(%)', 'Significativo']
base_vals = [base['delta_pct'], sum(1 for d in base_deltas if d > 0)/15*100, 0]
optimal_vals = [optimal['delta_pct'], optimal['win_rate']*100, 1]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, base_vals, width, label='BASE', color='#808080')
bars2 = ax.bar(x + width/2, optimal_vals, width, label='OPTIMAL', color='#2ecc71')

# Add values on bars
for bar, val in zip(bars1, base_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=11)

for bar, val in zip(bars2, optimal_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}' if val != 1 else 'Sí (***)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Valor')
ax.set_title('Resumen: Configuración BASE vs OPTIMAL')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 110)

# Add text box with config details
config_text = (
    "OPTIMAL Config:\n"
    f"  K_max = 18\n"
    f"  K_neighbors = 200\n"
    f"  Temp = 0.9\n"
    f"  Budget = 20%\n"
    f"  Weights = 2.0/0.8/0.3"
)
ax.text(0.98, 0.98, config_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(plots_dir / 'optimal_summary_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(plots_dir / 'optimal_summary_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: optimal_summary_comparison.pdf/png")

print("\nDone! Generated plots for optimal configuration.")
