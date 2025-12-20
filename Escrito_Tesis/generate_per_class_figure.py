#!/usr/bin/env python3
"""Generate per-class F1 improvement figure for thesis."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load FULL_SUMMARY.json
with open('/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results/FULL_SUMMARY.json', 'r') as f:
    data = json.load(f)

# Find the best config (W5_many_shot_10 - prompting with 10 examples)
best_config = None
for wave_data in data.values():
    for config in wave_data:
        if config['config'] == 'W5_many_shot_10':
            best_config = config
            break
    if best_config:
        break

if not best_config:
    # Use the first config with per_class_delta as fallback
    for wave_data in data.values():
        for config in wave_data:
            if 'per_class_delta' in config:
                best_config = config
                break
        if best_config:
            break

# Extract per-class deltas (convert to pp by multiplying by 100)
per_class_delta = best_config['per_class_delta']
classes = list(per_class_delta.keys())
deltas_pp = [per_class_delta[c] * 100 for c in classes]  # Convert to pp

# Sort by delta value
sorted_data = sorted(zip(classes, deltas_pp), key=lambda x: x[1], reverse=True)
classes_sorted = [x[0] for x in sorted_data]
deltas_sorted = [x[1] for x in sorted_data]

# Identify problem classes (ESFJ, ESFP, ESTJ)
problem_classes = {'ESFJ', 'ESFP', 'ESTJ'}
colors = ['orange' if c in problem_classes else '#2E86AB' for c in classes_sorted]

# Create figure
plt.figure(figsize=(12, 6))
bars = plt.bar(classes_sorted, deltas_sorted, color=colors, edgecolor='black', linewidth=0.5)

# Add horizontal line at 0
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

# Labels and title
plt.xlabel('Clase MBTI', fontsize=12)
plt.ylabel('$\Delta$ F1 (pp)', fontsize=12)
plt.title('Mejora en F1 por Clase Individual (Prompting 10 Ejemplos)', fontsize=14)

# Rotate x labels for readability
plt.xticks(rotation=45, ha='right')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', edgecolor='black', label='Clases con mejora'),
    Patch(facecolor='orange', edgecolor='black', label='Clases sin mejora (problema)')
]
plt.legend(handles=legend_elements, loc='upper right')

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Figures/fig_per_class_improvement.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Figures/fig_per_class_improvement.png',
            format='png', dpi=300, bbox_inches='tight')

print(f"Figure saved successfully!")
print(f"Config used: {best_config['config']}")
print(f"\nPer-class deltas (pp):")
for c, d in sorted_data:
    marker = " [PROBLEMA]" if c in problem_classes else ""
    print(f"  {c}: {d:+.2f} pp{marker}")
