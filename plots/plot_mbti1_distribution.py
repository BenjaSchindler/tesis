#!/usr/bin/env python3
"""
Genera un gráfico de distribución de clases para el dataset mbti_1.csv
Similar al estilo del MBTI_500 distribution plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Cargar dataset
df = pd.read_csv('/home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv')

# Contar distribución de clases
class_counts = df['type'].value_counts().sort_values(ascending=False)

print("Distribución de clases MBTI_1:")
print(class_counts)
print(f"\nTotal: {class_counts.sum()}")
print(f"Ratio desbalance: {class_counts.max()}:{class_counts.min()} = {class_counts.max()/class_counts.min():.1f}:1")

# Crear figura
fig, ax = plt.subplots(figsize=(16, 10))

# Crear gradiente de colores (verde a rojo)
n_classes = len(class_counts)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_classes))

# Crear barras
bars = ax.bar(range(n_classes), class_counts.values, color=colors, edgecolor='black', linewidth=0.5)

# Configurar etiquetas del eje X
ax.set_xticks(range(n_classes))
ax.set_xticklabels(class_counts.index, fontsize=12, fontweight='bold')

# Añadir valores encima de las barras
for i, (bar, val) in enumerate(zip(bars, class_counts.values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Identificar clases clave
max_class = class_counts.index[0]
max_val = class_counts.values[0]
min_class = class_counts.index[-1]
min_val = class_counts.values[-1]

# Encontrar umbral minoritario (por ejemplo, clases con menos de 1000 samples si existe, o usar percentil)
umbral = 300  # Ajustar según el dataset
minority_threshold_idx = None
for i, val in enumerate(class_counts.values):
    if val < umbral:
        minority_threshold_idx = i
        break

# Línea de umbral minoritario
ax.axhline(y=umbral, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax.text(n_classes - 0.5, umbral + 20, f'Umbral Minoritaria ({umbral})',
        ha='right', va='bottom', fontsize=11, color='red', fontweight='bold')

# Anotación clase mayoritaria
ax.annotate(f'{max_class}: {max_val:,}\n(Mayoritaria)',
            xy=(0, max_val), xytext=(2.5, max_val * 0.85),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#90EE90', edgecolor='green', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='green', linewidth=2))

# Anotación clase minoritaria (la más pequeña)
ax.annotate(f'{min_class}: {min_val}\n(Ultra Minoritaria)',
            xy=(n_classes-1, min_val), xytext=(n_classes-3, min_val + 400),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF6B6B', edgecolor='darkred', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='darkred', linewidth=2))

# Buscar una clase intermedia cerca del umbral para anotar
for i, (cls, val) in enumerate(class_counts.items()):
    if val < umbral * 1.5 and val >= umbral:
        ax.annotate(f'{cls}: {val}\n(Minoritaria)',
                    xy=(i, val), xytext=(i+2.5, val + 400),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFA500', edgecolor='darkorange', linewidth=2),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='darkorange', linewidth=2))
        break

# Títulos y etiquetas
ratio = max_val / min_val
ax.set_title(f'Distribución de Clases MBTI 1 Dataset\nDesbalance Extremo: Ratio {ratio:.0f}:1 ({max_class}/{min_class})',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tipo MBTI', fontsize=14, fontweight='bold')
ax.set_ylabel('Cantidad de Samples (escala lineal)', fontsize=14, fontweight='bold')

# Ajustar límites
ax.set_ylim(0, max_val * 1.15)
ax.set_xlim(-0.5, n_classes - 0.5)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Ajustar layout
plt.tight_layout()

# Guardar
output_path = '/home/benja/Desktop/Tesis/SMOTE-LLM/plots/fig_mbti1_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nGráfico guardado en: {output_path}")

# También guardar en PDF
pdf_path = '/home/benja/Desktop/Tesis/SMOTE-LLM/plots/fig_mbti1_distribution.pdf'
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"PDF guardado en: {pdf_path}")

plt.show()
