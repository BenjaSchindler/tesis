#!/usr/bin/env python3
"""
Genera graficos para todos los experimentos de Phase F Validation.
Crea visualizaciones de calidad para la tesis (en espanol).
Todos los graficos usan Delta F1 como metrica principal.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Configurar estilo para graficos de tesis
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paleta de colores
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'baseline': '#6c757d',
    'significant': '#28a745',
    'not_significant': '#dc3545',
}

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_clustering_validation():
    """Grafico 1: Validacion de K_max para clustering - Delta F1 (EXPANDED)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Datos EXPANDIDOS exp01 (K_max 1-24)
    k_max = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 24]
    delta_pct = [0.64, 0.68, 1.01, 0.94, 1.78, 1.40, 1.62, 0.49, 0.88, 1.55, -0.03, 1.11, 0.37, 1.92, 1.05]
    silhouette = [0, 0.057, 0.051, 0.047, 0.046, 0.042, 0.042, 0.040, 0.040, 0.039, 0.039, 0.039, 0.038, 0.037, 0.037]
    significant = [True, False, True, False, True, True, True, False, False, True, False, True, False, True, False]

    # Grafico 1: Delta F1 vs K_max (barras)
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]
    bars = ax1.bar(range(len(k_max)), delta_pct, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Resaltar el mejor (K_max=18)
    best_idx = np.argmax(delta_pct)
    bars[best_idx].set_color(COLORS['accent'])
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    # Resaltar rango optimo
    ax1.axvspan(4, 7, alpha=0.15, color=COLORS['accent'], label='Zona Optima (K=5-7)')

    ax1.set_xticks(range(len(k_max)))
    ax1.set_xticklabels([str(k) for k in k_max], fontsize=8)
    ax1.set_xlabel('K_max (Clusters Maximos por Clase)')
    ax1.set_ylabel('Delta F1 (%)')
    ax1.set_title('(a) Mejora F1 vs Numero de Clusters\n(Mejor: K_max=18, +1.92%)')
    ax1.set_ylim(-0.5, 2.5)

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Optimo (+1.92%)')
    ax1.legend(handles=[sig_patch, best_patch], loc='upper right', fontsize=9)

    # Grafico 2: Delta % con silhouette (linea)
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(k_max, delta_pct, 'o-', color=COLORS['secondary'],
                     linewidth=2, markersize=6, label='Delta F1')
    line2 = ax2_twin.plot(k_max, silhouette, 's--', color=COLORS['success'],
                         linewidth=2, markersize=5, label='Coef. Silhouette')

    # Marcar optimo
    ax2.scatter([18], [1.92], s=150, c=COLORS['accent'], edgecolors='black', linewidth=2, zorder=5)

    ax2.set_xlabel('K_max (Clusters Maximos por Clase)')
    ax2.set_ylabel('Delta F1 (%)', color=COLORS['secondary'])
    ax2_twin.set_ylabel('Coeficiente Silhouette', color=COLORS['success'])
    ax2.set_title('(b) Mejora vs Calidad de Clusters')

    # Combinar leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp01_clustering_validation.png')
    plt.savefig(PLOTS_DIR / 'exp01_clustering_validation.pdf')
    plt.close()
    print("  Guardado: exp01_clustering_validation.png/pdf")


def plot_anchor_strategies():
    """Grafico 2: Comparacion de estrategias de seleccion de anclas - Delta F1."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Datos
    strategies = ['Aleatorio', 'Vecino\nCercano', 'Medoide', 'Filtrado por\nCalidad', 'Diverso', 'Ensamble']
    delta_pct = [0.83, 1.07, 1.63, -0.05, 0.98, 0.15]
    significant = [False, False, True, False, False, False]

    x = np.arange(len(strategies))
    width = 0.6

    # Crear barras con colores segun significancia
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]
    bars = ax.bar(x, delta_pct, width, color=colors, alpha=0.8, edgecolor='black')

    # Linea base (delta=0)
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Agregar marcadores de significancia y etiquetas delta
    for i, (delta, sig) in enumerate(zip(delta_pct, significant)):
        color = COLORS['significant'] if sig else 'black'
        marker = '*' if sig else ''
        offset = 0.08 if delta >= 0 else -0.15
        ax.annotate(f'{delta:+.2f}%{marker}', (i, delta + offset), ha='center', fontsize=10,
                    fontweight='bold' if sig else 'normal', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_xlabel('Estrategia de Seleccion de Ancla')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Comparacion de Estrategias de Ancla\n(* = estadisticamente significativo, p < 0.05)')
    ax.set_ylim(-0.5, 2.2)

    # Leyenda para significancia
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    ns_patch = mpatches.Patch(color=COLORS['primary'], label='No Significativo')
    ax.legend(handles=[sig_patch, ns_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp02_anchor_strategies.png')
    plt.savefig(PLOTS_DIR / 'exp02_anchor_strategies.pdf')
    plt.close()
    print("  Guardado: exp02_anchor_strategies.png/pdf")


def plot_k_neighbors():
    """Grafico 3: Validacion de K vecinos (contexto del prompt) - Delta F1 (EXPANDED)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Datos EXPANDIDOS exp03 (K 5-200)
    k_values = [5, 8, 10, 12, 15, 18, 20, 25, 30, 50, 75, 100, 125, 150, 200]
    delta_pct = [0.84, 0.90, 0.51, 0.49, 0.68, 0.99, 0.73, 0.82, 0.76, 0.04, 0.95, 0.84, 1.09, 0.49, 1.60]
    acceptance = [99, 96, 95, 98, 96, 96, 97, 96, 98, 97, 97, 96, 99, 96, 97]
    context_type = ['Insuf', 'Lim', 'Lim', 'Opt', 'Opt', 'Opt', 'Opt', 'Red', 'Red', 'Red', 'Ruid', 'Ruid', 'MuyR', 'MuyR', 'Exc']
    significant = [False, False, False, False, False, False, False, True, False, False, False, False, False, False, True]

    # Grafico 1: Delta F1 vs K (linea con puntos)
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]
    ax1.plot(k_values, delta_pct, '-', color=COLORS['primary'], linewidth=1.5, alpha=0.5)
    ax1.scatter(k_values, delta_pct, c=colors, s=80, edgecolors='black', linewidth=1, zorder=5)
    ax1.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Resaltar el mejor (K=200)
    best_idx = np.argmax(delta_pct)
    ax1.scatter([k_values[best_idx]], [delta_pct[best_idx]], s=200, c=COLORS['accent'],
                edgecolors='black', linewidth=2, zorder=6, label=f'Optimo K={k_values[best_idx]}')

    # Zonas de contexto
    ax1.axvspan(5, 10, alpha=0.1, color='red', label='Insuficiente')
    ax1.axvspan(12, 20, alpha=0.1, color='green', label='Optimo')
    ax1.axvspan(75, 200, alpha=0.1, color='orange', label='Ruidoso')

    ax1.set_xlabel('K (Ejemplos en el Prompt)')
    ax1.set_ylabel('Delta F1 (%)')
    ax1.set_title('(a) Mejora F1 vs Tamano del Contexto\n(Mejor: K=200, +1.60%)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(-0.2, 2.0)

    # Grafico 2: Barras con tipo de contexto
    x = np.arange(len(k_values))
    bars = ax2.bar(x, delta_pct, color=colors, alpha=0.8, edgecolor='black')
    bars[best_idx].set_color(COLORS['accent'])

    ax2.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in k_values], fontsize=7, rotation=45)
    ax2.set_xlabel('K (Numero de Ejemplos)')
    ax2.set_ylabel('Delta F1 (%)')
    ax2.set_title('(b) Mejora por Valor de K\n(* = significativo)')

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo')
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Optimo (+1.60%)')
    ax2.legend(handles=[sig_patch, best_patch], loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp03_k_neighbors.png')
    plt.savefig(PLOTS_DIR / 'exp03_k_neighbors.pdf')
    plt.close()
    print("  Guardado: exp03_k_neighbors.png/pdf")


def plot_filter_cascade():
    """Grafico 4: Cascada de filtros - Solo v3 (adaptive ranking)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Datos v3 (ranking adaptativo) - SOLO V3
    configs = ['Solo\nLongitud', 'Longitud+\nSimilitud', 'Tres\nFiltros', 'Cascada\nCompleta']
    delta_v3 = [1.65, 1.40, 1.38, 1.76]
    n_synth_v3 = [127, 127, 129, 125]
    significant_v3 = [True, True, True, True]
    p_values_v3 = [0.012, 0.040, 0.002, 0.0009]

    x = np.arange(len(configs))
    width = 0.6

    # Crear barras con colores segun significancia
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant_v3]
    bars = ax.bar(x, delta_v3, width, color=colors, alpha=0.8, edgecolor='black')

    # Resaltar el mejor
    best_idx = np.argmax(delta_v3)
    bars[best_idx].set_color(COLORS['accent'])

    # Linea base
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Agregar anotaciones
    for i, (d, n, p) in enumerate(zip(delta_v3, n_synth_v3, p_values_v3)):
        p_str = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
        ax.annotate(f'+{d:.2f}%*\n({p_str})\nn={n}', (i, d + 0.08), ha='center', fontsize=10,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_xlabel('Configuracion de Filtros')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Cascada de Filtros con Ranking Adaptativo (v3)\n(Todos estadisticamente significativos)')
    ax.set_ylim(0, 2.5)

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Mejor configuracion')
    ax.legend(handles=[sig_patch, best_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp04_filter_cascade.png')
    plt.savefig(PLOTS_DIR / 'exp04_filter_cascade.pdf')
    plt.close()
    print("  Guardado: exp04_filter_cascade.png/pdf")


def plot_adaptive_thresholds():
    """Grafico 5: Umbrales adaptativos - Solo v2 (adaptive relaxation)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Datos v2 - SOLO V2
    configs = ['Estricto\nRelajado', 'Medio\nRelajado', 'Permisivo\nRelajado', 'Pureza\nAdaptativa']
    delta_v2 = [1.13, 0.67, 0.83, 0.99]
    n_synth_v2 = [96, 107, 100, 97]
    significant_v2 = [False, False, True, True]
    p_values_v2 = [0.056, 0.102, 0.007, 0.011]

    x = np.arange(len(configs))
    width = 0.6

    # Crear barras con colores segun significancia
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant_v2]
    bars = ax.bar(x, delta_v2, width, color=colors, alpha=0.8, edgecolor='black')

    # Resaltar el mejor
    best_idx = np.argmax(delta_v2)
    bars[best_idx].set_color(COLORS['accent'])

    # Linea base
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Agregar anotaciones
    for i, (d, n, sig, p) in enumerate(zip(delta_v2, n_synth_v2, significant_v2, p_values_v2)):
        marker = '*' if sig else ''
        p_str = f'p={p:.3f}'
        ax.annotate(f'+{d:.2f}%{marker}\n({p_str})\nn={n}', (i, d + 0.05), ha='center', fontsize=10,
                    fontweight='bold' if sig else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_xlabel('Configuracion de Umbral')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Umbrales con Relajacion Adaptativa (v2)\n(* = estadisticamente significativo)')
    ax.set_ylim(0, 1.6)

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    ns_patch = mpatches.Patch(color=COLORS['primary'], label='No Significativo')
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Mejor configuracion')
    ax.legend(handles=[sig_patch, ns_patch, best_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp05_adaptive_thresholds.png')
    plt.savefig(PLOTS_DIR / 'exp05_adaptive_thresholds.pdf')
    plt.close()
    print("  Guardado: exp05_adaptive_thresholds.png/pdf")


def plot_tier_impact():
    """Grafico 6: Analisis de impacto por tier - Delta F1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Datos por tier
    tiers = ['BAJO\n(<0.20)', 'MEDIO\n(0.20-0.45)', 'ALTO\n(>=0.45)']
    delta_avg = [5.93, 0.06, 0.11]
    n_classes = [9, 6, 1]
    std = [13.5, 0.6, 0.0]

    # Datos por clase para tier BAJO
    low_classes = ['ENTJ', 'ISTJ', 'ISFJ', 'ISFP', 'ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP']
    low_delta = [27.3, 32.0, 6.2, -12.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    low_baseline = [0.054, 0.055, 0.160, 0.184, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Grafico 1: Comparacion por tier
    colors = [COLORS['accent'], COLORS['primary'], COLORS['secondary']]
    bars = ax1.bar(range(len(tiers)), delta_avg, color=colors, alpha=0.8, edgecolor='black',
                   yerr=std, capsize=5)

    for i, (d, n, s) in enumerate(zip(delta_avg, n_classes, std)):
        ax1.annotate(f'{n} clases\n+{d:.2f}%', (i, d + s + 0.5), ha='center', fontsize=10)

    ax1.set_xticks(range(len(tiers)))
    ax1.set_xticklabels(tiers)
    ax1.set_xlabel('Tier de Rendimiento (Rango F1 Base)')
    ax1.set_ylabel('Delta F1 Promedio (%)')
    ax1.set_title('(a) Impacto de Aumento por Tier')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Grafico 2: Desglose tier BAJO
    colors_low = [COLORS['significant'] if d > 0 else
                  (COLORS['not_significant'] if d < 0 else COLORS['baseline'])
                  for d in low_delta]
    bars2 = ax2.bar(range(len(low_classes)), low_delta, color=colors_low, alpha=0.8, edgecolor='black')

    ax2.set_xticks(range(len(low_classes)))
    ax2.set_xticklabels(low_classes, rotation=45)
    ax2.set_xlabel('Tipo MBTI (Tier BAJO)')
    ax2.set_ylabel('Delta F1 (%)')
    ax2.set_title('(b) Impacto por Clase en Tier BAJO')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Agregar anotaciones de baseline
    for i, (d, b) in enumerate(zip(low_delta, low_baseline)):
        if b > 0:
            ax2.annotate(f'base={b:.2f}', (i, d + (2 if d > 0 else -4)), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp06_tier_impact.png')
    plt.savefig(PLOTS_DIR / 'exp06_tier_impact.pdf')
    plt.close()
    print("  Guardado: exp06_tier_impact.png/pdf")


def plot_weight_validation():
    """Grafico 7a: Validacion de peso por tier de rendimiento - Delta F1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Datos exp07a: weight_by_tier
    configs = ['Uniforme\n(1.0/1.0/1.0)', 'Boost LOW\n(1.5/1.0/0.5)',
               'Boost Extremo\n(2.0/0.8/0.3)', 'Solo LOW\n(1.0/0/0)']
    delta_pct = [2.64, 3.73, 5.12, 3.02]
    p_values = [0.0076, 0.0019, 0.00095, 0.0025]
    significant = [True, True, True, True]

    x = np.arange(len(configs))
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]

    bars = ax.bar(x, delta_pct, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Resaltar el mejor (tier_boost_extreme)
    best_idx = np.argmax(delta_pct)
    bars[best_idx].set_color(COLORS['accent'])
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    # Agregar etiquetas delta con p-values
    for i, (d, p) in enumerate(zip(delta_pct, p_values)):
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        ax.annotate(f'+{d:.2f}%*\n({p_str})', (i, d + 0.15), ha='center', fontsize=10,
                    fontweight='bold', color=COLORS['significant'])

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_xlabel('Configuracion de Pesos por Tier (LOW/MID/HIGH)')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Peso Diferenciado por Tier de Rendimiento\n(Mayor peso a clases difíciles mejora resultados)')
    ax.set_ylim(0, 6.5)

    # Leyenda
    legend_elements = [
        mpatches.Patch(color=COLORS['accent'], label='Optimo (+5.12%)'),
        mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp07a_weight_by_tier.png')
    plt.savefig(PLOTS_DIR / 'exp07a_weight_by_tier.pdf')
    plt.close()
    print("  Guardado: exp07a_weight_by_tier.png/pdf")


def plot_temperature_validation():
    """Grafico 7b: Validacion de temperatura del LLM con metricas de diversidad - Delta F1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Datos exp07b: temperature_diversity
    temps = [0.3, 0.5, 0.7, 0.9]
    delta_pct = [1.52, 1.16, 1.82, 2.03]
    vocab_diversity = [0.153, 0.157, 0.159, 0.162]
    p_values = [0.0089, 0.0512, 0.0045, 0.0031]
    significant = [True, False, True, True]

    x = np.arange(len(temps))

    # Grafico 1: Delta F1 (barras)
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]
    bars = ax1.bar(x, delta_pct, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    ax1.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Resaltar el mejor (T=0.9)
    best_idx = np.argmax(delta_pct)
    bars[best_idx].set_color(COLORS['accent'])

    for i, (d, sig, p) in enumerate(zip(delta_pct, significant, p_values)):
        marker = '*' if sig else ''
        ax1.annotate(f'+{d:.2f}%{marker}', (i, d + 0.08), ha='center', fontsize=10,
                    fontweight='bold' if sig else 'normal')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'τ={t}' for t in temps])
    ax1.set_xlabel('Temperatura del LLM')
    ax1.set_ylabel('Delta F1 (%)')
    ax1.set_title('(a) Mejora F1 vs Temperatura\n(* = significativo, p<0.05)')
    ax1.set_ylim(0, 2.8)

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['accent'], label='Optimo (+2.03%)')
    ns_patch = mpatches.Patch(color=COLORS['primary'], label='No Significativo')
    ax1.legend(handles=[sig_patch, ns_patch], loc='upper left')

    # Grafico 2: Delta vs Diversidad de Vocabulario
    ax2.scatter(vocab_diversity, delta_pct, s=200, c=colors, edgecolors='black', linewidth=2)
    ax2.scatter([vocab_diversity[best_idx]], [delta_pct[best_idx]], s=200,
                c=[COLORS['accent']], edgecolors='black', linewidth=2)

    for i, (v, d, t) in enumerate(zip(vocab_diversity, delta_pct, temps)):
        ax2.annotate(f'τ={t}', (v + 0.001, d + 0.05), fontsize=10)

    ax2.set_xlabel('Diversidad de Vocabulario')
    ax2.set_ylabel('Delta F1 (%)')
    ax2.set_title('(b) Mayor Diversidad = Mayor Mejora')

    # Linea de tendencia
    z = np.polyfit(vocab_diversity, delta_pct, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(vocab_diversity), max(vocab_diversity), 100)
    ax2.plot(x_line, p(x_line), '--', color=COLORS['baseline'], alpha=0.7)

    ax2.annotate('Correlacion positiva\nT alta = mas diversidad', (0.154, 1.9), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp07b_temperature_diversity.png')
    plt.savefig(PLOTS_DIR / 'exp07b_temperature_diversity.pdf')
    plt.close()
    print("  Guardado: exp07b_temperature_diversity.png/pdf")


def plot_budget_validation():
    """Grafico 7c: Validacion del presupuesto de generacion - Delta F1 (EXPANDED)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Datos EXPANDIDOS exp07c (5-30%)
    budgets = ['5%', '8%', '10%', '12%', '15%', '18%', '20%', '25%', '30%']
    budget_vals = [5, 8, 10, 12, 15, 18, 20, 25, 30]
    delta_pct = [0.49, 0.16, 1.98, 1.36, 1.87, 1.18, 2.35, 1.83, 2.12]
    n_synth = [381, 617, 775, 950, 1129, 1463, 1509, 2039, 2338]
    significant = [False, False, True, False, True, False, True, False, False]

    x = np.arange(len(budgets))

    # Grafico 1: Delta F1 (barras)
    colors = [COLORS['significant'] if sig else COLORS['primary'] for sig in significant]
    bars = ax1.bar(x, delta_pct, color=colors, alpha=0.8, edgecolor='black', width=0.7)
    ax1.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Resaltar el mejor (20%)
    best_idx = np.argmax(delta_pct)
    bars[best_idx].set_color(COLORS['accent'])

    for i, (d, sig) in enumerate(zip(delta_pct, significant)):
        marker = '*' if sig else ''
        ax1.annotate(f'+{d:.2f}%{marker}', (i, d + 0.08), ha='center', fontsize=9,
                    fontweight='bold' if sig else 'normal')

    ax1.set_xticks(x)
    ax1.set_xticklabels(budgets, fontsize=9)
    ax1.set_xlabel('Presupuesto (% del Tamano de Clase)')
    ax1.set_ylabel('Delta F1 (%)')
    ax1.set_title('(a) Mejora F1 vs Presupuesto\n(Mejor: 20%, +2.35%)')
    ax1.set_ylim(0, 3.0)

    # Leyenda
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    ns_patch = mpatches.Patch(color=COLORS['primary'], label='No Significativo')
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Optimo (+2.35%)')
    ax1.legend(handles=[sig_patch, ns_patch, best_patch], loc='upper left', fontsize=9)

    # Grafico 2: N Sinteticos vs Delta (scatter con tendencia)
    ax2.scatter(n_synth, delta_pct, s=150, c=colors, edgecolors='black', linewidth=2)
    ax2.scatter([n_synth[best_idx]], [delta_pct[best_idx]], s=250, c=[COLORS['accent']],
                edgecolors='black', linewidth=2, zorder=5)

    for i, (n, d, b) in enumerate(zip(n_synth, delta_pct, budgets)):
        offset = 50 if i != best_idx else 80
        ax2.annotate(b, (n + offset, d), fontsize=9)

    ax2.set_xlabel('Numero de Muestras Sinteticas')
    ax2.set_ylabel('Delta F1 (%)')
    ax2.set_title('(b) Mejora vs Cantidad de Muestras')

    # Agregar linea de tendencia
    z = np.polyfit(n_synth, delta_pct, 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(n_synth), max(n_synth), 100)
    ax2.plot(x_line, p(x_line), '--', color=COLORS['baseline'], alpha=0.7, label='Tendencia')

    # Anotar optimo
    ax2.annotate('Optimo\n(20%)', (n_synth[best_idx], delta_pct[best_idx]), fontsize=10, ha='center',
                 xytext=(n_synth[best_idx] + 300, delta_pct[best_idx] - 0.5),
                 arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp07c_budget_validation.png')
    plt.savefig(PLOTS_DIR / 'exp07c_budget_validation.pdf')
    plt.close()
    print("  Guardado: exp07c_budget_validation.png/pdf")


def plot_summary_comparison():
    """Grafico 8: Resumen de todos los hallazgos significativos - Delta F1 (ACTUALIZADO)."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Todos los resultados significativos (ACTUALIZADOS con todos los experimentos expandidos)
    experiments = [
        'Clustering\n(K_max=18)', 'K Vecinos\n(K=200)', 'Ancla\nMedoide', 'Cascada\nCompleta',
        'Pureza\nAdaptativa', 'Peso Tier\n(2.0/0.8/0.3)', 'Temp.\n(τ=0.9)', 'Presup.\n(20%)'
    ]
    delta = [1.92, 1.60, 1.63, 1.76, 0.99, 5.12, 2.03, 2.35]
    p_values = [0.0032, 0.0085, 0.005, 0.0009, 0.011, 0.00095, 0.0031, 0.0063]

    x = np.arange(len(experiments))

    # Color por nivel de p-value
    colors = []
    for p in p_values:
        if p < 0.001:
            colors.append(COLORS['accent'])
        elif p < 0.01:
            colors.append(COLORS['significant'])
        else:
            colors.append(COLORS['primary'])

    bars = ax.bar(x, delta, color=colors, alpha=0.8, edgecolor='black', width=0.6)

    # Agregar anotaciones de p-value
    for i, (d, p) in enumerate(zip(delta, p_values)):
        if p < 0.001:
            p_str = 'p<0.001'
        else:
            p_str = f'p={p:.3f}'
        ax.annotate(f'+{d:.2f}%\n({p_str})', (i, d + 0.15), ha='center', fontsize=10,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_xlabel('Experimento / Parametro Optimo')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Resumen: Mejores Configuraciones por Experimento\n(Todas estadisticamente significativas)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 6.5)

    # Leyenda para colores de p-value
    legend_elements = [
        mpatches.Patch(color=COLORS['accent'], label='p < 0.001'),
        mpatches.Patch(color=COLORS['significant'], label='p < 0.01'),
        mpatches.Patch(color=COLORS['primary'], label='p < 0.05'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Nivel de Significancia')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_all_significant.png')
    plt.savefig(PLOTS_DIR / 'summary_all_significant.pdf')
    plt.close()
    print("  Guardado: summary_all_significant.png/pdf")


def plot_optimal_config():
    """Grafico 9: Grafico radar de configuracion optima."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Parametros y sus valores normalizados (escala 0-1)
    categories = ['Clustering\n(K_max)', 'Estrategia\nde Ancla', 'K Vecinos',
                  'Cascada\nde Filtros', 'Umbral', 'Peso', 'Temperatura', 'Presupuesto']
    N = len(categories)

    # Valores optimos (normalizados)
    optimal = [0.5, 1.0, 0.6, 1.0, 0.8, 1.0, 0.3, 0.8]

    # Valores por defecto
    default = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.7, 0.5]

    # Angulos
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    optimal += optimal[:1]
    default += default[:1]

    # Graficar
    ax.plot(angles, optimal, 'o-', linewidth=2, label='Optimo', color=COLORS['accent'])
    ax.fill(angles, optimal, alpha=0.25, color=COLORS['accent'])

    ax.plot(angles, default, 'o-', linewidth=2, label='Por Defecto', color=COLORS['baseline'])
    ax.fill(angles, default, alpha=0.15, color=COLORS['baseline'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('Configuracion Optima vs Por Defecto\n(Escala Normalizada)')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'optimal_configuration_radar.png')
    plt.savefig(PLOTS_DIR / 'optimal_configuration_radar.pdf')
    plt.close()
    print("  Guardado: optimal_configuration_radar.png/pdf")


def main():
    """Genera todos los graficos."""
    print("=" * 60)
    print("  Generando Graficos de Validacion Phase F")
    print("  (Todos con Delta F1 como metrica principal)")
    print("=" * 60)
    print(f"Directorio de salida: {PLOTS_DIR}")
    print()

    print("Generando graficos...")
    plot_clustering_validation()
    plot_anchor_strategies()
    plot_k_neighbors()
    plot_filter_cascade()
    plot_adaptive_thresholds()
    plot_tier_impact()
    plot_weight_validation()
    plot_temperature_validation()
    plot_budget_validation()
    plot_summary_comparison()
    plot_optimal_config()

    print()
    print("=" * 60)
    print(f"  Todos los graficos guardados en: {PLOTS_DIR}")
    print("=" * 60)

    # Listar archivos generados
    print("\nArchivos generados:")
    for f in sorted(PLOTS_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
