#!/usr/bin/env python3
"""
Genera graficos para Phase G - Experimentos de Ensambles.
Crea visualizaciones de calidad academica para la tesis (en espanol).
Estilo limpio y profesional sin elementos decorativos.
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
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paleta de colores
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'significant': '#28a745',
    'baseline': '#6c757d',
    'not_significant': '#dc3545',
}

PLOTS_DIR = Path(__file__).parent
PLOTS_DIR.mkdir(exist_ok=True)


def plot_comparacion_estrategias():
    """Grafico 1: Comparacion de estrategias de generacion individuales."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Datos ordenados por Delta
    configs = [
        'Prompts Contrastivos',
        'Umbral Permisivo',
        'Volumen Ultra Alto',
        '25 Vecinos (K=25)',
        'Banda de Confianza',
        'Solo Clases Objetivo',
        'Filtros Relajados',
        'Forzar Problematicas',
        'Many-Shot (10)',
        'Sin Filtro',
        'Few-Shot (3)',
        'Sin Deduplicacion',
        'Zero-Shot',
    ]
    delta_pct = [2.69, 2.56, 2.46, 2.46, 2.46, 2.37, 2.36, 2.29, 1.87, 1.87, 1.56, 1.54, 1.43]
    n_synth = [43, 48, 50, 41, 46, 39, 33, 48, 41, 46, 45, 40, 32]

    y_pos = np.arange(len(configs))
    colors = [COLORS['accent'] if i == 0 else COLORS['significant'] for i in range(len(configs))]

    bars = ax.barh(y_pos, delta_pct, color=colors, alpha=0.8, edgecolor='black')

    # Anotaciones simples
    for i, (d, n) in enumerate(zip(delta_pct, n_synth)):
        ax.annotate(f'+{d:.2f}% (n={n})', (d + 0.05, i), va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs)
    ax.set_xlabel('Delta F1 (%)')
    ax.set_title('Comparacion de Estrategias de Generacion')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 3.5)

    # Leyenda
    best_patch = mpatches.Patch(color=COLORS['accent'], label='Mejor (+2.69%)')
    sig_patch = mpatches.Patch(color=COLORS['significant'], label='Significativo (p<0.05)')
    ax.legend(handles=[best_patch, sig_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g01_comparacion_estrategias.png')
    plt.savefig(PLOTS_DIR / 'exp_g01_comparacion_estrategias.pdf')
    plt.close()
    print("  Guardado: exp_g01_comparacion_estrategias.png/pdf")


def plot_estrategias_prompts():
    """Grafico 2: Impacto del numero de ejemplos en el prompt."""
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ['Zero-Shot\n(0 ejemplos)', 'Few-Shot\n(3 ejemplos)', 'Many-Shot\n(10 ejemplos)']
    delta_pct = [1.43, 1.56, 1.87]
    n_synth = [32, 45, 41]

    x = np.arange(len(strategies))
    width = 0.6

    colors = [COLORS['primary'], COLORS['primary'], COLORS['accent']]
    bars = ax.bar(x, delta_pct, width, color=colors, alpha=0.8, edgecolor='black')

    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Anotaciones simples
    for i, (d, n) in enumerate(zip(delta_pct, n_synth)):
        ax.annotate(f'+{d:.2f}%', (i, d + 0.05), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Impacto del Numero de Ejemplos en el Prompt')
    ax.set_ylim(0, 2.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g02_estrategias_prompts.png')
    plt.savefig(PLOTS_DIR / 'exp_g02_estrategias_prompts.pdf')
    plt.close()
    print("  Guardado: exp_g02_estrategias_prompts.png/pdf")


def plot_umbrales_filtros():
    """Grafico 3: Efecto de umbrales y filtros de calidad."""
    fig, ax = plt.subplots(figsize=(10, 5))

    configs = ['Sin Filtro', 'Umbral Permisivo', 'Filtros Relajados', 'Sin Deduplicacion']
    delta_pct = [1.87, 2.56, 2.36, 1.54]
    n_synth = [46, 48, 33, 40]

    x = np.arange(len(configs))
    width = 0.6

    colors = [COLORS['primary'], COLORS['accent'], COLORS['significant'], COLORS['primary']]
    bars = ax.bar(x, delta_pct, width, color=colors, alpha=0.8, edgecolor='black')

    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    for i, (d, n) in enumerate(zip(delta_pct, n_synth)):
        ax.annotate(f'+{d:.2f}%', (i, d + 0.05), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Efecto de Umbrales y Filtros de Calidad')
    ax.set_ylim(0, 3.0)

    best_patch = mpatches.Patch(color=COLORS['accent'], label='Mejor (+2.56%)')
    ax.legend(handles=[best_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g03_umbrales_filtros.png')
    plt.savefig(PLOTS_DIR / 'exp_g03_umbrales_filtros.pdf')
    plt.close()
    print("  Guardado: exp_g03_umbrales_filtros.png/pdf")


def plot_volumen_sinteticos():
    """Grafico 4: Relacion entre volumen de sinteticos y mejora."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_synth = [43, 48, 50, 41, 46, 39, 33, 48, 41, 46, 45, 40, 32]
    delta_pct = [2.69, 2.56, 2.46, 2.46, 2.46, 2.37, 2.36, 2.29, 1.87, 1.87, 1.56, 1.54, 1.43]

    ax.scatter(n_synth, delta_pct, s=100, c=COLORS['primary'], alpha=0.7, edgecolors='black', linewidth=1)

    # Mejor punto
    best_idx = 0
    ax.scatter([n_synth[best_idx]], [delta_pct[best_idx]], s=150, c=COLORS['accent'],
              edgecolors='black', linewidth=2, zorder=5, label='Mejor (Contrastivo)')

    # Linea de tendencia
    z = np.polyfit(n_synth, delta_pct, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(n_synth), max(n_synth), 100)
    ax.plot(x_line, p(x_line), '--', color=COLORS['baseline'], alpha=0.7, label='Tendencia')

    ax.set_xlabel('Numero de Muestras Sinteticas')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Relacion entre Volumen y Mejora')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g04_volumen_sinteticos.png')
    plt.savefig(PLOTS_DIR / 'exp_g04_volumen_sinteticos.pdf')
    plt.close()
    print("  Guardado: exp_g04_volumen_sinteticos.png/pdf")


def plot_evolucion_ensambles():
    """Grafico 5: Evolucion de ensambles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ensembles = ['Ensamble Top-3', 'Super Ensamble v1', 'Super Ensamble v2']
    delta_kfold = [5.98, 9.00, 10.03]
    delta_holdout = [8.40, 9.00, 14.33]
    n_synth = [178, 287, 327]

    x = np.arange(len(ensembles))
    width = 0.35

    bars1 = ax.bar(x - width/2, delta_kfold, width, label='K-Fold CV',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, delta_holdout, width, label='Hold-out',
                   color=COLORS['accent'], alpha=0.8, edgecolor='black')

    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    # Anotaciones
    for i, (dk, dh, n) in enumerate(zip(delta_kfold, delta_holdout, n_synth)):
        ax.annotate(f'+{dk:.1f}%', (i - width/2, dk + 0.3), ha='center', fontsize=9)
        ax.annotate(f'+{dh:.1f}%', (i + width/2, dh + 0.3), ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ensembles)
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Evolucion de Ensambles')
    ax.set_ylim(0, 17)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g05_evolucion_ensambles.png')
    plt.savefig(PLOTS_DIR / 'exp_g05_evolucion_ensambles.pdf')
    plt.close()
    print("  Guardado: exp_g05_evolucion_ensambles.png/pdf")


def plot_robustez_replicaciones():
    """Grafico 6: Robustez - varianza entre replicaciones."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ensembles = ['Ensamble Top-3', 'Ensamble Extendido', 'Super Ensamble v2']

    # Datos de las 3 replicaciones
    data = [
        [10.10, 10.20, 4.91],   # Top-3
        [9.82, 10.01, 12.23],   # Extendido
        [15.01, 14.12, 13.85],  # Super v2
    ]

    positions = [1, 2, 3]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

    # Colores
    colors_box = [COLORS['primary'], COLORS['primary'], COLORS['accent']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Puntos individuales
    for i, (pos, d) in enumerate(zip(positions, data)):
        ax.scatter([pos]*len(d), d, s=80, c='black', marker='o', zorder=5)

    # CV annotations
    cvs = [29.4, 10.2, 3.5]
    means = [8.40, 10.69, 14.33]
    for i, (pos, cv, m) in enumerate(zip(positions, cvs, means)):
        ax.annotate(f'CV={cv:.1f}%', (pos, max(data[i]) + 0.8), ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(ensembles)
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Robustez: Varianza entre 3 Replicaciones')
    ax.set_ylim(0, 18)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g06_robustez_replicaciones.png')
    plt.savefig(PLOTS_DIR / 'exp_g06_robustez_replicaciones.pdf')
    plt.close()
    print("  Guardado: exp_g06_robustez_replicaciones.png/pdf")


def plot_mejora_por_clase():
    """Grafico 7: Mejora por tipo MBTI."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Datos ordenados por delta
    clases = ['ISFJ', 'ESTP', 'ENTJ', 'ENFP', 'INTP', 'ENTP', 'ISTJ', 'ISFP',
              'INFJ', 'INFP', 'ISTP', 'INTJ', 'ENFJ', 'ESFJ', 'ESFP', 'ESTJ']
    delta_pp = [19.93, 19.05, 5.79, 0.18, 0.11, 0.03, 0.0, 0.0,
                -0.07, -0.13, -0.32, -1.35, 0.0, 0.0, 0.0, 0.0]

    # Ordenar
    sorted_indices = np.argsort(delta_pp)[::-1]
    clases = [clases[i] for i in sorted_indices]
    delta_pp = [delta_pp[i] for i in sorted_indices]

    y_pos = np.arange(len(clases))

    colors = []
    for d in delta_pp:
        if d >= 5:
            colors.append(COLORS['significant'])
        elif d > 0:
            colors.append(COLORS['primary'])
        elif d < -0.5:
            colors.append(COLORS['not_significant'])
        else:
            colors.append(COLORS['baseline'])

    bars = ax.barh(y_pos, delta_pp, color=colors, alpha=0.8, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Anotaciones solo para valores significativos
    for i, d in enumerate(delta_pp):
        if abs(d) >= 1:
            ax.annotate(f'{d:+.1f}pp', (d + 0.5 if d > 0 else d - 2.5, i),
                       va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clases)
    ax.set_xlabel('Delta F1 (puntos porcentuales)')
    ax.set_title('Mejora por Tipo MBTI (Super Ensamble v2)')
    ax.set_xlim(-5, 25)

    # Leyenda
    great_patch = mpatches.Patch(color=COLORS['significant'], label='Mejora >5pp')
    good_patch = mpatches.Patch(color=COLORS['primary'], label='Mejora 0-5pp')
    stable_patch = mpatches.Patch(color=COLORS['baseline'], label='Sin cambio')
    worse_patch = mpatches.Patch(color=COLORS['not_significant'], label='Empeora')
    ax.legend(handles=[great_patch, good_patch, stable_patch, worse_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g07_mejora_por_clase.png')
    plt.savefig(PLOTS_DIR / 'exp_g07_mejora_por_clase.pdf')
    plt.close()
    print("  Guardado: exp_g07_mejora_por_clase.png/pdf")


def plot_resumen_hallazgos():
    """Grafico 8: Resumen comparativo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Comparacion simple: Individual vs Ensambles
    categories = ['Mejor\nIndividual', 'Ensamble\nTop-3', 'Super\nEnsamble v1', 'Super\nEnsamble v2']
    delta = [2.69, 8.40, 9.00, 14.33]
    n_synth = [43, 178, 287, 327]

    x = np.arange(len(categories))
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['secondary'], COLORS['accent']]

    bars = ax.bar(x, delta, color=colors, alpha=0.8, edgecolor='black', width=0.6)

    ax.axhline(y=0, color=COLORS['baseline'], linestyle='-', linewidth=1)

    for i, (d, n) in enumerate(zip(delta, n_synth)):
        ax.annotate(f'+{d:.2f}%\n(n={n})', (i, d + 0.3), ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Comparacion: Estrategia Individual vs Ensambles')
    ax.set_ylim(0, 17)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_g08_resumen_hallazgos.png')
    plt.savefig(PLOTS_DIR / 'exp_g08_resumen_hallazgos.pdf')
    plt.close()
    print("  Guardado: exp_g08_resumen_hallazgos.png/pdf")


def main():
    """Genera todos los graficos."""
    print("=" * 60)
    print("  Generando Graficos de Phase G")
    print("=" * 60)
    print(f"Directorio de salida: {PLOTS_DIR}")
    print()

    print("Generando graficos...")
    plot_comparacion_estrategias()
    plot_estrategias_prompts()
    plot_umbrales_filtros()
    plot_volumen_sinteticos()
    plot_evolucion_ensambles()
    plot_robustez_replicaciones()
    plot_mejora_por_clase()
    plot_resumen_hallazgos()

    print()
    print("=" * 60)
    print(f"  Graficos guardados en: {PLOTS_DIR}")
    print("=" * 60)

    print("\nArchivos generados:")
    for f in sorted(PLOTS_DIR.glob('exp_g*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
