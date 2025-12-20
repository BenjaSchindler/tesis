#!/usr/bin/env python3
"""
Genera figuras para las diapositivas actualizadas de la tesis SMOTE-LLM.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración general
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colores consistentes
COLORS = {
    'primary': '#2E86AB',      # Azul
    'secondary': '#A23B72',    # Magenta
    'success': '#28A745',      # Verde
    'warning': '#F18F01',      # Naranja
    'danger': '#C73E1D',       # Rojo
    'neutral': '#6C757D',      # Gris
    'highlight': '#FFD700',    # Dorado
}

OUTPUT_DIR = '/home/benja/Desktop/Tesis/SMOTE-LLM/plots/slides_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_nshot_pattern():
    """Slide 11a: Patrón no-monotónico del N-shot"""

    # Datos del experimento N-shot
    n_shots = [0, 3, 10, 20, 60, 100, 140, 200]
    deltas = [0.29, 0.32, 0.38, 0.49, 1.22, 0.98, 1.32, 1.15]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores: resaltar el óptimo (140)
    colors = [COLORS['primary'] if n != 140 else COLORS['success'] for n in n_shots]

    bars = ax.bar(range(len(n_shots)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Resaltar el máximo
    max_idx = deltas.index(max(deltas))
    bars[max_idx].set_edgecolor(COLORS['highlight'])
    bars[max_idx].set_linewidth(3)

    # Etiquetas
    ax.set_xticks(range(len(n_shots)))
    ax.set_xticklabels([str(n) for n in n_shots])
    ax.set_xlabel('Número de Ejemplos en el Prompt', fontsize=13)
    ax.set_ylabel('Δ Macro F1 (pp)', fontsize=13)
    ax.set_title('Impacto del Número de Ejemplos In-Context en la Calidad', fontsize=14, fontweight='bold')

    # Valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        height = bar.get_height()
        label = f'+{val:.2f}'
        if i == max_idx:
            label += '\nÓPTIMO'
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=11,
                       color=COLORS['success'])
        else:
            ax.annotate(f'+{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10)

    # Línea de tendencia suavizada
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(0, len(n_shots)-1, 100)
    spl = make_interp_spline(range(len(n_shots)), deltas, k=3)
    y_smooth = spl(x_smooth)
    ax.plot(x_smooth, y_smooth, '--', color=COLORS['secondary'], alpha=0.5, linewidth=2)

    ax.set_ylim(0, max(deltas) * 1.25)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_nshot_pattern.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_nshot_pattern.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_nshot_pattern.png/pdf")


def fig_ensemble_comparison():
    """Slide 11b: Comparación Ensembles vs Individual"""

    # Datos con nombres comprensibles
    configs = [
        'Ensemble Top-5 Extendido\n(Hold-out Correcto)',
        'Super Ensemble v2\n(K-fold 5×3)',
        'Ensemble 4 Estrategias\n(Validación)',
        'Prompting 140 Ejemplos\n(Individual)'
    ]
    deltas = [2.78, 2.05, 1.61, 1.32]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores: ensembles en azul, individual en gris
    colors = [COLORS['success'], COLORS['primary'], COLORS['primary'], COLORS['neutral']]

    bars = ax.barh(range(len(configs)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    # Resaltar el mejor
    bars[0].set_edgecolor(COLORS['highlight'])
    bars[0].set_linewidth(3)

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_xlabel('Δ Macro F1 (pp)', fontsize=13)
    ax.set_title('Ensembles Superan Configuraciones Individuales', fontsize=14, fontweight='bold')

    # Valores al lado de las barras
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        width = bar.get_width()
        label = f'+{val:.2f} pp'
        if i == 0:
            label += ' MEJOR'
        ax.annotate(label, xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=11, fontweight='bold' if i == 0 else 'normal')

    # Línea divisoria entre ensembles e individual
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(max(deltas)*0.95, 2.6, 'Ensembles', ha='right', va='bottom', fontsize=10, style='italic')
    ax.text(max(deltas)*0.95, 2.4, 'Individual', ha='right', va='top', fontsize=10, style='italic')

    # Anotación de mejora
    improvement = ((1.61 - 1.32) / 1.32) * 100
    ax.annotate(f'+{improvement:.0f}% mejor\nvs individual',
                xy=(1.61, 2), xytext=(2.2, 2.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary']),
                fontsize=10, color=COLORS['secondary'], fontweight='bold')

    ax.set_xlim(0, max(deltas) * 1.3)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_ensemble_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_ensemble_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_ensemble_comparison.png/pdf")


def fig_class_improvement():
    """Slide 11c: Mejoras por clase (breakthroughs)"""

    # Datos de mejoras por clase (Hold-out correcto)
    classes = ['ISFJ', 'ESTP', 'ENTJ', 'ISTJ', 'ESFJ', 'ENFJ', 'ENFP', 'INFP', 'INTP', 'ESFP', 'ESTJ']
    deltas = [19.9, 19.0, 6.6, 6.5, 2.4, 2.0, 0.5, 0.1, 0.1, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colores según magnitud de mejora
    colors = []
    for d in deltas:
        if d >= 15:
            colors.append(COLORS['success'])  # Breakthrough
        elif d >= 5:
            colors.append(COLORS['primary'])  # Buena mejora
        elif d > 0:
            colors.append(COLORS['warning'])  # Mejora leve
        else:
            colors.append(COLORS['danger'])   # Sin mejora

    bars = ax.bar(range(len(classes)), deltas, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_xlabel('Clase MBTI', fontsize=13)
    ax.set_ylabel('Δ F1 (pp)', fontsize=13)
    ax.set_title('Mejoras por Clase con SMOTE-LLM (Hold-out Correcto)', fontsize=14, fontweight='bold')

    # Valores sobre las barras
    for bar, val, cls in zip(bars, deltas, classes):
        height = bar.get_height()
        if val >= 15:
            label = f'+{val:.1f}\nBREAK'
            ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=9,
                       color=COLORS['success'])
        elif val > 0:
            ax.annotate(f'+{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
        else:
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                       ha='center', va='bottom', fontsize=9, color=COLORS['danger'])

    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], edgecolor='black', label='Breakthrough (≥15 pp)'),
        Patch(facecolor=COLORS['primary'], edgecolor='black', label='Buena mejora (5-15 pp)'),
        Patch(facecolor=COLORS['warning'], edgecolor='black', label='Mejora leve (<5 pp)'),
        Patch(facecolor=COLORS['danger'], edgecolor='black', label='Sin mejora (0 pp)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_ylim(0, max(deltas) * 1.2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_class_improvement.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_class_improvement.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_class_improvement.png/pdf")


def fig_class_heatmap():
    """Slide 11c alternativo: Heatmap de mejoras por clase"""

    # Datos organizados por tier
    tiers = {
        'Bajo (F1<0.20)': ['ISFJ', 'ESTP', 'ENTJ', 'ISTJ', 'ESFJ', 'ENFJ', 'ESFP', 'ESTJ', 'ISTP'],
        'Medio (0.20-0.45)': ['ENFP', 'INTP', 'ENTP', 'INFJ', 'INTJ', 'ISFP'],
        'Alto (F1≥0.45)': ['INFP']
    }

    deltas_by_class = {
        'ISFJ': 19.9, 'ESTP': 19.0, 'ENTJ': 6.6, 'ISTJ': 6.5, 'ESFJ': 2.4,
        'ENFJ': 2.0, 'ESFP': 0.0, 'ESTJ': 0.0, 'ISTP': -0.3,
        'ENFP': 0.5, 'INTP': 0.1, 'ENTP': -0.1, 'INFJ': -0.1, 'INTJ': -0.2, 'ISFP': -0.7,
        'INFP': 0.1
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 2, 0.5]})

    for ax, (tier_name, classes) in zip(axes[:2], list(tiers.items())[:2]):
        deltas = [deltas_by_class[c] for c in classes]
        colors = []
        for d in deltas:
            if d >= 15:
                colors.append(COLORS['success'])
            elif d >= 5:
                colors.append(COLORS['primary'])
            elif d > 0:
                colors.append(COLORS['warning'])
            elif d == 0:
                colors.append(COLORS['neutral'])
            else:
                colors.append(COLORS['danger'])

        bars = ax.barh(range(len(classes)), deltas, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_title(tier_name, fontweight='bold')
        ax.set_xlabel('Δ F1 (pp)')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        for bar, val in zip(bars, deltas):
            width = bar.get_width()
            ax.annotate(f'{val:+.1f}', xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(3 if val >= 0 else -3, 0), textcoords='offset points',
                       ha='left' if val >= 0 else 'right', va='center', fontsize=9)

    # Tier Alto (solo INFP)
    ax = axes[2]
    ax.barh([0], [0.1], color=COLORS['warning'], edgecolor='black', linewidth=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(['INFP'])
    ax.set_title('Alto (F1≥0.45)', fontweight='bold')
    ax.set_xlabel('Δ F1 (pp)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.annotate('+0.1', xy=(0.1, 0), xytext=(3, 0), textcoords='offset points',
               ha='left', va='center', fontsize=9)

    plt.suptitle('Mejoras por Clase Organizadas por Tier de Rendimiento', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_class_heatmap_by_tier.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_class_heatmap_by_tier.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_class_heatmap_by_tier.png/pdf")


def fig_methodology_comparison():
    """Comparación K-fold vs Hold-out"""

    configs = ['ENS_SUPER_G5_F7_v2', 'ENS_TopG5_Extended', 'TOP_all_common']
    kfold = [2.05, None, 1.61]  # None = no evaluado con K-fold
    holdout = [2.70, 2.78, None]  # None = no evaluado con hold-out

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    # Solo mostrar los que tienen ambos valores
    bars1 = ax.bar(x - width/2, [2.05, 0, 1.61], width, label='K-Fold CV (5×3)',
                   color=COLORS['primary'], edgecolor='black')
    bars2 = ax.bar(x + width/2, [2.70, 2.78, 0], width, label='Hold-out Correcto',
                   color=COLORS['success'], edgecolor='black')

    ax.set_ylabel('Δ Macro F1 (pp)', fontsize=13)
    ax.set_title('K-Fold vs Hold-out: El Hold-out Muestra Mejoras Reales', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['ENS_SUPER\n_G5_F7_v2', 'ENS_TopG5\n_Extended', 'TOP_all\n_common'], fontsize=10)
    ax.legend(loc='upper right')

    # Valores sobre las barras
    for bar in bars1:
        if bar.get_height() > 0:
            ax.annotate(f'+{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        if bar.get_height() > 0:
            ax.annotate(f'+{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10,
                       fontweight='bold', color=COLORS['success'])

    # Flecha mostrando diferencia
    ax.annotate('', xy=(0.175, 2.70), xytext=(0.175, 2.05),
               arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2))
    ax.text(0.35, 2.35, '+0.65 pp\n(+32%)', fontsize=9, color=COLORS['secondary'], fontweight='bold')

    ax.set_ylim(0, 3.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_methodology_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_methodology_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_methodology_comparison.png/pdf")


def fig_summary_results():
    """Figura resumen con todos los resultados clave"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. N-shot pattern (arriba izquierda)
    ax = axes[0, 0]
    n_shots = [0, 3, 10, 20, 60, 100, 140, 200]
    deltas = [0.29, 0.32, 0.38, 0.49, 1.22, 0.98, 1.32, 1.15]
    colors = [COLORS['primary'] if n != 140 else COLORS['success'] for n in n_shots]
    ax.bar(range(len(n_shots)), deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(n_shots)))
    ax.set_xticklabels([str(n) for n in n_shots])
    ax.set_xlabel('N-shot')
    ax.set_ylabel('Δ F1 (pp)')
    ax.set_title('A) Patrón N-shot: 140 es Óptimo', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 2. Ensemble vs Individual (arriba derecha)
    ax = axes[0, 1]
    configs = ['Ensemble\n4 Estrategias', 'Prompting\n140 Ejemplos']
    deltas = [1.61, 1.32]
    colors = [COLORS['success'], COLORS['neutral']]
    bars = ax.bar(configs, deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Δ F1 (pp)')
    ax.set_title('B) Ensemble > Individual (+22%)', fontweight='bold')
    for bar, val in zip(bars, deltas):
        ax.annotate(f'+{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 3. Mejoras por clase (abajo izquierda)
    ax = axes[1, 0]
    classes = ['ISFJ', 'ESTP', 'ENTJ', 'ISTJ', 'ESFJ', 'ENFJ']
    deltas = [19.9, 19.0, 6.6, 6.5, 2.4, 2.0]
    colors = [COLORS['success'] if d >= 15 else COLORS['primary'] if d >= 5 else COLORS['warning'] for d in deltas]
    ax.barh(range(len(classes)), deltas, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Δ F1 (pp)')
    ax.set_title('C) Breakthroughs por Clase', fontweight='bold')
    ax.invert_yaxis()
    for i, val in enumerate(deltas):
        ax.annotate(f'+{val:.1f}', xy=(val, i), xytext=(3, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 4. Resumen de hallazgos (abajo derecha)
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = """
    RESUMEN DE RESULTADOS (en pp)
    ══════════════════════════════════════

    Mejor Individual:     Prompting 140 ej.  +1.32 pp
    Mejor Ensemble:       4 Estrategias      +1.61 pp
    Hold-out Correcto:    Ensemble Top-5 Ext +2.78 pp

    ──────────────────────────────────────

    BREAKTHROUGHS POR CLASE:
    • ISFJ: +19.9 pp (de 0% a ~20%)
    • ESTP: +19.0 pp (ahora detectable)

    LIMITACIÓN CONOCIDA:
    • ESFP: 0% mejora (48 samples)

    ══════════════════════════════════════
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('D) Resumen de Hallazgos', fontweight='bold')

    plt.suptitle('SMOTE-LLM: Resultados de Validación Phase G', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_summary_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_summary_results.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Guardado: fig_summary_results.png/pdf")


if __name__ == '__main__':
    print("Generando figuras para slides...")
    print(f"Directorio de salida: {OUTPUT_DIR}\n")

    fig_nshot_pattern()
    fig_ensemble_comparison()
    fig_class_improvement()
    fig_class_heatmap()
    fig_methodology_comparison()
    fig_summary_results()

    print(f"\n✅ Todas las figuras generadas en {OUTPUT_DIR}/")
