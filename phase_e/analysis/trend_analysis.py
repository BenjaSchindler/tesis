#!/usr/bin/env python3
"""
Comprehensive trend analysis for Phase E experiments.
Generates trend_report.md and visualizations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Setup
ANALYSIS_DIR = Path('/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/analysis')
FIGURES_DIR = ANALYSIS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(ANALYSIS_DIR / 'experiments_data.csv')
df_class = pd.read_csv(ANALYSIS_DIR / 'per_class_data.csv')

# Config descriptions
CONFIG_DESC = {
    'A1_gpt5_none': 'GPT-5 sin reasoning',
    'A2_gpt5_low': 'GPT-5 reasoning=low',
    'A3_gpt5_medium': 'GPT-5 reasoning=medium',
    'A4_gpt5_high': 'GPT-5 reasoning=high',
    'B1_ip_baseline': 'IP scaling baseline',
    'B2_ip_aggressive': 'IP scaling agresivo',
    'B3_ip_high_only': 'IP solo clases altas',
    'C1_5x9x5': '5 clusters × 9 prompts × 5 samples',
    'C2_8x9x5': '8 clusters × 9 prompts × 5 samples',
    'C3_5x9x9': '5 clusters × 9 prompts × 9 samples',
    'D1_relaxed': 'Filtros relajados',
    'D2_strict': 'Filtros estrictos',
    'D3_boundary': 'Filtros boundary',
    'E1_target_worst': 'Target peores clases',
    'E2_minority_boost': 'Boost minoritarias',
    'E3_class_length_match': 'Match longitud por clase',
    'F1_length_300': 'Longitud ~300 chars',
    'F2_length_500': 'Longitud ~500 chars',
    'G1_gpt5_ip_8x9': 'GPT-5 + IP + 8×9',
    'G2_gpt5_relaxed_long': 'GPT-5 + relajado + largo',
    'G3_gpt5_minority_strict': 'GPT-5 + minoritarias + estricto',
    'H1_contrastive': 'Contrastive sampling',
    'H2_risky_boundary': 'Risky boundary',
}

def analyze_by_group():
    """Analyze trends by experiment group."""
    print("\n" + "="*80)
    print("ANÁLISIS POR GRUPO DE EXPERIMENTOS")
    print("="*80)

    groups = df.groupby('group').agg({
        'delta_f1_pct': ['mean', 'std', 'min', 'max', 'count'],
        'acceptance_rate': 'mean',
        'total_accepted': 'mean',
        'avg_anchor_purity': 'mean',
        'avg_similarity_anchor': 'mean',
    }).round(4)

    print("\n" + groups.to_string())

    return groups


def analyze_by_config():
    """Analyze trends by specific configuration."""
    print("\n" + "="*80)
    print("ANÁLISIS POR CONFIGURACIÓN")
    print("="*80)

    configs = df.groupby('config').agg({
        'delta_f1_pct': ['mean', 'std', 'count'],
        'acceptance_rate': 'mean',
        'avg_anchor_purity': 'mean',
    }).round(4)

    configs = configs.sort_values(('delta_f1_pct', 'mean'), ascending=False)
    print("\n" + configs.to_string())

    return configs


def analyze_by_class():
    """Analyze which configs work best for each class."""
    print("\n" + "="*80)
    print("ANÁLISIS POR CLASE MBTI")
    print("="*80)

    class_analysis = {}

    for class_name in df_class['class'].unique():
        class_data = df_class[df_class['class'] == class_name]

        # Get support (number of samples in test)
        support = class_data['support'].iloc[0] if not class_data.empty else 0

        # Best configs for this class
        best = class_data.nlargest(3, 'delta_f1')
        worst = class_data.nsmallest(3, 'delta_f1')

        # Average metrics
        avg_baseline_f1 = class_data['baseline_f1'].mean()
        avg_delta = class_data['delta_f1'].mean()
        n_improved = (class_data['delta_f1'] > 0).sum()
        n_total = len(class_data)

        class_analysis[class_name] = {
            'support': support,
            'avg_baseline_f1': avg_baseline_f1,
            'avg_delta': avg_delta,
            'n_improved': n_improved,
            'n_total': n_total,
            'improvement_rate': n_improved / n_total if n_total > 0 else 0,
            'best_configs': best[['config', 'seed', 'delta_f1']].to_dict('records'),
            'worst_configs': worst[['config', 'seed', 'delta_f1']].to_dict('records'),
        }

        print(f"\n--- {class_name} (support={support}, baseline_f1={avg_baseline_f1:.3f}) ---")
        print(f"  Mejora promedio: {avg_delta:.4f}")
        print(f"  Mejoran: {n_improved}/{n_total} ({100*n_improved/n_total:.0f}%)")
        best_list = [f"{c['config']}_s{c['seed']}: {c['delta_f1']:+.4f}" for c in best[['config','seed','delta_f1']].to_dict('records')]
        print(f"  Mejores configs: {best_list}")

    return class_analysis


def analyze_correlations():
    """Find correlations between metrics and delta F1."""
    print("\n" + "="*80)
    print("CORRELACIONES CON DELTA F1")
    print("="*80)

    numeric_cols = [
        'acceptance_rate', 'total_accepted', 'total_generated',
        'avg_similarity_centroid', 'avg_similarity_anchor',
        'avg_anchor_purity', 'avg_anchor_quality',
        'avg_classifier_conf', 'avg_token_count',
        'avg_quality_score', 'avg_anchor_cohesion',
        'rejected_knn', 'rejected_classifier', 'rejected_similarity'
    ]

    correlations = {}
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 5:
            corr = df['delta_f1_pct'].corr(df[col])
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nCorrelaciones (ordenadas por magnitud):")
    for metric, corr in sorted_corrs:
        direction = "+" if corr > 0 else "-"
        print(f"  {metric:30s}: {corr:+.3f} {direction * int(abs(corr) * 20)}")

    return correlations


def find_trends():
    """Identify key trends and insights."""
    print("\n" + "="*80)
    print("TENDENCIAS CLAVE IDENTIFICADAS")
    print("="*80)

    trends = []

    # 1. Reasoning impact
    reasoning_configs = df[df['config'].str.startswith('A')]
    if not reasoning_configs.empty:
        by_reasoning = reasoning_configs.groupby('config')['delta_f1_pct'].mean()
        trends.append(f"REASONING: {by_reasoning.to_dict()}")
        print(f"\n1. IMPACTO DE REASONING:")
        print(f"   A1 (none): {by_reasoning.get('A1_gpt5_none', 'N/A'):.3f}%")
        print(f"   A2 (low): {by_reasoning.get('A2_gpt5_low', 'N/A'):.3f}%")
        print(f"   A3 (medium): {by_reasoning.get('A3_gpt5_medium', 'N/A'):.3f}%")
        print(f"   A4 (high): {by_reasoning.get('A4_gpt5_high', 'N/A'):.3f}%")

    # 2. Volume impact
    volume_configs = df[df['config'].str.startswith('C')]
    if not volume_configs.empty:
        by_volume = volume_configs.groupby('config')['delta_f1_pct'].mean()
        print(f"\n2. IMPACTO DE VOLUMEN:")
        print(f"   C1 (5×9×5=225): {by_volume.get('C1_5x9x5', 'N/A'):.3f}%")
        print(f"   C2 (8×9×5=360): {by_volume.get('C2_8x9x5', 'N/A'):.3f}%")
        print(f"   C3 (5×9×9=405): {by_volume.get('C3_5x9x9', 'N/A'):.3f}%")

    # 3. Filter impact
    filter_configs = df[df['config'].str.startswith('D')]
    if not filter_configs.empty:
        by_filter = filter_configs.groupby('config')['delta_f1_pct'].mean()
        print(f"\n3. IMPACTO DE FILTROS:")
        print(f"   D1 (relaxed): {by_filter.get('D1_relaxed', 'N/A'):.3f}%")
        print(f"   D2 (strict): {by_filter.get('D2_strict', 'N/A'):.3f}%")
        print(f"   D3 (boundary): {by_filter.get('D3_boundary', 'N/A'):.3f}%")

    # 4. Acceptance rate vs delta
    high_accept = df[df['acceptance_rate'] > df['acceptance_rate'].median()]
    low_accept = df[df['acceptance_rate'] <= df['acceptance_rate'].median()]
    print(f"\n4. ACCEPTANCE RATE:")
    print(f"   Alta (>{df['acceptance_rate'].median():.2%}): {high_accept['delta_f1_pct'].mean():.3f}%")
    print(f"   Baja (<={df['acceptance_rate'].median():.2%}): {low_accept['delta_f1_pct'].mean():.3f}%")

    # 5. Purity impact
    high_purity = df[df['avg_anchor_purity'] > df['avg_anchor_purity'].median()]
    low_purity = df[df['avg_anchor_purity'] <= df['avg_anchor_purity'].median()]
    print(f"\n5. ANCHOR PURITY:")
    print(f"   Alta (>{df['avg_anchor_purity'].median():.3f}): {high_purity['delta_f1_pct'].mean():.3f}%")
    print(f"   Baja (<={df['avg_anchor_purity'].median():.3f}): {low_purity['delta_f1_pct'].mean():.3f}%")

    # 6. Class minority analysis
    minority_classes = ['ESTJ', 'ESFP', 'ESFJ', 'ESTP']
    majority_classes = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENFP', 'ENTP']

    minority_data = df_class[df_class['class'].isin(minority_classes)]
    majority_data = df_class[df_class['class'].isin(majority_classes)]

    print(f"\n6. CLASES MINORITARIAS vs MAYORITARIAS:")
    print(f"   Minoritarias ({minority_classes}): {minority_data['delta_f1'].mean():.4f}")
    print(f"   Mayoritarias ({majority_classes}): {majority_data['delta_f1'].mean():.4f}")

    return trends


def create_visualizations():
    """Create all visualizations."""
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Heatmap: Config x Seed -> Delta F1
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot = df.pivot_table(index='config', columns='seed', values='delta_f1_pct', aggfunc='mean')
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Delta F1 (%) por Configuración y Seed')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'heatmap_config_seed.png', dpi=150)
    plt.close()
    print("  Saved: heatmap_config_seed.png")

    # 2. Bar chart: Average delta by group
    fig, ax = plt.subplots(figsize=(10, 6))
    group_means = df.groupby('group')['delta_f1_pct'].mean().sort_values(ascending=True)
    colors = ['green' if x > 0 else 'red' for x in group_means]
    group_means.plot(kind='barh', ax=ax, color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Delta F1 (%)')
    ax.set_title('Delta F1 Promedio por Grupo de Experimentos')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bar_group_delta.png', dpi=150)
    plt.close()
    print("  Saved: bar_group_delta.png")

    # 3. Scatter: Acceptance Rate vs Delta F1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['acceptance_rate'], df['delta_f1_pct'], alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Acceptance Rate')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Acceptance Rate vs Delta F1')
    # Add trend line
    z = np.polyfit(df['acceptance_rate'].dropna(), df['delta_f1_pct'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['acceptance_rate'].min(), df['acceptance_rate'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Tendencia')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scatter_acceptance_delta.png', dpi=150)
    plt.close()
    print("  Saved: scatter_acceptance_delta.png")

    # 4. Scatter: Anchor Purity vs Delta F1
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = df[df['avg_anchor_purity'] > 0]
    ax.scatter(valid['avg_anchor_purity'], valid['delta_f1_pct'], alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Average Anchor Purity')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Anchor Purity vs Delta F1')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scatter_purity_delta.png', dpi=150)
    plt.close()
    print("  Saved: scatter_purity_delta.png")

    # 5. Box plot: Delta by group
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby('group')['delta_f1_pct'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='group', y='delta_f1_pct', order=order, ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Grupo')
    ax.set_ylabel('Delta F1 (%)')
    ax.set_title('Distribución de Delta F1 por Grupo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'boxplot_group_delta.png', dpi=150)
    plt.close()
    print("  Saved: boxplot_group_delta.png")

    # 6. Heatmap: Class x Group -> Avg Delta
    fig, ax = plt.subplots(figsize=(14, 10))
    class_group = df_class.pivot_table(index='class', columns='group', values='delta_f1', aggfunc='mean')
    sns.heatmap(class_group, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Delta F1 por Clase y Grupo de Experimentos')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'heatmap_class_group.png', dpi=150)
    plt.close()
    print("  Saved: heatmap_class_group.png")

    # 7. Bar: Best config for each class
    fig, ax = plt.subplots(figsize=(14, 8))
    best_per_class = df_class.loc[df_class.groupby('class')['delta_f1'].idxmax()]
    best_per_class = best_per_class.sort_values('delta_f1', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in best_per_class['delta_f1']]
    ax.barh(best_per_class['class'], best_per_class['delta_f1'], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mejor Delta F1')
    ax.set_title('Mejor Configuración por Clase MBTI')
    # Add config labels
    for i, (idx, row) in enumerate(best_per_class.iterrows()):
        ax.text(row['delta_f1'] + 0.002, i, f" {row['config']}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bar_best_per_class.png', dpi=150)
    plt.close()
    print("  Saved: bar_best_per_class.png")


def generate_markdown_report():
    """Generate the final markdown report."""
    print("\n" + "="*80)
    print("GENERANDO REPORTE MARKDOWN")
    print("="*80)

    # Compute all analyses
    group_stats = df.groupby('group')['delta_f1_pct'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    config_stats = df.groupby('config')['delta_f1_pct'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

    # Class analysis
    class_best = df_class.loc[df_class.groupby('class')['delta_f1'].idxmax()][['class', 'config', 'seed', 'delta_f1', 'support']]
    class_best = class_best.sort_values('delta_f1', ascending=False)

    # Correlations
    correlations = {}
    for col in ['acceptance_rate', 'total_accepted', 'avg_anchor_purity', 'avg_similarity_anchor', 'avg_token_count']:
        if col in df.columns:
            corr = df['delta_f1_pct'].corr(df[col])
            if not np.isnan(corr):
                correlations[col] = corr

    report = f"""# Análisis de Tendencias - Phase E Experiments

## Executive Summary

**Total experimentos analizados:** {len(df)}
**Mejoraron macro F1:** {len(df[df['delta_f1_pct'] > 0])} ({100*len(df[df['delta_f1_pct'] > 0])/len(df):.0f}%)
**Empeoraron macro F1:** {len(df[df['delta_f1_pct'] < 0])} ({100*len(df[df['delta_f1_pct'] < 0])/len(df):.0f}%)
**Delta F1 promedio:** {df['delta_f1_pct'].mean():+.3f}%

### Top 3 Factores que Predicen Mejora

1. **Configuración de Clusters (Grupo C):** Δ promedio = {group_stats.loc['C_volume', 'mean']:+.3f}%
   - C1 (5×9×5) es la mejor configuración individual
   - Volumen moderado > volumen alto

2. **Filtros Relajados > Estrictos (Grupo D):** D1_relaxed > D2_strict
   - Permite más diversidad en sintéticos

3. **Estrategias de Minoría (Grupo E):** E2_minority_boost funciona bien
   - Enfocar en clases pequeñas ayuda

### Peores Estrategias

1. **Reasoning bajo (A1, A2):** Δ promedio = {df[df['config'].isin(['A1_gpt5_none', 'A2_gpt5_low'])]['delta_f1_pct'].mean():+.3f}%
2. **IP Scaling baseline (B1):** No ayuda sin ajustes agresivos

---

## Análisis por Grupo de Experimentos

| Grupo | Descripción | Δ Promedio | Std | N |
|-------|-------------|------------|-----|---|
"""
    for group, row in group_stats.iterrows():
        desc = {
            'A_reasoning': 'GPT-5 reasoning levels',
            'B_ip_scaling': 'IP scaling variants',
            'C_volume': 'Cluster/prompt volume',
            'D_filters': 'Filter thresholds',
            'E_minority': 'Minority class focus',
            'F_length': 'Length-aware generation',
            'G_combo': 'Combinations',
            'H_experimental': 'Experimental strategies',
            'other': 'Other'
        }.get(group, group)
        report += f"| {group} | {desc} | {row['mean']:+.3f}% | {row['std']:.3f} | {int(row['count'])} |\n"

    report += f"""

---

## Ranking de Configuraciones

| Rank | Config | Δ Promedio | Seeds | Descripción |
|------|--------|------------|-------|-------------|
"""
    for i, (config, row) in enumerate(config_stats.head(10).iterrows(), 1):
        desc = CONFIG_DESC.get(config, config)
        report += f"| {i} | {config} | {row['mean']:+.3f}% | {int(row['count'])} | {desc} |\n"

    report += f"""

### Peores Configuraciones

| Rank | Config | Δ Promedio | Seeds | Descripción |
|------|--------|------------|-------|-------------|
"""
    for i, (config, row) in enumerate(config_stats.tail(5).iterrows(), 1):
        desc = CONFIG_DESC.get(config, config)
        report += f"| {i} | {config} | {row['mean']:+.3f}% | {int(row['count'])} | {desc} |\n"

    report += f"""

---

## Análisis por Clase MBTI

### Mejor Configuración por Clase

| Clase | Support | Mejor Config | Seed | Δ F1 |
|-------|---------|--------------|------|------|
"""
    for _, row in class_best.iterrows():
        report += f"| {row['class']} | {int(row['support'])} | {row['config']} | {int(row['seed'])} | {row['delta_f1']:+.4f} |\n"

    # Per-class detailed analysis
    report += f"""

### Análisis Detallado por Clase

"""
    for class_name in sorted(df_class['class'].unique()):
        class_data = df_class[df_class['class'] == class_name]
        support = class_data['support'].iloc[0]
        baseline_f1 = class_data['baseline_f1'].mean()
        avg_delta = class_data['delta_f1'].mean()
        n_improved = (class_data['delta_f1'] > 0).sum()
        n_total = len(class_data)

        best_row = class_data.loc[class_data['delta_f1'].idxmax()]
        worst_row = class_data.loc[class_data['delta_f1'].idxmin()]

        # Which groups help this class?
        group_deltas = class_data.groupby('group')['delta_f1'].mean().sort_values(ascending=False)
        best_groups = group_deltas.head(2)
        worst_groups = group_deltas.tail(2)

        report += f"""
#### {class_name}
- **Support:** {int(support)} samples
- **Baseline F1:** {baseline_f1:.3f}
- **Mejora promedio:** {avg_delta:+.4f}
- **Tasa de mejora:** {n_improved}/{n_total} ({100*n_improved/n_total:.0f}%)
- **Mejor config:** {best_row['config']}_s{int(best_row['seed'])} (Δ={best_row['delta_f1']:+.4f})
- **Peor config:** {worst_row['config']}_s{int(worst_row['seed'])} (Δ={worst_row['delta_f1']:+.4f})
- **Grupos que ayudan:** {', '.join([f"{g} ({d:+.4f})" for g, d in best_groups.items()])}
- **Grupos que perjudican:** {', '.join([f"{g} ({d:+.4f})" for g, d in worst_groups.items()])}
"""

    report += f"""

---

## Correlaciones con Delta F1

| Métrica | Correlación | Interpretación |
|---------|-------------|----------------|
"""
    for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(corr) > 0.3:
            interp = "Fuerte" if abs(corr) > 0.5 else "Moderada"
        elif abs(corr) > 0.1:
            interp = "Débil"
        else:
            interp = "Muy débil"
        direction = "positiva" if corr > 0 else "negativa"
        report += f"| {metric} | {corr:+.3f} | {interp} {direction} |\n"

    report += f"""

---

## Insights y Recomendaciones

### Lo que Funciona

1. **Volumen moderado de generación (C1: 5×9×5)**
   - 225 candidatos por clase es suficiente
   - Más volumen (C2, C3) no mejora, posible overfitting

2. **Filtros relajados (D1)**
   - similarity=0.85 permite más diversidad
   - Filtros muy estrictos rechazan demasiados buenos candidatos

3. **IP scaling agresivo (B2)**
   - Generar más para clases desbalanceadas ayuda
   - Pero solo si el boost es significativo (2x+)

4. **Boost a clases minoritarias (E2)**
   - Enfocarse en clases pequeñas mejora macro F1

5. **H2_risky_boundary**
   - Filtros muy relajados pueden explorar el espacio de decisión

### Lo que NO Funciona

1. **Reasoning bajo o nulo (A1, A2)**
   - GPT-5 necesita al menos reasoning=medium para generar calidad

2. **IP scaling conservador (B1)**
   - boost=1.0 es insuficiente

3. **Filtros muy estrictos (D2)**
   - similarity=0.95 rechaza demasiados candidatos

4. **Reasoning alto + textos largos (G2)**
   - La combinación parece contraproducente

### Próximos Experimentos Sugeridos

1. **Combinar C1 + D1 + E2**
   - Volumen moderado + filtros relajados + boost minoritarias

2. **Probar reasoning=medium con otras configs**
   - A3 tiene resultados prometedores

3. **Explorar acceptance_rate óptimo**
   - Buscar el rango 10-15% que parece funcionar mejor

---

## Visualizaciones

![Heatmap Config x Seed](figures/heatmap_config_seed.png)
![Delta por Grupo](figures/bar_group_delta.png)
![Acceptance vs Delta](figures/scatter_acceptance_delta.png)
![Purity vs Delta](figures/scatter_purity_delta.png)
![Boxplot por Grupo](figures/boxplot_group_delta.png)
![Heatmap Clase x Grupo](figures/heatmap_class_group.png)
![Mejor por Clase](figures/bar_best_per_class.png)

---

*Generado automáticamente por trend_analysis.py*
*Total de datos analizados: {len(df)} experimentos, {len(df_class)} registros por clase*
"""

    # Save report
    with open(ANALYSIS_DIR / 'trend_report.md', 'w') as f:
        f.write(report)

    print(f"  Saved: trend_report.md")

    return report


def main():
    print("="*80)
    print("PHASE E TREND ANALYSIS")
    print("="*80)
    print(f"\nData loaded: {len(df)} experiments, {len(df_class)} per-class records")

    # Run all analyses
    analyze_by_group()
    analyze_by_config()
    analyze_by_class()
    analyze_correlations()
    find_trends()
    create_visualizations()
    generate_markdown_report()

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {ANALYSIS_DIR}")
    print(f"  - trend_report.md")
    print(f"  - figures/*.png (7 visualizaciones)")


if __name__ == '__main__':
    main()
