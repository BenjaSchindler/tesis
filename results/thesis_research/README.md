# LLM Augmentation Filter Comparison Research

## Objetivo
Encontrar la mejor combinación de filtros geométricos + configuración de generación LLM para superar SMOTE en clasificación de texto low-resource.

## Estructura de Carpetas

```
thesis_research/
├── README.md                    # Este archivo
├── configurations/              # Configuraciones exactas de experimentos
│   ├── filter_configs.json     # Todas las configuraciones de filtros
│   ├── generation_configs.json # Configuraciones de generación LLM
│   └── datasets_used.json      # Datasets y sus características
├── raw_results/                 # Resultados sin procesar
│   ├── filter_comparison/      # Experimento principal
│   ├── benchmark_results.json  # Resultados por % LLM
│   └── mbti_experiments/       # Experimentos originales MBTI
├── analysis/                    # Análisis y sumarios
│   ├── summary.json            # Resumen ejecutivo
│   ├── filter_ranking.json     # Ranking de filtros
│   └── recommendations.json    # Recomendaciones finales
├── datasets_info/              # Información de datasets
│   └── dataset_stats.json      # Estadísticas por dataset
└── figures/                    # Gráficos (para generar)
```

## Hallazgos Principales

### 1. LLM supera a SMOTE en low-resource (<25 samples/clase)

| Dataset | Mejor Config | vs SMOTE |
|---------|--------------|----------|
| 20newsgroups_10shot | cascade 100% LLM | +7.25 pp |
| sms_spam_10shot | 5% LLM | +4.89 pp |

### 2. Ranking de Filtros (20newsgroups_10shot)

| Filtro | Mean vs SMOTE | Win Rate |
|--------|---------------|----------|
| none | +5.51 pp | 100% |
| embedding_guided | +4.49 pp | 100% |
| cascade | +4.41 pp | 98.3% |
| lof | +4.13 pp | 94.4% |
| combined | +0.79 pp | 62.5% |

### 3. Configuraciones Óptimas

- **Ultra low-resource (10 samples/clase)**: 100% LLM, 10-shot, cascade level=1
- **Low-resource (25 samples/clase)**: 25-50% LLM, 25-shot, LOF o embedding_guided
- **Medium resource (50+ samples/clase)**: SMOTE puro o 5% LLM

## Archivos de Experimentos

1. `exp_filter_comparison.py` - Experimento exhaustivo (1440 configs)
2. `exp_benchmark_llm_pct.py` - Prueba de % LLM vs SMOTE
3. `exp_targeted_llm_v3.py` - Generación dirigida a gaps

## Reproducibilidad

```bash
# Ejecutar experimento completo
python experiments/exp_filter_comparison.py

# Ver resultados parciales
cat results/filter_comparison/partial_results.json | python -m json.tool
```
