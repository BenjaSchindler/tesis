# Filter Comparison for LLM Text Augmentation

Investigación sobre qué filtros geométricos funcionan mejor para augmentación de texto con LLMs en clasificación low-resource.

## Estructura

```
filters/
├── data/
│   └── benchmarks/          # Datasets de prueba (10/25/50-shot)
├── experiments/
│   ├── exp_filter_comparison.py   # Experimento principal (1440 configs)
│   ├── exp_benchmark_llm_pct.py   # Prueba de % LLM
│   └── download_datasets_v2.py    # Descarga de datasets
├── results/
│   ├── thesis_research/     # Resultados organizados para tesis
│   ├── filter_comparison/   # Resultados crudos
│   └── benchmark_results.json
├── src/
│   ├── core/
│   │   ├── geometric_filter.py      # LOFFilter, CombinedGeometricFilter
│   │   ├── filter_cascade.py        # FilterCascade (levels 0-4)
│   │   ├── embedding_guided_sampler.py  # Coverage-based selection
│   │   ├── llm_providers.py         # OpenAI, Gemini providers
│   │   └── validation_runner.py     # Evaluation utilities
│   └── baselines.py                 # SMOTEBaseline
└── cache/
    └── llm_generations/     # Cache de generaciones LLM
```

## Resultados Principales

### Win Rate por Dataset

| Dataset | Win Rate | Mean vs SMOTE |
|---------|----------|---------------|
| 20newsgroups_10shot | **90.4%** | **+3.61pp** |
| 20newsgroups_25shot | 88.3% | +0.78pp |
| hate_speech_davidson_10shot | 52.1% | -3.06pp |
| sms_spam_10shot | 65.4% | -1.17pp |
| sms_spam_25shot | 45.4% | -1.40pp |

### Ranking de Filtros

| Filtro | Mean vs SMOTE | Win Rate |
|--------|---------------|----------|
| **cascade** | **+1.45pp** | **83.3%** |
| lof | +0.88pp | 75.8% |
| none | +0.72pp | 62.9% |
| embedding_guided | -1.29pp | 55.0% |
| combined | -3.28pp | 52.5% |

### Mejor Configuración Global

- **Filter**: cascade (level=1)
- **LLM%**: 100%
- **N-shot**: 10
- **Resultado**: +7.25pp vs SMOTE

## Uso

```bash
# Ejecutar experimento completo
cd filters
python experiments/exp_filter_comparison.py

# Ver resultados
cat results/thesis_research/analysis/final_summary.md
```

## Conclusiones

1. **LLM supera a SMOTE en clasificación multi-clase low-resource** (90% win rate)
2. **Filtros simples > complejos**: cascade level=1 (solo distancia) es óptimo
3. **Filtro combined (LOF+sim) es contraproducente**: demasiado restrictivo
4. **Clasificación binaria es más difícil** para LLM augmentation
