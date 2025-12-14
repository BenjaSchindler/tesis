# Phase G Validation - Índice de Archivos

**Fecha de creación**: 2025-12-13

---

## Documentación Principal

| Archivo | Descripción | Idioma |
|---------|-------------|--------|
| **README.md** | Guía principal de Phase G | Inglés |
| **TECHNICAL_DOCUMENTATION.md** | Documentación técnica completa (16 secciones) | Inglés |
| **RESULTS_SUMMARY.md** | Resumen ejecutivo de resultados | Inglés |
| **RESUMEN_EJECUTIVO_ES.md** | Resumen ejecutivo en español para la tesis | Español |
| **LATEX_TABLES.md** | 12 tablas LaTeX listas para copiar-pegar | LaTeX |
| **INDEX.md** | Este archivo - índice de todos los archivos | Español |

---

## Scripts y Herramientas

| Archivo | Propósito | Tipo |
|---------|-----------|------|
| **validation_runner.py** | Framework de validación cruzada K-fold (5×3) | Python |
| **base_config.py** | Clases de configuración y parámetros | Python |
| **compile_results.py** | Compila todos los resultados en FULL_SUMMARY.json | Python |
| **generate_plots.py** | Genera 6 visualizaciones automáticamente | Python |

---

## Experimentos

| Archivo | Experimento | Configs | Descripción |
|---------|-------------|---------|-------------|
| **experiments/exp13_rare_classes.py** | Exp 13 | 5 | Foco en ESFJ/ESFP/ESTJ con sobremuestreo masivo |
| **experiments/exp14b_mlp_xgboost.py** | Exp 14b | 5 | Comparación LogReg, MLP, XGBoost, LightGBM |

---

## Configuraciones (configs/)

### Por Oleada

| Carpeta | Configs | Foco | Mejor Config |
|---------|---------|------|--------------|
| **wave1/** | 3 | Umbrales de calidad | W1_low_gate (+3.48%) |
| **wave2/** | 2 | Sobremuestreo masivo | W2_ultra_vol (+3.55%) |
| **wave3/** | 2 | Deduplicación y filtrado | W3_permissive_filter (+4.35%) |
| **wave4/** | 1 | Generación focalizada | W4_target_only (+1.46%) |
| **wave5/** | 3 | Few-shot vs many-shot | **W5_many_shot_10 (+5.98%)** 🏆 |
| **wave6/** | 3 | Temperatura LLM | W6_temp_high (+5.57%) |
| **wave7/** | 2 | Sin filtrado (YOLO) | W7_yolo (+5.05%) |
| **wave8/** | 2 | GPT-4o reasoning | FALLÓ |
| **wave9/** | 2 | Aprendizaje contrastivo | W9_contrastive (+3.84%) |

### Otras Categorías

| Carpeta | Configs | Descripción |
|---------|---------|-------------|
| **component/** | 4 | Validación de componentes de Phase F |
| **pf_derived/** | 3 | Derivados del config óptimo de Phase F |
| **rare_class/** | 5 | Experimento 13 - clases raras |
| **ensembles/** | 6 | Combinaciones de mejores configs |

**Total de configuraciones**: 38

---

## Resultados (results/)

### Resúmenes Compilados

| Archivo | Descripción |
|---------|-------------|
| **FULL_SUMMARY.json** | Todos los 38 configs compilados con métricas |

### Por Categoría

| Carpeta | Archivos | Formato |
|---------|----------|---------|
| **wave1/** | 3 | *_kfold.json |
| **wave2/** | 2 | *_kfold.json |
| **wave3/** | 2 | *_kfold.json |
| **wave4/** | 1 | *_kfold.json |
| **wave5/** | 3 | *_kfold.json |
| **wave6/** | 3 | *_kfold.json |
| **wave7/** | 2 | *_kfold.json |
| **wave8/** | 2 | *_kfold.json |
| **wave9/** | 2 | *_kfold.json |
| **component/** | 4 | *_kfold.json |
| **pf_derived/** | 3 | *_kfold.json |
| **rare_class/** | 5 | *_kfold.json |
| **multiclassifier/** | 1 | exp14b_mlp_xgboost.json |
| **ensembles/** | 6 | *_kfold.json |

**Total de archivos JSON**: 39

---

## Visualizaciones (plots/)

| Archivo | Contenido | Dimensiones |
|---------|-----------|-------------|
| **top10_configs.png** | Gráfico de barras: Top 10 configs por delta % | 12×6 |
| **wave_comparison.png** | Comparación de mejora promedio por oleada | 10×6 |
| **rare_class_heatmap.png** | Mapa de calor: Mejoras ESFJ/ESFP/ESTJ top 15 | 8×10 |
| **pvalue_analysis.png** | Scatter + histogram de p-values vs deltas | 14×5 |
| **multiclassifier_comparison.png** | Comparación LogReg/MLP/XGBoost/LightGBM | 14×6 |
| **category_summary.png** | Delta promedio y tasa de significancia por categoría | 14×6 |

**Total de visualizaciones**: 6 (formato PNG, 300 DPI)

---

## Logs (logs/)

| Archivo | Experimento | Tamaño |
|---------|-------------|--------|
| **exp13_rare_classes.log** | Experimento 13 - clases raras | ~50 KB |
| **exp14b_mlp_xgboost.log** | Experimento 14b - multi-clasificador | ~80 KB |

---

## Cache (cache/)

| Archivo | Descripción | Tamaño |
|---------|-------------|--------|
| **embeddings_mpnet.npy** | Embeddings MPNet pre-calculados (8675×768) | ~50 MB |

---

## Guía de Uso Rápida

### Para Lectura Rápida
1. **README.md** - Visión general
2. **RESULTS_SUMMARY.md** - Resultados principales
3. **plots/** - Ver visualizaciones

### Para Tesis en Español
1. **RESUMEN_EJECUTIVO_ES.md** - Resumen en español
2. **LATEX_TABLES.md** - Copiar tablas necesarias
3. **plots/** - Incluir figuras relevantes

### Para Detalles Técnicos
1. **TECHNICAL_DOCUMENTATION.md** - Toda la metodología
2. **results/FULL_SUMMARY.json** - Datos crudos
3. **logs/** - Logs de ejecución

### Para Reproducir Resultados
1. **validation_runner.py** - Framework de evaluación
2. **configs/** - Todas las configuraciones
3. **experiments/** - Scripts de experimentos

---

## Estructura de Carpetas

```
phase_g_validation/
│
├── README.md                          ← Start here
├── TECHNICAL_DOCUMENTATION.md         ← Full details
├── RESULTS_SUMMARY.md                 ← Quick reference
├── RESUMEN_EJECUTIVO_ES.md            ← Para tesis (español)
├── LATEX_TABLES.md                    ← Para tesis (LaTeX)
├── INDEX.md                           ← Este archivo
│
├── validation_runner.py
├── base_config.py
├── compile_results.py
├── generate_plots.py
│
├── configs/                           ← 38 configs
│   ├── wave1/ (3)
│   ├── wave2/ (2)
│   ├── wave3/ (2)
│   ├── wave4/ (1)
│   ├── wave5/ (3) 🏆
│   ├── wave6/ (3)
│   ├── wave7/ (2)
│   ├── wave8/ (2)
│   ├── wave9/ (2)
│   ├── component/ (4)
│   ├── pf_derived/ (3)
│   ├── rare_class/ (5)
│   └── ensembles/ (6)
│
├── experiments/                       ← Experiment scripts
│   ├── exp13_rare_classes.py
│   └── exp14b_mlp_xgboost.py
│
├── results/                           ← All results (39 JSON files)
│   ├── FULL_SUMMARY.json             ← Compiled results
│   ├── wave1/
│   ├── wave2/
│   ├── ...
│   ├── rare_class/
│   ├── multiclassifier/
│   └── ensembles/
│
├── plots/                             ← 6 visualizations (PNG, 300 DPI)
│   ├── top10_configs.png
│   ├── wave_comparison.png
│   ├── rare_class_heatmap.png
│   ├── pvalue_analysis.png
│   ├── multiclassifier_comparison.png
│   └── category_summary.png
│
├── logs/                              ← Experiment logs
│   ├── exp13_rare_classes.log
│   └── exp14b_mlp_xgboost.log
│
└── cache/                             ← Cached embeddings
    └── embeddings_mpnet.npy
```

---

## Métricas Clave

| Métrica | Valor |
|---------|-------|
| Total configs probadas | 38 |
| Configs significativas | 30 (78.9%) |
| Mejor mejora | +5.98% (W5_many_shot_10) |
| Mejor p-value | 0.00001 (V4_ultra) |
| ESFJ mejorado | +12.42% (MLP_512_256_128) |
| ESFP mejorado | 0% (no resuelto) |
| ESTJ mejorado | +1.79% (MLP_512_256_128) |
| Archivos documentación | 6 MD |
| Archivos resultados | 39 JSON |
| Visualizaciones | 6 PNG |
| Scripts Python | 4 |
| Líneas de código | ~2,500 |

---

## Checklist para la Tesis

### Documentación Leída
- [ ] README.md
- [ ] TECHNICAL_DOCUMENTATION.md
- [ ] RESULTS_SUMMARY.md
- [ ] RESUMEN_EJECUTIVO_ES.md

### Figuras para Incluir
- [ ] top10_configs.png
- [ ] wave_comparison.png
- [ ] multiclassifier_comparison.png
- [ ] rare_class_heatmap.png

### Tablas para Incluir
- [ ] Tabla 1: Top 10 configs
- [ ] Tabla 2: Wave comparison
- [ ] Tabla 3: Multi-classifier
- [ ] Tabla 10: Phase F vs G
- [ ] Tabla 12: Problem classes summary

### Secciones Escritas
- [ ] Metodología (Waves 1-9)
- [ ] Resultados generales
- [ ] Análisis multi-clasificador
- [ ] Clases problemáticas
- [ ] Limitaciones (ESFP no resuelto)
- [ ] Conclusiones

---

**Última actualización**: 2025-12-13
**Total de archivos creados**: 60+
**Estado**: ✅ Documentación completa
