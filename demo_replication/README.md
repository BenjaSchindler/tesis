# SMOTE-LLM Replication Demo

Este directorio contiene todo lo necesario para replicar los experimentos de SMOTE-LLM y medir la variabilidad por estocasticidad de los LLMs.

## Resultados Esperados

| Ensemble | Media | Std | CV |
|----------|-------|-----|-----|
| ENS_SUPER_G5_F7_v2 | +14.33% | 0.50% | 3.5% |
| ENS_TopG5_Extended | +10.69% | 1.10% | 10.2% |
| ENS_Top3_G5 | +8.40% | 2.47% | 29.4% |

## Requisitos

- Python 3.10+
- GNU parallel (`sudo apt install parallel`)
- CUDA (opcional, para GPU)
- OpenAI API key
- Dataset `mbti_1.csv` en el directorio padre

### Dependencias Python

```bash
pip install pandas numpy scikit-learn sentence-transformers openai scipy
```

## Uso Rápido

```bash
# 1. Configurar API key
export OPENAI_API_KEY='sk-...'

# 2. Ejecutar demo (3 replicaciones por defecto)
cd demo_replication
./run_demo.sh 3

# O solo 1 replicación para prueba rápida
./run_demo.sh 1
```

## Estructura del Directorio

```
demo_replication/
├── README.md                 # Este archivo
├── run_demo.sh               # Script principal de ejecución
├── configs/                  # Configuraciones de experimentos
│   ├── base_config.sh        # Configuración base común
│   ├── CMB3_skip.sh          # Config 1 (Phase F)
│   ├── CF1_conf_band.sh      # Config 2 (Phase F)
│   ├── V4_ultra.sh           # Config 3 (Phase F)
│   ├── G5_K25_medium.sh      # Config 4 (Phase F)
│   ├── EXP7_hybrid_best.sh   # Config 5 (Phase F Experiments)
│   ├── W9_contrastive.sh     # Config 6 (Phase G)
│   ├── W1_low_gate.sh        # Config 7 (Phase G)
│   ├── W1_force_problem.sh   # Config 8 (Phase G)
│   └── W3_no_dedup.sh        # Config 9 (Phase G)
├── core/
│   └── runner_phase2.py      # Script principal de generación
├── scripts/
│   ├── run_replication.sh           # Script de una replicación
│   ├── run_all_replications.sh      # Script de múltiples replicaciones
│   ├── create_ensembles_replication.py  # Creador de ensembles
│   ├── analyze_replication_variance.py  # Análisis de variabilidad
│   └── eval_holdout_correct.py      # Evaluador hold-out
└── results/                  # Resultados (se crean al ejecutar)
```

## Ensembles

Los experimentos crean 3 ensembles combinando diferentes configuraciones:

### ENS_Top3_G5
- CMB3_skip + CF1_conf_band + V4_ultra + G5_K25_medium
- ~200 sintéticos

### ENS_TopG5_Extended
- ENS_Top3_G5 + W9_contrastive + W1_low_gate
- ~300 sintéticos

### ENS_SUPER_G5_F7_v2 (MÁS ESTABLE)
- ENS_Top3_G5 + W1_force_problem + EXP7_hybrid_best + W3_no_dedup
- ~360 sintéticos

## Costos y Tiempos

| Replicaciones | Tiempo Estimado | Costo API |
|---------------|-----------------|-----------|
| 1 | ~45 min | ~$6 |
| 3 | ~2.5 horas | ~$17 |
| 5 | ~4 horas | ~$28 |

## Outputs

Después de ejecutar, encontrarás:

```
demo_replication/
├── replication_run1/
│   ├── results/
│   │   ├── CMB3_skip_s42_synth.csv
│   │   ├── ...
│   │   ├── ENS_Top3_G5_synth.csv
│   │   ├── ENS_TopG5_Extended_synth.csv
│   │   ├── ENS_SUPER_G5_F7_v2_synth.csv
│   │   ├── ENS_Top3_G5_holdout.json
│   │   ├── ENS_TopG5_Extended_holdout.json
│   │   └── ENS_SUPER_G5_F7_v2_holdout.json
│   └── logs/
├── replication_run2/
│   └── ...
├── replication_run3/
│   └── ...
└── replication_variance_analysis.json  # Análisis final
```

## Interpretación de Resultados

El archivo `replication_variance_analysis.json` contiene:

```json
{
  "ENS_SUPER_G5_F7_v2": {
    "mean": 14.33,      // Media de delta % F1
    "std": 0.50,        // Desviación estándar
    "cv": 3.5,          // Coeficiente de variación (%)
    "min": 13.85,       // Mínimo
    "max": 15.01        // Máximo
  }
}
```

### Criterios de Robustez

| CV | Interpretación |
|----|----------------|
| < 10% | EXCELENTE - Muy reproducible |
| 10-20% | BUENO - Aceptable varianza |
| > 20% | ALTO - Resultados variables |

## Ejecutar Configuraciones Individuales

Si quieres ejecutar solo una configuración:

```bash
export OPENAI_API_KEY='sk-...'
cd demo_replication
SEED=42 bash configs/CMB3_skip.sh
```

## Troubleshooting

### Error: GNU parallel no encontrado
```bash
sudo apt install parallel
```

### Error: Dataset no encontrado
Asegúrate de que `mbti_1.csv` esté en el directorio padre (`SMOTE-LLM/`).

### Error: Dependencias Python
```bash
pip install pandas numpy scikit-learn sentence-transformers openai scipy
```

### Los procesos usan mucha memoria
Reduce `PARALLEL_JOBS` en el script (default: 10).

## Licencia

MIT License - Ver archivo LICENSE en el directorio raíz.
