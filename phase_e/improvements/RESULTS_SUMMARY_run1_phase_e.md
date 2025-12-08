# Phase E - Resultados de Experimentos (2024-12-01)

## Resumen Ejecutivo

| Experimento | Delta F1 | Sintéticos | Observación |
|-------------|----------|------------|-------------|
| **cfg12_sim_095** | **+0.82%** | 30 | 🏆 Mejor resultado |
| cfg10_sim_088 | +0.33% | 28 | Buen resultado |
| imp_relaxed_gate | +0.25% | 30 | Gate relajado |
| cfg11_sim_092 | +0.13% | 30 | - |
| cfg13_combined | +0.09% | 30 | - |
| cfg09_sim_085 | -0.10% | 30 | Demasiado permisivo |
| imp_more_synth | -0.28% | 30 | ⚠️ Budget limitado |

**Conclusión principal**: El threshold de similaridad alto (0.95) da mejor resultado que aumentar cantidad de sintéticos.

---

## Configuración Base (Común a todos)

```bash
--data-path ../MBTI_500.csv
--test-size 0.2
--embedding-model sentence-transformers/all-mpnet-base-v2
--device cuda
--embedding-batch-size 128
--llm-model gpt-4o-mini
--max-clusters 3                    # Default (excepto imp_more_synth)
--prompts-per-cluster 3             # Default (excepto imp_more_synth)
--prompt-mode mix
--use-ensemble-selection
--use-val-gating
--val-size 0.15
--val-tolerance 0.02
--enable-anchor-gate
--enable-anchor-selection
--anchor-selection-ratio 0.8
--anchor-outlier-threshold 1.5
--use-class-description
--use-f1-budget-scaling             # ⚠️ Activo en todos!
--f1-budget-thresholds 0.45 0.20
--f1-budget-multipliers 0.0 0.5 1.0 # HIGH=0×, MEDIUM=0.5×, LOW=1×
--min-classifier-confidence 0.10
--contamination-threshold 0.95
--synthetic-weight 0.5
--synthetic-weight-mode flat
--use-hard-anchors
--deterministic-quality-gate
```

---

## Experimentos Detallados

### 1. cfg09_sim_085 - Similarity 0.85
```bash
--anchor-quality-threshold 0.30
--similarity-threshold 0.85
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4528 |
| Delta | **-0.10%** |
| Sintéticos | 30 |

**Análisis**: Similarity muy bajo permite demasiado ruido.

---

### 2. cfg10_sim_088 - Similarity 0.88
```bash
--anchor-quality-threshold 0.30
--similarity-threshold 0.88
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4547 |
| Delta | **+0.33%** |
| Sintéticos | 28 |

---

### 3. cfg11_sim_092 - Similarity 0.92
```bash
--anchor-quality-threshold 0.30
--similarity-threshold 0.92
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4538 |
| Delta | **+0.13%** |
| Sintéticos | 30 |

---

### 4. cfg12_sim_095 - Similarity 0.95 🏆
```bash
--anchor-quality-threshold 0.30
--similarity-threshold 0.95
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4569 |
| Delta | **+0.82%** |
| Sintéticos | 30 |

**Análisis**: Mejor resultado! Alta similaridad = sintéticos de alta calidad.

---

### 5. cfg13_combined - Anchor 0.25 + Sim 0.85
```bash
--anchor-quality-threshold 0.25
--similarity-threshold 0.85
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4536 |
| Delta | **+0.09%** |
| Sintéticos | 30 |

---

### 6. imp_more_synth - Más Sintéticos ⚠️
```bash
--anchor-quality-threshold 0.30
--similarity-threshold 0.90
--max-clusters 5
--prompts-per-cluster 9
# Nota: F1-budget-scaling seguía activo!
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4520 |
| Delta | **-0.28%** |
| Sintéticos | 30 |

**⚠️ PROBLEMA IDENTIFICADO**:
- Se generaron 225 candidatos (5×9×5) por clase
- Pero F1-budget-scaling limitó a budget=10 × 0.5 = 5 por clase
- Total: 30 sintéticos (igual que todos los demás)
- El experimento NO probó realmente "más sintéticos"

**Corrección aplicada**: Ver `exp_more_synthetics.sh` corregido.

---

### 7. imp_relaxed_gate - Gate Relajado
```bash
--anchor-quality-threshold 0.15  # (vs 0.30 default)
--similarity-threshold 0.80      # (vs 0.90 default)
```
| Métrica | Valor |
|---------|-------|
| Baseline F1 | 0.4532 |
| Augmented F1 | 0.4543 |
| Delta | **+0.25%** |
| Sintéticos | 30 |

**Análisis**: Gate más permisivo mejora ligeramente vs default.

---

## Resultados por Clase (cfg12_sim_095 - Mejor)

| Clase | Baseline F1 | Augmented F1 | Delta |
|-------|-------------|--------------|-------|
| ISFJ | 0.252 | 0.283 | **+12.2%** |
| ISTJ | 0.248 | 0.257 | **+3.7%** |
| ENFJ | 0.215 | 0.218 | +1.3% |
| ISFP | 0.268 | 0.261 | -2.6% |
| ESFP | 0.344 | 0.318 | -7.4% |
| ESTP | 0.791 | 0.787 | -0.4% |
| ESTJ | 0.606 | 0.606 | 0.0% |
| ESFJ | 0.222 | 0.206 | -7.4% |

**Observaciones**:
- ISFJ (+12.2%) y ISTJ (+3.7%): Mejora significativa (alto IP)
- ESFP y ESFJ: Empeoran (bajo IP, pero recibieron sintéticos)
- ESTP y ESTJ: Sin cambio (alto F1 baseline)

---

## Comparación con Experimentos Previos (safe_20251129)

| Config | Seed 42 | Seed 100 |
|--------|---------|----------|
| cfg01_phaseA_default | +0.16% | +0.09% |
| cfg02_no_hard_anchors | +0.29% | +0.06% |
| cfg03_no_det_gate | +0.45% | -1.05% |
| cfg04_original_pre_phaseA | -0.32% | -0.02% |

**vs experimentos actuales**:
- cfg12_sim_095 (+0.82%) > cfg03_no_det_gate (+0.45%)
- El similarity threshold alto es el factor más importante

---

## Hallazgos Clave

### 1. Calidad > Cantidad
- 30 sintéticos de alta calidad (sim=0.95) → +0.82%
- 30 sintéticos de baja calidad (sim=0.85) → -0.10%

### 2. F1-Budget-Scaling Limitante
Todos los experimentos produjeron ~30 sintéticos porque:
```
Budget = 10 base × 0.5 multiplier = 5 por clase (MEDIUM tier)
8 clases target × ~4-5 aceptados = 30 total
```

### 3. Clases con Alto IP Mejoran
- IP = 1 - baseline_F1
- ISFJ (IP=0.75): +12.2%
- ISTJ (IP=0.75): +3.7%
- ESFP (IP=0.66): -7.4% (bajo IP, no debería recibir tanto)

---

## Experimentos Pendientes

### Con configuraciones corregidas:

1. **exp_more_synthetics.sh** (CORREGIDO)
   - `--cap-class-ratio 0.15` (sin F1-scaling)
   - Esperado: 66-156 sintéticos/clase

2. **exp_ratio_f1_combined.sh** (NUEVO)
   - `--cap-class-ratio 0.20 --use-f1-budget-scaling`
   - Multiplicadores: 0.0/1.0/2.0

3. **exp_length_aware.sh**
   - Sintéticos de ~500 palabras (vs ~40 actual)

4. **exp_ip_scaling.sh**
   - Budget proporcional a IP

---

## Recomendaciones

1. **Usar similarity-threshold 0.95** para maximizar calidad
2. **Remover F1-budget-scaling** o usar multiplicadores menos restrictivos
3. **Enfocar sintéticos en clases con IP > 0.7** (ISFJ, ISTJ, ESFJ, ENFJ)
4. **Evitar augmentation en ESTP, ESTJ** (ya tienen F1 alto)
5. **Probar length-aware** para sintéticos más representativos

---

## Archivos de Resultados

```
phase_e/results/cached_20251130_223116/
├── cfg09_sim_085_s42_*.{json,csv,log}
├── cfg10_sim_088_s42_*.{json,csv,log}
├── cfg11_sim_092_s42_*.{json,csv,log}
├── cfg12_sim_095_s42_*.{json,csv,log}  # 🏆 Mejor
├── cfg13_combined_s42_*.{json,csv,log}
├── imp_more_synth_s42_*.{json,csv,log}
└── imp_relaxed_gate_s42_*.{json,csv,log}
```
