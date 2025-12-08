# Phase E - Mejoras a Implementar

## Hallazgos del Análisis Preliminar (2025-11-30)

### 1. Métrica "Espacio para Mejorar" (Improvement Potential)

**Definición**: `IP = 1 - baseline_f1`

| Grupo | IP | Clases | Promedio Delta |
|-------|-----|--------|----------------|
| Alto potencial (IP > 0.7) | 0.73-0.79 | ISFJ, ISTJ, ENFJ, ESFJ, ISFP | **+2.08%** |
| Bajo potencial (IP ≤ 0.7) | 0.21-0.66 | ESTP, ESTJ, ESFP | **-1.52%** |

**Conclusión**: La métrica IP sí predice tendencias, aunque con alta varianza.

**Implementación sugerida**:
- [ ] Priorizar augmentation en clases con IP > 0.7
- [ ] Usar IP para asignar presupuesto de sintéticos por clase
- [ ] Reportar IP en métricas de salida

---

### 2. Problema: Deltas muy pequeños

**Diagnóstico**:
- 30 sintéticos sobre ~21K muestras = 0.14% del dataset
- Impacto esperado: Mínimo

**Soluciones a probar**:

#### Opción A: Más sintéticos por clase
- Aumentar `--prompts-per-cluster` de 3 a 9
- Aumentar `--max-clusters` de 3 a 5
- Resultado esperado: ~90+ sintéticos por clase vs ~5 actual

#### Opción B: Quality gate más permisivo
- Bajar `--anchor-quality-threshold` de 0.30 a 0.15
- Bajar `--similarity-threshold` de 0.90 a 0.80
- Resultado esperado: Más sintéticos aceptados

#### Opción C: Más clases target
- Actualmente: 8 clases minoritarias
- Propuesta: Incluir clases con F1 < 0.60 (agregaría ENTJ, INTJ, etc.)

---

## Experimentos Pendientes

### Exp 1: `more_synthetics` - Más sintéticos por clase (CORREGIDO)
```bash
--max-clusters 5 --prompts-per-cluster 9 --cap-class-ratio 0.15
```
**CORRECCIÓN**: El experimento original usaba F1-budget-scaling que limitaba a 5 sintéticos.
Ahora usa `--cap-class-ratio 0.15` para permitir hasta 15% del tamaño de cada clase:
| Clase | Tamaño | Budget máximo |
|-------|--------|---------------|
| ISFJ  | 442    | 66            |
| ESFP  | 245    | 36            |
| ISTJ  | 845    | 126           |
| ENFJ  | 1043   | 156           |

Estimación: 66-156 sintéticos/clase (vs 5 anterior)

### Exp 2: `relaxed_gate` - Quality gate relajado
```bash
--anchor-quality-threshold 0.15 --similarity-threshold 0.80
```
Estimación: 2-3x más sintéticos aceptados

### Exp 3: `ip_scaling` - Presupuesto basado en Improvement Potential ⭐
```python
IP = 1 - baseline_f1
ip_mult = 1 + ip_boost_factor * IP  (si IP > 0.7)
IP Rescue: min_budget = 10 + 20 * IP  (para clases con alto IP)
```
Impacto esperado por clase:
| Clase | IP | Budget (old) | Budget (IP) | Boost |
|-------|-----|--------------|-------------|-------|
| ISFJ  | 0.75 | 10 | 24 | +140% |
| ISTJ  | 0.77 | 10 | 25 | +150% |
| ESFJ  | 0.78 | 10 | 25 | +150% |
| ESTP  | 0.21 | 31 | 31 | 0% (ya bueno) |

### Exp 4: `length_aware` - Longitud apropiada de sintéticos ⭐⭐
**Problema identificado**: Los sintéticos son ~93% más cortos que los textos reales:
| Clase | Real (words) | Synth (words) | Diferencia |
|-------|-------------|---------------|------------|
| ESFJ  | 515         | 31            | -94%       |
| ESFP  | 495         | 37            | -92%       |
| ISFJ  | 501         | 50            | -90%       |
| ISTJ  | 499         | 47            | -91%       |

**Solución**: Agregar instrucciones de longitud al prompt:
```
IMPORTANT LENGTH REQUIREMENT: Each generated text MUST be approximately 500 words long
(minimum 400 words, maximum 600 words). This matches the typical length of real ISFJ texts.
Short responses will be rejected.
```

**Modos**:
- `strict`: Target +/- 20% (más restrictivo)
- `range`: Entre p25 y p75 de la clase
- `approximate`: Sugerencia suave

**Implementación**:
- `length_aware_generator.py`: Módulo para calcular stats y generar instrucciones
- `runner_phase2_length_aware.py`: Runner modificado con soporte de longitud
- `exp_length_aware.sh`: Script de experimento

### Exp 5: `ratio_f1_combined` - Budget combinado (Opción D) ⭐
**Hipótesis**: Combinar cap-class-ratio + F1-based scaling da mejor control del budget.

```bash
--cap-class-ratio 0.20 --use-f1-budget-scaling --f1-budget-multipliers 0.0 1.0 2.0
```

**Lógica**:
```python
budget_final = min(
    cap_class_ratio × class_size,  # Límite proporcional al tamaño
    f1_scaled_budget               # Límite basado en F1
)
```

**Multiplicadores F1 (menos restrictivos)**:
| Tier | F1 | Multiplicador | Efecto |
|------|-----|---------------|--------|
| HIGH | > 0.45 | 0.0× | Skip (ya bueno) |
| MEDIUM | 0.20-0.45 | 1.0× | Budget normal |
| LOW | ≤ 0.20 | 2.0× | 2× budget (más ayuda) |

**Ejemplo de cálculo**:
| Clase | n | F1 | Tier | ratio_budget | f1_budget | **final** |
|-------|---|-----|------|--------------|-----------|-----------|
| ISFJ | 442 | 0.252 | MEDIUM | 88 | 10 | **10** |
| ENFJ | 1043 | 0.215 | LOW | 208 | 20 | **20** |
| ESTP | 523 | 0.791 | HIGH | 104 | 0 | **0 (skip)** |

---

## Estado

- [x] Análisis preliminar completado
- [x] Identificación de métrica IP
- [x] Diagnóstico de deltas pequeños
- [x] Implementación IP Budget Calculator
- [x] Identificación de problema de longitud (sintéticos 93% más cortos)
- [x] Implementación Length-Aware Generator
- [x] Corrección de `more_synthetics` (era ineficiente, budget=5)
- [x] Experimento `relaxed_gate` (versión con F1-budget) → +0.25%
- [x] Experimento `imp_more_synth` (versión con F1-budget limitante) → -0.28%
- [x] **Experimento cfg12_sim_095** → **+0.82%** 🏆 MEJOR RESULTADO
- [ ] Experimento `more_synthetics` (corregido con cap-class-ratio)
- [ ] Experimento `ip_scaling`
- [ ] Experimento `length_aware` ⭐
- [ ] Experimento `ratio_f1_combined` (Opción D) ⭐
- [ ] Análisis comparativo final

**Ver resultados completos en: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)**

## Archivos Creados

```
improvements/
├── TODO_IMPROVEMENTS.md              # Este archivo
├── RESULTS_SUMMARY.md                # Resultados de experimentos (2025-12-01)
├── ip_budget_calculator.py           # IP-enhanced budget calculator
├── exp_more_synthetics.sh            # Exp 1: más sintéticos (CORREGIDO)
├── exp_relaxed_gate.sh               # Exp 2: gate relajado
├── exp_ip_scaling.sh                 # Exp 3: IP scaling
├── exp_length_aware.sh               # Exp 4: length-aware ⭐
├── exp_ratio_f1_combined.sh          # Exp 5: ratio + F1 combinado ⭐
├── length_aware_generator.py         # Módulo de estadísticas de longitud
├── runner_phase2_length_aware.py     # Runner modificado con soporte longitud
├── run_both_experiments.sh           # Ejecutar exp 1 y 2
├── compare_results.py                # Comparar resultados
└── results/                          # (se creará al ejecutar)
```

## Cómo Ejecutar

```bash
# Verificar GPU disponible
ps aux | grep runner_phase2 | grep -v grep

# Ejecutar experimentos individualmente
./exp_more_synthetics.sh 42     # Corregido: ahora usa cap-class-ratio
./exp_relaxed_gate.sh 42
./exp_ip_scaling.sh 42
./exp_length_aware.sh 42        # Prueba strict, range y baseline
./exp_ratio_f1_combined.sh 42   # Opción D: ratio + F1 combinados

# Comparar resultados
python3 compare_results.py
```
