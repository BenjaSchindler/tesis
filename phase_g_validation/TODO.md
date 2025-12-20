# TODO - Experimentos Pendientes

## Estado Actual (2024-12-15)

### Experimentos n_shot ✅ COMPLETADOS
- [x] n_shot = 0, 3, 10 (Wave 5 original)
- [x] n_shot = 20, 50, 100, 200 (primera ronda extendida)
- [x] n_shot = 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 150 (granularidad fina) ✅ COMPLETADO

### Mejor Resultado CONFIRMADO
- **n_shot = 60**: +1.67 pp (p=0.0003) ← ÓPTIMO GLOBAL
- n_shot = 80: +1.36 pp (segundo mejor)
- n_shot = 100: +1.36 pp (empate segundo)

### Patrón Observado
Rendimiento no-monotónico con pico en n_shot=60, caída en 70, recuperación parcial 80-100, decline gradual después.

---

### Experimentos Temperatura con n_shot=60 ✅ COMPLETADOS (CORREGIDO 2024-12-16)

| Config | Temperatura | Delta (pp) | p-valor | Estado |
|--------|-------------|------------|---------|--------|
| W5b_temp03_n60 | τ = 0.3 | **+1.18** | 0.00012 | ✅ MEJOR |
| W5b_temp12_n60 | τ = 1.2 | +1.12 | 0.0013 | ✅ |
| W5b_temp09_n60 | τ = 0.9 | +0.92 | 0.00073 | ✅ |
| W5b_temp15_n60 | τ = 1.5 | +0.82 | 0.00015 | ✅ |
| W5b_temp06_n60 | τ = 0.6 | +0.61 | 0.0104 | ✅ |

**Hallazgo clave**: Con n_shot=60, τ=0.3 es óptimo (+1.18 pp).
Patrón no-monotónico: temperaturas extremas (0.3, 1.2) superan a la estándar (0.9).

---

## Prioridad de Re-Ejecución (post temperatura)

### 🟡 MEDIA PRIORIDAD - En ejecución

#### 2. Ensemble representativo con n_shot óptimo ⏳ EN PROGRESO
**Razón**: Verificar que la conclusión "ensembles no ayudan" se mantiene con configuración óptima.

| Config | Descripción | Estado |
|--------|-------------|--------|
| ENS_BEST_OPT | Ensemble mejores configs con n_shot óptimo | [ ] En progreso |

**Resultado anterior** (con n_shot=10): Ensemble +0.90 pp vs Individual +1.22 pp

---

### 🟢 BAJA PRIORIDAD - En ejecución

#### 3. MLP Profundo con n_shot óptimo ⏳ EN PROGRESO
**Razón**: Es sobre el clasificador, no la generación.

| Config | Descripción | Estado |
|--------|-------------|--------|
| MLP_512_OPT | MLP (512-256-128) con n_shot óptimo | [ ] En progreso |

**Resultado anterior**: MLP +20.11% en clases raras

---

## Resultados Completados (2024-12-16)

### Ensemble con n_shot=60 ✅
| Config | Delta (pp) | p-valor | Sintéticos |
|--------|------------|---------|------------|
| W5_shot_60 | +0.86 | 0.00023 | 1447 |
| W5_shot_80 | +0.86 | 0.00052 | 1453 |
| W5_shot_100 | +1.01 | 0.00077 | 1436 |

**Conclusión**: Configuraciones individuales similares (~0.86-1.01 pp), no hay beneficio claro de ensemble.

### MLP con n_shot=60 ✅
| Métrica | Valor |
|---------|-------|
| Baseline MLP | 0.2361 |
| Augmented MLP | 0.2434 |
| Delta | +0.74 pp |
| Sintéticos | 1446 |

### RARE_MLP con n_shot=60 ✅ (COMPARACIÓN JUSTA - mismo K-Fold)
| Métrica | Original RARE_MLP | Con n_shot=60 |
|---------|-------------------|---------------|
| Baseline MLP | 0.2075 | **0.2075** (igual) |
| Augmented MLP | 0.2492 | 0.2325 |
| Delta | **+20.11%** | +12.06% |
| p-value | 0.00095 | 0.019 |
| Sintéticos | 3218 | 2405 |

**HALLAZGO CLAVE**: Para clases raras, n_shot=60 **EMPEORA** el rendimiento:
- De +20.11% (default) a +12.06% (n_shot=60)
- Las clases raras tienen ~40-50 muestras, insuficientes para llenar 60 ejemplos diversos
- La configuración RARE_massive_oversample original era mejor para este caso
- **Conclusión**: n_shot alto solo beneficia clases con abundantes muestras

## Notas

- Los experimentos de temperatura original usaron n_shot=10
- El óptimo anterior era τ=0.9 con n_shot=10 (+5.57%)
- **CORREGIDO**: Con n_shot=60, τ=0.3 es óptimo (+1.18 pp)
- Baseline Macro F1 (LogReg): 0.2045
- Baseline Macro F1 (MLP 512-256-128): varía según configuración K-Fold
