# Phase E - TODO y Observaciones

## Hallazgos de Experimentos A1-A3 (5 runs completados)

### Correlaciones Preocupantes

| Correlación | Valor | Implicación |
|-------------|-------|-------------|
| `classifier_confidence` vs `knn_similarity` | **-0.58** | Sintéticos con alta confianza tienen BAJA similitud KNN → posible overfitting al clasificador |
| `similarity_to_centroid` vs `classifier_confidence` | **-0.37** | Los más cercanos al centroide no son los más confiantes → el centroide puede no representar bien la clase |
| `IP` vs `Delta` | **-0.09** | No hay correlación clara entre potencial de mejora y mejora real |

### Patrones por Clase

| Clase | Base F1 | IP | Sintéticos/exp | Problema |
|-------|---------|-----|----------------|----------|
| ESFP | 0.00 | 1.00 | ~5 | F1 sigue en 0 a pesar de sintéticos |
| ESTJ | 0.10 | 0.90 | ~4 | Sin cambio |
| ESFJ | 0.13 | 0.87 | ~1 | Muy pocos sintéticos |
| ENFJ | 0.14 | 0.86 | ~10 | **PERJUDICA** (-1.5% avg) |
| ISTJ | 0.15 | 0.85 | ~10 | Única clase que mejora consistentemente (3/5) |

---

## TODOs - Cambios Propuestos

### Alta Prioridad

- [ ] **Investigar por qué ENFJ empeora**: Tiene 10 sintéticos pero delta negativo
  - Revisar calidad de anchors para ENFJ
  - Verificar si hay confusión con clases similares (INFJ?)

- [ ] **Aumentar sintéticos para clases IP>0.9**: ESFP, ESTJ, ESFJ reciben muy pocos
  - Experimentos B (IP scaling) deberían ayudar
  - Considerar relajar thresholds solo para estas clases

- [ ] **Revisar filtro de classifier_confidence**:
  - Correlación negativa con knn_similarity sugiere que estamos filtrando mal
  - Posible: sintéticos "confiantes" son los que el modelo ya clasifica bien (no aportan)
  - Idea: Preferir sintéticos con confianza MEDIA (0.3-0.7) no alta

### Media Prioridad

- [ ] **Probar threshold de similitud más bajo para clases IP>0.8**
  - Actualmente: 0.90 para todos
  - Propuesta: 0.80-0.85 para ESFP, ESTJ, ESFJ, ENFJ

- [ ] **Verificar longitud de sintéticos vs reales**
  - Sintéticos: ~47 tokens promedio
  - Reales: ~500 palabras (~700 tokens)
  - Gap enorme → experimentos F (length-aware) críticos

- [ ] **Analizar rejection_analysis**:
  - ¿Por qué se rechazan tantos candidatos?
  - ¿Qué threshold es el más restrictivo?

### Baja Prioridad

- [ ] **Probar sin val-gating para clases minoritarias**
  - Val set pequeño puede dar señales ruidosas

- [ ] **Considerar contrastive prompting solo para clases confusas**
  - ENFJ vs INFJ
  - ESTJ vs ISTJ

---

## Configuración de Experimentos GPT-5-mini

| Experimento | reasoning_effort | output_verbosity | Resultado |
|-------------|------------------|------------------|-----------|
| A1 | none | medium | -0.37% (s100) |
| A2 | low | medium | -0.59% (s100), -3.31% (s123) |
| A3 | medium | high | **+1.05%** (s100), -0.97% (s123) |
| A4 | high | high | En progreso |

**Observación**: A3 (medium reasoning) fue el único positivo. ¿Es el output_verbosity=high o el reasoning_effort=medium?

---

## Experimentos Pendientes de Analizar

### Wave 1 (parcial)
- [x] A1_gpt5_none (2/3 seeds)
- [x] A2_gpt5_low (2/3 seeds)
- [x] A3_gpt5_medium (2/3 seeds)
- [ ] A4_gpt5_high (en progreso)
- [ ] B1_ip_baseline
- [ ] B2_ip_aggressive
- [ ] B3_ip_high_only

### Wave 2
- [ ] C1-C3 (cluster volume)
- [ ] D1-D3 (filtros)
- [ ] E1-E3 (minority focus) ← **Críticos para ESFP/ESTJ**

### Wave 3
- [ ] F1-F2 (length-aware) ← **Críticos para gap de longitud**
- [ ] G1-G3 (combinaciones)
- [ ] H1-H2 (contrastive/risky)

---

## Hipótesis a Validar

1. **IP scaling (B experiments)** generará más sintéticos para ESFP/ESTJ
2. **Relaxed thresholds (D1)** permitirá más sintéticos para minoritarias
3. **Length-aware (F2 500 words)** mejorará calidad de sintéticos
4. **Contrastive (H1)** ayudará a ENFJ a no confundirse con INFJ

---

## Notas Técnicas

- Cache de embeddings funcionando correctamente
- 3 experimentos en paralelo (RTX 3090 + Tier 3 API)
- **~32 min por run** (no 15-20 min como se estimó)

---

## Análisis de Capacidad API (Tier 3)

### Límites de API

| Modelo | Requests/min | Tokens/min | Tokens/día |
|--------|--------------|------------|------------|
| gpt-4o-mini | 5000 RPM | 4M TPM | 40M |
| gpt-5-mini | 5000 RPM | 4M TPM | 40M |

### Uso Actual vs Capacidad

| Métrica | Actual (3 paralelos) | Límite | Uso |
|---------|---------------------|--------|-----|
| Requests/min | 338 | 5000 | **6.8%** |
| Tokens/min | 202K | 4M | **5.1%** |

**Conclusión**: Solo usamos ~7% de la capacidad API disponible.

### Recomendación para Futuras Runs

| Paralelos | Tiempo Estimado | Uso API | Recomendación |
|-----------|-----------------|---------|---------------|
| 3 (actual) | ~32h | 6.8% | Muy conservador |
| **10** | **~10h** | **22%** | **Recomendado** |
| 20 | ~5h | 45% | Agresivo pero viable |
| 35 | ~3h | 80% | Máximo seguro |

### Cómo Aplicar

```bash
# En run_parallel.sh, cambiar línea 12:
PARALLEL_JOBS=${PARALLEL_JOBS:-10}  # En vez de 3
```

### Consideraciones de Hardware

| Recurso | Capacidad | Con 10 paralelos | Estado |
|---------|-----------|------------------|--------|
| GPU (RTX 3090) | 24GB VRAM | ~8GB usado | ✅ OK |
| RAM | ~32GB | ~20GB usado | ✅ OK |
| Disco | - | ~50MB/run | ✅ OK |
| API | 5000 RPM | 1125 RPM | ✅ OK |

---

## Lecciones Aprendidas

1. **Estimación de tiempo fue muy optimista**: 32 min/run vs 15 min estimado
2. **API está subutilizada**: Podríamos correr 10x más experimentos en paralelo
3. **Bottleneck real**: No es API, es tiempo de generación LLM por experimento
4. **Para futuras runs**: Usar PARALLEL_JOBS=10 desde el inicio

---

*Última actualización: 2025-12-02 22:05*
