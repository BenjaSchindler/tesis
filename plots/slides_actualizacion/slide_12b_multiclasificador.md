# Slide 12b: Validación Multi-Clasificador

## Título
**¿Qué Clasificador Explota Mejor los Datos Sintéticos?**

---

## Configuración del Experimento

- **Datos**: Sobremuestreo masivo (20×) para clases minoritarias
- **Embeddings**: MPNet 768 dimensiones
- **Validación**: K-Fold CV (5×3 = 15 folds)
- **Clasificadores evaluados**: 5 tipos diferentes

---

## Resultados Comparativos

| Clasificador | Línea Base | Aumentado | Δ F1 (pp) | p-valor |
|--------------|------------|-----------|-----------|---------|
| Regresión Logística | 0.2272 | 0.2308 | +0.36 | 0.3001 |
| MLP Medio (256-128) | 0.2273 | 0.2306 | +0.33 | 0.6224 |
| LightGBM | 0.1677 | 0.1667 | **-0.10** | 0.6672 |
| XGBoost | 0.1788 | 0.1745 | **-0.43** | 0.1792 |
| **MLP Profundo (512-256-128)** | **0.2075** | **0.2492** | **+4.17** | **0.0010** ✓ |

---

## Visualización de Mejoras

```
MLP Profundo    ████████████████████████████████████  +4.17 pp ***
LogReg          ██                                    +0.36 pp
MLP Medio       ██                                    +0.33 pp
LightGBM        ▌                                     -0.10 pp
XGBoost                                               -0.43 pp
                |--------|--------|--------|--------|
               -1       0        1        2        3        4   pp
```

---

## Hallazgos Críticos

### 1. Redes Neuronales Profundas son el GANADOR
- **MLP Profundo**: +4.17 pp (p=0.001) ← **ÚNICO SIGNIFICATIVO**
- Mejora **12× mayor** que Regresión Logística

### 2. Modelos Basados en Árboles FALLAN
- XGBoost y LightGBM tienen línea base inferior (0.17-0.18)
- **Empeoran** con augmentación (-0.10 a -0.43 pp)
- No manejan bien embeddings de alta dimensionalidad (768D)

### 3. Redes Más Profundas = Mejor Explotación
- MLP Medio (256-128): +0.33 pp
- MLP Profundo (512-256-128): +4.17 pp
- **Profundidad importa para datos sintéticos**

---

## Implicación Práctica

> **Para maximizar el impacto de la augmentación LLM:**
>
> Usar **MLP Profundo (512-256-128)** en lugar de clasificadores lineales o basados en árboles.

---

## Figura Sugerida
- `fig_multiclassifier_comparison.pdf` - Comparativa visual de 5 clasificadores

---

## Notas del Presentador
- Enfatizar el contraste dramático: +4.17pp vs +0.36pp
- Explicar por qué árboles fallan: alta dimensionalidad, sparsity
- Transición: "Con MLP Profundo, ¿podemos resolver las clases más difíciles?"
