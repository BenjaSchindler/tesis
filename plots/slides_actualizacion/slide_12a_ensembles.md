# Slide 12a: Análisis de Ensambles

## Título
**Ensembles de Configuraciones: ¿Mejoran los Resultados?**

---

## Pregunta de Investigación

> *"Si múltiples configuraciones funcionan bien individualmente, ¿combinarlas produce mejores resultados?"*

---

## Mejores Resultados por Metodología (en pp)

| Metodología | Configuración | Δ F1 (pp) | F1 Base → Aug |
|-------------|---------------|-----------|---------------|
| **Hold-out Correcto** | ENS_TopG5_Extended | **+2.78** | 0.2027 → 0.2306 |
| K-Fold (5×3) | ENS_SUPER_G5_F7_v2 | +2.05 | 0.2047 → 0.2252 |
| Validación Extendida | TOP_all_common | +1.61 | - |

---

## Comparación: Ensembles vs Individual

| Estrategia | Δ F1 (pp) | Resultado |
|------------|-----------|-----------|
| **Mejor Ensemble** (TOP_all_common) | **+1.61** | ← GANADOR |
| Mejor Individual (W5_shot_140) | +1.32 | -18% peor |

---

## Hallazgo Clave

### ✅ Los ensembles SÍ superan las configuraciones individuales

**Razón**: La **diversidad estratégica** de múltiples configuraciones mejora cobertura de clases.

---

## Top Ensembles (Validación Extendida)

| Ensemble | Estrategia | Δ F1 (pp) | p-valor |
|----------|------------|-----------|---------|
| **TOP_all_common** | Mejores configs comunes | **+1.61** | <0.05 |
| ENS_WaveChampions | Champions de cada wave | +0.90 | 0.0042 |
| ENS_Top3_G5 | Top 3 Phase G | +0.89 | 0.0002 |

---

## Visualización

```
ENS_TopG5_Extended (Hold-out)  ████████████████████████████  +2.78 pp
ENS_SUPER_G5_F7_v2 (K-fold)    ████████████████████          +2.05 pp
TOP_all_common (Validación)    ████████████████              +1.61 pp
W5_shot_140 (Individual)       █████████████                 +1.32 pp
```

---

## Mejoras por Clase (Hold-out Correcto)

| Clase | Δ F1 (pp) | Observación |
|-------|-----------|-------------|
| **ISFJ** | **+19.9** | Breakthrough |
| **ESTP** | **+19.0** | Ahora detectable |
| ENTJ | +6.6 | Mejorado |
| ISTJ | +6.5 | Mejorado |

---

## Implicación Práctica

> **Recomendación**: Usar ensembles estratégicos (TOP_all_common o ENS_TopG5_Extended) para maximizar mejoras.

---

## Notas del Presentador
- Los ensembles SÍ superan configuraciones individuales (+1.61 vs +1.32 pp)
- El hold-out correcto muestra mejoras aún mayores (+2.78 pp)
- Clases como ISFJ y ESTP muestran breakthroughs significativos
- Transición: "Pero ¿qué pasa si cambiamos el clasificador?"
