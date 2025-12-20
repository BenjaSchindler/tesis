# Slide 11: Resultados Principales (Regresión Logística)

## Título
**Resultados con Regresión Logística**

---

## Contenido Principal

### Configuración Base
- **Clasificador**: Regresión Logística
- **Validación**: K-Fold CV (5×3 = 15 folds) y Hold-out correcto
- **Métrica**: Macro F1-Score
- **Línea base**: 0.2045

---

### Mejores Resultados (en pp)

| Metodología | Configuración | Δ F1 (pp) | F1 Resultado |
|-------------|---------------|-----------|--------------|
| **Hold-out correcto** | ENS_TopG5_Extended | **+2.78** | 0.2306 |
| K-Fold (5×3) | ENS_SUPER_G5_F7_v2 | +2.05 | 0.2252 |
| Validación | TOP_all_common | +1.61 | - |
| Individual | W5_shot_140 | +1.32 | - |

---

### Mejoras por Clase (Hold-out Correcto)

| Clase | Δ F1 (pp) | Observación |
|-------|-----------|-------------|
| **ISFJ** | **+19.9** | Breakthrough |
| **ESTP** | **+19.0** | Ahora detectable |
| ENTJ | +6.6 | Mejorado |
| ISTJ | +6.5 | Mejorado |
| ESFJ | +2.4 | Mejorado |

---

### Hallazgos Clave

1. **Hold-out correcto** evita data leakage y muestra mejoras reales (+2.78 pp)
2. **Ensembles > Individual**: +1.61 pp vs +1.32 pp
3. **Clases ultra-minoritarias** (ISFJ, ESTP) muestran breakthroughs de ~20 pp

---

### Pregunta Abierta

> *"¿Y si usamos otros clasificadores más potentes?"*

**→ Siguiente slide: Análisis de Ensambles**

---

## Figuras Sugeridas
- Gráfico de barras: Mejoras por configuración
- Heatmap: Mejoras por clase

## Notas del Presentador
- Enfatizar diferencia entre K-fold (+2.05 pp) y hold-out correcto (+2.78 pp)
- ISFJ y ESTP muestran mejoras dramáticas de ~20 pp
- Ensembles superan configuraciones individuales
- Preparar transición hacia exploración de clasificadores
