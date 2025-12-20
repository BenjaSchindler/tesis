# Resumen: Mejores Resultados en Puntos Porcentuales (pp)

**Fecha de actualización**: 2025-12-17
**Métrica**: Macro F1-Score
**Línea base**: 0.2045

---

## Mejores Resultados Globales

| Rank | Metodología | Configuración | Δ F1 (pp) | F1 Base | F1 Aug | Significativo |
|------|-------------|---------------|-----------|---------|--------|---------------|
| 1 | Hold-out correcto | **ENS_TopG5_Extended** | **+2.78** | 0.2027 | 0.2306 | p<0.001 |
| 2 | Hold-out correcto | ENS_SUPER_G5_F7_v2 | +2.70 | - | - | p<0.001 |
| 3 | K-Fold (5×3) | ENS_SUPER_G5_F7_v2 | +2.05 | 0.2047 | 0.2252 | p=0.000003 |
| 4 | Validación extendida | TOP_all_common | +1.61 | - | - | p<0.05 |
| 5 | Individual | W5_shot_140 | +1.32 | - | - | p<0.05 |

---

## Mejoras por Clase (Hold-out Correcto)

| Clase | Δ F1 (pp) | Antes | Después | Estado |
|-------|-----------|-------|---------|--------|
| **ISFJ** | **+19.9** | ~0.00 | ~0.20 | BREAKTHROUGH |
| **ESTP** | **+19.0** | 0.00 | 0.19 | Ahora detectable |
| ENTJ | +6.6 | 0.059 | 0.125 | Mejorado |
| ISTJ | +6.5 | 0.049 | 0.114 | Mejorado |
| ISFJ | +4.6 | 0.166 | 0.212 | Mejorado |
| ESFJ | +2.4 | 0.00 | 0.024 | Ahora detectable |
| ENFJ | +2.0 | 0.016 | 0.036 | Mejorado |

---

## Comparación: Ensembles vs Individual

| Tipo | Mejor Configuración | Δ F1 (pp) | Resultado |
|------|---------------------|-----------|-----------|
| **Ensemble** | TOP_all_common | **+1.61** | GANADOR (+22%) |
| Individual | W5_shot_140 | +1.32 | -22% vs ensemble |

**Conclusión**: Los ensembles SÍ superan las configuraciones individuales.

---

## Mejores Resultados por Clasificador

| Clasificador | Δ F1 (pp) | p-valor | Observación |
|--------------|-----------|---------|-------------|
| **MLP (512-256-128)** | **+4.17** | 0.0010 | MEJOR |
| MLP (256-128) | +0.33 | >0.05 | No significativo |
| LogisticRegression | +0.36 | >0.05 | No significativo |
| XGBoost | -0.43 | >0.05 | EMPEORA |
| LightGBM | -0.11 | >0.05 | Sin efecto |

---

## Clases Problemáticas

| Clase | Samples | Estado | Mejor Resultado |
|-------|---------|--------|-----------------|
| ESFJ | 42 | RESUELTO | +12.42 pp con MLP Profundo |
| ESTJ | 39 | PARCIAL | +1.79 pp con MLP Profundo |
| ESFP | 48 | SIN RESOLVER | 0% en todas las configs |

---

## Hallazgos Clave

1. **Ensembles > Individual**: +1.61 pp vs +1.32 pp (ensembles ganan)
2. **Hold-out correcto**: Evita data leakage, muestra +2.78 pp (vs +2.05 pp K-fold)
3. **MLP Profundo**: 12× mejor que LogReg (+4.17 pp vs +0.36 pp)
4. **Clases ultra-minoritarias**: ISFJ y ESTP muestran breakthroughs de ~20 pp
5. **ESFP**: Limitación conocida, no mejora con ninguna estrategia

---

## Recomendaciones Finales

### Para máxima mejora global:
- Usar ensemble **ENS_TopG5_Extended** con metodología hold-out

### Para clases ultra-minoritarias:
- Usar **MLP Profundo (512-256-128)** + sobremuestreo 20×

### Para producción:
- Ensemble **TOP_all_common** es robusto y significativo (+1.61 pp)
