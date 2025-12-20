# Slide 13: Hallazgos Clave (ACTUALIZAR)

## Título
**Hallazgos Clave**

---

## Hallazgos Originales (mantener)
1. **Control de Calidad**: Cascada de 4 filtros crítica
2. **Configuración Inteligente**: Parámetros óptimos validados
3. **Generación Semántica**: LLM produce texto coherente

---

## NUEVOS Hallazgos a Agregar

### 4. MLPs Profundos Explotan Mejor los Sintéticos
> **+4.17 pp** con MLP (512-256-128) vs **+0.36 pp** con LogReg
>
> Las redes neuronales profundas aprovechan **12× mejor** los datos generados

### 5. 60 Ejemplos In-Context es Óptimo
> Patrón **no-monotónico**: 60 ejemplos > 70 > 100
>
> **+1.67 pp** (p=0.0003) - Mejor configuración individual

### 6. Ensembles SÍ Superan Mejor Individual
> Mejor ensemble (TOP_all_common): **+1.61 pp**
> Mejor individual (W5_shot_140): +1.32 pp
>
> La **diversidad estratégica** mejora cobertura de clases

### 7. Clases Ultra-Minoritarias Requieren MLPs
> **ESFJ resuelto** (+1.36 pp) con sobremuestreo 20× + MLP Profundo
>
> Estrategia específica para clases < 50 muestras

---

## Resumen Visual de Hallazgos

```
┌─────────────────────────────────────────────────────────────┐
│  HALLAZGO                           │  IMPACTO              │
├─────────────────────────────────────┼───────────────────────┤
│  MLP Profundo vs LogReg             │  12× mejor (+4.17pp)  │
│  60 ejemplos in-context             │  +1.67pp (p<0.001)    │
│  Ensembles vs Individual            │  Ensemble gana (+22%) │
│  Hold-out correcto (sin leakage)    │  +2.78pp (ISFJ +19.9) │
│  ESFJ con MLP + 20× oversample      │  Problema resuelto    │
│  ESFP                               │  Limitación conocida  │
└─────────────────────────────────────┴───────────────────────┘
```

---

## Implicaciones Prácticas

1. **Para máxima mejora**: Usar ensembles estratégicos + MLP Profundo
2. **Para metodología**: Hold-out correcto evita data leakage (+2.78 pp)
3. **Para clases raras**: Sobremuestreo masivo + clasificador no lineal

---

## Notas del Presentador
- Ensembles SÍ mejoran (+1.61 pp ensemble vs +1.32 pp individual)
- Hold-out correcto muestra mejoras aún mayores que K-fold
- ISFJ y ESTP muestran breakthroughs de ~20 pp
- Mencionar ESFP como limitación honesta (transparencia científica)
