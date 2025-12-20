# Slide 11a: Mejor Configuración Individual

## Titulo
**Mejor Configuración Individual**

---

## Resultado Principal

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Configuración: W5_shot_140                                │
│   (140 ejemplos in-context)                                 │
│                                                             │
│   ┌───────────────────────────────────────────────────┐     │
│   │                                                   │     │
│   │         Δ Macro F1: +1.32 pp                      │     │
│   │                                                   │     │
│   │         F1 Base: 0.2045                           │     │
│   │         F1 Augmentado: 0.2177                     │     │
│   │                                                   │     │
│   │         p-valor: <0.05 ✓                          │     │
│   │                                                   │     │
│   └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuración Técnica

| Parámetro | Valor |
|-----------|-------|
| Clasificador | Regresión Logística |
| Validación | K-Fold CV (5×3 = 15 folds) |
| N-shot | 140 ejemplos |
| Temperatura | 0.7 |
| Modelo LLM | gpt-4o-mini |

---

## Patrón No-Monotónico del N-shot

```
Ejemplos    Δ F1 (pp)
────────────────────────
   0        +0.29      ░░░
   3        +0.32      ░░░
  10        +0.38      ░░░░
  20        +0.49      ░░░░░
  60        +1.22      ░░░░░░░░░░░░
 100        +0.98      ░░░░░░░░░░
 140        +1.32      ░░░░░░░░░░░░░  ← ÓPTIMO
 200        +1.15      ░░░░░░░░░░░
```

**Hallazgo**: Más ejemplos NO siempre es mejor. 140 es el punto óptimo.

---

## Comparación con Otras Configuraciones

| Configuración | Δ F1 (pp) | Observación |
|---------------|-----------|-------------|
| **W5_shot_140** | **+1.32** | MEJOR INDIVIDUAL |
| W5_shot_60 | +1.22 | Segundo mejor |
| W6_temp_high (τ=1.2) | +1.14 | Temperatura alta |
| W7_yolo (sin filtro) | +1.03 | Sin filtrado |
| W5_shot_10 | +0.38 | Pocos ejemplos |

---

## Implicación

> **El contexto importa**: 140 ejemplos dan al LLM suficiente información para generar sintéticos de alta calidad sin saturar el prompt.

---

## Notas del Presentador
- Enfatizar que 140 ejemplos es el punto óptimo empírico
- El patrón no-monotónico es contraintuitivo (más ≠ mejor)
- Este es el mejor resultado CON UNA SOLA configuración
- Transición: "Pero ¿podemos combinar configuraciones?"
