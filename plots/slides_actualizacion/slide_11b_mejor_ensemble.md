# Slide 11b: Mejor Ensemble

## Titulo
**Ensembles: Combinando lo Mejor**

---

## Pregunta de Investigación

> *"Si una configuración logra +1.32 pp, ¿combinar varias logra más?"*

---

## Resultado: SÍ, los Ensembles Superan Individuales

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Ensemble: TOP_all_common                                  │
│   (Combinación de mejores configuraciones)                  │
│                                                             │
│   ┌───────────────────────────────────────────────────┐     │
│   │                                                   │     │
│   │         Δ Macro F1: +1.61 pp                      │     │
│   │                                                   │     │
│   │         vs Individual: +22% MEJOR                 │     │
│   │                                                   │     │
│   │         p-valor: <0.05 ✓                          │     │
│   │                                                   │     │
│   └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparación Visual

```
                                              Δ F1 (pp)
TOP_all_common (Ensemble)    ████████████████  +1.61 pp  ← GANADOR
W5_shot_140 (Individual)     █████████████     +1.32 pp
                             ─────────────────────────────
                                              +22% mejor
```

---

## Metodología Hold-out: Resultados Aún Mejores

| Metodología | Ensemble | Δ F1 (pp) | F1 Final |
|-------------|----------|-----------|----------|
| **Hold-out correcto** | ENS_TopG5_Extended | **+2.78** | 0.2306 |
| K-Fold (5×3) | ENS_SUPER_G5_F7_v2 | +2.05 | 0.2252 |
| Validación | TOP_all_common | +1.61 | - |

**Nota**: Hold-out correcto evita data leakage → mejoras reales

---

## ¿Por Qué Funcionan los Ensembles?

```
┌────────────────────────────────────────────────────────────┐
│  DIVERSIDAD ESTRATÉGICA                                    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Config A (N-shot=140)  →  Cubre clases X, Y               │
│       +                                                    │
│  Config B (Temp=1.2)    →  Cubre clases Y, Z               │
│       +                                                    │
│  Config C (Sin filtro)  →  Cubre clases X, Z               │
│       =                                                    │
│  ENSEMBLE               →  Cubre TODAS las clases          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Ranking de Ensembles

| Rank | Ensemble | Δ F1 (pp) | Estrategia |
|------|----------|-----------|------------|
| 1 | **TOP_all_common** | **+1.61** | Mejores configs comunes |
| 2 | ENS_WaveChampions | +0.90 | Champions de cada wave |
| 3 | ENS_Top3_G5 | +0.89 | Top 3 Phase G |
| 4 | ENS_ProblemClass | +0.75 | Foco en clases raras |

---

## Hallazgo Clave

> **La diversidad estratégica supera el volumen**: Combinar configuraciones con diferentes fortalezas cubre más clases que una sola configuración optimizada.

---

## Notas del Presentador
- Ensembles ganan por diversidad, no por volumen
- Hold-out correcto (+2.78 pp) evita data leakage del K-fold
- TOP_all_common combina las mejores configs de cada experimento
- Transición: "Pero ¿qué clases específicas mejoran más?"
