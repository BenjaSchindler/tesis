# Slide 11c: Mejora por Clases

## Titulo
**Breakthroughs en Clases Ultra-Minoritarias**

---

## El Desafío: Clases con F1 = 0%

Antes de SMOTE-LLM, algunas clases tenían **cero predicciones correctas**:

| Clase | Samples | F1 Baseline | Estado |
|-------|---------|-------------|--------|
| ISFJ | 181 | ~0.00 | Sin detección |
| ESTP | 89 | 0.00 | Sin detección |
| ESFJ | 42 | 0.00 | Sin detección |
| ESTJ | 39 | 0.00 | Sin detección |
| ESFP | 48 | 0.00 | Sin detección |

---

## Resultado: Breakthroughs con Hold-out Correcto

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   CLASE      ANTES    DESPUÉS     Δ F1 (pp)    ESTADO       │
│   ─────────────────────────────────────────────────────     │
│   ISFJ       ~0.00    ~0.20       +19.9 pp     ⭐ BREAK     │
│   ESTP        0.00     0.19       +19.0 pp     ⭐ BREAK     │
│   ENTJ        0.06     0.13       +6.6 pp      ✓ Mejorado   │
│   ISTJ        0.05     0.11       +6.5 pp      ✓ Mejorado   │
│   ESFJ        0.00     0.02       +2.4 pp      ✓ Detectable │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Visualización de Mejoras

```
                                                    Δ F1 (pp)
ISFJ    ████████████████████████████████████████    +19.9 pp  ⭐
ESTP    ███████████████████████████████████████     +19.0 pp  ⭐
ENTJ    █████████████                               +6.6 pp
ISTJ    ████████████                                +6.5 pp
ESFJ    █████                                       +2.4 pp
ENFJ    ████                                        +2.0 pp
```

---

## Clases Problemáticas: Estado Final

| Clase | Samples | Resultado | Solución |
|-------|---------|-----------|----------|
| **ESFJ** | 42 | **RESUELTO** | MLP Profundo + 20× oversample |
| **ESTJ** | 39 | **PARCIAL** | +1.79 pp solo con MLP |
| **ESFP** | 48 | **SIN RESOLVER** | Limitación conocida |

---

## ¿Por Qué ESFP No Mejora?

```
┌─────────────────────────────────────────────────────────────┐
│  HIPÓTESIS: ESFP es "textualmente ambiguo"                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • 944 sintéticos generados → 0% mejora                     │
│  • 5 clasificadores probados → 0% mejora                    │
│  • 38 configuraciones → 0% mejora                           │
│                                                             │
│  Posibles causas:                                           │
│  1. Overlap semántico con otras clases (ISFP, ESTP)         │
│  2. Características no capturables en texto                 │
│  3. Insuficientes muestras reales (48) para aprender        │
│                                                             │
│  → Limitación honesta del enfoque basado en texto           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Impacto por Tier

| Tier | Clases | Mejora Media | Observación |
|------|--------|--------------|-------------|
| **Bajo** (F1<0.20) | 9 clases | **+8.5 pp** | Mayor beneficio |
| Intermedio (0.20-0.45) | 6 clases | +0.3 pp | Neutral |
| Alto (F1≥0.45) | 1 clase | -0.1 pp | Protegido |

**Hallazgo**: SMOTE-LLM beneficia principalmente a clases de bajo rendimiento.

---

## Hallazgo Clave

> **Las clases más difíciles muestran los mayores breakthroughs**: ISFJ y ESTP pasan de F1=0% a F1~20%, demostrando que SMOTE-LLM puede "revivir" clases previamente indetectables.

---

## Notas del Presentador
- ISFJ y ESTP: de 0% a ~20% F1 es un breakthrough significativo
- ESFJ resuelto con MLP profundo + sobremuestreo masivo
- ESFP es una limitación honesta → transparencia científica
- Las clases de bajo rendimiento se benefician 28× más que las intermedias
- Transición: "Ahora veamos los hallazgos clave consolidados..."
