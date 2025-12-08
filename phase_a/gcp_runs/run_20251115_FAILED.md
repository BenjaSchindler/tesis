# GPU Run - 2025-11-15 - FAILED

**Status:** ❌ Failed (Configuration Error)
**Started:** 14:59 UTC
**Expected End:** ~22:30 UTC
**Seeds:** 20 (4 VMs × 5 seeds)
**Cost:** ~$20 USD

---

## Error Encontrado

### Problema: F1-Budget Multipliers Incorrectos

**Configuración Usada (INCORRECTA):**
```bash
--f1-budget-thresholds 0.35 0.20
--f1-budget-multipliers 30 70 100  # ← ERROR
```

**Configuración Correcta (Batch 5 Phase A):**
```bash
--f1-budget-thresholds 0.45 0.20
--f1-budget-multipliers 0.0 0.5 1.0  # ← CORRECTO
```

### Impacto

**Con multipliers incorrectos (30, 70, 100):**
- HIGH class: budget_base × 30 = 300 synthetics (debería ser 0)
- MID class: budget_base × 70 = 700 synthetics (debería ser ~5-15)
- LOW class: budget_base × 100 = 1000 synthetics (debería ser ~10-30)

**Resultado:**
- Todas las clases rechazadas por `anchor_quality < 0.50`
- 0/20 seeds generaron datos sintéticos
- Mensaje: "No se generaron ejemplos sintéticos tras el filtrado"

---

## Configuración Completa Usada

```bash
--data-path MBTI_500.csv
--test-size 0.2
--embedding-model sentence-transformers/all-mpnet-base-v2
--device cuda
--embedding-batch-size 256
--llm-model gpt-4o-mini
--max-clusters 3
--prompts-per-cluster 3
--prompt-mode mix

# Ensemble & Gating
--use-ensemble-selection
--use-val-gating
--val-size 0.15
--val-tolerance 0.02

# Anchor Quality
--enable-anchor-gate
--anchor-quality-threshold 0.50  # ← Correcto
--enable-anchor-selection
--anchor-selection-ratio 0.8
--anchor-outlier-threshold 1.5

# Quality Filters
--enable-adaptive-filters
--similarity-threshold 0.90
--min-classifier-confidence 0.10
--contamination-threshold 0.95

# F1-Budget Scaling (ERROR AQUÍ)
--use-f1-budget-scaling
--f1-budget-thresholds 0.35 0.20       # ← Debería ser 0.45 0.20
--f1-budget-multipliers 30 70 100      # ← ERROR: Debería ser 0.0 0.5 1.0

# Synthetic Weighting
--synthetic-weight 0.5
--synthetic-weight-mode flat

# Class Descriptions
--use-class-description
```

---

## Seeds Ejecutados

**Batch 1 (vm-batch1-gpu):** 42, 100, 123, 456, 789
**Batch 2 (vm-batch2-gpu):** 111, 222, 333, 444, 555
**Batch 3 (vm-batch3-gpu):** 1000, 2000, 3000, 4000, 5000
**Batch 4 (vm-batch4-gpu):** 7, 13, 21, 37, 101

**Todos completaron sin errores de ejecución.**
**Ninguno generó datos sintéticos.**

---

## Ejemplo de Log (Seed 42, batch1)

```
✨ Phase 2 Enhanced Quality Gate for ISTJ:
   Decision: skip (confidence: 0.26)
   Metrics: n=845, F1=0.247
   Quality: 0.308, Purity: 0.025, Cohesion: 0.743
   Reason: probabilistic_reject (confidence=0.26)
   Budget: 10 synthetics
   📊 Phase 3 F1-Based Budget Scaling:
      Baseline F1: 0.247 (MEDIUM)
      Budget multiplier: 70.0×
   💰 Phase 2 Enhanced Budget:
      probabilistic_reject (confidence=0.26)
      After F1 (70.0×) + contamination (1.0×): 700
⚠️  GATE: Skipping ISTJ - quality=0.308 < 0.500
    Cohesion=0.743, Purity=0.025, Separation=0.006

[... todas las demás clases también rechazadas ...]

No se generaron ejemplos sintéticos tras el filtrado.
✓ Seed 42 completed at Sat Nov 15 16:26:52 UTC 2025
```

---

## Análisis de Quality Scores

**Todas las clases tuvieron quality < 0.50:**

| Clase | Quality | Cohesion | Purity | Separation |
|-------|---------|----------|--------|------------|
| ISTJ  | 0.308   | 0.743    | 0.025  | 0.006      |
| ENFJ  | 0.308   | 0.736    | 0.033  | 0.004      |
| ESFJ  | 0.302   | 0.743    | 0.009  | 0.007      |
| ISFP  | 0.312   | 0.742    | 0.034  | 0.005      |
| ISFJ  | 0.306   | 0.745    | 0.016  | 0.006      |
| ESFP  | 0.323   | 0.764    | 0.041  | 0.005      |
| ESTJ  | 0.389   | 0.681    | 0.273  | 0.039      |
| ESTP  | 0.528   | 0.728    | 0.570  | -          |

ESTP único > 0.50, pero rechazado por F1 alto (0.789 > 0.6).

---

## Lecciones Aprendidas

1. **Multipliers son factores (0.0-1.0), NO cantidades absolutas**
   - 0.0 = skip augmentation
   - 0.5 = 50% del budget base
   - 1.0 = 100% del budget base

2. **Anchor quality threshold 0.50 es correcto**
   - Batch 5 Phase A usó el mismo threshold y funcionó
   - Diferencia: Batch 5 tenía multipliers correctos → budgets razonables

3. **Verificar configuración contra scripts validados**
   - Siempre comparar con batch_phase_a_config.sh de Batch 5
   - No asumir parámetros de documentación anterior sin validar

---

## Próximo Run (Corrección)

**Cambios necesarios:**
```bash
# ANTES (incorrecto)
--f1-budget-thresholds 0.35 0.20
--f1-budget-multipliers 30 70 100

# DESPUÉS (correcto)
--f1-budget-thresholds 0.45 0.20
--f1-budget-multipliers 0.0 0.5 1.0
```

**Expected Results:**
- HIGH classes (F1 ≥ 0.45): skip augmentation
- MID classes (0.20-0.45): ~5-15 synthetics cada una
- LOW classes (< 0.20): ~10-30 synthetics cada una
- Target: +1.00% ± 0.25% macro F1 (matching Batch 5)

---

**Documentado:** 2025-11-15 21:20 UTC
**VMs:** Corriendo últimos seeds (16/20 completados)
**Próximo paso:** Lanzar run corregido cuando estos terminen
