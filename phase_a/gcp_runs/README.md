# Phase A - GCP GPU Runs

Documentación de ejecuciones en Google Cloud Platform con GPUs NVIDIA T4.

---

## Runs

### [run_20251115_FAILED.md](run_20251115_FAILED.md) ❌
### [GPU_RUN_2025_11_15.md](GPU_RUN_2025_11_15.md) ❌ (Mismo run, detalles completos)

**Fecha:** 2025-11-15 14:59-22:30 UTC
**Seeds:** 20 (4 VMs × 5 seeds)
**Status:** Failed - Configuration Error
**Cost:** ~$20 USD (sin resultados útiles)

**Errores encontrados:**
1. F1-budget multipliers incorrectos (30, 70, 100 instead of 0.0, 0.5, 1.0)
2. Anchor-quality threshold demasiado estricto (0.50 rechazó 87.5% de clases)

**Resultado:** 0/20 seeds generaron datos sintéticos

**Lecciones:**
1. Multipliers son factores (0.0-1.0), no cantidades absolutas
2. Threshold 0.50 es demasiado estricto para datasets con overlap natural (MBTI)

**Análisis detallado:** Ver [ANCHOR_QUALITY_ANALYSIS.md](ANCHOR_QUALITY_ANALYSIS.md)

---

## Configuración Correcta (Actualizada 2025-11-15)

```bash
--f1-budget-thresholds 0.45 0.20
--f1-budget-multipliers 0.0 0.5 1.0
--anchor-quality-threshold 0.30      # Reducido de 0.50 → 0.30
--use-ensemble-selection
--use-val-gating
--val-size 0.15
--val-tolerance 0.02
```

**Justificación threshold 0.30:**
- Batch 5 Phase A NO usaba anchor-gate (implícito threshold=0.0)
- Threshold 0.50 rechaza 7/8 clases (purity scores muy bajos debido a overlap MBTI)
- Threshold 0.30 acepta 8/8 clases pero mantiene protección básica
- Ver análisis completo en [ANCHOR_QUALITY_ANALYSIS.md](ANCHOR_QUALITY_ANALYSIS.md)

**Expected Performance:**
- Mean Delta: +1.00% ± 0.25%
- Success Rate: 80-90% seeds
- Variance: < 5pp

---

## Costos

| Run | VMs | Tiempo | GPU Cost | Compute | API | Total |
|-----|-----|--------|----------|---------|-----|-------|
| 2025-11-15 | 4×T4 | ~7.5h | $10.50 | $5.70 | ~$4 | ~$20 |

**Lección:** Verificar configuración antes de lanzar para evitar gastos en runs fallidos.

---

## Próximo Run

**Archivo:** `run_YYYYMMDD_phaseA_corrected.md` (pendiente)

**Cambios:**
- F1-budget multipliers corregidos: 0.0, 0.5, 1.0
- F1-budget thresholds: 0.45, 0.20
- **Anchor-quality threshold reducido: 0.30** (en vez de 0.50)
- Mismo hardware: 4 VMs × NVIDIA T4
- Mismos seeds: 20 total

**Expected Cost:** ~$15-20 USD
**Expected Time:** ~6-8 hours
**Expected Results:** +1.00% ± 0.25% macro F1

**Scripts actualizados:**
- [run_batch_batch1_gpu.sh](../batch_scripts/run_batch_batch1_gpu.sh) ✅
- [run_batch_batch2_gpu.sh](../batch_scripts/run_batch_batch2_gpu.sh) ✅
- [run_batch_batch3_gpu.sh](../batch_scripts/run_batch_batch3_gpu.sh) ✅
- [run_batch_batch4_gpu.sh](../batch_scripts/run_batch_batch4_gpu.sh) ✅

---

**Última actualización:** 2025-11-15
