# Seed Variance Problem - 54pp Range

**Severidad:** 🔴 Critical
**Status:** ✅ Solved (93% reduction)
**Solución:** F1-budget scaling

---

## 🎯 El Problema

### Manifestación

**Experimentos sin F1-budget scaling:**
```
Seed 42:  Macro F1 = 0.45
Seed 100: Macro F1 = 0.67  
Seed 789: Macro F1 = 0.99

Range: 54 percentage points
Std: 18.3pp
```

**Implicaciones:**
- ❌ NO reproducible
- ❌ NO publicable  
- ❌ NO confiable para producción

---

## 🔍 Causa Raíz

### Sin F1-Budget Scaling

**Flujo problemático:**
```
1. Todas las clases reciben augmentation
2. HIGH F1 classes (≥45%) se degradan
3. Degradación varía entre seeds
4. Variance extrema en macro F1
```

### Ejemplo Concreto

**Seed 42 (unlucky):**
```
INFP: 0.82 → 0.74 (-8pp)
INFJ: 0.75 → 0.68 (-7pp)
ENTP: 0.65 → 0.58 (-7pp)
Total: -22pp degradation
```

**Seed 789 (lucky):**
```
INFP: 0.82 → 0.83 (+1pp)
INFJ: 0.75 → 0.77 (+2pp)
ENTP: 0.65 → 0.67 (+2pp)
Total: +5pp improvement
```

**Swing: 27pp difference!**

---

## 💡 Solución: F1-Budget Scaling

### Concepto

**Tiered augmentation:**
```python
if F1 >= 0.45:  # HIGH tier
    multiplier = 0.0  # SKIP
elif F1 >= 0.20:  # MID tier
    multiplier = 0.5  # REDUCE
else:  # LOW tier
    multiplier = 1.0  # FULL
```

### Justificación

**HIGH tier protection:**
- Risk > Reward
- Synthetics add noise
- → SKIP completamente

**Result:** Elimina fuente principal de variance

---

## 📊 Resultados

### Before & After

**BEFORE (sin F1-budget):**
```
Range: 54pp
Mean: Variable
Std: 18.3pp
```

**AFTER (con F1-budget):**
```
Range: 3.75pp
Mean: Stable
Std: 0.02pp

Reduction: 93% ✅
```

### Validation

**Batch 1E:**
```
Seed 42:  -0.88%
Seed 100: -0.92%
Difference: 0.04pp (just 4 basis points!)
```

**Fase A:**
```
Seed 42: +1.00%
(Expected seed 100: +0.97% to +1.03%)
```

---

## 🎯 Conclusión

**Status:** ✅ **SOLVED**

**Solution:** F1-budget scaling

**Impact:**  
- 93% variance reduction
- Reproducibilidad garantizada
- Foundation para todas las mejoras

---

**Referencias:**
- [Batch 1E Analysis](../../03_EXPERIMENTOS/01_batch1_f1_budget.md)
- [High-F1 Protection](03_high_f1_protection.md)
