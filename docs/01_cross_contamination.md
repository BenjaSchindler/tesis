# Cross-Contamination (Envenenamiento entre Clases)

**Severidad:** 🟡 High
**Status:** ⚠️ Mitigated (no completamente resuelto)
**Solución:** Strict thresholds + per-class gating

---

## 🎯 El Problema

### Definición

**Synthetics generados para clase A son clasificados como clase B**

### Manifestación

```python
# Generate for ENFP
synthetic = llm.generate("Create ENFP text...")

# But classifier predicts:
prediction = classifier.predict(synthetic)
# prediction = "INFP" (wrong!)

contamination_rate = 0.15  # 15% contaminated
```

---

## 🔍 Por Qué Ocurre

### 1. Similitud Semántica

**Pares problemáticos:**
```
ENFP ↔ INFP (E/I flip only)
ISTJ ↔ ISFJ (T/F flip only)
ENTP ↔ INTP (E/I flip only)
```

### 2. Embedding Space Overlap

- Boundaries difusas entre clases similares
- Solo 1 dimensión MBTI difiere

### 3. LLM Confusion

- GPT no es experto en MBTI
- Fine-grained distinctions difíciles

---

## 💡 Teoría: Proportional Contamination

### Hipótesis

```
contamination_rate ∝ semantic_similarity(A, B)
```

### Medición

```python
predictions = classifier.predict_proba(synthetics_A)
prob_B = predictions[:, class_B_idx]
contamination = np.mean(prob_B > threshold)
```

---

## 💡 Soluciones Implementadas

### 1. Strict Similarity Threshold

```python
similarity_threshold = 0.90  # vs 0.70 default

if cosine_similarity(synthetic, anchor) > 0.90:
    reject()  # Too similar = duplicate risk
```

**Effect:** Reduce accepted synthetics, improve quality

### 2. Contamination Threshold

```python
contamination_threshold = 0.95  # vs 0.80

predicted_proba = classifier.predict_proba(synthetic)
if predicted_proba[wrong_class] > 0.95:
    reject()  # Strong contamination
```

**Effect:** Filter severe contamination only

### 3. Per-Class Gating

```python
for cls in classes:
    if val_performance_degrades(cls):
        reject_synthetics(cls)
```

**Effect:** Early stopping per-class

### 4. Ensemble Selection

```python
F1_final[cls] = max(F1_baseline[cls], F1_augmented[cls])
```

**Effect:** Safety net si contamination ocurre

---

## 📊 Resultados

### Fase A

**MID tier:** -0.59% (posible contamination residual)

**Evidence:**
- Consistent degradation 0/4 classes
- Worst: ENTJ -1.91%
- Hypothesis: Contaminated by similar classes

### Impact

**Mitigated pero not eliminated:**
- Strict thresholds ayudan
- Pero contamination persiste en MID tier
- Fase B targets this con adaptive weighting

---

## 🎯 Status

**Current:** ⚠️ **Mitigated**

**What works:**
- Protege HIGH tier (100%)
- Reduce contamination severity

**What doesn't:**
- MID tier aún vulnerable
- Contamination from LOW tier

**Next:** Fase B adaptive weighting

---

**Referencias:**
- [MID Tier Vulnerability](04_mid_tier_vulnerability.md)
- [Problemas Identificados](../../01_CONTEXTO/02_problemas_identificados.md)
