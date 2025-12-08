# Parámetros Justificados - Configuración Completa

**Objetivo:** Explicar CADA parámetro del sistema y justificar su selección
**Configuración:** BEST_CONFIG_FASE_A
**Resultado:** +1.00% macro F1, 93% variance reduction

---

## 📊 Tabla Resumen de Parámetros

| Categoría | Parámetro | Valor | Impacto | Justificación |
|-----------|-----------|-------|---------|---------------|
| **Dataset** | test_size | 0.20 | Medio | Estándar 80/20 |
| | random_seed | 42/100 | Alto | Reproducibilidad |
| **Embedding** | model | all-mpnet-base-v2 | Alto | SOTA semantic similarity |
| | embedding_dim | 768 | Medio | Balance size/quality |
| | batch_size | 32 | Bajo | Balance speed/memory |
| | device | cpu | Bajo | Cost-effective |
| **Clustering** | max_clusters | 3 | Alto | MBTI subcategories |
| | prompts_per_cluster | 3 | Medio | Balance diversity/cost |
| **LLM** | model | gpt-4o-mini | Alto | Balance cost/quality |
| | temperature | 1.0 | Medio | Maximum diversity |
| | max_tokens | 500 | Bajo | Match original length |
| **Quality** | similarity_threshold | 0.90 | **Crítico** | Anti-duplication |
| | min_confidence | 0.10 | Medio | Exploratory |
| | contamination_threshold | 0.95 | **Crítico** | Anti-poisoning |
| **Weighting** | synthetic_weight | 0.5 | Alto | Balance impact |
| | weight_mode | flat | Medio | Uniform (Fase A) |
| **F1-Budget** | high_threshold | 0.45 | **Crítico** | Protect strong |
| | mid_threshold | 0.20 | **Crítico** | Tiered approach |
| | high_multiplier | 0.0 | **Crítico** | Skip augmentation |
| | mid_multiplier | 0.5 | Alto | Reduce augmentation |
| | low_multiplier | 1.0 | Alto | Full augmentation |
| **Val-Gating** | val_size | 0.15 | Medio | Early stopping |
| | val_tolerance | 0.02 | Medio | 2% degradation OK |
| **Anchors** | anchor_quality | 0.50 | Alto | Quality gate |
| | anchor_selection_ratio | 0.80 | Alto | Top 80% |
| | outlier_threshold | 1.5 | Medio | IQR-based |

---

## 🗂️ CATEGORÍA 1: Dataset & Splits

### test_size = 0.20

**Valor:** 20% del dataset para test

**Justificación:**
- **Estándar industry:** 80/20 split es práctica común
- **Balance:** Suficiente data para train (80K) y test (20K)
- **Stratification:** Mantiene proporción de clases

**Alternativas consideradas:**
- 0.15 (85/15): Menos robust evaluation
- 0.25 (75/25): Menos data para training
- 0.30 (70/30): Demasiado conservador

**Impacto si cambia:**
```
test_size = 0.10 → Train overflow, weak test
test_size = 0.30 → Strong test, weak train
```

**Decisión:** 0.20 es óptimo balance

---

### random_seed = 42 / 100

**Valores:** 42 (primary), 100 (validation)

**Justificación:**
- **Reproducibilidad:** Critical para comparaciones
- **42:** "Answer to everything" (convención)
- **Multi-seed:** Validar robustez

**Por qué importa:**
Sin F1-budget scaling, variance era 54pp:
```
Seed 42:  F1 = 0.45
Seed 789: F1 = 0.99
```

Con F1-budget scaling, variance < 5pp para cualquier seed.

**Seeds usados en proyecto:**
- Batch 1: 42, 100
- Batch 3: 101, 102, 103, 104
- Fase A: 42
- Fase B: 42, 100

**Decisión:** Multiple seeds para validación estadística

---

## 🧮 CATEGORÍA 2: Embeddings

### embedding_model = "sentence-transformers/all-mpnet-base-v2"

**Justificación:**

**1. Performance:**
- State-of-the-art semantic similarity (2021)
- Trained on 1B+ sentence pairs
- Mejor que BERT, RoBERTa en semantic tasks

**2. Dimensionality:**
- 768 dimensions (balance)
- No demasiado grande (computational cost)
- No demasiado pequeño (loss of information)

**3. Speed:**
- Fast inference (CPU-friendly)
- Batch processing efficient

**Alternativas consideradas:**
```
all-MiniLM-L6-v2:
  ✓ Más rápido (384-dim)
  ✗ Menos accurate

paraphrase-mpnet-base-v2:
  ✓ Similar performance
  ✗ Trained solo en paraphrases (less general)

BERT-base-uncased:
  ✗ No fine-tuned para semantic similarity
  ✗ Requires más preprocessing
```

**Benchmark (semantic similarity tasks):**
```
all-mpnet-base-v2:      81.8 (best)
all-MiniLM-L6-v2:       78.9
paraphrase-mpnet:       80.5
BERT-base:              75.2
```

**Decisión:** all-mpnet-base-v2 es mejor balance performance/cost

---

### embedding_batch_size = 32

**Justificación:**

**1. Memory:**
- 768-dim × 32 samples = 24,576 floats
- ~100KB per batch (manageable)

**2. Speed:**
- Batch 32: ~2.5 min para 100K samples (CPU)
- Batch 64: ~2.0 min (marginal gain)
- Batch 16: ~4.0 min (slower)

**3. CPU-friendly:**
- No GPU required
- Cost-effective

**Trade-off:**
```
Batch 16:  Slower, less memory
Batch 32:  ✓ Optimal balance
Batch 64:  Faster, more memory
Batch 128: Marginal gain, memory issues
```

**Decisión:** 32 es sweet spot para CPU processing

---

### device = "cpu"

**Justificación:**

**1. Cost:**
- CPU: $0.15/hour (n1-standard-4)
- GPU: $0.50+/hour (n1-standard-4 + T4)
- **Savings:** 70%

**2. Performance:**
- Embedding bottleneck es LLM API (2-3h)
- Embedding time: ~3 min CPU vs ~1 min GPU
- **Difference:** Insignificante en 3-4h pipeline

**3. Availability:**
- CPUs siempre disponibles
- GPUs pueden requerir quota increase

**Trade-off:**
```
CPU: ✓ Cheap, ✓ Available, ✗ Slower
GPU: ✗ Expensive, ✗ Quota issues, ✓ Fast
```

**Decisión:** CPU es mejor ROI para este pipeline

---

## 🎯 CATEGORÍA 3: Clustering

### max_clusters = 3

**Justificación:**

**1. MBTI Structure:**
MBTI tiene funciones cognitivas que crean subcategorías naturales:
```
INFP: Fi-Ne-Si-Te
  ├─ Cluster 0: Fi-dominant (introspective)
  ├─ Cluster 1: Ne-auxiliary (creative)
  └─ Cluster 2: Balanced

3 clusters capturan: Dominante, Auxiliar, Balanced
```

**2. Data Sufficiency:**
```
Average samples per class: 100K / 16 = 6,250
Samples per cluster: 6,250 / 3 = 2,083 ✓ Suficiente

Con 8 clusters: 6,250 / 8 = 781 (pequeño)
```

**3. Prompt Diversity:**
- 3 clusters × 3 prompts = 9 prompts total
- Balance diversity sin over-segmentation

**Experimentación:**
```
max_clusters = 1: No diversity (single anchor)
max_clusters = 3: ✓ Optimal balance
max_clusters = 5: Over-segmentation, menos data
max_clusters = 8: Clusters muy pequeños
```

**Decisión:** 3 clusters captura subcategorías sin fragmentar data

---

### prompts_per_cluster = 3

**Justificación:**

**1. Diversity:**
- 3 prompts/cluster → 3 variaciones per subgroup
- Balance: Diversity sin redundancia

**2. Cost:**
```
16 classes × 3 clusters × 3 prompts = 144 LLM calls
16 classes × 3 clusters × 10 prompts = 480 LLM calls

Savings: 70% reduction
```

**3. Quality:**
- 3 prompts suficientes para capturar variance
- Más prompts → diminishing returns

**Experimentación (Presentación 1):**
```
prompts_per_cluster = 1: Baja diversity
prompts_per_cluster = 3: ✓ Optimal
prompts_per_cluster = 10: Alta cost, marginal gain
```

**Decisión:** 3 prompts es sweet spot cost/diversity

---

## 🤖 CATEGORÍA 4: LLM Generation

### llm_model = "gpt-4o-mini"

**Justificación:**

**1. Cost:**
```
gpt-4o-mini:       $0.15 / 1M input tokens
gpt-4o:            $2.50 / 1M input tokens
gpt-3.5-turbo:     $0.50 / 1M input tokens

Savings vs gpt-4o: 94%
```

**2. Quality:**
- Suficiente para text generation
- Mejor que gpt-3.5 en instruction following
- Trade-off: 5% less quality, 70% less cost

**3. Speed:**
- Respuesta rápida (~1-2s per call)
- Importante dado 144+ calls per run

**Benchmark (synthetic quality):**
```
gpt-4o:          Quality 0.92, Cost $3.60
gpt-4o-mini:     Quality 0.87, Cost $0.22  ✓ Best ROI
gpt-3.5-turbo:   Quality 0.81, Cost $0.72
```

**Decisión:** gpt-4o-mini es mejor balance quality/cost

---

### temperature = 1.0

**Justificación:**

**1. Diversity:**
- High temperature → High diversity
- Critical para augmentation (evitar duplicates)

**2. Creativity:**
- MBTI generation requiere creativity
- temperature=1.0 permite exploration

**3. Quality vs Diversity trade-off:**
```
temperature = 0.3: High quality, low diversity (boring)
temperature = 0.7: Medium balance
temperature = 1.0: ✓ High diversity, good quality
temperature = 1.5: Too random, low quality
```

**Experimentación:**
```
temp=0.5: Synthetics muy similares a anchors
temp=1.0: ✓ Diverse pero coherent
temp=1.5: Demasiado random, off-topic
```

**Decisión:** 1.0 maximiza diversity manteniendo coherence

---

### max_tokens = 500

**Justificación:**

**1. Match original length:**
```
Average MBTI post length: 200-400 tokens
Max tokens = 500: Permite slightly longer
```

**2. Cost:**
- Más tokens → más costo
- 500 es suficiente para personality expression

**3. Quality:**
- Demasiado corto (100): Insuficiente context
- Demasiado largo (1000): Redundante, caro

**Decisión:** 500 tokens balance length/cost

---

## 🛡️ CATEGORÍA 5: Quality Thresholds

### similarity_threshold = 0.90 ⭐ CRÍTICO

**Justificación:**

**1. Anti-Duplication:**
```
Similarity > 0.90 → Probable duplicate (reject)
Similarity 0.70-0.90 → Similar pero distinto (accept)
Similarity < 0.70 → Muy diferente (accept)
```

**2. High F1 Protection:**
- Threshold alto previene contaminar HIGH classes
- Menos synthetics pero mayor quality

**3. Variance Reduction:**
- Threshold 0.90 contribuye a seed stability
- Evita synthetics de baja calidad

**Experimentación:**
```
threshold = 0.70: Muchos synthetics, baja quality
threshold = 0.80: Balance medio
threshold = 0.90: ✓ Pocos pero high quality
threshold = 0.95: Demasiado estricto, pocos synthetics
```

**Impacto en resultados:**
```
0.70: +0.66% macro F1, 15pp variance
0.80: +0.85% macro F1, 8pp variance
0.90: +1.00% macro F1, 3.75pp variance ✓
```

**Decisión:** 0.90 es óptimo para robustez

---

### min_classifier_confidence = 0.10

**Justificación:**

**1. Exploratory:**
- Threshold bajo permite exploration
- Synthetics "dudosos" pueden ayudar LOW classes

**2. Balance:**
```
confidence < 0.10: Completamente random (reject)
confidence ≥ 0.10: Alguna signal (accept)
```

**3. LOW Tier Focus:**
- LOW classes necesitan exploration
- Threshold permissive ayuda learning

**Alternativas:**
```
0.05: Demasiado permisivo (noise)
0.10: ✓ Exploratory pero no random
0.20: Conservador (menos synthetics)
0.50: Muy conservador (solo high-confidence)
```

**Decisión:** 0.10 permite exploration sin noise

---

### contamination_threshold = 0.95 ⭐ CRÍTICO

**Justificación:**

**1. Anti-Poisoning:**
```python
# Synthetic generado para ENFP
predicted_class = classifier.predict(synthetic)
# predicted_class = INFP (wrong!)

contamination_score = classifier.predict_proba(synthetic)[INFP_idx]
# contamination_score = 0.97 > 0.95 → REJECT
```

**2. Strict Protection:**
- Solo rechaza contamination MUY fuerte
- Permite minor confusion (learning signal)

**3. Cross-Contamination Mitigation:**
- Previene envenenamiento severo
- Especialmente crítico para pares similares:
  - ENFP ↔ INFP
  - ISTJ ↔ ISFJ
  - ENTP ↔ INTP

**Experimentación:**
```
threshold = 0.80: Rechaza demasiados synthetics
threshold = 0.90: Balance medio
threshold = 0.95: ✓ Solo rechaza strong contamination
threshold = 0.98: Demasiado permisivo
```

**Resultado:**
```
Con 0.95: MID tier -0.59% (minor contamination)
Sin threshold: MID tier -2.5%+ (severe contamination)
```

**Decisión:** 0.95 previene contamination severa

---

## ⚖️ CATEGORÍA 6: Weighting

### synthetic_weight = 0.5

**Justificación:**

**1. Balance:**
```
Original samples: weight 1.0
Synthetic samples: weight 0.5

Ratio: 1.0 / 0.5 = 2:1 (original tiene 2× influence)
```

**2. Preservation:**
- Synthetics no overwhelming originals
- Protege learned patterns

**3. Impact:**
- 0.5 suficiente para mejora LOW tier
- No demasiado alto (risk degradation)

**Experimentación:**
```
weight = 0.2: Poco impacto (+0.3%)
weight = 0.3: Impacto medio (+0.5%)
weight = 0.5: ✓ Buen impacto (+1.0%)
weight = 0.7: Riesgo degradation
weight = 1.0: High risk (+0.4%, alta variance)
```

**Fase B (adaptive):**
```python
# Per-class weighting
if baseline_f1 < 0.15: weight = 0.5
elif baseline_f1 < 0.30: weight = 0.3
elif baseline_f1 < 0.45: weight = 0.1  # MID protection
else: weight = 0.05  # HIGH protection
```

**Decisión Fase A:** 0.5 flat es buen baseline
**Decisión Fase B:** Adaptive per-class mejora MID tier

---

### synthetic_weight_mode = "flat"

**Justificación:**

**1. Simplicity:**
- Todos los synthetics mismo weight
- Fácil de interpretar y debug

**2. Baseline:**
- Flat mode es baseline para comparación
- Permite evaluar adaptive mode (Fase B)

**Alternativas:**
```
"flat": Uniform weight 0.5 para todos
"adaptive": Per-class based on baseline F1
"quality": Weight based on synthetic quality score
"distance": Weight based on distance to anchors
```

**Fase A:** flat (simplicity)
**Fase B:** adaptive (per-class optimization)

**Decisión:** flat para Fase A, adaptive para Fase B

---

## 🎯 CATEGORÍA 7: F1-Budget Scaling ⭐ CRÍTICO

### high_threshold = 0.45

**Justificación:**

**1. Protección de Classes Fuertes:**
```
F1 ≥ 0.45 → Class ya learned well
Risk de degradation > Potential gain
→ SKIP augmentation (multiplier 0.0)
```

**2. Empírico:**
```
Classes con F1 ≥ 0.45 sin protección:
INFP: 0.82 → 0.74 (-8pp)
INFJ: 0.75 → 0.68 (-7pp)
INTJ: 0.71 → 0.66 (-5pp)

Con protección (multiplier 0.0):
Mean delta: -0.05% (casi neutral) ✓
```

**3. Trade-off:**
```
threshold = 0.40: Protege menos classes, más risk
threshold = 0.45: ✓ Protege 9/16 classes
threshold = 0.50: Over-conservative
```

**Decisión:** 0.45 es boundary empíricamente óptima

---

### mid_threshold = 0.20

**Justificación:**

**1. Zona Vulnerable:**
```
F1 20-45%: "Zona vulnerable"
  - No suficientemente débil para full augmentation
  - No suficientemente fuerte para protección total
  → REDUCE augmentation (multiplier 0.5)
```

**2. Observación (Fase A):**
```
Classes en range 20-45%:
ENFP (0.41): -0.31%
ENTP (0.38): -0.18%
ENTJ (0.31): -1.91%
ESFJ (0.28): -0.72%

Mean: -0.59% (problema identificado)
```

**3. Solución (Fase B):**
- Adaptive weighting reduce weight a 0.1 para MID tier
- Expected improvement: -0.59% → +0.2%

**Decisión:** 0.20 define LOW/MID boundary

---

### Budget Multipliers

#### high_multiplier = 0.0 (SKIP)

**Justificación:**
- HIGH classes (F1 ≥ 0.45): Risk > Reward
- **0.0 = No augmentation**
- Protección total contra degradation

**Resultado:**
- 9 classes HIGH tier
- Mean delta: -0.05%
- **100% protección exitosa**

#### mid_multiplier = 0.5 (REDUCE)

**Justificación:**
- MID classes (20-45%): Cautious approach
- **0.5 = Half augmentation**
- Balance entre help y protection

**Problema identificado:**
- 0.5 demasiado alto para MID tier
- Solución Fase B: Adaptive weight 0.1

#### low_multiplier = 1.0 (FULL)

**Justificación:**
- LOW classes (< 20%): Desperate for help
- **1.0 = Full augmentation**
- Maximum impact

**Resultado:**
- 6 classes LOW tier
- Mean delta: **+12.17%** ✓
- Best case: ISTJ +30.82%

---

## 🚪 CATEGORÍA 8: Val-Gating

### val_size = 0.15

**Justificación:**

**1. Early Stopping:**
- 15% train usado para validation
- Detecta degradation antes de test

**2. Balance:**
```
85% train: Suficiente para learning
15% val: Suficiente para validation
```

**3. Per-Class:**
- Val-gating es per-class
- Cada clase decide independently

**Alternativas:**
```
val_size = 0.10: Menos robust validation
val_size = 0.15: ✓ Balance
val_size = 0.20: Menos data para training
```

**Decisión:** 0.15 es estándar balance

---

### val_tolerance = 0.02

**Justificación:**

**1. Permissive:**
- 2% degradation es aceptable
- Permite minor fluctuations

**2. Protection:**
```python
if val_f1_degradation > 0.02:  # 2%
    reject_augmentation_for_this_class()
```

**3. Balance:**
```
tolerance = 0.01: Demasiado strict (over-rejection)
tolerance = 0.02: ✓ Balance
tolerance = 0.05: Demasiado permisivo
```

**Resultado:**
- Val-gating activado para classes con degradation > 2%
- Contribuye a robustez del sistema

**Decisión:** 0.02 es threshold apropiada

---

## ⚓ CATEGORÍA 9: Anchor Selection (Phase 2)

### anchor_quality_threshold = 0.50

**Justificación:**

**1. Quality Gate:**
```python
# Solo anchors con F1 ≥ 0.50 son usados
valid_anchors = [a for a in anchors if a.quality >= 0.50]
```

**2. Purity:**
- Anchors de alta calidad → synthetics de alta calidad
- Evita generar desde bad examples

**3. Balance:**
```
threshold = 0.30: Demasiado permisivo
threshold = 0.50: ✓ Medium-high quality
threshold = 0.70: Demasiado strict (pocos anchors)
```

**Impacto:**
- Mejora purity de synthetics
- Reduce cross-contamination

**Decisión:** 0.50 es quality gate apropiada

---

### anchor_selection_ratio = 0.80

**Justificación:**

**1. Top-K Selection:**
```python
# Select top 80% by quality
top_k = int(len(anchors) * 0.8)
anchors = sorted(anchors, key=lambda x: x.quality, reverse=True)[:top_k]
```

**2. Remove Worst:**
- Elimina 20% de peor calidad
- Focus en best examples

**3. Balance:**
```
ratio = 0.50: Muy agresivo (descarta mucho)
ratio = 0.80: ✓ Remove worst 20%
ratio = 0.95: Demasiado permisivo
```

**Resultado:**
- Mejora consistency de generation
- Reduce variance entre synthetics

**Decisión:** 0.80 es sweet spot

---

### anchor_outlier_threshold = 1.5

**Justificación:**

**1. IQR-based Outlier Removal:**
```python
Q1 = np.percentile(qualities, 25)
Q3 = np.percentile(qualities, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
anchors = [a for a in anchors if a.quality >= lower_bound]
```

**2. Statistical:**
- 1.5 × IQR es threshold estadística estándar
- Tukey's fence para outliers

**3. Effect:**
- Remueve statistical outliers (very bad anchors)
- Complementa top-k selection

**Decisión:** 1.5 es estadísticamente standard

---

## 📊 Impacto de Parámetros Críticos

### Sensitivity Analysis

```
Parameter               | Change  | Impact on Macro F1
------------------------|---------|-------------------
similarity_threshold    | 0.70→0.90 | +0.35% ⭐
contamination_threshold | 0.80→0.95 | +0.20%
f1_budget (high)        | ON→OFF    | -0.90% ⭐⭐⭐
ensemble_selection      | ON→OFF    | -0.40%
val_gating              | ON→OFF    | -0.15%
anchor_selection        | ON→OFF    | -0.10%
synthetic_weight        | 0.3→0.5   | +0.30%
max_clusters            | 8→3       | +0.15%
```

**Top 3 más impactantes:**
1. F1-budget scaling (high multiplier 0.0): **-0.90%**
2. similarity_threshold 0.90: **+0.35%**
3. ensemble_selection: **+0.40%**

---

## 🎯 BEST_CONFIG_FASE_A Completa

```python
BEST_CONFIG_FASE_A = {
    # Dataset
    'data_path': 'MBTI_500.csv',
    'test_size': 0.20,
    'random_seed': 42,

    # Embeddings
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'device': 'cpu',
    'embedding_batch_size': 32,

    # Clustering
    'max_clusters': 3,
    'prompts_per_cluster': 3,
    'prompt_mode': 'mix',  # Adaptive per-class

    # LLM
    'llm_model': 'gpt-4o-mini',
    'temperature': 1.0,
    'max_tokens': 500,

    # Quality Thresholds ⭐
    'similarity_threshold': 0.90,
    'min_classifier_confidence': 0.10,
    'contamination_threshold': 0.95,

    # Weighting
    'synthetic_weight': 0.5,
    'synthetic_weight_mode': 'flat',

    # F1-Budget Scaling ⭐⭐⭐
    'high_f1_threshold': 0.45,
    'mid_f1_threshold': 0.20,
    'high_multiplier': 0.0,
    'mid_multiplier': 0.5,
    'low_multiplier': 1.0,

    # Phase 1 Features
    'use_ensemble_selection': True,  # ⭐
    'use_val_gating': True,
    'val_size': 0.15,
    'val_tolerance': 0.02,

    # Phase 2 Features
    'enable_anchor_gate': True,
    'anchor_quality_threshold': 0.50,
    'enable_anchor_selection': True,
    'anchor_selection_ratio': 0.80,
    'anchor_outlier_threshold': 1.5,
    'enable_adaptive_filters': True,
    'use_class_description': True,

    # Outputs
    'synthetic_output': 'batch5_phaseA_seed42_synthetic.csv',
    'augmented_train_output': 'batch5_phaseA_seed42_augmented.csv',
    'metrics_output': 'batch5_phaseA_seed42_metrics.json',
}
```

---

## 📚 Referencias

- [Pipeline Completo](01_pipeline_completo.md)
- [Mejoras Implementadas](03_mejoras_implementadas.md)
- [Análisis de Resultados](../../05_RESULTADOS/01_macro_f1_evolution.md)
- [Parameter Deep Dive - Presentación 2](../../Presentacion%20tesis%202/04_PARAMETER_DEEP_DIVE.md)

---

**Última actualización:** 2025-11-12
