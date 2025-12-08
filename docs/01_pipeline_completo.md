# Pipeline Completo - Data Augmentation System

**Sistema:** LLM-based Synthetic Data Augmentation for Text Classification
**Versión:** Phase 2 (Fase A)
**Dataset:** MBTI 500 (100K samples, 16 classes)

---

## 📊 Visión General del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA PREPARATION                │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  1. Load & Split Dataset             │
        │     - MBTI_500.csv (100K samples)    │
        │     - Train/Test split (80/20)       │
        │     - Random seed for reproducibility│
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  2. Train Baseline Classifier        │
        │     - Logistic Regression            │
        │     - On original embeddings         │
        │     - Compute baseline F1 per class  │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  3. Compute Embeddings               │
        │     - sentence-transformers          │
        │     - all-mpnet-base-v2 (768-dim)    │
        │     - Batch processing (32 samples)  │
        └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 2: ANCHOR SELECTION                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  4. Cluster Per-Class                │
        │     - K-means (max_clusters=3)       │
        │     - Per class independently        │
        │     - Find prototypical examples     │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  5. Select Anchors (Phase 2)         │
        │     - Quality gate: F1 ≥ 0.50        │
        │     - Selection: Top 80% by quality  │
        │     - Outlier removal: IQR × 1.5     │
        │     - Result: High-quality centroids │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  6. Apply F1-Budget Scaling          │
        │     - HIGH (F1 ≥ 45%): multiplier 0.0│
        │     - MID (20-45%): multiplier 0.5   │
        │     - LOW (<20%): multiplier 1.0     │
        │     - Protect strong classes         │
        └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                PHASE 3: SYNTHETIC GENERATION                │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  7. Generate Prompts                 │
        │     - Per cluster (3 prompts/cluster)│
        │     - Adaptive mode: mix/paraphrase  │
        │     - Include class descriptions     │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  8. LLM Generation                   │
        │     - gpt-4o-mini via OpenAI API     │
        │     - Based on anchor examples       │
        │     - Temperature = 1.0 (diversity)  │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  9. Quality Filtering (Multi-layer)  │
        │     Layer 1: Similarity ≤ 0.90       │
        │     Layer 2: Classifier conf ≥ 0.10  │
        │     Layer 3: Contamination ≤ 0.95    │
        │     Layer 4: Adaptive filters        │
        │     Result: High-purity synthetics   │
        └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 4: AUGMENTATION                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  10. Validation Split (Phase 1)      │
        │     - Split train: 85% / 15%         │
        │     - Val-gating tolerance: 2%       │
        │     - Early stopping per-class       │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  11. Weight Synthetics               │
        │     - Flat mode: uniform 0.5         │
        │     - Balance original vs synthetic  │
        │     - Per-class budget respected     │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  12. Train Augmented Classifier      │
        │     - Same architecture (LogReg)     │
        │     - On original + weighted synth   │
        │     - Compute augmented F1 per class │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  13. Ensemble Selection (Phase 1)    │
        │     - Per-class: max(F1_base, F1_aug)│
        │     - Mathematical no-degradation    │
        │     - Select best model per class    │
        └──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 5: EVALUATION                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  14. Compute Final Metrics           │
        │     - Per-class F1 scores            │
        │     - Macro F1 (mean across classes) │
        │     - Deltas vs baseline             │
        │     - Tier analysis (LOW/MID/HIGH)   │
        └──────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  15. Save Outputs                    │
        │     - Synthetic samples CSV          │
        │     - Augmented train set CSV        │
        │     - Metrics JSON                   │
        │     - Logs                           │
        └──────────────────────────────────────┘
                              ↓
                         [COMPLETE]
```

---

## 🔧 Componentes Detallados

### 1. Data Preparation

#### 1.1 Dataset Loading
```python
df = pd.read_csv('MBTI_500.csv')
# 100,000 samples
# 16 classes (MBTI personality types)
# Columns: 'type' (label), 'posts' (text)
```

**Características:**
- Balanceo: Desbalanceado natural (INFP más común)
- Texto: Posts concatenados por usuario
- Longitud: Variable (100-5000 caracteres)

#### 1.2 Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=seed,
    stratify=y  # Mantiene proporción de clases
)
```

**Justificación:**
- 80/20: Estándar para datasets medianos
- Stratify: Mantiene distribución de clases
- Random seed: Reproducibilidad

#### 1.3 Baseline Training
```python
baseline_classifier = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
baseline_classifier.fit(X_train_embeddings, y_train)
```

**Output:**
- F1 per class (16 scores)
- Identifica LOW/MID/HIGH tiers
- Base para F1-budget scaling

---

### 2. Embedding Generation

#### 2.1 Model Selection
```python
embedding_model = SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
```

**Razones de selección:**
- 768-dim embeddings (balance size/quality)
- State-of-the-art para semantic similarity
- Pre-trained en large corpus
- Rápido (batch 32 en CPU)

#### 2.2 Batch Processing
```python
embeddings = embedding_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    device='cpu'
)
```

**Optimizaciones:**
- Batch size 32: Balance speed/memory
- CPU mode: Más cost-effective que GPU para este dataset
- Progress bar: Tracking de proceso largo

---

### 3. Clustering & Anchor Selection

#### 3.1 Per-Class Clustering
```python
for class_label in classes:
    class_embeddings = embeddings[y_train == class_label]

    # K-means clustering
    optimal_k = min(max_clusters, len(class_embeddings) // 10)
    kmeans = KMeans(n_clusters=optimal_k, random_state=seed)
    clusters = kmeans.fit_predict(class_embeddings)
```

**Justificación:**
- Per-class: Cada clase tiene sus propios clusters
- max_clusters=3: Captura subcategorías sin over-segmentation
- min samples per cluster: 10 (evita clusters tiny)

#### 3.2 Anchor Quality Gate (Phase 2)
```python
if enable_anchor_gate:
    quality_threshold = 0.50
    # Solo anchors con F1 ≥ 0.50 son usados
    valid_anchors = [a for a in anchors if a.quality >= quality_threshold]
```

**Efecto:**
- Filtra anchors de baja calidad
- Mejora purity de synthetics generados
- Reduce contamination

#### 3.3 Anchor Selection (Phase 2)
```python
if enable_anchor_selection:
    # Top 80% por calidad
    top_k = int(len(anchors) * 0.8)
    anchors = sorted(anchors, key=lambda x: x.quality, reverse=True)[:top_k]

    # Outlier removal (IQR)
    Q1 = np.percentile(qualities, 25)
    Q3 = np.percentile(qualities, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    anchors = [a for a in anchors if a.quality >= lower_bound]
```

**Efecto:**
- Remueve 20% de peor calidad
- Remueve outliers estadísticos
- Mejora consistency de generación

---

### 4. F1-Budget Scaling

#### 4.1 Tier Classification
```python
def classify_tier(baseline_f1):
    if baseline_f1 >= 0.45:
        return 'HIGH'
    elif baseline_f1 >= 0.20:
        return 'MID'
    else:
        return 'LOW'
```

#### 4.2 Budget Multipliers
```python
def get_budget_multiplier(tier):
    multipliers = {
        'HIGH': 0.0,   # SKIP augmentation
        'MID': 0.5,    # REDUCE augmentation
        'LOW': 1.0     # FULL augmentation
    }
    return multipliers[tier]
```

**Justificación:**
- HIGH: Risk > Reward → Skip completamente
- MID: Cautious → Reduce a mitad
- LOW: Desperate for help → Full power

#### 4.3 Application
```python
for class_label in classes:
    baseline_f1 = baseline_f1_per_class[class_label]
    tier = classify_tier(baseline_f1)
    multiplier = get_budget_multiplier(tier)

    n_synthetics_target = base_synthetics_per_class * multiplier
```

**Efecto:**
- Protección automática de HIGH tier
- Reducción de variance (54pp → 3.75pp)
- Focus recursos en LOW tier

---

### 5. Synthetic Generation

#### 5.1 Adaptive Prompt Mode
```python
def get_prompt_mode(baseline_f1):
    if baseline_f1 >= 0.45:
        return 'paraphrase'  # Quality preservation
    else:
        return 'mix'  # Maximum diversity
```

**Justificación:**
- HIGH F1: Paraphrase mantiene quality
- LOW F1: Mix genera diversity para aprender

#### 5.2 Prompt Templates
```python
# Mix mode (diversity)
prompt_mix = f"""
Generate a new text sample for personality type {class_label}.

Context: {class_description}

Reference examples:
{anchor_examples}

Generate a NEW, DIVERSE sample that represents this personality.
"""

# Paraphrase mode (quality)
prompt_paraphrase = f"""
Paraphrase the following text while maintaining the personality traits:

Original: {anchor_text}

Requirements:
- Preserve key personality indicators
- Use different words and structure
- Maintain semantic meaning
"""
```

#### 5.3 LLM API Call
```python
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a personality analysis expert."},
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,  # High diversity
    max_tokens=500
)
synthetic_text = response['choices'][0]['message']['content']
```

**Parámetros:**
- gpt-4o-mini: Balance costo/calidad
- temperature=1.0: Maximizar diversity
- max_tokens=500: Similar length to originals

---

### 6. Quality Filtering (Multi-Layer)

#### Layer 1: Semantic Similarity
```python
similarity = cosine_similarity(synthetic_emb, anchor_emb)
if similarity > 0.90:
    reject()  # Too similar to original
```

**Threshold: 0.90 (strict)**
- Previene duplicates
- Fuerza diversity

#### Layer 2: Classifier Confidence
```python
confidence = classifier.predict_proba(synthetic_emb).max()
if confidence < 0.10:
    reject()  # Too ambiguous
```

**Threshold: 0.10 (permissive)**
- Permite exploration
- Evita completamente random samples

#### Layer 3: Contamination Check
```python
predicted_class = classifier.predict(synthetic_emb)
predicted_proba = classifier.predict_proba(synthetic_emb)

# Check if misclassified
if predicted_class != target_class:
    # Check contamination strength
    contamination_score = predicted_proba[predicted_class_idx]
    if contamination_score > 0.95:
        reject()  # Strong contamination
```

**Threshold: 0.95 (very strict)**
- Solo rechaza contamination fuerte
- Previene envenenamiento severo

#### Layer 4: Adaptive Filters (Phase 2)
```python
if enable_adaptive_filters:
    # Dynamic thresholding based on class performance
    if class_performance < target:
        relax_thresholds()  # More permissive
    else:
        tighten_thresholds()  # More strict
```

**Efecto:**
- Self-adjusting quality gates
- Balance quantity/quality per class

---

### 7. Augmentation & Training

#### 7.1 Validation Split (Val-Gating)
```python
if use_val_gating:
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,  # 15% for validation
        stratify=y_train
    )
```

**Purpose:**
- Early stopping per-class
- Detect degradation before test
- Tolerance: 2% degradation acceptable

#### 7.2 Synthetic Weighting
```python
# Flat weighting mode
synthetic_weight = 0.5

# Create weighted dataset
X_augmented = np.vstack([
    X_train,  # Original samples (weight 1.0)
    X_synthetic  # Synthetic samples (weight 0.5)
])

# Weights array
weights = np.concatenate([
    np.ones(len(X_train)),
    np.full(len(X_synthetic), synthetic_weight)
])
```

**Justificación:**
- 0.5: Balance preservation/augmentation
- Flat mode: Todos los synthetics igual weight
- Alternative: Adaptive per-class (Fase B)

#### 7.3 Augmented Training
```python
augmented_classifier = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
augmented_classifier.fit(
    X_augmented,
    y_augmented,
    sample_weight=weights
)
```

---

### 8. Ensemble Selection (Phase 1)

```python
for class_label in classes:
    f1_baseline = baseline_f1_per_class[class_label]
    f1_augmented = augmented_f1_per_class[class_label]

    if f1_augmented > f1_baseline:
        selected_model[class_label] = 'augmented'
    else:
        selected_model[class_label] = 'baseline'

    final_f1[class_label] = max(f1_baseline, f1_augmented)
```

**Garantía matemática:**
```
F1_final ≥ F1_baseline (always)
```

**Efecto:**
- Safety net contra degradation
- Per-class selection optimiza macro F1
- 100% protección high-F1 classes

---

## 📊 Métricas y Outputs

### Final Metrics JSON
```json
{
  "macro_f1_baseline": 0.XXXX,
  "macro_f1_augmented": 0.XXXX,
  "macro_f1_final": 0.XXXX,
  "improvement": "+X.XX%",

  "per_class": {
    "INFP": {
      "baseline_f1": 0.XX,
      "augmented_f1": 0.XX,
      "final_f1": 0.XX,
      "delta": "+X.XX%",
      "selected_model": "augmented|baseline",
      "tier": "HIGH|MID|LOW",
      "synthetics_generated": N
    },
    ...
  },

  "tier_analysis": {
    "LOW": {"mean_delta": "+XX.XX%", "n_classes": 6},
    "MID": {"mean_delta": "-X.XX%", "n_classes": 4},
    "HIGH": {"mean_delta": "-X.XX%", "n_classes": 6}
  },

  "seed": 42,
  "timestamp": "2025-11-12T..."
}
```

### Synthetic Output CSV
```
type,posts,source,quality_score,cluster_id
INFP,"synthetic text...",generated,0.87,0
ENFP,"synthetic text...",generated,0.92,1
...
```

### Augmented Train CSV
```
type,posts,source,weight
INFP,"original text...",original,1.0
INFP,"synthetic text...",synthetic,0.5
...
```

---

## 🔄 Pipeline Execution Flow

### Tiempo de Ejecución (Típico)

```
Phase 1 - Data Prep:        ~5 min
  ├─ Load & split:          30s
  ├─ Baseline training:     1 min
  └─ Embedding generation:  3 min

Phase 2 - Anchor Selection: ~10 min
  ├─ Clustering:            2 min
  ├─ Quality gate:          3 min
  └─ F1-budget scaling:     5 min

Phase 3 - Generation:       ~2-3 hours ← Bottleneck
  ├─ Prompt creation:       5 min
  ├─ LLM API calls:         2h (rate limited)
  └─ Quality filtering:     10 min

Phase 4 - Augmentation:     ~5 min
  ├─ Val-gating:            2 min
  ├─ Weighting:             1 min
  └─ Training augmented:    2 min

Phase 5 - Evaluation:       ~2 min
  ├─ Compute metrics:       1 min
  └─ Save outputs:          1 min

Total: ~3-4 hours
```

**Bottleneck:** LLM API calls (rate limits, latency)

---

## 🎯 Diferencias entre Fases

### Presentación 1
- 4 clases only
- No F1-budget scaling
- No ensemble selection
- No quality gates
- **Result:** +2.8% (4 classes)

### Batch 1-4
- 16 clases
- F1-budget scaling introduced
- Validation gating
- Basic quality filters
- **Result:** +0.66% (Batch 4)

### Fase A (Current)
- 16 clases
- F1-budget scaling
- **Ensemble selection** ← NEW
- **Adaptive prompt-mode** ← NEW
- **Phase 2 features** ← NEW
  - Anchor gate
  - Anchor selection
  - Adaptive filters
  - Class descriptions
- **Result:** +1.00%

### Fase B (In Progress)
- All Fase A features
- **Adaptive weighting** ← NEW
- Per-class synthetic weights
- **Target:** +1.20% ~ +1.40%

---

## 📚 Referencias

- [Parámetros Justificados](02_parametros_justificados.md)
- [Mejoras Implementadas](03_mejoras_implementadas.md)
- [Estrategias de Batches](04_estrategias_batches.md)
- [runner_phase2.py](../../Laptop%20Runs/runner_phase2.py)

---

**Última actualización:** 2025-11-12
