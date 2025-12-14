# Optimal Configuration for SMOTE-LLM

**Based on Phase F Validation Experiments**
**Date**: 2025-12-10

---

## Executive Summary

After 7 experiments with 41 configurations tested over 15-fold cross-validation, we identified the statistically optimal parameters for LLM-based data augmentation.

**Best achieved improvement**: **+3.17%** Macro F1 (p < 0.0001)

---

## Recommended Configuration

```python
OPTIMAL_PARAMS = {
    # Clustering
    "max_clusters": 12,              # K_max per class

    # Anchor Selection
    "anchor_strategy": "medoid",     # Most representative point

    # Prompt Context
    "k_neighbors": 15,               # Examples in LLM prompt

    # Quality Filtering
    "filter_method": "full_cascade", # All 4 filters
    "selection_method": "adaptive_ranking",  # Top-N by score

    # Thresholds
    "threshold_method": "purity_adaptive",

    # Training Parameters
    "synthetic_weight": 1.0,         # EQUAL to real samples
    "temperature": 0.3,              # LOW for coherence
    "budget": 0.12,                  # 12% of class size

    # LLM Settings
    "model": "gpt-4o-mini",
    "max_tokens": 150,
}
```

---

## Key Changes from Initial Assumptions

| Parameter | Initial Assumption | Optimal Value | Impact |
|-----------|-------------------|---------------|--------|
| **Weight** | 0.5 (down-weight synthetic) | **1.0** | +1.41% additional |
| **Temperature** | 0.7 (default) | **0.3** | +0.34% additional |
| **Budget** | 8% | **12%** | +0.69% additional |
| **Filter** | Simple (length only) | **Full cascade** | +1.76% (p=0.0009) |

---

## Statistical Evidence

### Statistically Significant Results (p < 0.05)

| Finding | Delta | p-value | Confidence |
|---------|-------|---------|------------|
| Medoid > Random anchor | +0.80% | 0.005 | High |
| Full cascade > Length only | +0.11% | 0.0009 | Very High |
| Purity adaptive > Fixed | +0.74% | 0.011 | High |
| Weight 1.0 > 0.5 | +1.41% | <0.0001 | Very High |
| Temp 0.3 > 0.7 | +0.34% | 0.027 | High |
| Budget 12% > 8% | +0.69% | 0.003 | High |

---

## Implementation Checklist

### 1. Clustering Setup
```python
from sklearn.cluster import KMeans

def cluster_class(embeddings, max_k=12):
    n_samples = len(embeddings)
    n_clusters = min(max_k, n_samples // 30)  # At least 30 per cluster
    n_clusters = max(1, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit(embeddings)
```

### 2. Medoid Selection
```python
def select_medoid(cluster_points, cluster_center):
    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
    medoid_idx = np.argmin(distances)
    return cluster_points[medoid_idx]
```

### 3. Quality Filtering (Full Cascade)
```python
def compute_quality(candidate, anchor, class_embeddings, class_centroid):
    # Length score
    length_score = 1 - (dist(candidate, anchor) / max_dist)

    # Similarity score
    similarity_score = cosine_similarity(candidate, anchor)

    # KNN purity
    knn_score = knn_purity(candidate, class_embeddings, k=10)

    # Confidence
    confidence_score = 1 - (dist(candidate, class_centroid) / max_dist)

    # Geometric mean
    return (length_score * similarity_score * knn_score * confidence_score) ** 0.25
```

### 4. Adaptive Ranking Selection
```python
def select_top_n(candidates, scores, target_count):
    top_indices = np.argsort(scores)[-target_count:]
    return candidates[top_indices]
```

### 5. Training with Optimal Weight
```python
from sklearn.linear_model import LogisticRegression

# Combine original and synthetic
X_train = np.vstack([X_original, X_synthetic])
y_train = np.concatenate([y_original, y_synthetic])

# Weight = 1.0 for synthetic (EQUAL to real)
weights = np.concatenate([
    np.ones(len(X_original)),
    np.ones(len(X_synthetic))  # NOT 0.5!
])

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train, sample_weight=weights)
```

### 6. LLM Generation
```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,  # LOW for coherence
    max_tokens=150,
)
```

---

## Expected Performance

### With Optimal Configuration

| Metric | Baseline | Augmented | Improvement |
|--------|----------|-----------|-------------|
| Macro F1 | 0.2045 | 0.2110 | **+3.17%** |
| LOW tier classes | 0.10 avg | 0.16 avg | +60% relative |
| Statistical significance | - | - | p < 0.0001 |

### Per-Tier Impact

| Tier | Baseline F1 | Expected Delta |
|------|-------------|----------------|
| LOW (< 0.20) | 0.10 | **+5.93%** |
| MID (0.20-0.45) | 0.30 | +0.06% |
| HIGH (≥ 0.45) | 0.50 | +0.11% |

**Note**: LLM augmentation primarily helps low-performing minority classes.

---

## Cost Estimation

For a dataset with 16 classes and 8,675 samples:

| Resource | Quantity | Cost |
|----------|----------|------|
| API calls | ~100 generations | ~$2 USD |
| Tokens | ~50K input, ~15K output | ~$1 USD |
| **Total** | | **~$3 USD** |

---

## Limitations

1. Results validated on MBTI dataset (text classification)
2. May vary with different embedding models
3. Requires OpenAI API access
4. Benefits diminish for high-performing classes

---

## Quick Start

```python
# 1. Load data
texts, labels = load_data("mbti_1.csv")

# 2. Compute embeddings (cached)
embeddings = embed_texts(texts, model="all-mpnet-base-v2")

# 3. Generate synthetic with optimal params
X_synth, y_synth = generate_synthetic(
    embeddings, labels, texts,
    max_clusters=12,
    anchor_strategy="medoid",
    filter_method="full_cascade",
    temperature=0.3,
    budget=0.12
)

# 4. Train with optimal weight
X_train = np.vstack([embeddings, X_synth])
y_train = np.concatenate([labels, y_synth])
weights = np.ones(len(X_train))  # Equal weight!

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train, sample_weight=weights)

# 5. Evaluate
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
```

---

## References

- Phase F Validation: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- Technical Details: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- Experiment Code: [experiments/](experiments/)
- LaTeX Tables: [latex_output/](latex_output/)

---

*Configuration validated with 15-fold cross-validation (5 splits × 3 repeats)*
