# Phase F Validation - Technical Documentation

**Version**: 1.0
**Date**: 2025-12-10
**Author**: Benjamin (Thesis Project)

---

## 1. Overview

Phase F Validation is a comprehensive experimental framework designed to validate hyperparameters and design decisions for LLM-based data augmentation in text classification. The framework uses rigorous K-fold cross-validation to ensure statistical robustness.

### 1.1 Objectives

1. Validate clustering parameters (K_max)
2. Compare anchor selection strategies
3. Optimize prompt context size (K neighbors)
4. Evaluate filter cascade configurations
5. Test adaptive threshold mechanisms
6. Analyze tier-based impact
7. Optimize training parameters (weight, temperature, budget)

### 1.2 Dataset

- **Source**: `mbti_1.csv`
- **Samples**: 8,675
- **Classes**: 16 (MBTI personality types)
- **Features**: Social media posts (text)
- **Baseline F1**: 0.2045 (Logistic Regression on sentence embeddings)

---

## 2. Methodology

### 2.1 Evaluation Protocol

**K-Fold Cross-Validation**:
- Splits: 5
- Repeats: 3
- Total folds: 15
- Stratified sampling to preserve class distribution

**Statistical Testing**:
- Paired t-test comparing baseline vs augmented F1 scores
- Significance level: α = 0.05
- Effect reported as percentage delta from baseline

### 2.2 Embedding Model

- **Model**: `all-mpnet-base-v2` (sentence-transformers)
- **Dimension**: 768
- **Caching**: Embeddings cached to `cache/embeddings_mpnet.npy`

### 2.3 LLM Configuration

- **Model**: `gpt-4o-mini`
- **Default Temperature**: 0.7
- **Max Tokens**: 150 per generation
- **API**: OpenAI Chat Completions

### 2.4 Classifier

- **Algorithm**: Logistic Regression
- **Solver**: lbfgs
- **Max Iterations**: 2000
- **Regularization**: Default (C=1.0)

---

## 3. Experiments

### 3.1 Experiment 1: Clustering Validation

**File**: `experiments/exp01_clustering.py`

**Objective**: Find optimal maximum clusters per class.

**Parameter Space**:
```python
K_MAX_VALUES = [1, 2, 3, 6, 12, 24]
```

**Results**:

| K_max | Silhouette | Coherence | Macro F1 | Delta |
|-------|------------|-----------|----------|-------|
| 1 | N/A | 67% | 0.2047 | +0.08% |
| 2 | 0.06 | 64% | 0.2067 | +1.06% |
| 3 | 0.05 | 64% | 0.2069 | +1.18% |
| 6 | 0.04 | 64% | 0.2068 | +1.09% |
| **12** | **0.04** | **64%** | **0.2083** | **+1.85%** |
| 24 | 0.04 | 64% | 0.2059 | +0.69% |

**Conclusion**: K_max=12 is optimal. Too many clusters fragment the data.

---

### 3.2 Experiment 2: Anchor Strategies

**File**: `experiments/exp02_anchor_strategies.py`

**Objective**: Compare anchor selection methods.

**Strategies Tested**:
1. **Random**: Random sample from cluster
2. **Nearest Neighbor**: Closest to centroid
3. **Medoid**: Most representative point
4. **Quality-gated**: High embedding quality only
5. **Diverse**: Maximize coverage
6. **Ensemble**: Weighted combination

**Results**:

| Strategy | Macro F1 | Delta | p-value | Sig? |
|----------|----------|-------|---------|------|
| Random | 0.2062 | +0.83% | 0.088 | No |
| Nearest Neighbor | 0.2067 | +1.07% | - | No |
| **Medoid** | **0.2079** | **+1.63%** | **0.005** | **Yes** |
| Quality-gated | 0.2044 | -0.05% | - | No |
| Diverse | 0.2065 | +0.98% | - | No |
| Ensemble | 0.2048 | +0.15% | 0.682 | No |

**Conclusion**: Medoid strategy significantly outperforms others (p=0.005).

---

### 3.3 Experiment 3: K Neighbors

**File**: `experiments/exp03_k_neighbors.py`

**Objective**: Determine optimal prompt context size.

**Parameter Space**:
```python
K_VALUES = [5, 10, 15, 25, 50, 75, 100]
```

**Results**:

| K | Macro F1 | Acceptance | Context Quality | Delta |
|---|----------|------------|-----------------|-------|
| 5 | 0.2054 | 99% | Insufficient | +0.41% |
| **10** | **0.2058** | **99%** | **Limited** | **+0.60%** |
| 15 | 0.2052 | 98% | Optimal | +0.31% |
| **25** | **0.2059** | **98%** | **Redundant** | **+0.66%** |
| 50 | 0.2050 | 98% | Noisy | +0.22% |
| 75 | 0.2051 | 98% | Noisy | +0.27% |
| 100 | 0.2053 | 97% | Excessive | +0.41% |

**Conclusion**: K=10-25 offers best trade-off. More context doesn't help.

---

### 3.4 Experiment 4: Filter Cascade

**File**: `experiments/exp04_filter_cascade.py` (v1)
**File**: `experiments/exp04_filter_cascade_v3.py` (v3 - final)

**Objective**: Evaluate progressive filtering impact.

**Filter Components**:
1. **Length**: Distance to anchor embedding
2. **Similarity**: Cosine similarity to anchor
3. **KNN**: K-nearest neighbor purity
4. **Confidence**: Distance to class centroid

#### v1 Results (Fixed Threshold):

**Problem**: Strict filters rejected all samples.

| Config | Acceptance | Delta |
|--------|------------|-------|
| length_only | 87% | +3.09%* |
| length_similarity | 1% | +0.60% |
| three_partial | 0% | +0.00% |
| full_cascade | 0% | +0.00% |

#### v3 Results (Adaptive Ranking):

**Solution**: Rank by composite quality score, select top-N.

| Config | N Filters | N Synth | Quality | Delta | p-value |
|--------|-----------|---------|---------|-------|---------|
| length_only | 1 | 127 | 0.154 | +1.65%* | 0.012 |
| length_similarity | 2 | 127 | 0.235 | +1.40%* | 0.040 |
| three_filters | 3 | 129 | 0.306 | +1.38%* | 0.002 |
| **full_cascade** | **4** | **125** | **0.231** | **+1.76%*** | **0.0009** |

**Conclusion**: With controlled sample count, full cascade performs best.

---

### 3.5 Experiment 5: Adaptive Thresholds

**File**: `experiments/exp05_adaptive_thresholds.py` (v1)
**File**: `experiments/exp05_adaptive_thresholds_v2.py` (v2 - final)

**Objective**: Compare fixed vs adaptive quality thresholds.

#### v1 Results (Fixed):

| Config | Threshold | N Synth | Delta |
|--------|-----------|---------|-------|
| fixed_permissive | 0.60 | 14 | +0.25% |
| fixed_medium | 0.70 | 1 | +0.31% |
| fixed_strict | 0.90 | 0 | +0.00% |

#### v2 Results (Adaptive Relaxation):

| Config | Avg Threshold | N Synth | Delta | p-value |
|--------|---------------|---------|-------|---------|
| strict_relaxing | 0.38 | 96 | +1.13% | 0.056 |
| medium_relaxing | 0.39 | 107 | +0.67% | 0.102 |
| permissive_relaxing | 0.38 | 100 | +0.83%* | 0.007 |
| **purity_adaptive** | **0.39** | **97** | **+0.99%*** | **0.011** |

**Conclusion**: Purity-based adaptation achieves best balance.

---

### 3.6 Experiment 6: Tier Impact

**File**: `experiments/exp06_tier_impact.py`

**Objective**: Measure augmentation impact by baseline performance.

**Tier Definitions**:
- **LOW**: F1 < 0.20 (9 classes)
- **MID**: 0.20 ≤ F1 < 0.45 (6 classes)
- **HIGH**: F1 ≥ 0.45 (1 class)

**Results**:

| Tier | Classes | Delta F1 Avg | Std |
|------|---------|--------------|-----|
| **LOW** | 9 | **+5.93%** | 13.5% |
| MID | 6 | +0.06% | 0.6% |
| HIGH | 1 | +0.11% | 0.0% |

**Per-Class Breakdown (LOW tier)**:
- ENTJ: +27.3% (baseline: 0.054)
- ISTJ: +32.0% (baseline: 0.055)
- ISFJ: +6.2% (baseline: 0.160)
- ISFP: -12.1% (baseline: 0.184)

**Conclusion**: LLM augmentation primarily benefits low-performing classes.

---

### 3.7 Experiment 7: Comprehensive Validation

**File**: `experiments/exp07_comprehensive.py`

**Objective**: Optimize weight, temperature, and budget with best ensemble.

#### 7a: Synthetic Weight

| Weight | N Synth | Macro F1 | Delta | p-value |
|--------|---------|----------|-------|---------|
| 0.3 | 125 | 0.2103 | +2.83%* | <0.0001 |
| 0.5 | 125 | 0.2081 | +1.76%* | 0.0009 |
| 0.7 | 125 | 0.2100 | +2.68%* | <0.0001 |
| **1.0** | **125** | **0.2110** | **+3.17%*** | **<0.0001** |

**Finding**: Equal weight (1.0) outperforms down-weighting.

#### 7b: LLM Temperature

| Temp | N Synth | Quality | Macro F1 | Delta |
|------|---------|---------|----------|-------|
| **0.3** | **125** | **0.245** | **0.2063** | **+0.88%*** |
| 0.5 | 125 | 0.238 | 0.2057 | +0.60% |
| 0.7 | 125 | 0.231 | 0.2056 | +0.54% |
| 0.9 | 125 | 0.219 | 0.2051 | +0.29% |

**Finding**: Lower temperature produces better samples.

#### 7c: Budget Multiplier

| Budget | N Synth | Quality | Macro F1 | Delta |
|--------|---------|---------|----------|-------|
| 5% | 63 | 0.242 | 0.2055 | +0.49% |
| 8% | 100 | 0.236 | 0.2061 | +0.78% |
| **12%** | **150** | **0.228** | **0.2075** | **+1.47%*** |
| 15% | 188 | 0.221 | 0.2069 | +1.17%* |

**Finding**: 12% budget is optimal.

---

## 4. Key Findings

### 4.1 Statistically Significant Results

| Parameter | Best Value | Delta | p-value |
|-----------|------------|-------|---------|
| Anchor Strategy | Medoid | +1.63% | 0.005 |
| Filter (v3) | full_cascade | +1.76% | 0.0009 |
| Threshold (v2) | purity_adaptive | +0.99% | 0.011 |
| Weight | 1.0 | +3.17% | <0.0001 |
| Temperature | 0.3 | +0.88% | 0.027 |
| Budget | 12% | +1.47% | 0.003 |

### 4.2 Surprising Findings

1. **Weight w=1.0 is best**: Contradicts assumption that synthetic samples need down-weighting.

2. **Lower temperature helps**: τ=0.3 produces more useful samples than default 0.7.

3. **Full cascade wins with adaptive ranking**: When sample count is controlled, more filters = better results.

### 4.3 Non-Significant Parameters

- K_max (clustering): All values similar, K_max=12 slightly best
- K neighbors: K=10-25 range all acceptable
- Tier impact: Not statistically testable (different class groups)

---

## 5. Optimal Configuration

Based on all experiments, the recommended configuration is:

```python
OPTIMAL_CONFIG = {
    # Clustering
    "max_clusters": 12,

    # Anchor Selection
    "anchor_strategy": "medoid",

    # Prompt Context
    "k_neighbors": 15,

    # Filtering
    "filter_cascade": "full_cascade",  # length + similarity + knn + confidence
    "filter_method": "adaptive_ranking",  # rank by quality, select top-N

    # Thresholds
    "threshold_method": "purity_adaptive",

    # Training
    "synthetic_weight": 1.0,  # Equal to real samples
    "temperature": 0.3,  # Lower = more coherent
    "budget_multiplier": 0.12,  # 12% of class size

    # LLM
    "model": "gpt-4o-mini",
    "max_tokens": 150,
}
```

**Expected Improvement**: +3.17% Macro F1 (statistically significant)

---

## 6. File Structure

```
phase_f_validation/
├── base_config.py              # Base parameters
├── validation_runner.py        # K-fold evaluation engine
├── RESULTS_SUMMARY.md          # Human-readable results
├── TECHNICAL_DOCUMENTATION.md  # This file
│
├── experiments/
│   ├── exp01_clustering.py
│   ├── exp02_anchor_strategies.py
│   ├── exp03_k_neighbors.py
│   ├── exp04_filter_cascade.py
│   ├── exp04_filter_cascade_v3.py  # Final version
│   ├── exp05_adaptive_thresholds.py
│   ├── exp05_adaptive_thresholds_v2.py  # Final version
│   ├── exp06_tier_impact.py
│   └── exp07_comprehensive.py
│
├── results/                    # JSON results per experiment
│   ├── clustering/
│   ├── anchor_strategies/
│   ├── k_neighbors/
│   ├── filter_cascade/
│   ├── filter_cascade_v3/
│   ├── adaptive_thresholds/
│   ├── adaptive_thresholds_v2/
│   ├── tier_impact/
│   └── comprehensive/
│
├── latex_output/               # LaTeX tables for thesis
│   ├── tab_clustering_validation.tex
│   ├── tab_anchor_strategies.tex
│   ├── tab_k_neighbors.tex
│   ├── tab_filter_cascade.tex
│   ├── tab_filter_cascade_v3.tex
│   ├── tab_adaptive_thresholds_v2.tex
│   ├── tab_tier_impact.tex
│   ├── tab_weight_validation.tex
│   ├── tab_temperature_validation.tex
│   └── tab_budget_validation.tex
│
└── cache/
    ├── embeddings_mpnet.npy    # Cached embeddings (8675 x 768)
    └── labels.npy              # Cached labels
```

---

## 7. Execution Statistics

| Metric | Value |
|--------|-------|
| Total Runtime | ~11 hours |
| API Calls | ~900 |
| Synthetic Samples | ~4,000 |
| Total Experiments | 7 |
| Total Configurations | 41 |
| Significant Results | 6 |

---

## 8. Reproducibility

### 8.1 Environment

```bash
# Python version
python --version  # 3.10+

# Key dependencies
pip install numpy scikit-learn sentence-transformers openai scipy
```

### 8.2 Running Experiments

```bash
cd phase_f_validation

# Set API key
export OPENAI_API_KEY='your-key'

# Run individual experiment
python3 -u experiments/exp01_clustering.py

# Run all experiments
./run_all.sh
```

### 8.3 Random Seeds

- K-fold: `random_state=42`
- KMeans: `random_state=42`
- Train/test split: Handled by RepeatedStratifiedKFold

---

## 9. Limitations

1. **Single dataset**: Results may not generalize to other text classification tasks.

2. **Fixed LLM**: Only tested gpt-4o-mini; other models may behave differently.

3. **Embedding model**: Results depend on sentence-transformers embedding quality.

4. **Cost**: Full experiment suite costs ~$25 USD in API calls.

---

## 10. Future Work

1. Test on larger datasets (MBTI_500.csv with 106K samples)
2. Compare different LLM models (GPT-4, Claude, Llama)
3. Explore dynamic weight scheduling
4. Implement confidence-based sample selection
5. Test on other classification tasks (sentiment, topic)

---

## Appendix A: Quality Score Computation

The adaptive ranking uses geometric mean of filter scores:

```python
def compute_quality_scores(candidates, anchor, embeddings, labels, target_class, filters):
    scores = {}

    # Length: Distance to anchor (closer = better)
    if "length" in filters:
        dists = norm(candidates - anchor, axis=1)
        scores["length"] = 1 - (dists / max(dists))

    # Similarity: Cosine to anchor
    if "similarity" in filters:
        scores["similarity"] = 1 - cdist(candidates, anchor, 'cosine')

    # KNN: Purity of nearest neighbors
    if "knn" in filters:
        scores["knn"] = compute_knn_purity(candidates, embeddings, labels, target_class)

    # Confidence: Distance to class centroid
    if "confidence" in filters:
        centroid = embeddings[labels == target_class].mean(axis=0)
        dists = norm(candidates - centroid, axis=1)
        scores["confidence"] = 1 - (dists / max(dists))

    # Geometric mean
    combined = np.prod([scores[f] for f in filters], axis=0)
    return np.power(combined, 1.0 / len(filters))
```

---

## Appendix B: Adaptive Threshold Relaxation

```python
def adaptive_relaxation(candidates, scores, target_count, initial_threshold=0.9):
    threshold = initial_threshold

    while threshold > 0.1:
        selected = candidates[scores >= threshold]
        if len(selected) >= target_count:
            return selected[:target_count], threshold
        threshold -= 0.05

    # Fallback: return top-N by score
    top_idx = np.argsort(scores)[-target_count:]
    return candidates[top_idx], threshold
```

---

*End of Technical Documentation*
