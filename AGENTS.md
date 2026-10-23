# Geometric Filters for LLM-Based Data Augmentation

## Project Overview

This research project investigates whether **geometric filtering** of LLM-generated synthetic text samples can improve classification performance in **low-resource scenarios** (few training examples per class).

### Core Hypothesis

LLMs can generate diverse synthetic training data, but not all generated samples are equally useful. By applying geometric filters in the embedding space, we can:
1. **Select high-quality samples** that are geometrically consistent with the real data distribution
2. **Reject outliers** that might confuse the classifier
3. **Outperform traditional augmentation** methods like SMOTE

### Key Research Questions

1. **Does geometric filtering improve LLM augmentation?** → Yes, but simple filters work best
2. **Which filter is optimal?** → Cascade level=1 (distance-only) with +1.45pp vs SMOTE
3. **When does LLM augmentation outperform SMOTE?** → In multi-class low-resource settings (≤10 samples/class)
4. **When does filtering become counterproductive?** → When too restrictive (combined filter: -3.28pp)

---

## Geometric Filters Implemented

### 1. LOF Filter (`src/core/geometric_filter.py`)
**Local Outlier Factor** - Density-based anomaly detection

```python
LOFFilter(n_neighbors=20, threshold=0.0)
```

- Trains LOF on real class embeddings
- Scores synthetic samples by local density
- Keeps samples with LOF score > threshold (inliers)
- **Correlation with F1**: r=0.923 (p=0.0011)

### 2. Filter Cascade (`src/core/filter_cascade.py`)
**4-level hierarchical scoring system**

```python
FilterCascade(filter_level=1, k_neighbors=10)
```

| Level | Metric | Description |
|-------|--------|-------------|
| 1 | Distance | Euclidean distance to anchor (closer = better) |
| 2 | Similarity | Cosine similarity to anchor |
| 3 | KNN Purity | Avg distance to k-nearest same-class neighbors |
| 4 | Confidence | Distance to class centroid |

- Uses geometric mean of active filter scores
- Selects top-N by ranking (no hard threshold)
- **Best result**: Level 1 only (+7.25pp vs SMOTE)

### 3. Combined Filter (`src/core/geometric_filter.py`)
**LOF + Cosine Similarity** (dual criteria)

```python
CombinedGeometricFilter(lof_threshold=0.0, sim_threshold=0.5)
```

- Must pass BOTH: LOF > threshold AND similarity > threshold
- **Too restrictive** → worst performance (-3.28pp)

### 4. Embedding Guided Sampler (`src/core/embedding_guided_sampler.py`)
**Coverage + Quality optimization**

```python
EmbeddingGuidedSampler(coverage_weight=0.6, quality_weight=0.4)
```

- Coverage: Fills gaps in embedding space
- Quality: Proximity to class centroid
- Greedy selection with diversity constraints

### 5. None (Control)
No geometric filtering - random selection from LLM outputs

---

## Key Findings

### When LLM Augmentation Works Best

| Condition | LLM Win Rate | Recommendation |
|-----------|--------------|----------------|
| Multi-class (4+ classes) | **88-90%** | Use 100% LLM with cascade |
| ≤10 samples/class | **90%** | Use cascade level=1 |
| 25 samples/class | 70-88% | Hybrid 25-50% LLM |
| Binary classification | 45-65% | Consider SMOTE |
| >50 samples/class | <50% | SMOTE is better |

### Filter Ranking (Global)

| Filter | Mean Δ vs SMOTE | Win Rate |
|--------|-----------------|----------|
| **cascade** | **+1.45pp** | **83.3%** |
| lof | +0.88pp | 75.8% |
| none | +0.72pp | 62.9% |
| embedding_guided | -1.29pp | 55.0% |
| combined | -3.28pp | 52.5% |

### Critical Insight: Simple Beats Complex

The simplest filter (cascade level=1, distance-only) outperforms all complex multi-criteria filters. **More filtering is not always better**.

---

## Project Structure

```
filters/
├── data/benchmarks/              # Test datasets (10/25/50-shot versions)
├── experiments/
│   ├── exp_filter_comparison.py  # Main experiment (1440 configurations)
│   ├── exp_fixed_output_count.py # NEW: Equal sample count experiment
│   └── download_datasets_v2.py   # Dataset downloader
├── src/core/
│   ├── geometric_filter.py       # LOFFilter, CombinedGeometricFilter
│   ├── filter_cascade.py         # FilterCascade (levels 0-4)
│   ├── embedding_guided_sampler.py
│   ├── llm_providers.py          # OpenAI, Gemini providers
│   └── validation_runner.py      # Evaluation utilities
├── results/
│   ├── thesis_research/          # Organized results for thesis
│   └── fixed_output_count/       # Results from equal-count experiment
└── cache/llm_generations/        # LLM generation cache
```

---

## Experiments

### 1. Main Filter Comparison (`exp_filter_comparison.py`)

Tests 1440 configurations across:
- 6 filter types × multiple parameters
- 4 LLM percentages (5%, 25%, 50%, 100%)
- 3 N-shot values (10, 25, 50)
- 6 datasets

```bash
python experiments/exp_filter_comparison.py
```

### 2. Fixed Output Count (`exp_fixed_output_count.py`)

**NEW**: Tests filters with equal sample counts to isolate filter quality from quantity.

- All filters produce exactly 50 samples per class
- Stricter filters need more LLM calls
- Measures efficiency: F1_improvement / LLM_calls

```bash
python experiments/exp_fixed_output_count.py
```

**Filters tested**: none, lof_relaxed, lof_strict, cascade_l1/l2/full, combined

**Key metrics**:
- F1 score (with equal sample counts)
- Total LLM calls needed
- Acceptance rate per filter
- Efficiency score

---

## Configuration

### LLM Provider

Uses Google Gemini 3 Flash by default:

```python
from core.llm_providers import create_provider
provider = create_provider("google", "gemini-3-flash-preview")
```

Set API key in `.env`:
```
GOOGLE_API_KEY=your_key_here
```

### Embedding Model

Uses `all-mpnet-base-v2` (768 dimensions) from SentenceTransformers.

---

## Datasets

| Dataset | Classes | Versions | Domain |
|---------|---------|----------|--------|
| 20newsgroups | 4 | 10/25/50-shot | News topics |
| sms_spam | 2 | 10/25/50-shot | SMS spam detection |
| hate_speech_davidson | 3 | 10/25/50-shot | Hate speech detection |
| ag_news | 4 | 10/25/50-shot | News classification |

---

## Practical Recommendations

1. **For multi-class, very low resource (≤10/class)**: Use 100% LLM with `cascade level=1`
2. **For moderate low resource (25/class)**: Hybrid approach with 25-50% LLM, `lof` filter
3. **For binary classification**: Test carefully, SMOTE may be better
4. **Avoid**: `combined` filter (too restrictive), `embedding_guided` (underperforms)
5. **Key insight**: Generate many candidates, filter lightly. Don't over-filter.

---

## References

- LOF: Breunig et al., "LOF: Identifying Density-Based Local Outliers" (2000)
- SMOTE: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
- SentenceTransformers: Reimers & Gurevych, "Sentence-BERT" (2019)
