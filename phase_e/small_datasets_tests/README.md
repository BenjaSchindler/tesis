# Small Datasets Tests - Phase E

## Overview

This folder documents experiments optimizing SMOTE-LLM for small datasets (< 10K samples).
We discovered that the original `auto_params` module was overriding cluster settings, limiting
synthetic generation. After fixing this and adding purity-aware clustering, we achieved
**+4.15% delta** (up from +0.26%).

## Dataset

- **File**: `mbti_1.csv` (subset of MBTI_500.csv)
- **Total samples**: ~8,675
- **Classes**: 16 MBTI personality types
- **Train/Val/Test split**: 80% / 15% val / 20% test
- **Smallest class**: ~39 samples (ESTJ)
- **Largest class**: ~1,832 samples (INFP)

## Problem Identified

The `auto_params.py` module was designed to adjust parameters for small datasets, but it had
a critical bug: it **overrode** `--max-clusters` and `--prompts-per-cluster` to 2x2 for any
dataset < 10K samples, regardless of CLI arguments.

```python
# BUG: This overrode user-specified cluster settings
if profile.total_samples < 10000:
    max_clusters = 2
    prompts_per_cluster = 2  # Forces only 4 prompts per class!
```

## Fixes Applied

### 1. Fixed auto_params.py (no cluster override)

```python
# FIXED: Don't override cluster settings
return AutoParams(
    anchor_quality_threshold=...,
    cap_class_ratio=...,
    max_clusters=None,  # Don't override - let CLI args handle it
    prompts_per_cluster=None,  # Don't override
    min_cluster_samples=min_cluster_samples,  # For purity-aware
)
```

### 2. Added Purity-Aware Clustering

Low-purity classes (< 0.03) or small classes (< 30 samples) get fewer clusters to
prevent dilution of good anchors:

```python
# In runner_phase2.py
if purity < 0.03 or n_samples < 30:
    max_clusters_for_size = max(3, n_samples // min_cluster_samples)
    if n_clusters_est > max_clusters_for_size:
        n_clusters_est = max_clusters_for_size
        print(f"Purity-aware: {cls} clusters reduced (purity={purity:.3f})")
```

## Test Configurations

### Original (Broken auto-params)
```bash
--auto-params                    # Forced max-clusters=2, prompts-per-cluster=2
--anchor-quality-threshold 0.10  # Auto-adjusted for small dataset
--cap-class-ratio 0.30          # Auto-adjusted
--similarity-threshold 0.85     # Auto-adjusted
```

### Test A: Fixed auto-params + 5x9 clusters + purity-aware
```bash
--auto-params                    # Now respects CLI args
--max-clusters 5                 # 5 clusters per class
--prompts-per-cluster 9          # 9 prompts per cluster = 45 total
--anchor-quality-threshold 0.10  # Auto-adjusted
--cap-class-ratio 0.30          # Auto-adjusted
--similarity-threshold 0.85     # Auto-adjusted
# Purity-aware clustering enabled automatically
```

### Test B: Fixed auto-params + 3x3 clusters + purity-aware
```bash
--auto-params
--max-clusters 3
--prompts-per-cluster 3          # 9 total prompts (baseline config)
--anchor-quality-threshold 0.10
--cap-class-ratio 0.30
--similarity-threshold 0.85
```

## Results

| Configuration | Baseline F1 | Augmented F1 | Delta | Synthetics |
|--------------|-------------|--------------|-------|------------|
| Original (broken, 2x2) | 0.2488 | 0.2494 | +0.26% | 31 |
| Test A (fixed, 5x9) | 0.2268 | 0.2362 | **+4.15%** | 48 |
| Test B (fixed, 3x3) | 0.2268 | 0.2275 | +0.31% | 41 |

### Per-Class Synthetics

| Class | Purity | Original | Test A (5x9) | Test B (3x3) |
|-------|--------|----------|--------------|--------------|
| ENTJ | 0.059 | 4 | 10 | 5 |
| ISTJ | 0.051 | 5 | 10 | 10 |
| ISFJ | 0.043 | 6 | 7 | 10 |
| ENFJ | 0.041 | 4 | 8 | 10 |
| ESTP | 0.021 | 4 | 5 | 1 |
| ESFJ | 0.017 | 6 | 6 | 4 |
| ESTJ | 0.005 | 2 | 2 | 1 |

## Key Insights

### 1. More Candidates Helps High-Purity Classes
Classes with purity > 0.04 (ENTJ, ISTJ, ISFJ, ENFJ) benefit significantly from more
prompt candidates (5x9 = 45 vs 3x3 = 9). More diversity + good anchors = better synthetics.

### 2. Purity-Aware Prevents Dilution
Low-purity classes (ESTJ: 0.005, ESFJ: 0.017) don't benefit from more clusters.
The purity-aware adjustment reduces clusters to concentrate good anchors:
```
Purity-aware: ESTJ clusters 6->5 (purity=0.005, n=26)
```

### 3. The Winning Formula
```
IF purity > 0.04 AND n_samples > 50:
    USE more clusters (5x9) for diversity
ELSE:
    USE fewer clusters to concentrate anchors
```

## Files Modified

1. **`core/auto_params.py`**
   - Removed cluster override for small datasets
   - Added `min_cluster_samples` parameter

2. **`core/runner_phase2.py`**
   - Added purity-aware cluster adjustment (lines 2624-2633)
   - Clusters reduced for low-purity or small classes

## How to Reproduce

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_e

# Test A: Best configuration (5x9 + purity-aware)
python3 -u core/runner_phase2.py \
    --data-path ../mbti_1.csv \
    --test-size 0.2 \
    --random-seed 42 \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 128 \
    --cache-dir embeddings_cache \
    --llm-model gpt-4o-mini \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --use-class-description \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --auto-params \
    --synthetic-output results/testA_synth.csv \
    --augmented-train-output results/testA_aug.csv \
    --metrics-output results/testA_metrics.json
```

## Conclusion

For small datasets (< 10K samples):
1. **Don't limit clusters** - let the elbow method adapt
2. **Use more prompts** (5x9) for classes with good purity
3. **Enable purity-aware clustering** to protect low-purity classes
4. **Lower quality thresholds** (0.10-0.15) to allow generation

This combination achieved **+4.15% F1 improvement** vs +0.26% with the broken configuration.
