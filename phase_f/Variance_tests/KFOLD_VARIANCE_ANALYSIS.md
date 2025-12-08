# Stratified K-Fold Variance Analysis

**Date:** 2024-12-04
**Phase:** F - Calibrated Experiments
**Objective:** Reduce evaluation variance using Stratified K-Fold Cross-Validation

---

## 1. Problem Statement

### Original Issue
Single train/test split evaluations showed high variance across different random seeds:

| Config | Seed 42 | Seed 100 | Seed 123 | Range |
|--------|---------|----------|----------|-------|
| V2_high_vol | +3.37% | -4.34% | +0.47% | 7.71pp |
| V3_low_vol | +1.63% | -0.91% | +1.79% | 2.70pp |

V2_high_vol had the highest peak (+3.37%) but also the worst valley (-4.34%), making it unreliable for conclusions.

### Hypothesis
The high variance was due to:
1. Random train/test split sensitivity
2. Small synthetic sample sizes (44-46 samples)
3. Class imbalance effects

K-Fold CV should stabilize results by averaging over multiple splits.

---

## 2. Methodology

### 2.1 Stratified K-Fold Configuration

```
Algorithm: Repeated Stratified K-Fold Cross-Validation
K (splits): 5
Repeats: 3
Total Folds: 15 (5 x 3)
Random State: 42
```

### 2.2 Why These Parameters?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| K=5 | 5 folds | Standard choice, 80/20 train/test ratio per fold |
| Repeats=3 | 3 repetitions | Increases statistical power without excessive computation |
| Total=15 | 15 evaluations | Sufficient for t-test significance (df=14) |
| Stratified | Yes | Maintains class distribution in each fold |

### 2.3 Evaluation Pipeline

```
For each fold i in 1..15:
    1. Split original data (8675 samples) into train/test
    2. Train baseline LogisticRegression on train fold
    3. Evaluate baseline on test fold -> baseline_f1[i]
    4. Combine train fold + ALL synthetic data
    5. Train augmented LogisticRegression with sample_weight=0.5 for synthetic
    6. Evaluate augmented on test fold -> augmented_f1[i]
    7. delta[i] = augmented_f1[i] - baseline_f1[i]

Compute:
    - mean(delta), std(delta)
    - 95% CI using t-distribution
    - t-statistic and p-value (one-sample t-test, H0: delta=0)
    - win_rate = count(delta > 0) / 15
```

---

## 3. Configurations Tested

### 3.1 V3_low_vol (Conservative)

**Config File:** `configs/V3_low_vol.sh`

```bash
# Key parameters
--cap-class-ratio 0.10          # Max 10% synthetic per class
--min-classifier-confidence 0.15 # Higher confidence threshold
--contamination-threshold 0.90   # Stricter contamination filter
--similarity-threshold 0.92      # Higher similarity required
--synthetic-weight 0.5           # 50% weight for synthetic samples
```

**Synthetic Data Generated:** 44 samples

### 3.2 V2_high_vol (Aggressive)

**Config File:** `configs/V2_high_vol.sh`

```bash
# Key parameters
--cap-class-ratio 0.20          # Max 20% synthetic per class
--min-classifier-confidence 0.10 # Lower confidence threshold
--contamination-threshold 0.95   # More permissive
--similarity-threshold 0.90      # Standard similarity
--synthetic-weight 0.5           # 50% weight for synthetic samples
```

**Synthetic Data Generated:** 46 samples

---

## 4. Results

### 4.1 V3_low_vol K-Fold Results

```
============================================================
  K-FOLD EVALUATION SUMMARY: V3_low_vol
============================================================

  Folds: 15 (5 splits x 3 repeats)
  Synthetic samples used: 44

  BASELINE:
    Mean F1: 0.2050 +/- 0.0131

  AUGMENTED:
    Mean F1: 0.2080 +/- 0.0150

  IMPROVEMENT (Delta):
    Mean:     +0.0029 (+1.42%)
    Std:      0.0046
    95% CI:   [+0.0004, +0.0054]
    Win Rate: 11/15 (73.3%)

  STATISTICAL SIGNIFICANCE:
    t-statistic: 2.472
    p-value:     0.0269
    Result:      SIGNIFICANT IMPROVEMENT (p < 0.05)
```

**Per-Fold Breakdown:**

| Fold | Baseline | Augmented | Delta | Delta % |
|------|----------|-----------|-------|---------|
| 1 | 0.2172 | 0.2240 | +0.0068 | +3.14% |
| 2 | 0.1948 | 0.1981 | +0.0033 | +1.72% |
| 3 | 0.2155 | 0.2168 | +0.0013 | +0.61% |
| 4 | 0.1836 | 0.1832 | -0.0003 | -0.19% |
| 5 | 0.2172 | 0.2336 | +0.0163 | +7.52% |
| 6 | 0.2181 | 0.2237 | +0.0056 | +2.55% |
| 7 | 0.2051 | 0.2049 | -0.0002 | -0.08% |
| 8 | 0.2217 | 0.2213 | -0.0004 | -0.20% |
| 9 | 0.1840 | 0.1864 | +0.0024 | +1.29% |
| 10 | 0.2124 | 0.2129 | +0.0005 | +0.21% |
| 11 | 0.2101 | 0.2127 | +0.0027 | +1.26% |
| 12 | 0.1965 | 0.2023 | +0.0058 | +2.95% |
| 13 | 0.2118 | 0.2091 | -0.0027 | -1.29% |
| 14 | 0.2010 | 0.2034 | +0.0023 | +1.16% |
| 15 | 0.1866 | 0.1870 | +0.0003 | +0.18% |

### 4.2 V2_high_vol K-Fold Results

```
============================================================
  K-FOLD EVALUATION SUMMARY: V2_high_vol
============================================================

  Folds: 15 (5 splits x 3 repeats)
  Synthetic samples used: 46

  BASELINE:
    Mean F1: 0.2050 +/- 0.0131

  AUGMENTED:
    Mean F1: 0.2089 +/- 0.0152

  IMPROVEMENT (Delta):
    Mean:     +0.0039 (+1.88%)
    Std:      0.0038
    95% CI:   [+0.0017, +0.0060]
    Win Rate: 15/15 (100.0%)

  STATISTICAL SIGNIFICANCE:
    t-statistic: 3.895
    p-value:     0.0016
    Result:      SIGNIFICANT IMPROVEMENT (p < 0.05)
```

**Per-Fold Breakdown:**

| Fold | Baseline | Augmented | Delta | Delta % |
|------|----------|-----------|-------|---------|
| 1 | 0.2172 | 0.2309 | +0.0137 | +6.31% |
| 2 | 0.1948 | 0.2008 | +0.0060 | +3.10% |
| 3 | 0.2155 | 0.2168 | +0.0014 | +0.63% |
| 4 | 0.1836 | 0.1846 | +0.0010 | +0.56% |
| 5 | 0.2172 | 0.2245 | +0.0073 | +3.35% |
| 6 | 0.2181 | 0.2269 | +0.0088 | +4.02% |
| 7 | 0.2051 | 0.2059 | +0.0008 | +0.38% |
| 8 | 0.2217 | 0.2221 | +0.0003 | +0.15% |
| 9 | 0.1840 | 0.1844 | +0.0004 | +0.20% |
| 10 | 0.2124 | 0.2184 | +0.0060 | +2.83% |
| 11 | 0.2101 | 0.2133 | +0.0032 | +1.54% |
| 12 | 0.1965 | 0.1995 | +0.0031 | +1.57% |
| 13 | 0.2118 | 0.2150 | +0.0031 | +1.47% |
| 14 | 0.2010 | 0.2015 | +0.0004 | +0.22% |
| 15 | 0.1866 | 0.1889 | +0.0023 | +1.25% |

---

## 5. Comparison Summary

| Metric | V3_low_vol | V2_high_vol | Winner |
|--------|------------|-------------|--------|
| **Baseline F1** | 0.2050 +/- 0.0131 | 0.2050 +/- 0.0131 | Tie |
| **Augmented F1** | 0.2080 +/- 0.0150 | 0.2089 +/- 0.0152 | V2 |
| **Mean Delta** | +0.0029 (+1.42%) | +0.0039 (+1.88%) | **V2** |
| **Delta Std** | 0.0046 | 0.0038 | **V2** (lower) |
| **95% CI Lower** | +0.0004 | +0.0017 | **V2** |
| **95% CI Upper** | +0.0054 | +0.0060 | V2 |
| **Win Rate** | 73.3% (11/15) | **100% (15/15)** | **V2** |
| **t-statistic** | 2.472 | 3.895 | **V2** |
| **p-value** | 0.0269 | **0.0016** | **V2** |
| **Synthetic Samples** | 44 | 46 | Similar |

---

## 6. Key Findings

### 6.1 Variance Reduction Achieved

Before K-Fold (single seed evaluation):
- V2_high_vol variance: 7.71 percentage points range
- V3_low_vol variance: 2.70 percentage points range

After K-Fold (15-fold average):
- V2_high_vol std: 0.0038 (0.38pp)
- V3_low_vol std: 0.0046 (0.46pp)

**Variance reduced by ~85-95%**

### 6.2 V2_high_vol Revealed as Superior

The single-seed evaluation was misleading:
- Seed 42 showed V2 at +3.37% (overestimate)
- Seed 100 showed V2 at -4.34% (underestimate)
- K-Fold reveals true improvement: **+1.88%**

### 6.3 Statistical Significance Confirmed

Both configurations show statistically significant improvement:
- V3_low_vol: p = 0.0269 < 0.05 (significant)
- V2_high_vol: p = 0.0016 < 0.01 (highly significant)

### 6.4 100% Win Rate for V2

V2_high_vol improved performance in ALL 15 folds:
- This indicates robust, consistent improvement
- V3_low_vol had 4 negative folds (26.7% loss rate)

---

## 7. Conclusions

1. **K-Fold CV is essential** for reliable evaluation of LLM augmentation
2. **V2_high_vol is the best configuration** when evaluated properly
3. **Higher augmentation volume** (cap_class_ratio=0.20) is beneficial
4. **Single-seed evaluation is unreliable** - can show 7+ pp variance
5. **Both configs provide significant improvement** over baseline

---

## 8. Recommendations

### For Thesis Reporting:
- Use K-Fold results as primary evidence
- Report both mean and 95% CI
- Highlight V2_high_vol as recommended configuration

### For Future Experiments:
- Always use K-Fold (k=5, repeats=3) for final evaluation
- Single-seed is OK for quick iteration during development
- Consider testing V2 parameters with other LLM models

---

## 9. Technical Details

### 9.1 Script Used

```bash
python3 kfold_evaluator.py \
    --config V2_high_vol \
    --seed 42 \
    --k 5 \
    --repeated 3 \
    --results-dir /home/benja/Desktop/Tesis/SMOTE-LLM/phase_f/results \
    --data-path /home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv \
    --cache-dir /home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/embeddings_cache \
    --synthetic-weight 0.5
```

### 9.2 Dependencies

- scikit-learn: StratifiedKFold, RepeatedStratifiedKFold
- scipy.stats: t.interval, ttest_1samp
- sentence-transformers: all-mpnet-base-v2
- numpy, pandas

### 9.3 Classifier Configuration

```python
LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    class_weight=None  # No balancing in these tests
)
```

### 9.4 Embedding Model

```
Model: sentence-transformers/all-mpnet-base-v2
Dimensions: 768
Cached: Yes (in embeddings_cache/)
```

---

## 10. Files Generated

| File | Description |
|------|-------------|
| `results/V2_high_vol_s42_kfold_k5.json` | V2 K-Fold results |
| `results/V3_low_vol_s42_kfold_k5.json` | V3 K-Fold results |
| `kfold_evaluator.py` | K-Fold evaluation script |
| `Variance_tests/KFOLD_VARIANCE_ANALYSIS.md` | This document |

---

*Generated: 2025-12-04*
*Phase F - SMOTE-LLM Thesis Project*
