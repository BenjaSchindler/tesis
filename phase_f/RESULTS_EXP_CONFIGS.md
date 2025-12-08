# Phase F - EXP Configs Results (December 2025)

## Overview

8 new experimental configurations combining best strategies from Phase E and Phase F.
Evaluated with K-fold cross-validation (k=5, repeats=3, total=15 folds).

**Run Date:** 2025-12-05 (overnight)
**Evaluation:** K-fold with RandomForest classifier

## Results Summary

| Config | Synth | Delta | 95% CI | p-value | Win Rate | Significant |
|--------|-------|-------|--------|---------|----------|-------------|
| **EXP7_hybrid_best** | 61 | **+0.53%** | [+0.30, +0.77] | 0.000244 | 93% | ✓ |
| EXP3_minority_focus | 70 | +0.49% | [+0.25, +0.74] | 0.000763 | 93% | ✓ |
| **EXP8_intj_protect** | 45 | +0.39% | [+0.25, +0.53] | 0.000034 | **100%** | ✓ |
| EXP4_ultra_relaxed | 47 | +0.38% | [+0.21, +0.55] | 0.000274 | 93% | ✓ |
| EXP1_phaseE_port | 47 | +0.37% | [+0.14, +0.61] | 0.003861 | 73% | ✓ |
| EXP2_relaxed_plus | 44 | +0.34% | [+0.18, +0.50] | 0.000412 | 87% | ✓ |
| EXP5_quality_focus | 44 | +0.30% | [+0.09, +0.52] | 0.008336 | 80% | ✓ |
| EXP6_knn_strict | - | FAILED | - | - | - | ✗ |

**Note:** EXP6 failed due to invalid `--filter-mode knn_only` (should be `knn`).

## Config Descriptions

### EXP1_phaseE_port
Port of Phase E C1_5x9x5 config.
```bash
--max-clusters 5 --prompts-per-cluster 9 --samples-per-prompt 5
```

### EXP2_relaxed_plus
D1_relaxed improved with relaxed filters.
```bash
--similarity-threshold 0.85 --contamination-threshold 0.90
--min-classifier-confidence 0.05 --anchor-quality-threshold 0.20
```

### EXP3_minority_focus
Aggressive F1-budget-scaling for minority classes.
```bash
--use-f1-budget-scaling --f1-budget-thresholds 0.40 0.25
--f1-budget-multipliers 0.0 0.3 3.0 --max-clusters 8
```

### EXP4_ultra_relaxed
V4 volume + D1 relaxed filters combined.
```bash
--max-clusters 8 --prompts-per-cluster 12 --samples-per-prompt 7
--similarity-threshold 0.85 --contamination-threshold 0.90
```

### EXP5_quality_focus
Strict quality filters for fewer but better synthetics.
```bash
--similarity-threshold 0.92 --contamination-threshold 0.98
--min-classifier-confidence 0.25 --anchor-quality-threshold 0.70
```

### EXP7_hybrid_best ⭐
Best performing - combines CMB3 + Phase E strategies.
```bash
--max-clusters 6 --prompts-per-cluster 10 --samples-per-prompt 6
--auto-anchor-margin 0.05 --min-classifier-confidence 0.05
--filter-mode hybrid --use-f1-budget-scaling
--f1-budget-thresholds 0.35 0.20 --f1-budget-multipliers 0.0 0.5 2.5
--similarity-threshold 0.88
```

### EXP8_intj_protect ⭐
100% win rate - protects INTJ from degradation.
```bash
--max-clusters 5 --prompts-per-cluster 9 --samples-per-prompt 5
--contamination-threshold 0.98 --min-classifier-confidence 0.20
```

## Key Findings

1. **All 7 configs statistically significant** (p < 0.05)
2. **EXP8_intj_protect has 100% win rate** - most consistent
3. **EXP7_hybrid_best has highest delta** (+0.53%)
4. **F1-budget-scaling** in EXP3/EXP7 generated more synthetics but mixed results on per-class

## Comparison to Previous Best

| Metric | ENS_Top3_G5 (prev best) | EXP7_hybrid_best |
|--------|-------------------------|------------------|
| Delta | +1.29% | +0.53% |
| Synthetics | 199 | 61 |
| Win Rate | 93% | 93% |

**Note:** EXP configs are individual runs, not ensembles. Combining them with Phase G configs may improve results.

## Files

- Synthetic CSVs: `results/EXP*_s42_synth.csv`
- K-fold JSONs: `results/EXP*_s42_kfold_k5.json`
- Config scripts: `experiments/configs/EXP*.sh`
