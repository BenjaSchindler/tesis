# Phase F: K-Fold Cross-Validation and Configuration Optimization

## Overview

Phase F establishes a rigorous evaluation framework using Repeated Stratified K-Fold Cross-Validation (k=5, repeats=3) to systematically compare different SMOTE-LLM configurations. This phase discovered that **ensemble combinations of multiple configurations provide synergistic improvements**.

## Key Achievement

**Best Result: ENS_Top3_G5 ensemble achieves +1.29% macro F1 improvement (p=0.000004)**

## Problem Statement

Previous phases used single train/test splits which could be affected by:
- Random seed selection bias
- Overfitting to specific data partitions
- Limited statistical confidence

Phase F addresses this by:
1. Using 15 evaluation folds (5-fold x 3 repeats)
2. Computing statistical significance via paired t-test
3. Testing multiple configuration variants systematically

## Methodology

### Evaluation Protocol

```
Evaluation:         Repeated Stratified K-Fold
Folds:              k = 5
Repeats:            r = 3
Total evaluations:  15 per configuration
Synthetic weight:   0.5 (down-weighted)
Classifier:         RandomForest (n_estimators=100)
Statistical test:   Paired t-test (alpha=0.05)
```

### Configuration Categories

| Category | Description | Configs |
|----------|-------------|---------|
| **Volume (V)** | Vary cluster/prompt/sample counts | V1-V5 |
| **Confidence Filter (CF)** | Different filter modes | CF1-CF3 |
| **Combinations (CMB)** | Multiple features combined | CMB1-CMB3 |
| **GPT-5-mini (G5)** | Reasoning model experiments | G5_K5-G5_K100 |

## Results Summary

### Top Configurations

| Rank | Config | Delta% | p-value | Samples | Type |
|------|--------|--------|---------|---------|------|
| 1 | **ENS_Top3_G5** | **+1.29%** | 0.000004 | 199 | Ensemble |
| 2 | **ENS_Top3** | **+1.22%** | <0.00001 | 146 | Ensemble |
| 3 | ENS_CMB3_V2 | +0.93% | 0.00002 | 107 | Ensemble |
| 4 | ENS_CMB3_CF1 | +0.87% | 0.00001 | 103 | Ensemble |
| 5 | CMB3_skip | +0.57% | 0.0004 | 61 | Single |
| 6 | CF1_conf_band | +0.56% | 0.0009 | 42 | Single |

### Ensemble Compositions

```
ENS_Top3_G5 = CMB3_skip + CF1_conf_band + V4_ultra + G5_K25_medium
              (61 samples) + (42 samples) + (43 samples) + (53 samples) = 199 total

ENS_Top3    = CMB3_skip + CF1_conf_band + V4_ultra
              (61 samples) + (42 samples) + (43 samples) = 146 total
```

## Key Findings

### 1. Ensemble Approach Provides 2.26x Improvement

```
Best single config:   CMB3_skip    = +0.57%
Best ensemble:        ENS_Top3_G5  = +1.29%
Multiplier:           2.26x better
```

### 2. Diversity > Volume

More candidates doesn't guarantee better results:
- V5_extreme (1500 candidates): +0.27%
- V4_ultra (672 candidates): +0.45%
- CMB3_skip (225 candidates): +0.57%

Quality filtering and targeted generation beat brute-force volume.

### 3. F1-Budget-Scaling is Effective

CMB3_skip uses resource allocation based on class difficulty:
- Skip high F1 classes (>0.35): multiplier 0.0
- Half budget for medium F1 (0.20-0.35): multiplier 0.5
- 2.5x budget for low F1 (<0.20): multiplier 2.5

### 4. GPT-5-mini Adds Unique Diversity

While G5 alone underperforms gpt-4o-mini, it contributes to ensembles:
- G5_K25_medium standalone: +0.30%
- Adding G5 to Top3: +1.22% -> +1.29% (+0.07%)

## Directory Structure

```
phase_f/
  configs/              # 24 configuration files
    V1_baseline.sh      # Base config (225 candidates)
    V2_high_vol.sh      # High volume (270 candidates)
    V4_ultra.sh         # Ultra volume (672 candidates)
    CF1_conf_band.sh    # Confidence band filter
    CMB3_skip.sh        # Best single config
    G5_K25_medium.sh    # GPT-5-mini with reasoning
    ...
  core/                 # Runner and utilities
  results/              # Output files and K-Fold JSONs
  kfold_evaluator.py    # K-Fold evaluation script
  create_ensembles.py   # Ensemble creation utility
  base_config.sh        # Template for all configs
  RESULTS_PHASE_F.md    # Detailed results documentation
  PER_CLASS_ANALYSIS.md # Per-class breakdown
  README.md             # This file
```

## Usage

### Run Single Configuration

```bash
cd phase_f
export OPENAI_API_KEY='...'

# Generate synthetic data
SEED=42 bash configs/CMB3_skip.sh

# Evaluate with K-Fold
python3 kfold_evaluator.py --config CMB3_skip --seed 42 --k 5 --repeated 3
```

### Create Ensemble

```bash
python3 create_ensembles.py --configs CMB3_skip,CF1_conf_band,V4_ultra --name ENS_Top3
```

### Run All Configurations

```bash
nohup bash run_overnight.sh > overnight_run.log 2>&1 &
tail -f overnight_run.log
```

## Best Configuration Parameters

### CMB3_skip (Best Single)

```bash
--auto-anchor-margin 0.05
--max-clusters 5
--prompts-per-cluster 9
--samples-per-prompt 5
--min-classifier-confidence 0.05
--filter-mode hybrid
--use-f1-budget-scaling
--f1-budget-thresholds 0.35 0.20
--f1-budget-multipliers 0.0 0.5 2.5
```

### ENS_Top3_G5 (Best Overall)

Combines synthetic data from 4 configurations:
1. CMB3_skip: F1-budget-scaling, hybrid filter
2. CF1_conf_band: Tight confidence band (0.02 margin)
3. V4_ultra: High volume (8x12x7 = 672 candidates)
4. G5_K25_medium: GPT-5-mini with medium reasoning

## Statistical Analysis

### ENS_Top3_G5 Detailed Results

```
BASELINE F1 (macro):
  Mean: 0.205045
  Std:  0.013148

AUGMENTED F1 (macro):
  Mean: 0.217964
  Std:  0.014368

DELTA:
  Mean:          +1.29%
  95% CI:        [+0.92%, +1.67%]
  t-statistic:   7.37
  p-value:       0.0000035
  Win Rate:      14/15 folds (93.3%)
```

## Files Reference

- [RESULTS_PHASE_F.md](RESULTS_PHASE_F.md) - Complete results with all configs
- [PER_CLASS_ANALYSIS.md](PER_CLASS_ANALYSIS.md) - Per-class F1 breakdown
- [TODO.md](TODO.md) - Remaining experiments
- [STATE_OF_THE_ART_SOURCES.md](STATE_OF_THE_ART_SOURCES.md) - Literature references

## Next Steps

Phase F findings led to:
- **Phase G**: Focus on problematic classes (ENFJ, ESFJ, ESFP, ESTJ, ISTJ)
- **Phase H**: Compare ML/DL classifier architectures
- **Phase I**: Multi-LLM provider comparison

## Created

2025-12-03 to 2025-12-05
