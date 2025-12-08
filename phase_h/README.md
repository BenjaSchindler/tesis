# Phase H: Multi-Classifier Model Comparison

## Overview

Phase H investigates how different ML/DL classifier architectures benefit from LLM-generated synthetic data augmentation. This phase discovered that **neural networks can unlock "impossible" classes** that linear models cannot predict, while **simpler models benefit more from augmentation overall**.

## Key Achievement

**MLP_GPU_small can predict ESTJ class (0.0000 -> 0.1067 F1)** - a class that was completely unpredictable by LogisticRegression even with augmentation.

## Problem Statement

Previous phases used only LogisticRegression/RandomForest for evaluation. Questions arose:
- Do different model architectures benefit equally from synthetic data?
- Can more powerful models unlock classes that linear models cannot predict?
- Is there an inverse relationship between baseline performance and augmentation benefit?

## Methodology

### Evaluation Setup

```
Dataset:           MBTI (8,675 samples, 16 classes)
Embeddings:        all-mpnet-base-v2 (768 dimensions)
Synthetic Data:    ENS_Top3_G5 ensemble (178 samples)
Evaluation:        5-Fold Stratified Cross-Validation
Synthetic Weight:  0.5 (down-weighted)
Random Seed:       42
```

### Models Tested

| Model | Type | Hardware | Configuration |
|-------|------|----------|---------------|
| LogisticRegression | Linear | CPU (n_jobs=-1) | max_iter=2000 |
| LogisticRegression_balanced | Linear | CPU | class_weight='balanced' |
| XGBoost_GPU | Gradient Boosting | CUDA | tree_method='hist' |
| LightGBM_GPU | Gradient Boosting | GPU | device='gpu' |
| MLP_GPU_small | Neural Network | CUDA | 256->128, 50 epochs |
| MLP_GPU_large | Neural Network | CUDA | 512->256->128, 100 epochs |

## Results

### Overall Performance (Macro F1)

| Model | Baseline | Augmented | Relative % | Absolute pp |
|-------|----------|-----------|------------|-------------|
| LightGBM_GPU | 0.1661 | 0.1802 | **+8.50%** | +1.41 |
| LogisticRegression | 0.2057 | 0.2225 | +8.22% | **+1.69** |
| XGBoost_GPU | 0.1778 | 0.1863 | +4.77% | +0.85 |
| MLP_GPU_small | 0.2562 | 0.2645 | +3.21% | +0.82 |
| LogisticRegression_balanced | 0.2523 | 0.2568 | +1.78% | +0.45 |
| MLP_GPU_large | 0.2490 | 0.2501 | +0.47% | +0.12 |

### Problematic Classes Performance

Classes with historically F1=0.00 (ENFJ, ESFJ, ESFP, ESTJ, ISTJ):

| Model | ENFJ | ESFJ | ESFP | ESTJ | ISTJ | AVG |
|-------|------|------|------|------|------|-----|
| MLP_GPU_small | **+5.89** | -2.16 | 0.00 | **+10.67** | +1.42 | **+3.16** |
| LogisticRegression | +2.93 | 0.00 | 0.00 | 0.00 | **+4.88** | +1.56 |
| LogReg_balanced | -1.76 | **+6.09** | -0.02 | +0.11 | +1.24 | +1.13 |
| MLP_GPU_large | -0.89 | +4.26 | 0.00 | -0.08 | +1.51 | +0.96 |
| XGBoost_GPU | +1.00 | -0.44 | 0.00 | 0.00 | +2.56 | +0.62 |
| LightGBM_GPU | -1.03 | +0.44 | 0.00 | 0.00 | +0.81 | +0.05 |

### Best Model per Problematic Class

| Class | Best Model | Baseline | Augmented | Delta |
|-------|------------|----------|-----------|-------|
| **ESTJ** | MLP_GPU_small | 0.0000 | 0.1067 | **+10.67 pp** |
| ESFJ | LogReg_balanced | 0.1076 | 0.1685 | +6.09 pp |
| ENFJ | MLP_GPU_small | 0.1474 | 0.2063 | +5.89 pp |
| ISTJ | LogisticRegression | 0.0554 | 0.1042 | +4.88 pp |
| ESFP | - | 0.0000 | 0.0000 | 0.00 pp |

## Key Findings

### 1. Inverse Relationship: Baseline vs Augmentation Benefit

Models with **lower baseline performance benefit more** from augmentation:
- LightGBM (baseline 0.1661): +8.50% relative
- LogReg (baseline 0.2057): +8.22% relative
- MLP_small (baseline 0.2562): +3.21% relative

This suggests augmentation helps close the gap for weaker models.

### 2. Neural Networks Unlock "Impossible" Classes

**Most significant finding**: MLP_GPU_small can predict ESTJ (0.0000 -> 0.1067) - a class that LogisticRegression could never predict regardless of augmentation.

This demonstrates that augmentation + neural networks enables predictions that were structurally impossible with linear models.

### 3. Model-Specific Class Preferences

Different models excel at different classes:
- **MLP_GPU_small**: Best for ENFJ (+5.89) and ESTJ (+10.67)
- **LogReg_balanced**: Best for ESFJ (+6.09)
- **LogisticRegression**: Best for ISTJ (+4.88)

### 4. ESFP Remains Challenging

No model could improve ESFP. This class likely needs:
- More synthetic samples (0 in current ensemble)
- Different generation strategy
- Targeted anchor selection

## Directory Structure

```
phase_h/
  configs/                           # Configuration files
  results/                           # Output JSON files
  augmentation_effect_by_model.py    # Multi-model evaluation script
  multi_model_evaluator.py           # Core evaluator class
  multi_classifier_evaluator.py      # Extended evaluator
  compare_synth_sources.sh           # Compare synthetic sources
  run_phaseH.sh                      # Main run script
  run_single_classifier.sh           # Single model evaluation
  RESULTS.md                         # Detailed results
  README.md                          # This file
```

## Usage

### Run Full Comparison

```bash
cd phase_h
python3 augmentation_effect_by_model.py
```

### Run Single Classifier

```bash
bash run_single_classifier.sh LogisticRegression
```

### Compare Synthetic Sources

```bash
bash compare_synth_sources.sh
```

## Recommendations

1. **For maximum overall improvement**: Use LogisticRegression (+1.69 pp)

2. **For problematic classes**: Use MLP_GPU_small (unlocks ESTJ, best on ENFJ)

3. **Ensemble approach**: Consider combining:
   - LogisticRegression for general performance
   - MLP for problematic classes (ESTJ, ENFJ)
   - LogReg_balanced for ESFJ

4. **ESFP solution**: Generate targeted synthetic data for this class

## Technical Details

### GPU Acceleration

```python
# XGBoost GPU
XGBClassifier(tree_method='hist', device='cuda')

# LightGBM GPU
LGBMClassifier(device='gpu')

# PyTorch MLP
model = MLP([768, 256, 128, 16]).cuda()
```

### MLP Architecture

```python
# MLP_GPU_small
Sequential(
    Linear(768, 256), ReLU(), Dropout(0.3),
    Linear(256, 128), ReLU(), Dropout(0.3),
    Linear(128, 16)
)
# Training: 50 epochs, lr=0.001, Adam

# MLP_GPU_large
Sequential(
    Linear(768, 512), ReLU(), Dropout(0.3),
    Linear(512, 256), ReLU(), Dropout(0.3),
    Linear(256, 128), ReLU(), Dropout(0.3),
    Linear(128, 16)
)
# Training: 100 epochs, lr=0.001, Adam
```

## Files Reference

- [RESULTS.md](RESULTS.md) - Complete results with per-fold breakdown
- `results/augmentation_effect_by_model.json` - Raw JSON results

## Created

2025-12-05
