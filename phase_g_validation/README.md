# Phase G Validation

**Advanced Strategies for Problem Class Improvement & LLM-Based Data Augmentation**

---

## Overview

Phase G Validation extends Phase F by specifically targeting **rare/difficult classes** (ESFJ, ESFP, ESTJ with <50 samples each) and testing advanced LLM prompting strategies.

**Key Achievement**: Improved overall F1 from +2.07% (Phase F) to **+5.98%** (Phase G W5_many_shot_10), and achieved **+12.42% ESFJ improvement** using neural networks.

---

## Quick Start

### 1. View Results Summary
```bash
cat RESULTS_SUMMARY.md
```
Quick reference with top configs, wave results, and recommendations.

### 2. Read Full Documentation
```bash
cat TECHNICAL_DOCUMENTATION.md
```
Complete technical details of all 38 configurations tested.

### 3. View Plots
```bash
ls plots/
# top10_configs.png
# wave_comparison.png
# rare_class_heatmap.png
# pvalue_analysis.png
# multiclassifier_comparison.png
# category_summary.png
```

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **Best Config** | W5_many_shot_10 (+5.98%, p<0.0001) |
| **Total Configs Tested** | 38 |
| **Significant Results** | 30/38 (78.9%) |
| **Baseline F1** | 0.2045 |
| **Best F1** | 0.2167 (W5_many_shot_10) |
| **ESFJ Improvement** | +12.42% (MLP_512_256_128) |
| **ESFP Improvement** | 0% (unsolved across all configs) |
| **ESTJ Improvement** | +1.79% (MLP_512_256_128) |

---

## Top 5 Configurations

| Rank | Config | Delta | p-value | Category |
|------|--------|-------|---------|----------|
| 🥇 | **W5_many_shot_10** | +5.98% | 0.00005 | Prompting |
| 🥈 | **W6_temp_high** | +5.57% | 0.0002 | Temperature |
| 🥉 | **W5_few_shot_3** | +5.34% | 0.0005 | Prompting |
| 4 | **V4_ultra** | +5.22% | 0.00001 | Volume |
| 5 | **W7_yolo** | +5.05% | 0.0003 | No Filter |

---

## Experiments Conducted

### Wave Experiments (1-9)
- **Wave 1**: Quality gate thresholds (3 configs)
- **Wave 2**: Volume oversampling (2 configs)
- **Wave 3**: Deduplication & filtering (2 configs)
- **Wave 4**: Targeted generation (1 config)
- **Wave 5**: Few-shot vs many-shot prompting 🏆 (3 configs)
- **Wave 6**: Temperature diversity (3 configs)
- **Wave 7**: YOLO - no filtering (2 configs)
- **Wave 8**: GPT-4o reasoning mode (2 configs, failed)
- **Wave 9**: Contrastive learning (2 configs)

### Special Experiments
- **Component Validation**: Phase F component testing (4 configs)
- **Phase F Derived**: Variations of Phase F optimal (3 configs)
- **Rare Class (Exp 13)**: ESFJ/ESFP/ESTJ focus (5 configs)
- **Multi-Classifier (Exp 14b)**: Neural nets vs tree-based (5 classifiers)
- **Ensembles**: Combining top configs (6 ensembles)

---

## Key Findings

### ✅ What Works Best

1. **Many-shot prompting** (10 in-context examples)
   - W5_many_shot_10: +5.98%, p<0.0001
   - Far superior to few-shot (+5.34%) and zero-shot (+1.82%)

2. **Temperature = 1.2**
   - W6_temp_high: +5.57%, p=0.0002
   - Sweet spot for diversity without losing coherence

3. **Neural Networks for rare classes**
   - MLP_512_256_128: +12.41% overall, +12.42% ESFJ
   - First solution to ESFJ problem class

4. **High volume generation**
   - V4_ultra (4,000+ synthetics): +5.22%
   - W2_ultra_vol: +3.55%

5. **Permissive filtering**
   - W3_permissive_filter: +4.35%
   - Low quality gates increase useful samples

### ❌ What Doesn't Work

1. **Tree-based models** (XGBoost, LightGBM)
   - Baseline F1 only 0.16-0.18 (vs 0.22 LogReg)
   - Augmentation **hurts** performance (negative deltas)
   - Fail catastrophically with 768D embeddings

2. **Forcing problem classes** in standard generation
   - W1_force_problem: +1.63% (vs +3.48% without forcing)
   - Reduces overall quality

3. **Targeted generation** (only low-F1 classes)
   - W4_target_only: +1.46%
   - Full generation (+5.98%) much better

4. **Ensembles**
   - Don't surpass best individual configs
   - Best ensemble: +4.40% < W5_many_shot_10: +5.98%

### 🤔 Surprising Findings

1. **YOLO (no filtering) works well**
   - W7_yolo: +5.05%
   - Quality gates may remove good samples

2. **ESFP is unsolvable**
   - 0% improvement across **all 38 configs**
   - Even with 944 synthetic samples + neural networks
   - May require domain-specific features

3. **Deduplication hurts**
   - W3_no_dedup: +3.88%
   - W3_permissive_filter (with dedup): +4.35% but less than expected

---

## Problem Classes Deep Dive

### ESFJ (42 samples): ✅ SOLVED
- **Standard configs**: 0% improvement
- **RARE_massive_oversample**: +8.02% (20x multiplier)
- **MLP_512_256_128**: **+12.42%** ← Best result
- **Solution**: Massive oversampling + Neural Networks

### ESFP (48 samples): ❌ UNSOLVED
- **All 38 configs**: 0% improvement
- **944 synthetic samples**: 0% improvement
- **All 5 classifiers**: 0% improvement
- **Status**: Exceptional difficulty, possible limitation of text-based approach

### ESTJ (39 samples): ⚠️ PARTIAL
- **Standard configs**: 0% improvement
- **MLP_512_256_128**: +1.79%
- **Status**: Slight improvement with neural networks, still challenging

---

## Recommended Configurations

### For Maximum Overall Performance
```python
CONFIG_MAX_F1 = {
    "prompting": "many_shot_10",      # 10 in-context examples
    "temperature": 1.2,                # Optimal diversity
    "quality_gate": 0.50,              # Permissive
    "tier_budgets": [30, 20, 15],     # High volume
    "K_max": 12,
    "K_neighbors": 25,
    "classifier": "LogisticRegression"
}
# Expected: +5.98% (F1: 0.2045 → 0.2167)
# p-value: 0.00005 (highly significant)
```

### For Rare Class Improvement
```python
CONFIG_RARE_CLASS = {
    "prompting": "many_shot_10",
    "temperature": 1.2,
    "rare_class_multiplier": 20,       # 20x oversampling
    "rare_class_threshold": 50,        # classes with <50 samples
    "K_max": 12,
    "K_neighbors": 25,
    "classifier": "MLP_512_256_128"    # Neural network
}
# Expected: +12.41% overall, +12.42% ESFJ, +1.79% ESTJ
# p-value: 0.0077 (significant)
```

### For Cost-Effective Balance
```python
CONFIG_COST_EFFECTIVE = {
    "prompting": "few_shot_3",         # Fewer examples
    "temperature": 1.2,
    "quality_gate": 0.50,
    "tier_budgets": [20, 12, 8],      # Medium volume
    "K_max": 12,
    "K_neighbors": 25,
    "classifier": "LogisticRegression"
}
# Expected: +5.34%
# Cost: 70% less API calls than many_shot_10
```

---

## File Structure

```
phase_g_validation/
│
├── README.md                          # This file
├── TECHNICAL_DOCUMENTATION.md         # Full technical details
├── RESULTS_SUMMARY.md                 # Quick reference summary
│
├── validation_runner.py               # K-fold CV framework
├── base_config.py                     # Configuration classes
├── compile_results.py                 # Results aggregation
├── generate_plots.py                  # Visualization generator
│
├── configs/                           # All configuration files
│   ├── wave1/
│   ├── wave2/
│   ├── ...
│   └── rare_class/
│
├── experiments/                       # Experiment scripts
│   ├── exp13_rare_classes.py         # Rare class experiments
│   └── exp14b_mlp_xgboost.py         # Multi-classifier eval
│
├── results/                           # All experimental results
│   ├── wave1/
│   ├── wave2/
│   ├── ...
│   ├── rare_class/
│   ├── multiclassifier/
│   ├── ensembles/
│   └── FULL_SUMMARY.json             # Compiled results
│
├── plots/                             # Generated visualizations
│   ├── top10_configs.png
│   ├── wave_comparison.png
│   ├── rare_class_heatmap.png
│   ├── pvalue_analysis.png
│   ├── multiclassifier_comparison.png
│   └── category_summary.png
│
├── logs/                              # Experiment logs
│   ├── exp13_rare_classes.log
│   └── exp14b_mlp_xgboost.log
│
└── cache/                             # Cached embeddings
    └── embeddings_mpnet.npy
```

---

## Running Experiments

### Generate All Plots
```bash
python3 generate_plots.py
# Output: plots/*.png (6 visualizations)
```

### Compile Results Summary
```bash
python3 compile_results.py
# Output: results/FULL_SUMMARY.json
```

### Run Rare Class Experiment
```bash
python3 experiments/exp13_rare_classes.py
# Tests 5 rare class focused configurations
# Results: results/rare_class/*.json
```

### Run Multi-Classifier Experiment
```bash
python3 experiments/exp14b_mlp_xgboost.py
# Tests LogReg, MLP, XGBoost, LightGBM
# Results: results/multiclassifier/exp14b_mlp_xgboost.json
```

---

## Statistical Summary

### Overall Performance

| Delta Range | Count | % of Total |
|-------------|-------|------------|
| > +5.0% | 5 | 13.2% |
| +4.0 to +5.0% | 6 | 15.8% |
| +3.0 to +4.0% | 11 | 28.9% |
| +1.0 to +3.0% | 8 | 21.1% |
| 0 to +1.0% | 3 | 7.9% |
| Negative | 5 | 13.2% |

### Significance Levels

| p-value | Count | % of Total |
|---------|-------|------------|
| < 0.001 | 8 | 21.1% |
| 0.001 - 0.01 | 16 | 42.1% |
| 0.01 - 0.05 | 6 | 15.8% |
| > 0.05 | 8 | 21.1% |

---

## Key Insights for Thesis

### Main Contributions

1. **Identified optimal prompting strategy**
   - Many-shot (10 examples) > few-shot (3) > zero-shot
   - Temperature=1.2 for best diversity/coherence balance

2. **Solved ESFJ rare class problem**
   - Required 20x oversampling + neural networks
   - Achieved +12.42% improvement (p<0.01)

3. **Proved neural networks superior for high-D embeddings**
   - MLP_512 >> LogReg for synthetic data exploitation
   - Tree-based models fail catastrophically with 768D

4. **Quantified limits of text-based augmentation**
   - ESFP: 0% improvement despite all efforts
   - May require multimodal features or domain knowledge

### Limitations

1. **ESFP remains unsolvable** with current approach
2. **Computational cost** of many-shot + neural networks
3. **API costs** increase significantly with 10 examples/generation
4. **Class imbalance** not fully solved (2/3 problem classes persist)

### Future Work

1. Test alternative embeddings (BERT, RoBERTa, domain-specific)
2. Explore multimodal features for ESFP
3. Try meta-learning or few-shot learning frameworks
4. Investigate if ESFP class definition is inherently ambiguous

---

## Citation

If using this work, please cite:

```
Phase G Validation: Advanced LLM-Based Data Augmentation for Problem Classes
Author: Benjamin
Institution: [Your Institution]
Date: December 2025
```

---

## Contact & Support

For questions or issues:
- See `TECHNICAL_DOCUMENTATION.md` for detailed methodology
- Check `RESULTS_SUMMARY.md` for quick reference
- Review generated plots in `plots/`
- Examine raw results in `results/FULL_SUMMARY.json`

---

**Phase G Complete**: 38 configurations tested, 30 significant, +5.98% best improvement, 1/3 rare classes solved.
