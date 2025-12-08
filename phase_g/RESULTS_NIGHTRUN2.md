# Phase G - Nightrun2 Results (December 2025)

## Overview

20 experimental configurations (W1-W9 series) testing various strategies.
Evaluated with K-fold cross-validation using kfold_multimodel.py.

**Run Date:** 2025-12-05/06 (overnight)
**Evaluation:** K-fold (k=5, repeats=3) with LogisticRegression

---

## NEW RECORD: ENS_SUPER_G5_F7_v2 (+10.03%)

**Date:** 2025-12-06 | **Status:** BREAKTHROUGH

| Metric | Value |
|--------|-------|
| **Delta** | **+10.03%** |
| **Synthetics** | 327 |
| **p-value** | 0.000003 |
| **Win Rate** | 100% (15/15) |

**Components:**
- ENS_Top3_G5 (178 synth) - Phase G best ensemble (CMB3+CF1+V4+G5)
- W1_force_problem (48 synth) - Forces problem classes
- EXP7_hybrid_best (61 synth) - Phase F best config
- W3_no_dedup (40 synth) - No deduplication, ENTJ-safe

**Key Breakthroughs:**
- ESTP: 0.000 → 0.082 (+8.2pp) - Class now detectable!
- ESFJ: 0.000 → 0.024 (+2.4pp) - Class now detectable!
- ENTJ: 0.059 → 0.125 (+6.6pp)
- ISTJ: 0.049 → 0.114 (+6.5pp)

**Evolution:**
| Version | Synth | Delta | Added Component |
|---------|-------|-------|-----------------|
| ENS_Top3_G5 | 178 | +5.98% | Base |
| ENS_SUPER_G5_F7 | 287 | +9.00% | +W1_force_problem +EXP7 |
| ENS_SUPER_G5_F7_v2 | 327 | +10.03% | +W3_no_dedup |

See: `results/ensembles_v2/ENSEMBLE_PLAN.md` for full details.

---

## Results Summary (Sorted by Delta)

| Config | Synth | Delta | p-value | Win Rate | Significant |
|--------|-------|-------|---------|----------|-------------|
| **ENS_SUPER_G5_F7_v2** | 327 | **+10.03%** | 0.000003 | 100% | ✓ |
| **ENS_SUPER_G5_F7** | 287 | **+9.00%** | 0.000016 | 100% | ✓ |
| **ENS_Top3_G5** | 178 | **+5.98%** | 0.000000 | 100% | ✓ |
| **W9_contrastive** | 43 | **+2.69%** | 0.000265 | 93% | ✓ |
| **W1_low_gate** | 48 | **+2.56%** | 0.000547 | 93% | ✓ |
| W2_ultra_vol | 50 | +2.46% | 0.007351 | 73% | ✓ |
| G5_K25_medium | 41 | +2.46% | 0.001097 | 80% | ✓ |
| CF1_conf_band | 46 | +2.46% | 0.000160 | 93% | ✓ |
| W4_target_only | 39 | +2.37% | 0.002090 | 80% | ✓ |
| W3_permissive_filter | 33 | +2.36% | 0.000571 | 87% | ✓ |
| **W1_force_problem** | 48 | +2.29% | 0.002412 | 80% | ✓ |
| W2_mega_vol | 47 | +2.26% | 0.002127 | 87% | ✓ |
| W1_no_gate | 46 | +1.87% | 0.001146 | 87% | ✓ |
| W5_many_shot_10 | 41 | +1.87% | 0.000014 | 100% | ✓ |
| W9_best_combo | 52 | +1.76% | 0.001683 | 80% | ✓ |
| W7_yolo | 33 | +1.75% | 0.005034 | 80% | ✓ |
| W7_yolo_force | 52 | +1.60% | 0.016113 | 80% | ✓ |
| W5_few_shot_3 | 45 | +1.56% | 0.018514 | 73% | ✓ |
| W3_no_dedup | 40 | +1.54% | 0.007620 | 87% | ✓ |
| W5_zero_shot | 32 | +1.43% | 0.020925 | 73% | ✓ |
| V4_ultra | 47 | +1.22% | 0.001149 | 87% | ✓ |
| CMB3_skip | 44 | +1.17% | 0.012280 | 67% | ✓ |
| W8_gpt5_high | 47 | +0.80% | 0.050071 | 67% | - |
| W8_gpt5_reasoning | 50 | +0.65% | 0.113454 | 60% | - |

## Key Statistics

- **Total configs:** 22
- **Positive delta:** 22/22 (100%)
- **Statistically significant (p<0.05):** 20/22 (91%)

## Config Series Descriptions

### W1: Gate Threshold Experiments
- `W1_no_gate`: No quality gate filtering
- `W1_low_gate`: Low threshold gate (permissive)
- `W1_force_problem`: Force generation for problematic classes

### W2: Volume Experiments
- `W2_ultra_vol`: Ultra high volume generation
- `W2_mega_vol`: Maximum volume attempt

### W3: Filter Experiments
- `W3_permissive_filter`: Relaxed contamination/similarity filters
- `W3_no_dedup`: No deduplication of synthetics

### W4: Target Experiments
- `W4_target_only`: Target only specific classes

### W5: Prompt Experiments
- `W5_zero_shot`: Zero-shot prompting
- `W5_few_shot_3`: 3-shot prompting
- `W5_many_shot_10`: 10-shot prompting

### W7: YOLO Experiments
- `W7_yolo`: Aggressive generation, minimal filtering
- `W7_yolo_force`: Even more aggressive

### W8: Model Experiments
- `W8_gpt5_high`: Using gpt-5-mini with high temp
- `W8_gpt5_reasoning`: Using gpt-5-mini reasoning mode

### W9: Combination Experiments
- `W9_contrastive`: Contrastive prompting strategy ⭐
- `W9_best_combo`: Best parameter combination

## Notable Findings

### 1. W9_contrastive is Best Individual (+2.69%)
Uses contrastive prompting to generate more distinctive synthetics.

### 2. W1_force_problem Protects ENTJ
Only config that improves ENTJ (+13.1%) instead of degrading it.
Per-class metrics:
- ISFJ: +27.1%
- ESTP: +11.1%
- ENTJ: +13.1% ✓

### 3. GPT-5 Models Underperform
W8_gpt5_high and W8_gpt5_reasoning have lowest deltas, not significant.

### 4. LogisticRegression vs RandomForest
Results with LogisticRegression are higher than previous RandomForest evaluations.
ENS_Top3_G5: +5.98% (LR) vs +1.29% (RF)

## Per-Class Analysis (Simple Metrics)

| Config | ISFJ | ESTP | ENTJ | INTJ |
|--------|------|------|------|------|
| W1_force_problem | +27% | +11% | **+13%** | 0% |
| W2_ultra_vol | +50% | +21% | -27% | 0% |
| W9_contrastive | +27% | +18% | -12% | 0% |
| W3_no_dedup | +3% | +18% | **+12%** | 0% |

**ENTJ-safe configs:** W1_force_problem, W3_no_dedup

## Files

- Results JSON: `results/kfold_nightrun2_results.json`
- Synthetic CSVs: `results/W*_s42_synth.csv`
- Config scripts: `configs/W*.sh`
- Run script: `run_nightrun2.sh`
