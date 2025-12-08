# Phase G: Improving Problematic Classes

## Objective

Improve F1 scores for classes that consistently fail to improve despite augmentation:

| Class | Samples | Current Issue | Priority |
|-------|---------|---------------|----------|
| **ENFJ** | 190 | Gets 31-41 synth, stays F1=0.00 | HIGH |
| **ESFJ** | 42 | Gets 8-11 synth, stays F1=0.00 | HIGH |
| **ESFP** | 48 | Gets 0 synth (purity=0.009) | HIGH |
| **ESTJ** | 39 | Gets 1 synth, stays F1=0.00 | HIGH |
| **ISTJ** | 205 | Gets 35-45 synth, stays F1=0.05 | MEDIUM |

## Phase F Findings (Why They Fail)

### 1. ESFP - Zero Generation Problem
- **Root cause:** Anchor purity = 0.009 (too low)
- **Filter behavior:** No samples pass quality gate
- **Evidence:** `ESFP: accepted=0/0 (no valid samples)`

### 2. ENFJ/ESFJ/ESTJ - Generation Without Impact
- **Root cause:** Synthetic quality insufficient to shift classifier boundary
- **Evidence:** ENFJ gets 41 samples in ENS_Top3_G5 but F1 stays 0.00
- **Hypothesis:**
  - Samples too similar to majority classes
  - Not enough samples to overcome class imbalance (190 vs 1832 INFP)

### 3. ISTJ - Augmentation Ineffective
- **Root cause:** Unknown - needs investigation
- **Evidence:** 45 synth samples, F1 stays at 0.0465 (tiny baseline)

## Counter-intuitive Findings from Phase F

1. **Classes that improve have LOWER quality scores:**
   - ESTP improves +10.5% with avg quality 0.42
   - ISFJ improves +19.9% with avg quality 0.45

2. **Classes that don't improve have HIGHER quality scores:**
   - ENFJ stays 0.00 with avg quality 0.52
   - ISTJ stays 0.05 with avg quality 0.48

3. **Indirect effects exist:**
   - INFJ improves +0.85% with 0 synthetic samples
   - INTJ degrades -1.19% with 0 synthetic samples

## Proposed Strategies

### Strategy A: Lower Quality Thresholds
**Hypothesis:** Overly strict quality gates reject good diversity

```bash
--min-classifier-confidence 0.02  # Lower from 0.05
--quality-threshold 0.30          # If exists
```

### Strategy B: Increase Sample Volume
**Hypothesis:** Need critical mass to overcome class imbalance

```bash
--prompts-per-cluster 15          # Up from 9
--samples-per-prompt 8            # Up from 5
--f1-budget-multipliers 0.0 1.0 5.0  # Higher multiplier for worst classes
```

### Strategy C: Class-Specific Prompting
**Hypothesis:** Generic prompts don't capture class distinctiveness

- Implement per-class prompt templates in `mbti_class_descriptions.py`
- Add contrastive examples (what makes ENFJ different from INFJ)

### Strategy D: Relaxed Anchor Selection for Problem Classes
**Hypothesis:** ESFP fails due to strict anchor purity

```bash
--min-anchor-purity 0.005         # Lower from default
--anchor-selection-method "relaxed"
```

### Strategy E: Target Only Problem Classes
**Hypothesis:** Focus resources on classes that need help

```bash
--target-classes ENFJ,ESFJ,ESFP,ESTJ,ISTJ
--f1-threshold-skip 0.10          # Only skip if F1 > 0.10
```

### Strategy F: Multi-Round Generation
**Hypothesis:** Iterative refinement helps quality

1. Generate first batch
2. Train intermediate classifier
3. Use intermediate classifier to filter
4. Generate second batch targeting remaining gaps

## Experiments to Run

### Wave 1: Quick Tests (1 hour each)
| Config | Strategy | Key Change |
|--------|----------|------------|
| G1_low_conf | A | min-classifier-confidence=0.02 |
| G2_high_vol | B | prompts=15, samples=8 |
| G3_relaxed_anchor | D | min-anchor-purity=0.005 |

### Wave 2: Combined Approaches (2 hours each)
| Config | Strategy | Key Changes |
|--------|----------|-------------|
| G4_combo_AB | A+B | Low conf + High volume |
| G5_target_only | E | Target only problem classes |

### Wave 3: Advanced (if needed)
| Config | Strategy | Key Changes |
|--------|----------|-------------|
| G6_multi_round | F | 2-pass generation |
| G7_contrastive | C | Custom prompts per class |

## Success Metrics

| Metric | Target |
|--------|--------|
| ENFJ F1 | > 0.05 (any improvement from 0) |
| ESFJ F1 | > 0.05 |
| ESFP F1 | > 0.05 |
| ESTJ F1 | > 0.05 |
| ISTJ F1 | > 0.10 (double baseline) |
| Macro F1 | Maintain > +1.5% overall improvement |

## Files

```
phase_g/
├── core/              # Copy of Phase F core (frozen)
├── configs/           # New experiment configurations
├── results/           # Output files
├── kfold_evaluator.py # K-Fold evaluation script
├── base_config.sh     # Base configuration template
└── README.md          # This file
```

## Usage

```bash
cd phase_g
export OPENAI_API_KEY='...'
SEED=42 bash configs/G1_low_conf.sh
python3 kfold_evaluator.py --config G1_low_conf --seed 42 --k 5 --repeated 3
```

## Notes

- Core is a copy from Phase F to avoid contamination
- All experiments use K-Fold CV (established as essential in Phase F)
- Start with single seed (42), expand to multi-seed if promising
