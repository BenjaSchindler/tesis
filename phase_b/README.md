# Phase B - Adaptive Weighting SMOTE-LLM

## Overview

Phase B builds on Phase A by adding **adaptive per-class synthetic weighting** based on baseline F1 performance. Classes with lower F1 scores receive higher weights to balance the dataset more effectively.

## Expected Performance

**Target**: +1.20% to +1.40% macro F1 improvement over baseline

**Previous Results**:
- Phase A (flat weighting): +1.00% ± 0.25%
- Phase B (adaptive weighting): Expected +0.20-0.40% additional improvement

## Key Difference from Phase A

### Synthetic Weighting Strategy

**Phase A (Flat Mode)**:
- All classes receive the same weight: **0.5**
- Simple, consistent approach

**Phase B (Adaptive Mode)**:
- **HIGH F1 classes** (≥0.60): weight = **0.05** (very low - already performing well)
- **MID F1 classes** (0.35-0.60): weight = **0.3** (medium - moderate boost needed)
- **LOW F1 classes** (<0.35): weight = **0.5** (high - need significant help)

This adaptive strategy focuses synthetic data generation on classes that need it most while avoiding over-augmentation of already strong classes.

## Configuration

All Phase A features PLUS:

### Enabled:
- `--enable-adaptive-weighting`
- Ensemble selection
- Validation gating (15% validation split, 2% tolerance)
- Anchor quality gating (threshold 0.50)
- Anchor selection with outlier removal (ratio 0.8, IQR 1.5)
- Adaptive quality filters
- F1-budget scaling
- Class descriptions

### Adaptive Weighting Logic

```python
if baseline_f1 >= 0.60:
    weight = 0.05  # HIGH F1 - minimal augmentation
elif baseline_f1 >= 0.35:
    weight = 0.3   # MID F1 - moderate augmentation
else:
    weight = 0.5   # LOW F1 - strong augmentation
```

## Local Execution

### Prerequisites

```bash
# Install dependencies
pip install numpy pandas scikit-learn sentence-transformers openai

# Set API key
export OPENAI_API_KEY='your-openai-api-key'
```

### Run with GPU (Recommended)

```bash
cd phase_b
./local_run_gpu.sh DATASET SEED

# Example:
./local_run_gpu.sh ../MBTI_500.csv 42
```

This script:
1. Checks for CUDA availability
2. Uses GPU for embedding generation if available
3. Falls back to CPU if no GPU detected
4. Runs complete Phase B experiment with adaptive weighting

### Output Files

- `phaseB_seed{SEED}_metrics.json` - Performance metrics with per-class weights
- `phaseB_seed{SEED}_synthetic.csv` - Generated synthetic data
- `phaseB_seed{SEED}_augmented.csv` - Combined training data

## Phase B Configuration Parameters

```bash
python3 ../core/runner_phase2.py \
    --data-path MBTI_500.csv \
    --test-size 0.2 \
    --random-seed 42 \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \  # or cpu
    --embedding-batch-size 64 \
    --llm-model gpt-4o-mini \
    --max-clusters 3 \
    --prompts-per-cluster 3 \
    --prompt-mode mix \

    # All Phase A features
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --anchor-quality-threshold 0.50 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --enable-adaptive-filters \
    --use-class-description \
    --f1-budget-scaling \
    --f1-high-threshold 0.35 \
    --f1-low-threshold 0.20 \

    # NEW: Adaptive weighting (Phase B)
    --enable-adaptive-weighting \
    --synthetic-weight 0.5 \  # Base weight (overridden by adaptive)
    --synthetic-weight-mode adaptive \

    # Quality filters
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95
```

## Understanding Adaptive Weighting

### Example MBTI Distribution

Typical MBTI baseline F1 distribution:

| Type | Baseline F1 | Category | Adaptive Weight | Synthetics (budget=100) | Effective Count |
|------|-------------|----------|----------------|------------------------|-----------------|
| ENFJ | 0.68 | HIGH | 0.05 | 100 | 5 |
| INFP | 0.62 | HIGH | 0.05 | 100 | 5 |
| INTJ | 0.55 | MID | 0.30 | 100 | 30 |
| ENTP | 0.48 | MID | 0.30 | 70 | 21 |
| ISTJ | 0.33 | LOW | 0.50 | 70 | 35 |
| ESFP | 0.28 | LOW | 0.50 | 100 | 50 |

**Result**: Low-performing classes receive more synthetic samples, helping balance the dataset more effectively than flat weighting.

### Why This Works

1. **Prevents over-fitting strong classes**: HIGH F1 classes get minimal augmentation (only 5%)
2. **Targeted improvement**: LOW F1 classes get maximum boost (50% of budget)
3. **Balanced approach**: MID F1 classes get moderate help (30%)
4. **Macro F1 optimization**: Focuses resources where they have most impact

## Troubleshooting Phase B

### Issue: Performance worse than Phase A

**Possible causes**:
1. **Over-augmentation of weak classes**: Too many low-quality synthetics
   - Solution: Reduce base synthetic weight from 0.5 to 0.4
   - Solution: Increase anchor quality threshold from 0.50 to 0.60

2. **Contamination in LOW F1 classes**: Harder to generate quality synthetics
   - Solution: Check contamination metrics in output JSON
   - Solution: Increase contamination threshold from 0.95 to 0.97

3. **Validation gating too aggressive**: Rejecting good synthetics
   - Solution: Increase val-tolerance from 0.02 to 0.03
   - Solution: Check validation metrics in output

### Issue: Not enough improvement over Phase A

**Possible causes**:
1. **Adaptive weights too conservative**: Not enough difference between tiers
   - Modify weights in runner_phase2.py:
     - HIGH: 0.05 → 0.03
     - MID: 0.30 → 0.40
     - LOW: 0.50 → 0.60

2. **F1 budget not scaled enough**: LOW classes not getting enough synthetics
   - Increase budget thresholds:
     - `--f1-high-threshold 0.35` → `0.40`
     - `--f1-low-threshold 0.20` → `0.25`

## Comparison: Phase A vs Phase B

| Feature | Phase A | Phase B |
|---------|---------|---------|
| Synthetic Weight | 0.5 (all classes) | 0.05/0.3/0.5 (adaptive) |
| Focus | Equal augmentation | Targeted augmentation |
| Complexity | Simple | Moderate |
| Expected Improvement | +1.00% | +1.20-1.40% |
| Best for | Initial validation | Production deployment |
| Robustness | High | Moderate (more tuning needed) |

## Next Steps

1. **Validate Phase B** with 5-10 different seeds
2. **Compare with Phase A** results to verify improvement
3. **Analyze per-class improvements** in metrics JSON
4. **Tune adaptive weights** if needed based on results
5. Consider **ensemble of Phase A + Phase B** for maximum robustness

## Additional Documentation

- [INSTRUCCIONES.md](INSTRUCCIONES.md) - Detailed Spanish instructions
- [COMANDOS.md](COMANDOS.md) - Command reference
- [../README.md](../README.md) - Project overview
- [../CLAUDE.md](../CLAUDE.md) - GCP deployment guide
