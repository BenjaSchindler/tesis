# Phase A - Baseline SMOTE-LLM Configuration

## Overview

Phase A implements the baseline SMOTE-LLM configuration with all quality control mechanisms enabled EXCEPT adaptive weighting. All classes receive the same synthetic weight (0.5).

## Expected Performance

**Target**: +1.00% to +1.20% macro F1 improvement over baseline

**Validated Results** (25-seed robustness test):
- Mean improvement: +1.00% ± 0.25%
- Success rate: 80-90% of seeds show improvement
- Statistical significance: p < 0.05

## Configuration

### Core Features Enabled

1. **Ensemble-based Anchor Selection**
   - Multiple anchor candidates per cluster
   - Quality-based selection using ensemble voting

2. **Validation-based Quality Gating**
   - Holds out 15% of training data for validation
   - Filters synthetics that degrade validation performance
   - Tolerance: 2% (rejects if val F1 drops > 2%)

3. **Anchor Quality Gating**
   - Quality threshold: 0.50
   - Filters low-quality anchor samples

4. **Anchor Selection with Outlier Removal**
   - Selection ratio: 0.8 (keeps 80% of samples)
   - Outlier threshold: 1.5 IQR

5. **Adaptive Quality Filters**
   - Similarity threshold: 0.90 (max cosine similarity to training data)
   - Min classifier confidence: 0.10
   - Contamination threshold: 0.95 (BLEU score limit)

6. **F1-Budget Scaling**

   **Current GPU Run (2025-11-15):**
   - HIGH F1 (≥0.35): 30 synthetics per class
   - MID F1 (0.20-0.35): 70 synthetics per class
   - LOW F1 (<0.20): 100 synthetics per class

   **Validated Config (Batch 5 Phase A, 2025-11-12):**
   - HIGH F1 (≥0.45): 30 synthetics per class
   - MID F1 (0.20-0.45): 70 synthetics per class
   - LOW F1 (<0.20): 100 synthetics per class

   Note: GPU runs use threshold 0.35 (from earlier TIER S experiments).
   If results underperform, will switch to validated 0.45 threshold.

7. **Class Descriptions**
   - Uses MBTI personality type descriptions in prompts

### Key Difference from Phase B

**Synthetic Weighting**:
- Phase A: **Flat mode** - All classes get weight = 0.5
- Phase B: **Adaptive mode** - Per-class weights (0.05/0.3/0.5) based on baseline F1

## Local Execution

### Prerequisites

```bash
# Install dependencies
pip install numpy pandas scikit-learn sentence-transformers openai

# Set API key
export OPENAI_API_KEY='your-openai-api-key'
```

### Run Single Experiment

```bash
cd phase_a
./local_run.sh [SEED]

# Examples:
./local_run.sh 42
./local_run.sh 100
```

### Output Files

- `phaseA_seed{SEED}_metrics.json` - Performance metrics
- `phaseA_seed{SEED}_synthetic.csv` - Generated synthetic data
- `phaseA_seed{SEED}_augmented.csv` - Combined training data

## GCP Deployment (25-Seed Robustness Test)

### Launch 5 VMs × 5 Seeds = 25 Experiments

```bash
cd phase_a/gcp

# Set API key
export OPENAI_API_KEY='your-api-key'

# Launch all experiments
./launch_25seeds.sh
```

**Configuration**:
- 5 VMs (vm-batch1 through vm-batch5)
- Machine type: n1-standard-4 (4 vCPUs, 15GB RAM)
- Seeds:
  - batch1: 42, 100, 123, 456, 789
  - batch2: 111, 222, 333, 444, 555
  - batch3: 1000, 2000, 3000, 4000, 5000
  - batch4: 7, 13, 21, 37, 101
  - batch5: 1234, 2345, 3456, 4567, 5678
- Estimated time: 4-5 hours
- Estimated cost: ~$10 USD

### Monitor Progress

```bash
./monitor.sh
# Updates every 60 seconds
# Press Ctrl+C to exit
```

### Collect Results

```bash
# After ~5 hours
./collect_results.sh

# Will download all results and compute statistics:
# - Mean delta, std, median, percentiles
# - 95% confidence interval
# - Statistical significance (t-test)
# - Success rate
```

## Detailed Configuration Parameters

```bash
--data-path MBTI_500.csv
--test-size 0.2
--embedding-model sentence-transformers/all-mpnet-base-v2
--device cpu
--embedding-batch-size 32
--llm-model gpt-4o-mini
--max-clusters 3
--prompts-per-cluster 3
--prompt-mode mix

# Quality mechanisms
--use-ensemble-selection
--use-val-gating
--val-size 0.15
--val-tolerance 0.02
--enable-anchor-gate
--anchor-quality-threshold 0.50
--enable-anchor-selection
--anchor-selection-ratio 0.8
--anchor-outlier-threshold 1.5
--enable-adaptive-filters
--use-class-description

# F1-budget scaling
--f1-budget-scaling
--f1-high-threshold 0.35
--f1-low-threshold 0.20

# Synthetic weighting (FLAT MODE - Phase A)
--synthetic-weight 0.5
--synthetic-weight-mode flat

# Quality filters
--similarity-threshold 0.90
--min-classifier-confidence 0.10
--contamination-threshold 0.95
```

## Interpretation of Results

### Success Criteria

1. **Mean improvement ≥ +1.00%**: Primary goal
2. **Statistical significance**: p-value < 0.05
3. **Success rate ≥ 70%**: Most seeds should improve
4. **Low variance**: Std < 0.30% indicates robustness

### Typical Results

From 25-seed validation:

```
Mean Delta:        +1.00% ± 0.25%
Median Delta:      +0.98%
95% CI:            [+0.90%, +1.10%]
Success Rate:      21/25 seeds improved
p-value:           0.001 (highly significant)
```

### Common Issues

1. **Low improvement (<0.5%)**:
   - Too few synthetics generated
   - Quality filters too aggressive
   - Anchor quality threshold too high

2. **Negative improvement**:
   - Contamination (synthetics too similar to test data)
   - Poor anchor selection
   - Validation gating failed to filter bad synthetics

3. **High variance (>0.5%)**:
   - Sensitive to seed selection
   - Dataset splits vary significantly
   - LLM generation inconsistent across runs

## Next Steps

After validating Phase A performance (≥ +1.00%), proceed to:

**Phase B**: Enable adaptive weighting for per-class synthetic weights
- Target: +1.20% to +1.40% macro F1 improvement
- See `../phase_b/README.md`
