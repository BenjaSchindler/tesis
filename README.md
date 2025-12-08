# SMOTE-LLM: LLM-Based Synthetic Data Generation for Text Classification

An advanced synthetic data augmentation pipeline that leverages Large Language Models (LLMs) to generate high-quality synthetic training samples for imbalanced text classification tasks.

## Overview

SMOTE-LLM combines clustering-based anchor selection with LLM-powered text generation to create synthetic training data that improves model performance on minority classes. The system includes sophisticated quality control mechanisms to ensure generated samples are diverse, relevant, and free from contamination.

## Key Features

- **Ensemble-based Anchor Selection**: Selects representative samples using clustering and quality scoring
- **Validation-based Quality Gating**: Filters synthetics that degrade validation performance
- **Adaptive Quality Filters**: Multi-layer filtering (semantic similarity, classifier confidence, BLEU contamination)
- **F1-Budget Scaling**: Allocates more synthetics to low-performing classes
- **Per-Class Adaptive Weighting**: Dynamically adjusts synthetic sample weights based on baseline F1
- **Contamination Prevention**: Ensures generated samples don't leak test information

## Project Structure

```
SMOTE-LLM/
├── README.md                # This file
├── CLAUDE.md                # GCP deployment guide
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── MBTI_500.csv            # Dataset (106K MBTI personality samples)
│
├── core/                   # Shared modules
│   ├── runner_phase2.py               # Main pipeline orchestrator
│   ├── ensemble_anchor_selector.py    # Anchor selection logic
│   ├── contamination_aware_filter.py  # Anti-contamination filtering
│   ├── enhanced_quality_gate.py       # Quality threshold gating
│   ├── anchor_quality_improvements.py # Anchor quality scoring
│   ├── quality_gate_predictor.py      # Validation-based gating
│   └── mbti_class_descriptions.py     # MBTI type descriptions
│
├── phase_a/                # Phase A: Baseline configuration
│   ├── README.md           # Phase A documentation
│   ├── local_run.sh        # Local execution script
│   ├── gcp/                # GCP cloud deployment
│   │   ├── launch_25seeds.sh    # Launch 5 VMs × 5 seeds
│   │   ├── monitor.sh           # Monitor progress
│   │   └── collect_results.sh   # Collect and analyze results
│   └── batch_scripts/      # VM execution scripts
│       ├── run_batch_batch1.sh  # Seeds: 42, 100, 123, 456, 789
│       ├── run_batch_batch2.sh  # Seeds: 111, 222, 333, 444, 555
│       ├── run_batch_batch3.sh  # Seeds: 1000, 2000, 3000, 4000, 5000
│       ├── run_batch_batch4.sh  # Seeds: 7, 13, 21, 37, 101
│       └── run_batch_batch5.sh  # Seeds: 1234, 2345, 3456, 4567, 5678
│
├── phase_b/                # Phase B: Adaptive weighting
│   ├── README.md           # Phase B documentation
│   ├── INSTRUCCIONES.md    # Spanish instructions
│   ├── COMANDOS.md         # Command reference
│   └── local_run_gpu.sh    # GPU execution script
│
├── phase_c/                # Phase C: MID-tier optimization (SOTA 2024-2025) ✨ NEW
│   ├── README.md           # Phase C documentation
│   ├── IMPLEMENTATION_SUMMARY.md  # Implementation details
│   └── local_run_phaseC.sh # Test script with adaptive temperature
│
├── docs/                   # Technical documentation
│   ├── README.md           # Documentation index
│   ├── 00_RESUMEN_EJECUTIVO.md      # Executive summary (Spanish)
│   ├── 01_TECHNICAL_OVERVIEW.md     # System architecture
│   ├── 02_PROBLEM_STATEMENT.md      # Core problems & theory
│   ├── 04_PARAMETER_DEEP_DIVE.md    # Parameter justification (45 pages)
│   ├── 01-05_*.md          # Methodology, results, analysis (17+ docs)
│   └── ...                 # Problems, solutions, validation
│
└── scripts/                # Analysis and visualization
    ├── README.md           # Scripts documentation
    └── generate_all_plots.py # Generate visualizations
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### Phase A: Baseline (Flat Weighting)

Run a single experiment locally:

```bash
cd phase_a
./local_run.sh 42
```

Launch 25-seed robustness test on GCP (recommended):

```bash
cd phase_a/gcp
export OPENAI_API_KEY='your-key'
./launch_25seeds.sh

# Monitor progress (in new terminal)
./monitor.sh

# Collect results after ~5 hours
./collect_results.sh
```

**Expected Result**: +1.00% ± 0.25% macro F1 improvement

### Phase B: Advanced (Adaptive Weighting)

Run with GPU acceleration:

```bash
cd phase_b
./local_run_gpu.sh ../MBTI_500.csv 42
```

**Expected Result**: +1.20% to +1.40% macro F1 improvement

### Phase C: MID-Tier Optimization (SOTA 2024-2025) ✨ NEW

Addresses MID-tier class degradation using state-of-the-art techniques:

```bash
cd phase_c
./local_run_phaseC.sh ../MBTI_500.csv 42
```

**Key Feature**: Adaptive temperature (0.5 for MID-tier vs 1.0 default)
**Expected Result**: MID-tier +0.10% to +0.25%, Overall +1.10% to +1.25%

## Phase Comparison

| Aspect | Phase A | Phase B | Phase C ✨ |
|--------|---------|---------|----------|
| **Weighting** | Flat (0.5) | Adaptive (0.05/0.1/0.5) | Adaptive (0.05/0.2/0.5) |
| **Temperature** | Fixed (1.0) | Fixed (1.0) | **Adaptive (0.3/0.5/0.8)** |
| **MID-tier Result** | -0.59% | -0.59% | **+0.10% to +0.25%** ⬆️ |
| **Overall Gain** | +1.00% | +1.00% | +1.10% to +1.25% |
| **Complexity** | Simple | Moderate | Moderate |
| **Robustness** | High | Moderate | Moderate-High |
| **Best Use** | Initial validation | Production (if MID-tier OK) | **MID-tier fix** |
| **Research Basis** | Established | Standard | **SOTA 2024-2025** |
| **Cost per seed** | ~$0.50 | ~$0.50 | ~$0.50 |

### When to Use Each Phase

**Use Phase A if**:
- You want to validate the SMOTE-LLM approach works for your dataset
- You need high robustness and reproducibility
- You're running initial experiments
- Budget is limited

**Use Phase B if**:
- Phase A showed positive results (+0.8% or better)

**Use Phase C if** ✨:
- Phase A/B showed **MID-tier degradation** (classes with F1 0.20-0.45 degrade)
- You want to apply **SOTA 2024-2025 research** findings
- You need to **fix the "vulnerable zone" problem**
- You're willing to test adaptive temperature approach
- Your dataset has significant class imbalance
- You can tune adaptive weight thresholds
- You want maximum performance gain

## Dataset

**MBTI_500.csv**:
- 106,396 samples
- 16 classes (MBTI personality types)
- Text classification task
- Imbalanced distribution (some types are rare)

Classes: INFP, INFJ, INTP, INTJ, ENFP, ENFJ, ENTP, ENTJ, ISFP, ISFJ, ISTP, ISTJ, ESFP, ESFJ, ESTP, ESTJ

## Configuration Details

### Core Architecture

1. **Baseline Classifier**: Logistic Regression with TF-IDF features
2. **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768 dims)
3. **Clustering**: K-Means with k=3 per class
4. **LLM**: OpenAI GPT-4o-mini
5. **Prompts**: 3 prompts per cluster (systematic, creative, mixed)

### Quality Control Mechanisms

1. **Ensemble Anchor Selection**
   - Selects 80% of samples per cluster
   - Removes outliers using 1.5 IQR threshold
   - Quality threshold: 0.50

2. **Validation Gating**
   - Holds out 15% of training data
   - Rejects synthetics if validation F1 drops >2%

3. **Contamination Filters**
   - Max semantic similarity: 0.90 (cosine distance to training data)
   - Min classifier confidence: 0.10
   - BLEU contamination threshold: 0.95

4. **F1-Budget Scaling**
   - HIGH F1 (≥0.35): 30 synthetics
   - MID F1 (0.20-0.35): 70 synthetics
   - LOW F1 (<0.20): 100 synthetics

## Results Interpretation

### Success Metrics

1. **Macro F1 improvement**: Primary metric (target: ≥ +1.00%)
2. **Statistical significance**: p-value < 0.05 (t-test vs baseline)
3. **Success rate**: ≥70% of seeds show improvement
4. **Robustness**: Std dev < 0.30%

### Example Output

```json
{
  "baseline": {
    "macro_f1": 0.5234,
    "weighted_f1": 0.6123
  },
  "augmented": {
    "macro_f1": 0.5286,
    "weighted_f1": 0.6198
  },
  "improvement": {
    "f1_delta": 0.0052,
    "f1_delta_pct": 1.00
  },
  "synthetic_data": {
    "accepted_count": 847,
    "per_class_weights": {...}
  }
}
```

## Deployment Options

### Local Execution
- Best for: Testing, development
- Requirements: 8GB RAM, ~2-3 hours
- Cost: API calls only (~$0.50 per run)

### GCP Cloud (5 VMs)
- Best for: Multi-seed validation
- Requirements: GCP account, gcloud CLI
- Runs: 25 experiments in parallel
- Time: ~5 hours
- Cost: ~$10 USD total
- See [CLAUDE.md](CLAUDE.md) for detailed guide

## Dependencies

```txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
openai>=1.0.0
python-dotenv>=1.0.0
```

## Troubleshooting

### Low Improvement (<0.5%)

**Possible causes**:
- Too few synthetics generated → Increase budget
- Quality filters too aggressive → Lower thresholds
- Poor anchor selection → Increase anchor selection ratio
- LLM generation quality → Check prompt engineering

### Negative Improvement

**Possible causes**:
- Contamination → Increase contamination threshold
- Overfitting → Reduce synthetic count
- Validation gating failed → Check validation metrics
- Dataset too small → Need more training data

### High Variance (>0.5% std)

**Possible causes**:
- Dataset splits inconsistent → Use stratified splits
- LLM generation non-deterministic → Temperature is too high
- Sensitive to seed → Run more seeds for statistical power

## Advanced Tuning

### Adjusting Adaptive Weights (Phase B)

Edit [runner_phase2.py:1523](core/runner_phase2.py#L1523):

```python
# Current thresholds
if baseline_f1 >= 0.60:
    weight = 0.05  # HIGH F1
elif baseline_f1 >= 0.35:
    weight = 0.3   # MID F1
else:
    weight = 0.5   # LOW F1

# More aggressive variant
if baseline_f1 >= 0.60:
    weight = 0.03  # Reduce HIGH even more
elif baseline_f1 >= 0.35:
    weight = 0.40  # Increase MID
else:
    weight = 0.60  # Increase LOW
```

### Modifying F1-Budget Scaling

Edit command-line arguments:

```bash
# Default
--f1-high-threshold 0.35 --f1-low-threshold 0.20

# More aggressive (more synthetics for weak classes)
--f1-high-threshold 0.40 --f1-low-threshold 0.25
```

## Citation

If you use this code, please cite:

```bibtex
@misc{smote-llm-2025,
  title={SMOTE-LLM: LLM-Based Synthetic Data Generation for Imbalanced Text Classification},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/SMOTE-LLM}}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test changes with Phase A validation
4. Submit pull request with results

## Contact

For questions or issues:
- GitHub Issues: [github.com/yourusername/SMOTE-LLM/issues](https://github.com/yourusername/SMOTE-LLM/issues)
- Email: your.email@domain.com

## Acknowledgments

- OpenAI GPT-4o-mini for synthetic text generation
- sentence-transformers for embedding models
- scikit-learn for baseline classifiers
- MBTI dataset contributors
