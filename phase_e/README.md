# Phase E - Clean Phase A Baseline with GPT-5-mini Support

Phase E is a clean snapshot of the Phase A baseline with added support for GPT-5-mini reasoning models and configurable Phase A improvements.

## Why Phase E?

- **Phase A is the winner**: +1.00% macro F1 improvement (25-seed validated)
- **Phase C/D all degraded**: Due to using evolving main `core/` with regressions
- **Phase E restores**: The clean Phase A baseline from `phase_a/core/`
- **Plus GPT-5-mini**: Native support for reasoning models with `reasoning_effort` parameter
- **Configurable improvements**: Phase A improvements can be enabled/disabled via CLI

## Key Features

### Clean Phase A Configuration
- F1-budget thresholds: 0.45, 0.20
- F1-budget multipliers: 0.0, 0.5, 1.0 (NOT 30, 70, 100)
- Anchor-quality threshold: 0.30 (NOT 0.50)
- Flat synthetic weight: 0.5
- All Phase 2 quality mechanisms enabled

### Phase A Improvements (Configurable)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-hard-anchors` | True | Enable 4th anchor strategy (hard anchors near decision boundary) |
| `--no-hard-anchors` | - | Disable hard anchors (use original 3-strategy ensemble) |
| `--deterministic-quality-gate` | True | Use seeded RNG for reproducible quality gate decisions |
| `--no-deterministic-quality-gate` | - | Disable seeded RNG (non-deterministic) |
| `--verbose-logging` | False | Enable detailed per-synthetic quality metrics |

### GPT-5-mini Support
For reasoning models (GPT-5-mini, o1, o3, etc.):
- Uses `reasoning_effort`: "low", "medium", "high"
- Uses `max_completion_tokens` instead of `max_tokens`
- Does NOT use `temperature` or `top_p`

For standard models (GPT-4o-mini, GPT-4o):
- Uses `temperature`, `top_p`, `max_tokens` as usual

## Directory Structure

```
phase_e/
  core/                 # Clean Phase A baseline (9 Python files)
    runner_phase2.py    # Main runner with GPT-5-mini support
    contamination_aware_filter.py
    enhanced_quality_gate.py
    ensemble_anchor_selector.py
    anchor_quality_improvements.py
    quality_gate_predictor.py
    mbti_class_descriptions.py
    multi_seed_ensemble.py
    adversarial_discriminator.py
  batch_scripts/        # GCP batch scripts (TBD)
  gcp/                  # GCP deployment scripts (TBD)
  results/              # Output files
  local_run.sh          # Single run script
  run_4variants.sh      # 4-variant comparison script
  README.md             # This file
```

## Usage

### Single Run
```bash
# With GPT-4o-mini (baseline)
./local_run.sh 42 gpt-4o-mini

# With GPT-5-mini (low reasoning effort)
./local_run.sh 42 gpt-5-mini low

# With GPT-5-mini (high reasoning effort)
./local_run.sh 42 gpt-5-mini high
```

### 4-Variant Comparison
```bash
# Run all 4 variants with seed 42
./run_4variants.sh 42
```

Variants:
- **A**: GPT-4o-mini (baseline)
- **B**: GPT-5-mini reasoning_effort=low
- **C**: GPT-5-mini reasoning_effort=medium
- **D**: GPT-5-mini reasoning_effort=high

## GPT-5-mini API Parameters

Per OpenAI documentation:
- `reasoning_effort`: Controls how much reasoning the model performs
  - `"low"`: Fast, cheap, minimal reasoning tokens
  - `"medium"`: Balanced (default)
  - `"high"`: Thorough, more expensive, more reasoning tokens
- `max_completion_tokens`: Replaces `max_tokens` for reasoning models

**Important**: Reasoning models do NOT use `temperature` or `top_p`.

## Expected Results

Phase E with GPT-4o-mini should replicate Phase A:
- Mean improvement: +1.00% +/- 0.25%
- Success rate: 80-90% of seeds
- Synthetics accepted: ~850 per run

GPT-5-mini variants are experimental - expected to:
- Potentially generate higher quality synthetics
- May have different acceptance rates
- Cost varies by reasoning_effort level

## Configuration Comparison

| Parameter | Phase A (Winner) | Phase C/D (Failed) |
|-----------|------------------|---------------------|
| f1-budget-multipliers | 0.0 0.5 1.0 | 30 70 100 |
| anchor-quality-threshold | 0.30 | 0.50 |
| Core | phase_a/core/ (clean) | core/ (regressed) |

## Created

2025-11-25

## Reference

- [Phase Results Analysis](../.claude/plans/phase-results-analysis.md)
- [OpenAI Reasoning Guide](https://platform.openai.com/docs/guides/reasoning)
