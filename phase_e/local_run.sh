#!/bin/bash
# Phase E - Local Run Script
# Clean Phase A baseline with GPT-5-mini support
#
# Usage: ./local_run.sh [SEED] [MODEL]
#   SEED: Random seed (default: 42)
#   MODEL: gpt-4o-mini, gpt-5-mini (default: gpt-4o-mini)
#
# Examples:
#   ./local_run.sh                      # Seed 42, gpt-4o-mini
#   ./local_run.sh 123                  # Seed 123, gpt-4o-mini
#   ./local_run.sh 42 gpt-5-mini        # Seed 42, gpt-5-mini with default reasoning_effort=low

set -e

SEED="${1:-42}"
MODEL="${2:-gpt-4o-mini}"
REASONING_EFFORT="${3:-low}"  # low, medium, high (only for gpt-5-mini)

# Verify API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Change to phase_e directory
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════"
echo "  Phase E - Clean Phase A Baseline"
echo "  Model: $MODEL"
echo "  Seed: $SEED"
if [[ "$MODEL" == *"gpt-5"* ]]; then
    echo "  Reasoning Effort: $REASONING_EFFORT"
fi
echo "════════════════════════════════════════════════════════"
echo ""

# Build command
CMD="python3 core/runner_phase2.py \
    --data-path ../MBTI_500.csv \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cpu \
    --embedding-batch-size 64 \
    --llm-model $MODEL \
    --max-clusters 3 \
    --prompts-per-cluster 3 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --anchor-quality-threshold 0.30 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --enable-adaptive-filters \
    --use-class-description \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.45 0.20 \
    --f1-budget-multipliers 0.0 0.5 1.0 \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --synthetic-output results/phaseE_${MODEL}_seed${SEED}_synthetic.csv \
    --augmented-train-output results/phaseE_${MODEL}_seed${SEED}_augmented.csv \
    --metrics-output results/phaseE_${MODEL}_seed${SEED}_metrics.json"

# Add GPT-5-mini specific parameters
if [[ "$MODEL" == *"gpt-5"* ]]; then
    CMD="$CMD \
    --reasoning-effort $REASONING_EFFORT \
    --max-completion-tokens 1024"
fi

# Run
echo "Running command..."
eval $CMD

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase E Complete!"
echo "  Results saved to:"
echo "    - results/phaseE_${MODEL}_seed${SEED}_metrics.json"
echo "    - results/phaseE_${MODEL}_seed${SEED}_synthetic.csv"
echo "════════════════════════════════════════════════════════"
