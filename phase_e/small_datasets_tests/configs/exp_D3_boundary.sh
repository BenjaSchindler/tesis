#!/bin/bash
# =============================================================================
# Experiment D3: Balanced Boundary Exploration
# =============================================================================
# Balanced settings: sim=0.88, contam=0.95, anchor=0.25
# Hypothesis: Sweet spot between quality and diversity
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "D3_boundary" "$SEED" \
        --llm-model gpt-4o-mini \
        --similarity-threshold 0.88 \
        --contamination-threshold 0.95 \
        --anchor-quality-threshold 0.25 \
        --min-classifier-confidence 0.08
done
