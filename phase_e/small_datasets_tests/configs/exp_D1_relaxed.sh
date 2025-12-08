#!/bin/bash
# =============================================================================
# Experiment D1: Relaxed Filters
# =============================================================================
# Lower similarity (0.85), lower contamination (0.90), lower anchor quality (0.20)
# Hypothesis: Allow more "risky" texts that may help boundary exploration
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "D1_relaxed" "$SEED" \
        --llm-model gpt-4o-mini \
        --similarity-threshold 0.85 \
        --contamination-threshold 0.90 \
        --anchor-quality-threshold 0.20 \
        --min-classifier-confidence 0.05
done
