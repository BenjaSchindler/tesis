#!/bin/bash
# =============================================================================
# Experiment D2: Strict Filters
# =============================================================================
# High similarity (0.95), high contamination (0.98), high anchor quality (0.40)
# Hypothesis: Only accept highest quality synthetics
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "D2_strict" "$SEED" \
        --llm-model gpt-4o-mini \
        --similarity-threshold 0.95 \
        --contamination-threshold 0.98 \
        --anchor-quality-threshold 0.40 \
        --min-classifier-confidence 0.20
done
