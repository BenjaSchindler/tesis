#!/bin/bash
# =============================================================================
# Experiment F2: Length-Aware Generation (500 words)
# =============================================================================
# Target 500 words per synthetic (matching real data average)
# Hypothesis: Full-length texts match real data distribution better
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "F2_length_500" "$SEED" \
        --llm-model gpt-4o-mini \
        --length-aware \
        --length-target-words 500 \
        --llm-max-tokens 4000
done
