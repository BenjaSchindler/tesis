#!/bin/bash
# =============================================================================
# Experiment F1: Length-Aware Generation (300 words)
# =============================================================================
# Target 300 words per synthetic (vs ~40 current)
# Hypothesis: Medium-length texts are more natural and useful
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "F1_length_300" "$SEED" \
        --llm-model gpt-4o-mini \
        --length-aware \
        --length-target-words 300 \
        --llm-max-tokens 2000
done
