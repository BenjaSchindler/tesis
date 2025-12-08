#!/bin/bash
# =============================================================================
# Experiment A3: GPT-5-mini with reasoning_effort=medium
# =============================================================================
# Hypothesis: More thoughtful generation improves quality
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "A3_gpt5_medium" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort medium \
        --output-verbosity high
done
