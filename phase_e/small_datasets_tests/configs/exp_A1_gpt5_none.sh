#!/bin/bash
# =============================================================================
# Experiment A1: GPT-5-mini with reasoning_effort=none
# =============================================================================
# Hypothesis: Fast baseline with no reasoning overhead
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "A1_gpt5_none" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort none \
        --output-verbosity medium
done
