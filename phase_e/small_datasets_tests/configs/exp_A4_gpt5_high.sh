#!/bin/bash
# =============================================================================
# Experiment A4: GPT-5-mini with reasoning_effort=high
# =============================================================================
# Hypothesis: Maximum quality, worth the extra latency
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "A4_gpt5_high" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort high \
        --output-verbosity high
done
