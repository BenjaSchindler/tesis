#!/bin/bash
# =============================================================================
# Experiment A2: GPT-5-mini with reasoning_effort=low
# =============================================================================
# Hypothesis: Balanced speed/quality tradeoff
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "A2_gpt5_low" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort low \
        --output-verbosity medium
done
