#!/bin/bash
# =============================================================================
# Experiment C1: Standard Volume (5x9x5 = 225 candidates/class)
# =============================================================================
# Best config from small dataset tests
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "C1_5x9x5" "$SEED" \
        --llm-model gpt-4o-mini \
        --max-clusters 5 \
        --prompts-per-cluster 9 \
        --samples-per-prompt 5
done
