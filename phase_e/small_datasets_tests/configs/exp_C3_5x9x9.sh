#!/bin/bash
# =============================================================================
# Experiment C3: Very High Volume (5x9x9 = 405 candidates/class)
# =============================================================================
# More samples per prompt for maximum diversity
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "C3_5x9x9" "$SEED" \
        --llm-model gpt-4o-mini \
        --max-clusters 5 \
        --prompts-per-cluster 9 \
        --samples-per-prompt 9
done
