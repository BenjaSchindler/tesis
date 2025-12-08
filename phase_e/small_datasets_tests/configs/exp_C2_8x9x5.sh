#!/bin/bash
# =============================================================================
# Experiment C2: High Volume (8x9x5 = 360 candidates/class)
# =============================================================================
# More clusters for more diversity
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "C2_8x9x5" "$SEED" \
        --llm-model gpt-4o-mini \
        --max-clusters 8 \
        --prompts-per-cluster 9 \
        --samples-per-prompt 5
done
