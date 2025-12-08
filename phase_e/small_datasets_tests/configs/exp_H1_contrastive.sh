#!/bin/bash
# =============================================================================
# Experiment H1: Contrastive Prompting
# =============================================================================
# Uses --use-contrastive-prompting to generate against confuser classes
# Hypothesis: Contrastive context helps generate more distinctive texts
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "H1_contrastive" "$SEED" \
        --llm-model gpt-4o-mini \
        --use-contrastive-prompting \
        --contrastive-top-k 2
done
