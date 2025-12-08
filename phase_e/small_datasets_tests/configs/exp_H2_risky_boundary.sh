#!/bin/bash
# =============================================================================
# Experiment H2: Risky Boundary Exploration
# =============================================================================
# Very relaxed filters: sim=0.80, anchor=0.15, confidence=0.03
# Hypothesis: Allow more diverse/risky texts for boundary exploration
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "H2_risky_boundary" "$SEED" \
        --llm-model gpt-4o-mini \
        --similarity-threshold 0.80 \
        --contamination-threshold 0.85 \
        --anchor-quality-threshold 0.15 \
        --min-classifier-confidence 0.03
done
