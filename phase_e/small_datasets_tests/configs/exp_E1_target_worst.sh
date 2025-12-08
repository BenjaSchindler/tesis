#!/bin/bash
# =============================================================================
# Experiment E1: Target Only Worst Classes
# =============================================================================
# Only generate for ESTJ (F1=0), ESFP (F1=0), ESFJ (F1=0.29)
# Hypothesis: Focus resources on classes that need most help
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "E1_target_worst" "$SEED" \
        --llm-model gpt-4o-mini \
        --target-classes ESTJ ESFP ESFJ \
        --use-ip-scaling \
        --ip-threshold 0.7 \
        --ip-boost-factor 2.0 \
        --ip-minimum-base 15 \
        --ip-minimum-scale 25.0
done
