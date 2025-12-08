#!/bin/bash
# =============================================================================
# Experiment G3: GPT-5 High + Minority Only + Strict Filters
# =============================================================================
# Combines: reasoning=high, target=ESTJ/ESFP, sim=0.95
# Hypothesis: Max quality reasoning for worst classes only
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "G3_gpt5_minority_strict" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort high \
        --output-verbosity high \
        --target-classes ESTJ ESFP ESFJ ENFJ ISFJ ISTJ ESTP \
        --similarity-threshold 0.95 \
        --contamination-threshold 0.98 \
        --anchor-quality-threshold 0.30 \
        --use-ip-scaling \
        --ip-threshold 0.8 \
        --ip-boost-factor 2.5 \
        --ip-minimum-base 20 \
        --ip-minimum-scale 30.0
done
