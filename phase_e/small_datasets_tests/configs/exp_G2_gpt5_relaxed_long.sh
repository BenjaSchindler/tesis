#!/bin/bash
# =============================================================================
# Experiment G2: GPT-5 Medium + Relaxed Filters + Long Texts
# =============================================================================
# Combines: reasoning=medium, sim=0.85, length=500 words
# Hypothesis: Quality reasoning + diverse texts + full length
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "G2_gpt5_relaxed_long" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort medium \
        --output-verbosity high \
        --similarity-threshold 0.85 \
        --contamination-threshold 0.90 \
        --anchor-quality-threshold 0.20 \
        --length-aware \
        --length-target-words 500 \
        --llm-max-tokens 4000
done
