#!/bin/bash
# =============================================================================
# Experiment B2: Aggressive IP Scaling
# =============================================================================
# ip_threshold=0.5 (lower), ip_boost_factor=2.0 (higher), minimum=15+30*IP
# Targets more classes with larger boosts
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "B2_ip_aggressive" "$SEED" \
        --llm-model gpt-4o-mini \
        --use-ip-scaling \
        --ip-threshold 0.5 \
        --ip-boost-factor 2.0 \
        --ip-minimum-base 15 \
        --ip-minimum-scale 30.0
done
