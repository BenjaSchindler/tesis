#!/bin/bash
# =============================================================================
# Experiment B3: High-IP Only Boost
# =============================================================================
# ip_threshold=0.8 (high), ip_boost_factor=1.5 (big), minimum=20+25*IP
# Only boosts the worst classes (ESTJ, ESFP with IP=1.0)
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "B3_ip_high_only" "$SEED" \
        --llm-model gpt-4o-mini \
        --use-ip-scaling \
        --ip-threshold 0.8 \
        --ip-boost-factor 1.5 \
        --ip-minimum-base 20 \
        --ip-minimum-scale 25.0
done
