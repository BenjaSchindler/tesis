#!/bin/bash
# =============================================================================
# Experiment E2: Triple Boost for IP>0.9 Classes
# =============================================================================
# ip_threshold=0.9, ip_boost_factor=3.0, high minimums
# Hypothesis: Massive budget increase for worst classes without hurting others
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "E2_minority_boost" "$SEED" \
        --llm-model gpt-4o-mini \
        --use-ip-scaling \
        --ip-threshold 0.9 \
        --ip-boost-factor 3.0 \
        --ip-minimum-base 25 \
        --ip-minimum-scale 35.0
done
