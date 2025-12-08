#!/bin/bash
# =============================================================================
# Experiment B1: IP Scaling Baseline
# =============================================================================
# ip_threshold=0.7, ip_boost_factor=1.0, minimum=10+20*IP
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "B1_ip_baseline" "$SEED" \
        --llm-model gpt-4o-mini \
        --use-ip-scaling \
        --ip-threshold 0.7 \
        --ip-boost-factor 1.0 \
        --ip-minimum-base 10 \
        --ip-minimum-scale 20.0
done
