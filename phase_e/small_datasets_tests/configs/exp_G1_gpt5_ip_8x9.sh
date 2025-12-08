#!/bin/bash
# =============================================================================
# Experiment G1: GPT-5 High + IP Boost + 8x9 Clusters
# =============================================================================
# Combines: reasoning=high, IP boost 2.0, 8x9x5 clusters
# Hypothesis: Best ideas combined for maximum improvement
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "G1_gpt5_ip_8x9" "$SEED" \
        --llm-model gpt-5-mini \
        --reasoning-effort high \
        --output-verbosity high \
        --use-ip-scaling \
        --ip-threshold 0.7 \
        --ip-boost-factor 2.0 \
        --ip-minimum-base 15 \
        --ip-minimum-scale 25.0 \
        --max-clusters 8 \
        --prompts-per-cluster 9 \
        --samples-per-prompt 5
done
