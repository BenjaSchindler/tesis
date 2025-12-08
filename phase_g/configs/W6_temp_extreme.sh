#!/bin/bash
# W6_temp_extreme: Wave 6 - Extreme temperature (1.0) for maximum diversity
source "$(dirname "$0")/../base_config.sh"

run_experiment "W6_temp_extreme" \
    --disable-quality-gate \
    --llm-temperature 1.0 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
