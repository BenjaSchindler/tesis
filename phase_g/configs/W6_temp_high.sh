#!/bin/bash
# W6_temp_high: Wave 6 - High temperature (0.9) for more diversity
source "$(dirname "$0")/../base_config.sh"

run_experiment "W6_temp_high" \
    --disable-quality-gate \
    --llm-temperature 0.9 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
