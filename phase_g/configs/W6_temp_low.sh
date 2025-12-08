#!/bin/bash
# W6_temp_low: Wave 6 - Low temperature (0.3) for more deterministic output
source "$(dirname "$0")/../base_config.sh"

run_experiment "W6_temp_low" \
    --disable-quality-gate \
    --llm-temperature 0.3 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
