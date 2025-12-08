#!/bin/bash
# gpt4o_V4: Replica of V4_ultra config with gpt-4o-mini
# Component 3 of ENS_Top3_G5: High volume generation (8x12x7=672 candidates)
source "$(dirname "$0")/../base_config.sh"

run_experiment "gpt4o_V4" \
    --llm-model gpt-4o-mini \
    --auto-anchor-margin 0.02 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
