#!/bin/bash
# V4_ultra: Component 3 of ENS_Top3_G5
# High volume generation (8x12x7=672 candidates)
source "$(dirname "$0")/../base_config.sh"

run_experiment "V4_ultra" \
    --auto-anchor-margin 0.02 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
