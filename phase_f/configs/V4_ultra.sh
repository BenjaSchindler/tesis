#!/bin/bash
# V4_ultra: Higher volume (8×12×7 = 672 candidates) - 2.5x more than V2
source "$(dirname "$0")/../base_config.sh"
run_experiment "V4_ultra" \
    --auto-anchor-margin 0.02 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
