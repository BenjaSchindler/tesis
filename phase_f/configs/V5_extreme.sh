#!/bin/bash
# V5_extreme: Maximum volume (10×15×10 = 1500 candidates) - 5.5x more than V2
source "$(dirname "$0")/../base_config.sh"
run_experiment "V5_extreme" \
    --auto-anchor-margin 0.02 \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10
