#!/bin/bash
# V1: Baseline volume (5x9x5 = 225 candidates) - Same as C1
source "$(dirname "$0")/../base_config.sh"

run_experiment "V1_baseline" \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5
