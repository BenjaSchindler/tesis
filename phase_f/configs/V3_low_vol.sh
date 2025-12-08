#!/bin/bash
# V3: Lower volume (4x9x5 = 180 candidates)
source "$(dirname "$0")/../base_config.sh"

run_experiment "V3_low_vol" \
    --auto-anchor-margin 0.02 \
    --max-clusters 4 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5
