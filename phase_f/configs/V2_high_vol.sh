#!/bin/bash
# V2: Higher volume (6x9x5 = 270 candidates)
source "$(dirname "$0")/../base_config.sh"

run_experiment "V2_high_vol" \
    --auto-anchor-margin 0.02 \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5
