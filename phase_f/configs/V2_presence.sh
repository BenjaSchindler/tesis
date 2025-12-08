#!/bin/bash
# V2_presence: V2 base + presence_penalty=0.3 for diversity
# Hypothesis: presence_penalty encourages more diverse outputs
source "$(dirname "$0")/../base_config.sh"
run_experiment "V2_presence" \
    --auto-anchor-margin 0.02 \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --presence-penalty 0.3
