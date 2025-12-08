#!/bin/bash
# V2_f1scaled: V2 base + f1-budget-scaling from CMB3_skip
# Skip high F1 (>0.35), half medium, 2.5x budget for low F1 classes
source "$(dirname "$0")/../base_config.sh"
run_experiment "V2_f1scaled" \
    --auto-anchor-margin 0.02 \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 0.0 0.5 2.5
