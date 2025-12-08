#!/bin/bash
# IP2: 3x budget for very low F1 classes (F1 < 0.10)
# thresholds: [0.25, 0.10] => HIGH if > 0.25, LOW if < 0.10
# multipliers: [0.3, 1.0, 3.0] => 0.3x for HIGH, 1.0x for MID, 3.0x for LOW
source "$(dirname "$0")/../base_config.sh"

run_experiment "IP2_3x" \
    --auto-anchor-margin 0.00 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.25 0.10 \
    --f1-budget-multipliers 0.3 1.0 3.0
