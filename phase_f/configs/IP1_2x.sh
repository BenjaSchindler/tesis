#!/bin/bash
# IP1: 2x budget for low F1 classes (F1 < 0.15)
# thresholds: [0.30, 0.15] => HIGH if > 0.30, LOW if < 0.15
# multipliers: [0.5, 1.0, 2.0] => 0.5x for HIGH, 1.0x for MID, 2.0x for LOW
source "$(dirname "$0")/../base_config.sh"

run_experiment "IP1_2x" \
    --auto-anchor-margin 0.00 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.30 0.15 \
    --f1-budget-multipliers 0.5 1.0 2.0
