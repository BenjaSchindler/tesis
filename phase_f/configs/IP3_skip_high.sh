#!/bin/bash
# IP3: Skip high F1 classes, 2.5x for low F1
# thresholds: [0.35, 0.20] => HIGH if > 0.35, LOW if < 0.20
# multipliers: [0.0, 0.5, 2.5] => SKIP HIGH, 0.5x for MID, 2.5x for LOW
source "$(dirname "$0")/../base_config.sh"

run_experiment "IP3_skip_high" \
    --auto-anchor-margin 0.00 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 0.0 0.5 2.5
