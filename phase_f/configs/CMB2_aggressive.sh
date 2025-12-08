#!/bin/bash
# CMB2: V2 + CF2 + IP2 (More volume + KNN only + 3x IP)
# Most aggressive approach - bypass classifier, max budget for weak classes
source "$(dirname "$0")/../base_config.sh"

run_experiment "CMB2_aggressive" \
    --auto-anchor-margin 0.05 \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --filter-mode knn \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.25 0.10 \
    --f1-budget-multipliers 0.3 1.0 3.0
