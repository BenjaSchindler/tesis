#!/bin/bash
# CMB3: V1 + CF3 + IP3 (Baseline + very relaxed conf 0.05 + skip high F1)
# Focus resources on weak classes only
source "$(dirname "$0")/base_config.sh"

run_experiment "CMB3_skip" \
    --auto-anchor-margin 0.05 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.05 \
    --filter-mode hybrid \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 0.0 0.5 2.5
