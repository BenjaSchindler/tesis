#!/bin/bash
# CMB1: V1 + CF1 + IP1 (Baseline volume + conf band + 2x IP)
# Balanced approach combining key insights
source "$(dirname "$0")/../base_config.sh"

run_experiment "CMB1_balanced" \
    --auto-anchor-margin 0.05 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.3 \
    --max-classifier-confidence 0.7 \
    --filter-mode hybrid \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.30 0.15 \
    --f1-budget-multipliers 0.5 1.0 2.0
