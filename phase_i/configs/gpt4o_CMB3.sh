#!/bin/bash
# gpt4o_CMB3: Replica of CMB3_skip config with gpt-4o-mini
# Component 1 of ENS_Top3_G5: F1-budget scaling with relaxed confidence
source "$(dirname "$0")/../base_config.sh"

run_experiment "gpt4o_CMB3" \
    --llm-model gpt-4o-mini \
    --auto-anchor-margin 0.05 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.05 \
    --filter-mode hybrid \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 0.0 0.5 2.5
