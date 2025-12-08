#!/bin/bash
# gpt4o_CF1: Replica of CF1_conf_band config with gpt-4o-mini
# Component 2 of ENS_Top3_G5: Confidence band 0.3-0.7
source "$(dirname "$0")/../base_config.sh"

run_experiment "gpt4o_CF1" \
    --llm-model gpt-4o-mini \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.3 \
    --max-classifier-confidence 0.7 \
    --filter-mode hybrid
