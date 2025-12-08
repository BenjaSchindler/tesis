#!/bin/bash
# gpt4o_G5: Replica of G5_K25_medium config with gpt-4o-mini
# Component 4 of ENS_Top3_G5: High K=25 samples per prompt
# Note: Since gpt-4o-mini doesn't support reasoning, no --reasoning-effort param
source "$(dirname "$0")/../base_config.sh"

run_experiment "gpt4o_G5" \
    --llm-model gpt-4o-mini \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 25 \
    --max-completion-tokens 2048
