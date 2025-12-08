#!/bin/bash
# G5_K5_none: GPT-5-mini baseline (same K as gpt-4o-mini)
# Tests GPT-5-mini with same parameters as default experiments
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K5_none" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 5 \
    --reasoning-effort none \
    --output-verbosity high \
    --max-completion-tokens 1024 \
    --max-clusters 5 \
    --prompts-per-cluster 9
