#!/bin/bash
# G5_K15_none: GPT-5-mini with 3x more context (K=15)
# Leverages larger context window for more cluster examples
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K15_none" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 15 \
    --reasoning-effort none \
    --output-verbosity high \
    --max-completion-tokens 1024 \
    --max-clusters 5 \
    --prompts-per-cluster 9
