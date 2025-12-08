#!/bin/bash
# G5_K25_none: GPT-5-mini with 5x more context (K=25)
# High context, no reasoning overhead
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K25_none" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 25 \
    --reasoning-effort none \
    --output-verbosity high \
    --max-completion-tokens 1024 \
    --max-clusters 5 \
    --prompts-per-cluster 9
