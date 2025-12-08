#!/bin/bash
# G5_K25_medium: GPT-5-mini with K=25, medium reasoning
# High context + medium thinking for quality generation
source "$(dirname "$0")/base_config.sh"

run_experiment "G5_K25_medium" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 25 \
    --reasoning-effort medium \
    --output-verbosity high \
    --max-completion-tokens 2048 \
    --max-clusters 5 \
    --prompts-per-cluster 9
