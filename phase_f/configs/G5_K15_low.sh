#!/bin/bash
# G5_K15_low: GPT-5-mini with K=15, low reasoning
# High context + light thinking for better quality
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K15_low" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 15 \
    --reasoning-effort low \
    --output-verbosity high \
    --max-completion-tokens 1536 \
    --max-clusters 5 \
    --prompts-per-cluster 9
