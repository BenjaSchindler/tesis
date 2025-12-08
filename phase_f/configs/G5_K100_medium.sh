#!/bin/bash
# G5_K100_medium: TRYHARD - GPT-5-mini with K=100 (full cluster context)
# Leverages 400K context window to give model ALL cluster examples
# Hypothesis: Maximum context = maximum understanding of class patterns
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K100_medium" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --samples-per-prompt 100 \
    --reasoning-effort medium \
    --output-verbosity high \
    --max-completion-tokens 4096 \
    --max-clusters 5 \
    --prompts-per-cluster 9
