#!/bin/bash
# G5V2_K25_med: gpt-5-mini with V2 volume + medium reasoning
# Hypothesis: V2 structure + G5 context + reasoning for better quality
source "$(dirname "$0")/../base_config.sh"
run_experiment "G5V2_K25_med" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 25 \
    --reasoning-effort medium \
    --output-verbosity high \
    --max-completion-tokens 2048
