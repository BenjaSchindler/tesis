#!/bin/bash
# G5V2_K25: gpt-5-mini with V2 volume (6×9=54 prompts) and K=25 context
# Hypothesis: V2 structure + G5 larger context window
source "$(dirname "$0")/../base_config.sh"
run_experiment "G5V2_K25" \
    --auto-anchor-margin 0.02 \
    --llm-model gpt-5-mini \
    --max-clusters 6 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 25 \
    --reasoning-effort none \
    --output-verbosity high \
    --max-completion-tokens 2048
