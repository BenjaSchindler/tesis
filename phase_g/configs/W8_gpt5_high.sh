#!/bin/bash
# W8_gpt5_high: Wave 8 - GPT-5-mini with high reasoning
source "$(dirname "$0")/../base_config.sh"

run_experiment "W8_gpt5_high" \
    --disable-quality-gate \
    --llm-model gpt-5-mini \
    --reasoning-effort high \
    --max-clusters 6 \
    --prompts-per-cluster 8 \
    --samples-per-prompt 4
