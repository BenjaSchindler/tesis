#!/bin/bash
# W8_gpt5_reasoning: Wave 8 - GPT-5-mini with medium reasoning
source "$(dirname "$0")/../base_config.sh"

run_experiment "W8_gpt5_reasoning" \
    --disable-quality-gate \
    --llm-model gpt-5-mini \
    --reasoning-effort medium \
    --max-clusters 8 \
    --prompts-per-cluster 10 \
    --samples-per-prompt 5
