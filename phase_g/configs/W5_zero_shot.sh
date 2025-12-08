#!/bin/bash
# W5_zero_shot: Wave 5 - Zero-shot prompting (no examples)
source "$(dirname "$0")/../base_config.sh"

run_experiment "W5_zero_shot" \
    --disable-quality-gate \
    --n-shot 0 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
