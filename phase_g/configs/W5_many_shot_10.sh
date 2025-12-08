#!/bin/bash
# W5_many_shot_10: Wave 5 - 10-shot prompting
source "$(dirname "$0")/../base_config.sh"

run_experiment "W5_many_shot_10" \
    --disable-quality-gate \
    --n-shot 10 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
