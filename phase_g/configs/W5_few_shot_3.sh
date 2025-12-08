#!/bin/bash
# W5_few_shot_3: Wave 5 - 3-shot prompting
source "$(dirname "$0")/../base_config.sh"

run_experiment "W5_few_shot_3" \
    --disable-quality-gate \
    --n-shot 3 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
