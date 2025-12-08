#!/bin/bash
# W2_ultra_vol: Wave 2 - Ultra high volume generation
# 10 clusters x 15 prompts x 10 samples = 1500 candidates per class
source "$(dirname "$0")/../base_config.sh"

run_experiment "W2_ultra_vol" \
    --disable-quality-gate \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10 \
    --min-classifier-confidence 0.01
