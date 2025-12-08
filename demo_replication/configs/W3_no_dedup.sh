#!/bin/bash
# W3_no_dedup: Wave 3 - Disable deduplication entirely
source "$(dirname "$0")/base_config.sh"

run_experiment "W3_no_dedup" \
    --disable-quality-gate \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7 \
    --min-classifier-confidence 0.01 \
    --dedup-embed-sim 1.0 \
    --duplicate-threshold 1.0
