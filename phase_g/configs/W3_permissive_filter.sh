#!/bin/bash
# W3_permissive_filter: Wave 3 - Very permissive filters
source "$(dirname "$0")/../base_config.sh"

run_experiment "W3_permissive_filter" \
    --disable-quality-gate \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7 \
    --min-classifier-confidence 0.01 \
    --similarity-threshold 0.98 \
    --dedup-embed-sim 0.99 \
    --filter-mode classifier
