#!/bin/bash
# W7_yolo: Wave 7 - YOLO mode - NO filters at all!
source "$(dirname "$0")/../base_config.sh"

run_experiment "W7_yolo" \
    --disable-quality-gate \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10 \
    --min-classifier-confidence 0.0 \
    --similarity-threshold 0.99 \
    --dedup-embed-sim 1.0 \
    --filter-mode classifier \
    --cap-class-ratio 0.50
