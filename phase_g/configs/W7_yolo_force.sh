#!/bin/bash
# W7_yolo_force: Wave 7 - YOLO mode + force problem classes
source "$(dirname "$0")/../base_config.sh"

run_experiment "W7_yolo_force" \
    --disable-quality-gate \
    --force-generation-classes "ENFJ,ESFJ,ESFP,ESTJ,ISTJ" \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10 \
    --min-classifier-confidence 0.0 \
    --similarity-threshold 0.99 \
    --dedup-embed-sim 1.0 \
    --cap-class-ratio 0.50
