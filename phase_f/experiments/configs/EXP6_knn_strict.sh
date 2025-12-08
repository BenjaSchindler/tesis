#!/bin/bash
# =============================================================================
# EXP6: KNN Strict - Prioritize KNN similarity
# =============================================================================
# Hypothesis: KNN similarity is a better predictor of quality
# Based on: Using knn_only filter mode with strict KNN threshold
# Goal: Only accept synthetics that are close to real neighbors
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP6_knn_strict" \
    --max-clusters 8 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --filter-mode knn_only \
    --knn-similarity-threshold 0.55 \
    --similarity-threshold 0.88
