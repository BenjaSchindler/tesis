#!/bin/bash
# CF2: KNN only filtering (bypass classifier confidence entirely)
# Hypothesis: classifier confidence may be counterproductive
source "$(dirname "$0")/../base_config.sh"

run_experiment "CF2_knn_only" \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --filter-mode knn
