#!/bin/bash
# CF1: Confidence band 0.3-0.7 (avoid overly confident predictions)
# Hypothesis: classifier_conf vs knn_similarity = -0.58 suggests high conf = bad
source "$(dirname "$0")/../base_config.sh"

run_experiment "CF1_conf_band" \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.3 \
    --max-classifier-confidence 0.7 \
    --filter-mode hybrid
