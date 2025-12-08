#!/bin/bash
# CF3: Very relaxed confidence threshold (0.05 - almost no filtering)
# Base uses 0.10, this tests even lower to accept more borderline synthetics
source "$(dirname "$0")/../base_config.sh"

run_experiment "CF3_relaxed" \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --min-classifier-confidence 0.05 \
    --filter-mode hybrid
