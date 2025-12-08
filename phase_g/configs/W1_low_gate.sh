#!/bin/bash
# W1_low_gate: Wave 1 - Very low gate threshold (0.05)
source "$(dirname "$0")/../base_config.sh"

run_experiment "W1_low_gate" \
    --anchor-quality-threshold 0.05 \
    --purity-gate-threshold 0.005 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7 \
    --min-classifier-confidence 0.01
