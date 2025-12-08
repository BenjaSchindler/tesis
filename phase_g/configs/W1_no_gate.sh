#!/bin/bash
# W1_no_gate: Wave 1 - Disable ALL quality gates
# Hypothesis: Quality gate blocks ESFP/ESTJ entirely
source "$(dirname "$0")/../base_config.sh"

run_experiment "W1_no_gate" \
    --disable-quality-gate \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7 \
    --min-classifier-confidence 0.01 \
    --filter-mode hybrid
