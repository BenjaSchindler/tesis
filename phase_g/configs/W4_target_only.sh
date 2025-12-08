#!/bin/bash
# W4_target_only: Wave 4 - Target ONLY problematic classes with force generation
source "$(dirname "$0")/../base_config.sh"

run_experiment "W4_target_only" \
    --force-generation-classes "ENFJ,ESFJ,ESFP,ESTJ,ISTJ" \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.20 0.10 \
    --f1-budget-multipliers 0.0 0.5 3.0 \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10
