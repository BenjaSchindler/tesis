#!/bin/bash
# W1_force_problem: Wave 1 - Force generation for problem classes
source "$(dirname "$0")/base_config.sh"

run_experiment "W1_force_problem" \
    --force-generation-classes "ENFJ,ESFJ,ESFP,ESTJ,ISTJ" \
    --max-clusters 8 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 7 \
    --min-classifier-confidence 0.01
