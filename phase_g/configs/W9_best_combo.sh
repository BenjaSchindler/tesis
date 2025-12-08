#!/bin/bash
# W9_best_combo: Wave 9 - Best combination from previous phases
source "$(dirname "$0")/../base_config.sh"

run_experiment "W9_best_combo" \
    --disable-quality-gate \
    --force-generation-classes "ENFJ,ESFJ,ESFP,ESTJ,ISTJ" \
    --use-contrastive-prompting \
    --contrastive-top-k 2 \
    --max-clusters 10 \
    --prompts-per-cluster 15 \
    --samples-per-prompt 10 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.25 0.15 \
    --f1-budget-multipliers 0.0 0.5 3.0
