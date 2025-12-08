#!/bin/bash
# W9_contrastive: Wave 9 - Contrastive prompting to differentiate from confusers
source "$(dirname "$0")/base_config.sh"

run_experiment "W9_contrastive" \
    --disable-quality-gate \
    --use-contrastive-prompting \
    --contrastive-top-k 3 \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7
