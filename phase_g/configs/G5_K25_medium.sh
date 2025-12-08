#!/bin/bash
# G5_K25_medium: Component 4 of ENS_Top3_G5
# High K=25 samples per prompt with reasoning
source "$(dirname "$0")/../base_config.sh"

run_experiment "G5_K25_medium" \
    --auto-anchor-margin 0.02 \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 25 \
    --max-completion-tokens 2048
