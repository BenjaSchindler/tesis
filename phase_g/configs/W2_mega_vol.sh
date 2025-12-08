#!/bin/bash
# W2_mega_vol: Wave 2 - Mega volume (3000+ candidates)
source "$(dirname "$0")/../base_config.sh"

run_experiment "W2_mega_vol" \
    --disable-quality-gate \
    --max-clusters 12 \
    --prompts-per-cluster 20 \
    --samples-per-prompt 12 \
    --min-classifier-confidence 0.01 \
    --cap-class-ratio 0.30
