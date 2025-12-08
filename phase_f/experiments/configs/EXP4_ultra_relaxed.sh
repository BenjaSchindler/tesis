#!/bin/bash
# =============================================================================
# EXP4: Ultra Relaxed - V4 high volume + D1 relaxed filters
# =============================================================================
# Hypothesis: High volume + relaxed filters = more diversity
# Based on: Combining V4_ultra (8x12x7) with D1_relaxed thresholds
# Total candidates: 8 * 12 * 7 = 672 per class
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP4_ultra_relaxed" \
    --max-clusters 8 \
    --prompts-per-cluster 12 \
    --samples-per-prompt 7 \
    --similarity-threshold 0.85 \
    --contamination-threshold 0.90 \
    --min-classifier-confidence 0.05
