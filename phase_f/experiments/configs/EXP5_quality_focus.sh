#!/bin/bash
# =============================================================================
# EXP5: Quality Focus - Only high-quality synthetics
# =============================================================================
# Hypothesis: Fewer but better synthetics may work better
# Based on: Strict thresholds for all quality metrics
# Goal: Only accept top-tier synthetics
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP5_quality_focus" \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --similarity-threshold 0.92 \
    --contamination-threshold 0.98 \
    --min-classifier-confidence 0.25 \
    --anchor-quality-threshold 0.70
