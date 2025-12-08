#!/bin/bash
# =============================================================================
# EXP2: D1_relaxed improved - Relaxed filters for more diversity
# =============================================================================
# Hypothesis: Relaxed filters allow more diversity in synthetics
# Based on: Phase E D1_relaxed which achieved +3.27%
# Changes: Lower thresholds for similarity, contamination, confidence
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP2_relaxed_plus" \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --similarity-threshold 0.85 \
    --contamination-threshold 0.90 \
    --min-classifier-confidence 0.05 \
    --anchor-quality-threshold 0.20
