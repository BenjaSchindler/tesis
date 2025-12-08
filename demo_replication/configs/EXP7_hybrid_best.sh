#!/bin/bash
# =============================================================================
# EXP7: Hybrid Best - Combine best of CMB3 + Phase E
# =============================================================================
# Hypothesis: Combining proven strategies should work best
# Based on: CMB3_skip (best single config) + Phase E tuning
# Features: F1-budget-scaling + hybrid filter + relaxed confidence
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

run_experiment "EXP7_hybrid_best" \
    --max-clusters 6 \
    --prompts-per-cluster 10 \
    --samples-per-prompt 6 \
    --auto-anchor-margin 0.05 \
    --min-classifier-confidence 0.05 \
    --filter-mode hybrid \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 0.0 0.5 2.5 \
    --similarity-threshold 0.88
