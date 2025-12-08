#!/bin/bash
# =============================================================================
# EXP3: Minority Focus - Aggressive boost for minority classes
# =============================================================================
# Hypothesis: ISFJ improves +19.93%, force more generation for other minorities
# Based on: CMB3_skip F1-budget-scaling but more aggressive
# Changes: Higher multiplier for low-F1 classes (3.0x vs 2.5x)
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP3_minority_focus" \
    --max-clusters 8 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.40 0.25 \
    --f1-budget-multipliers 0.0 0.3 3.0
