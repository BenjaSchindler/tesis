#!/bin/bash
# =============================================================================
# EXP1: Phase E Port - C1_5x9x5 config on full dataset
# =============================================================================
# Hypothesis: The 5x9x5 volume worked well in Phase E, test on full dataset
# Based on: Phase E C1_5x9x5 which achieved +4.74% (but high variance)
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP1_phaseE_port" \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5
