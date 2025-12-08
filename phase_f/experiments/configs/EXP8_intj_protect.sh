#!/bin/bash
# =============================================================================
# EXP8: INTJ Protect - Avoid synthetics that invade INTJ space
# =============================================================================
# Hypothesis: INTJ always degrades because neighbor synthetics invade its space
# Based on: Phase F analysis showing INTJ has -0.95% avg degradation
# Strategy: Stricter contamination + higher confidence to reject ambiguous synth
# =============================================================================

set -e
source "$(dirname "$0")/../../base_config.sh"

run_experiment "EXP8_intj_protect" \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --samples-per-prompt 5 \
    --contamination-threshold 0.98 \
    --min-classifier-confidence 0.20
