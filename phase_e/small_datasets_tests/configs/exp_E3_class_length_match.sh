#!/bin/bash
# =============================================================================
# Experiment E3: Class-Average Length Matching
# =============================================================================
# Uses --use-class-length-average to calculate per-class average word count
# and generate synthetics matching that length for each class.
# Targets all minority classes (F1 < 0.5) with relaxed thresholds.
# =============================================================================

set -e
source "$(dirname "$0")/base_config.sh"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

SEEDS="${1:-42 100 123}"

for SEED in $SEEDS; do
    run_experiment "E3_class_length_match" "$SEED" \
        --length-aware \
        --use-class-length-average \
        --llm-max-tokens 4000 \
        --minority-f1-threshold 0.5 \
        --similarity-threshold 0.85 \
        --anchor-quality-threshold 0.15 \
        --contamination-threshold 0.90
done
