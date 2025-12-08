#!/bin/bash
# SMOTE-LLM Demo - Base config
# This file is sourced by individual experiment configs

# Use environment variables or auto-detect paths
DEMO_DIR="${DEMO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$DEMO_DIR")}"
BASE_DIR="$DEMO_DIR"

BASE_ARGS=(
    --data-path "$PROJECT_ROOT/mbti_1.csv"
    --test-size 0.2
    --embedding-model sentence-transformers/all-mpnet-base-v2
    --device cuda
    --embedding-batch-size 128
    --cache-dir "$PROJECT_ROOT/phase_e/embeddings_cache"
    --llm-model gpt-4o-mini
    --prompt-mode mix
    --use-ensemble-selection
    --use-val-gating
    --val-size 0.15
    --val-tolerance 0.02
    --enable-anchor-gate
    --auto-anchor-threshold
    --auto-params
    --verbose-logging
    --enable-anchor-selection
    --anchor-selection-ratio 0.8
    --anchor-outlier-threshold 1.5
    --use-class-description
    --cap-class-ratio 0.15
    --similarity-threshold 0.90
    --min-classifier-confidence 0.10
    --contamination-threshold 0.95
    --synthetic-weight 0.5
    --synthetic-weight-mode flat
)

# Helper function to run experiment
run_experiment() {
    local EXP_NAME="$1"
    shift
    local EXTRA_ARGS=("$@")

    if [ -z "$SEED" ]; then
        echo "ERROR: SEED not set"
        exit 1
    fi

    echo "=== Running $EXP_NAME with seed $SEED ==="

    mkdir -p "$DEMO_DIR/results"

    python3 -u "$DEMO_DIR/core/runner_phase2.py" \
        "${BASE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        --random-seed "$SEED" \
        --synthetic-output "$DEMO_DIR/results/${EXP_NAME}_s${SEED}_synth.csv" \
        --augmented-train-output "$DEMO_DIR/results/${EXP_NAME}_s${SEED}_aug.csv" \
        --metrics-output "$DEMO_DIR/results/${EXP_NAME}_s${SEED}_metrics.json" \
        2>&1 | tee "$DEMO_DIR/results/${EXP_NAME}_s${SEED}.log"
}
