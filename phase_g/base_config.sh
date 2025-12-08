#!/bin/bash
# Phase G - Base config (copy from Phase F, targeting problem classes)
# This file is sourced by individual experiment configs

# Use absolute paths for reliability
PROJECT_ROOT="/home/benja/Desktop/Tesis/SMOTE-LLM"
BASE_DIR="$PROJECT_ROOT/phase_g"

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
    --auto-anchor-threshold  # Phase F: Auto-adjust threshold based on dataset
    --auto-params            # Phase F: Auto-adjust thresholds based on dataset size
    --verbose-logging        # Phase F: Save detailed per-synthetic quality metrics
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

    python3 -u "$BASE_DIR/core/runner_phase2.py" \
        "${BASE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        --random-seed "$SEED" \
        --synthetic-output "$BASE_DIR/results/${EXP_NAME}_s${SEED}_synth.csv" \
        --augmented-train-output "$BASE_DIR/results/${EXP_NAME}_s${SEED}_aug.csv" \
        --metrics-output "$BASE_DIR/results/${EXP_NAME}_s${SEED}_metrics.json" \
        2>&1 | tee "$BASE_DIR/results/${EXP_NAME}_s${SEED}.log"
}
