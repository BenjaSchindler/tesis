#!/bin/bash
# =============================================================================
# Base Configuration for Phase E Small Dataset Experiments
# =============================================================================
# This script defines common parameters used across all experiments.
# Individual experiment scripts source this file and override specific params.
# =============================================================================

# Common paths
PHASE_E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${PHASE_E_DIR}/../mbti_1.csv"
CACHE_DIR="${PHASE_E_DIR}/embeddings_cache"
RESULTS_DIR="${PHASE_E_DIR}/small_datasets_tests/results"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Function to run experiment with given parameters
# Usage: run_experiment EXP_NAME SEED [extra params...]
run_experiment() {
    local EXP_NAME="$1"
    local SEED="$2"
    shift 2

    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUT="${RESULTS_DIR}/${EXP_NAME}_s${SEED}_${TIMESTAMP}"

    echo "========================================"
    echo "  Experiment: $EXP_NAME"
    echo "  Seed: $SEED"
    echo "  Output: $OUT"
    echo "========================================"

    python3 -u "${PHASE_E_DIR}/core/runner_phase2.py" \
        --data-path "$DATA_PATH" \
        --test-size 0.2 \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 128 \
        --cache-dir "$CACHE_DIR" \
        --max-clusters 5 \
        --prompts-per-cluster 9 \
        --samples-per-prompt 5 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --anchor-quality-threshold 0.10 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --auto-params \
        --verbose-logging \
        --random-seed "$SEED" \
        --synthetic-output "${OUT}_synth.csv" \
        --augmented-train-output "${OUT}_aug.csv" \
        --metrics-output "${OUT}_metrics.json" \
        "$@" \
        2>&1 | tee "${OUT}.log"

    # Print summary
    echo ""
    echo "========================================"
    echo "  Results:"
    python3 -c "
import json
try:
    with open('${OUT}_metrics.json') as f:
        d = json.load(f)
    b = d['baseline']['macro_f1']
    a = d['augmented']['macro_f1']
    s = d.get('synthetic_data', {}).get('accepted_count', 0)
    print(f'  Baseline F1:  {b:.4f}')
    print(f'  Augmented F1: {a:.4f}')
    print(f'  Delta:        {(a-b)/b*100:+.2f}%')
    print(f'  Synthetics:   {s}')
except Exception as e:
    print(f'  Error reading results: {e}')
"
    echo "========================================"
    echo ""
}
