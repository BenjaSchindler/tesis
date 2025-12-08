#!/bin/bash
# Phase G Extended - Overnight Run with K-Fold Evaluation
# Strategy: 1 seed for generation, K-Fold CV for evaluation
# Total: 20 configs x 1 seed = 20 generation runs
# Evaluation: 5-fold x 3 repeats = 15 evaluations per config

set -e

cd "$(dirname "$0")"

# Configuration
PARALLEL_JOBS=10
SEED=42
RESULTS_DIR="results"
JOB_LOG="$RESULTS_DIR/overnight_kfold_joblog.txt"

# Get list of W* configs
CONFIGS=($(ls configs/W*.sh | xargs -n1 basename | sed 's/.sh$//'))

echo "=============================================="
echo "Phase G Extended - Overnight K-Fold Run"
echo "=============================================="
echo "Start time: $(date)"
echo "Configs: ${#CONFIGS[@]}"
echo "Seed: $SEED (single)"
echo "Evaluation: 5-fold x 3 repeats = 15 evals/config"
echo "Total generation jobs: ${#CONFIGS[@]}"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "=============================================="
echo ""

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Create job list (single seed)
JOB_FILE="/tmp/phase_g_kfold_jobs.txt"
> "$JOB_FILE"

for cfg in "${CONFIGS[@]}"; do
    echo "SEED=$SEED bash configs/${cfg}.sh" >> "$JOB_FILE"
done

echo "Job list created: $JOB_FILE"
echo "Total generation jobs: $(wc -l < "$JOB_FILE")"
echo ""
echo "=== PHASE 1: GENERATION ==="
echo ""

# Run generation with GNU parallel
cat "$JOB_FILE" | parallel -j "$PARALLEL_JOBS" --joblog "$JOB_LOG"

echo ""
echo "=== PHASE 2: K-FOLD EVALUATION ==="
echo ""

# Run K-Fold evaluation for all synthetic files
python3 kfold_multimodel.py --all --k 5 --repeats 3 \
    --models LogisticRegression MLP_small \
    --output kfold_overnight_results.json

echo ""
echo "=============================================="
echo "OVERNIGHT K-FOLD RUN COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  Generation log: $JOB_LOG"
echo "  K-Fold results: $RESULTS_DIR/kfold_overnight_results.json"
echo ""

# Summary of synthetic files
echo "Synthetic files generated:"
ls -lh "$RESULTS_DIR"/*_synth.csv 2>/dev/null | wc -l
echo ""
