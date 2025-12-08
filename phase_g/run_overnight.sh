#!/bin/bash
# Phase G Extended - Overnight Run Script
# Runs all W* configs with 5 seeds in parallel
# Total: 20 configs x 5 seeds = 100 experiments
# Estimated time: 6-8 hours with 10 parallel jobs

set -e

cd "$(dirname "$0")"

# Configuration
PARALLEL_JOBS=10
SEEDS=(42 100 123 456 789)
RESULTS_DIR="results"
JOB_LOG="$RESULTS_DIR/overnight_joblog.txt"
SUMMARY_LOG="$RESULTS_DIR/overnight_summary.txt"

# Get list of W* configs
CONFIGS=($(ls configs/W*.sh | xargs -n1 basename | sed 's/.sh$//'))

echo "=============================================="
echo "Phase G Extended - Overnight Run"
echo "=============================================="
echo "Start time: $(date)"
echo "Configs: ${#CONFIGS[@]}"
echo "Seeds: ${#SEEDS[@]}"
echo "Total jobs: $((${#CONFIGS[@]} * ${#SEEDS[@]}))"
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

# Create job list
JOB_FILE="/tmp/phase_g_jobs.txt"
> "$JOB_FILE"

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "SEED=$seed bash configs/${cfg}.sh" >> "$JOB_FILE"
    done
done

echo "Job list created: $JOB_FILE"
echo "Total jobs: $(wc -l < "$JOB_FILE")"
echo ""
echo "Starting parallel execution..."
echo ""

# Run with GNU parallel
cat "$JOB_FILE" | parallel -j "$PARALLEL_JOBS" --progress --joblog "$JOB_LOG" --bar

# Summary
echo ""
echo "=============================================="
echo "OVERNIGHT RUN COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Job log: $JOB_LOG"
echo ""

# Count results
echo "Results summary:" | tee "$SUMMARY_LOG"
for cfg in "${CONFIGS[@]}"; do
    synth_count=$(ls -1 "$RESULTS_DIR/${cfg}_s*_synth.csv" 2>/dev/null | wc -l)
    echo "  $cfg: $synth_count/5 seeds completed" | tee -a "$SUMMARY_LOG"
done

echo ""
echo "To analyze results, run:"
echo "  python3 kfold_multimodel.py"
echo ""
echo "To create ensembles, run:"
echo "  python3 create_ensembles.py"
