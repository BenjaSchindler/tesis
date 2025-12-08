#!/bin/bash
# =============================================================================
# Phase E: Parallel Experiment Runner
# =============================================================================
# Runs experiments in parallel (3 concurrent jobs by default)
# Requires: GNU parallel (install with: sudo apt install parallel)
# =============================================================================

set -e

# Configuration
PARALLEL_JOBS=${PARALLEL_JOBS:-3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$SCRIPT_DIR/run_parallel_$(date +%Y%m%d_%H%M%S).log"
SEEDS="42 100 123"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel is not installed"
    echo "Install with: sudo apt install parallel"
    echo ""
    echo "Alternatively, run the sequential version:"
    echo "  ./run_all_experiments.sh"
    exit 1
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Make all scripts executable
chmod +x "$CONFIG_DIR"/*.sh

# List of all experiments
EXPERIMENTS=(
    # Wave 1: GPT-5-mini reasoning (A1-A4) + IP scaling (B1-B3)
    "exp_A1_gpt5_none.sh"
    "exp_A2_gpt5_low.sh"
    "exp_A3_gpt5_medium.sh"
    "exp_A4_gpt5_high.sh"
    "exp_B1_ip_baseline.sh"
    "exp_B2_ip_aggressive.sh"
    "exp_B3_ip_high_only.sh"

    # Wave 2: Cluster volume (C1-C3) + Filters (D1-D3) + Minority (E1-E3)
    "exp_C1_5x9x5.sh"
    "exp_C2_8x9x5.sh"
    "exp_C3_5x9x9.sh"
    "exp_D1_relaxed.sh"
    "exp_D2_strict.sh"
    "exp_D3_boundary.sh"
    "exp_E1_target_worst.sh"
    "exp_E2_minority_boost.sh"
    "exp_E3_class_length_match.sh"

    # Wave 3: Length-aware (F1-F2) + Combinations (G1-G3) + Risky (H1-H2)
    "exp_F1_length_300.sh"
    "exp_F2_length_500.sh"
    "exp_G1_gpt5_ip_8x9.sh"
    "exp_G2_gpt5_relaxed_long.sh"
    "exp_G3_gpt5_minority_strict.sh"
    "exp_H1_contrastive.sh"
    "exp_H2_risky_boundary.sh"
)

TOTAL=${#EXPERIMENTS[@]}
START_TIME=$(date +%s)

echo "=============================================================="
echo "  Phase E Parallel Experiment Runner"
echo "  22 experiments × 3 seeds = 66 runs"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Estimated time: ~$((11 / PARALLEL_JOBS + 1)) hours"
echo "  Log file: $LOG_FILE"
echo "=============================================================="
echo ""

# Create a temporary file with experiment list
EXPERIMENT_FILE=$(mktemp)
for exp in "${EXPERIMENTS[@]}"; do
    echo "$exp" >> "$EXPERIMENT_FILE"
done

# Function to run a single experiment
run_single_experiment() {
    local exp="$1"
    local config_dir="$2"
    local seeds="$3"

    echo "[$(date +%H:%M:%S)] Starting: $exp"
    if bash "$config_dir/$exp" "$seeds" 2>&1; then
        echo "[$(date +%H:%M:%S)] Completed: $exp"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $exp"
    fi
}
export -f run_single_experiment

# Run experiments in parallel
echo "Starting parallel execution with $PARALLEL_JOBS jobs..."
echo ""

cat "$EXPERIMENT_FILE" | parallel -j "$PARALLEL_JOBS" --bar \
    "run_single_experiment {} '$CONFIG_DIR' '$SEEDS'" 2>&1 | tee "$LOG_FILE"

# Cleanup
rm -f "$EXPERIMENT_FILE"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=============================================================="
echo "  ALL EXPERIMENTS COMPLETED"
echo "  Total time: $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "=============================================================="

# Run analysis
echo ""
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze_results.py" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Done! Check $RESULTS_DIR for all outputs."
echo "Monitor live logs with: tail -f $RESULTS_DIR/*.log"
