#!/bin/bash
#
# Phase F Validation - Run All Experiments
#
# Usage: ./run_all.sh [experiment_number]
#   - No args: Run all experiments
#   - With arg: Run specific experiment (1-6)
#
# Logs are saved to results/validation.log
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure OPENAI_API_KEY is set (for experiments that need LLM)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Some experiments may fail."
fi

# Create directories
mkdir -p results latex_output cache

LOG_FILE="results/validation_$(date +%Y%m%d_%H%M%S).log"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Phase F Validation - Thesis Methodology Experiments      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Log file: $LOG_FILE"
echo "  Started: $(date)"
echo ""

run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local exp_script=$3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$exp_num/6] Running: $exp_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run with unbuffered output
    python3 -u "experiments/$exp_script" 2>&1 | tee -a "$LOG_FILE"

    echo "  ✓ Completed: $exp_name"
}

# Parse arguments
if [ $# -eq 0 ]; then
    # Run all experiments
    EXPERIMENTS="1 2 3 4 5 6"
else
    EXPERIMENTS="$@"
fi

# Run selected experiments
for exp in $EXPERIMENTS; do
    case $exp in
        1)
            run_experiment 1 "Clustering Validation (K_max)" "exp01_clustering.py"
            ;;
        2)
            run_experiment 2 "Anchor Strategies" "exp02_anchor_strategies.py"
            ;;
        3)
            run_experiment 3 "K Neighbors Validation" "exp03_k_neighbors.py"
            ;;
        4)
            run_experiment 4 "Filter Cascade" "exp04_filter_cascade.py"
            ;;
        5)
            run_experiment 5 "Adaptive Thresholds" "exp05_adaptive_thresholds.py"
            ;;
        6)
            run_experiment 6 "Tier Impact Analysis" "exp06_tier_impact.py"
            ;;
        *)
            echo "Unknown experiment: $exp (valid: 1-6)"
            ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   All Experiments Completed!                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results: results/"
echo "  LaTeX:   latex_output/"
echo "  Log:     $LOG_FILE"
echo ""
echo "  Finished: $(date)"
