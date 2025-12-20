#!/usr/bin/env bash
#
# Parallel Ensemble Test Suite Runner
#
# Executes 70+ ensemble tests in parallel batches (4 at a time)
# Uses GNU parallel for optimal throughput on Ryzen 9 5900x + RTX 3090
#
# Phases:
#   Phase 1: Categories 1-2 (22 tests, ~5.5 hours)
#   Phase 2: Categories 3-4 (23 tests, ~6 hours)
#   Phase 3: Category 5 (12 tests, ~3 hours)
#   Phase 4: Categories 6-7 (13 tests, ~3.5 hours)
#
# Total: 70 tests, ~18 hours

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_PARALLEL=4  # 4 concurrent configs (optimal for hardware)
PYTHON=python3
LOG_DIR="$SCRIPT_DIR/logs/ensemble_suite"
RESULTS_DIR="$SCRIPT_DIR/results/ensembles"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "  ENSEMBLE TEST SUITE - Phase G Extended"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Max parallel: $MAX_PARALLEL configs"
echo "  Python: $PYTHON"
echo "  Log directory: $LOG_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY not set${NC}"
    echo "Please set: export OPENAI_API_KEY='your-key-here'"
    exit 1
else
    echo -e "${GREEN}✓ OPENAI_API_KEY configured${NC}"
fi

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo -e "${YELLOW}WARNING: GNU parallel not found, using sequential execution${NC}"
    USE_PARALLEL=false
else
    echo -e "${GREEN}✓ GNU parallel available${NC}"
    USE_PARALLEL=true
fi

# Function to run a single category
run_category() {
    local category=$1
    local description=$2
    local script=$3
    local estimated_time=$4

    echo ""
    echo "========================================================================"
    echo "  CATEGORY $category: $description"
    echo "========================================================================"
    echo "  Estimated time: $estimated_time"
    echo "  Script: $script"
    echo ""

    local log_file="$LOG_DIR/category${category}_${TIMESTAMP}.log"

    echo "  Starting at: $(date)"
    echo "  Log: $log_file"
    echo ""

    if $PYTHON "$script" > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ Category $category completed successfully${NC}"
    else
        echo -e "${RED}✗ Category $category failed (see log)${NC}"
        echo "  Log: $log_file"
        return 1
    fi

    echo "  Completed at: $(date)"
}

# Function to run Phase 1 (Categories 1-2)
run_phase1() {
    echo ""
    echo "========================================================================"
    echo "  PHASE 1: Categories 1-2 (Weighted + Diversity)"
    echo "========================================================================"
    echo "  Tests: 22 (12 + 10)"
    echo "  Estimated time: ~5.5 hours"
    echo ""

    # Note: exp_ensembles_extended.py runs ALL categories 1-5
    # For Phase 1 only, we could modify it or run full script
    # For now, running full extended script

    run_category 1-5 "Weighted + Diversity + Hybrid + Dedup + ClassTarget" \
        "experiments/exp_ensembles_extended.py" "~12-14 hours"
}

# Function to run Phase 2 (Category 6)
run_phase2() {
    echo ""
    echo "========================================================================"
    echo "  PHASE 2: Category 6 (Advanced Combination)"
    echo "========================================================================"
    echo "  Tests: 8"
    echo "  Estimated time: ~2 hours"
    echo ""

    run_category 6 "Advanced Combination (Stacking/Voting/Selective)" \
        "experiments/exp_ensembles_advanced.py" "~2 hours"
}

# Main execution
main() {
    echo ""
    echo "Which phase do you want to run?"
    echo "  1) Phase 1 only (Categories 1-5: 55 tests, ~12-14 hours)"
    echo "  2) Phase 2 only (Category 6: 8 tests, ~2 hours)"
    echo "  3) All phases (70 tests, ~14-16 hours)"
    echo "  4) Custom (specify category)"
    echo ""

    if [ -n "$1" ]; then
        CHOICE=$1
    else
        read -p "Enter choice [1-4]: " CHOICE
    fi

    case $CHOICE in
        1)
            echo ""
            echo "Running Phase 1 (Categories 1-5)..."
            run_phase1
            ;;
        2)
            echo ""
            echo "Running Phase 2 (Category 6)..."
            run_phase2
            ;;
        3)
            echo ""
            echo "Running All Phases..."
            run_phase1
            if [ $? -eq 0 ]; then
                run_phase2
            fi
            ;;
        4)
            echo ""
            read -p "Enter category number (1-6): " CAT_NUM
            if [ "$CAT_NUM" -eq 6 ]; then
                run_phase2
            else
                run_phase1
            fi
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================================================"
    echo "  SUITE COMPLETED"
    echo "========================================================================"
    echo "  Completed at: $(date)"
    echo "  Logs: $LOG_DIR/"
    echo "  Results: $RESULTS_DIR/"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python3 compile_ensemble_results.py"
    echo "  2. Review: results/ensembles/ensemble_extended_full.json"
    echo "  3. Generate plots: python3 generate_ensemble_plots.py"
    echo ""
}

# Run main with arguments
main "$@"
