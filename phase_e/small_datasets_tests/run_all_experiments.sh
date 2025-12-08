#!/bin/bash
# =============================================================================
# Phase E: Run All 22 Experiments (66 runs with 3 seeds each)
# =============================================================================
# Estimated runtime: ~11 hours sequential, ~4 hours with parallel (3 jobs)
# =============================================================================

set -e

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$SCRIPT_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

# Make all scripts executable
chmod +x "$CONFIG_DIR"/*.sh

# Seeds to run
SEEDS="42 100 123"

echo "=============================================================="
echo "  Phase E Comprehensive Experiments"
echo "  22 experiments × 3 seeds = 66 runs"
echo "  Estimated time: ~11 hours (sequential), ~4 hours (parallel)"
echo "  Log file: $LOG_FILE"
echo "=============================================================="
echo ""

# List of all experiments in order
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
CURRENT=0
START_TIME=$(date +%s)

for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "=============================================================" | tee -a "$LOG_FILE"
    echo "  [$CURRENT/$TOTAL] Running: $exp" | tee -a "$LOG_FILE"
    echo "  Time: $(date)" | tee -a "$LOG_FILE"
    echo "=============================================================" | tee -a "$LOG_FILE"

    # Run experiment with all seeds
    if bash "$CONFIG_DIR/$exp" "$SEEDS" 2>&1 | tee -a "$LOG_FILE"; then
        echo "  ✓ $exp completed successfully" | tee -a "$LOG_FILE"
    else
        echo "  ✗ $exp failed!" | tee -a "$LOG_FILE"
    fi

    # Progress estimate
    ELAPSED=$(($(date +%s) - START_TIME))
    AVG_TIME=$((ELAPSED / CURRENT))
    REMAINING=$(((TOTAL - CURRENT) * AVG_TIME))
    echo "  Progress: $CURRENT/$TOTAL experiments" | tee -a "$LOG_FILE"
    echo "  Elapsed: $(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))" | tee -a "$LOG_FILE"
    echo "  ETA: $(printf '%02d:%02d:%02d' $((REMAINING/3600)) $((REMAINING%3600/60)) $((REMAINING%60)))" | tee -a "$LOG_FILE"
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "=============================================================" | tee -a "$LOG_FILE"
echo "  ALL EXPERIMENTS COMPLETED" | tee -a "$LOG_FILE"
echo "  Total time: $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))" | tee -a "$LOG_FILE"
echo "=============================================================" | tee -a "$LOG_FILE"

# Run analysis
echo "" | tee -a "$LOG_FILE"
echo "Running analysis..." | tee -a "$LOG_FILE"
python3 "$SCRIPT_DIR/analyze_results.py" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Done! Check $RESULTS_DIR for all outputs." | tee -a "$LOG_FILE"
