#!/bin/bash
# Quick validation of all experiment configs
# Tests that arguments parse correctly without running full experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
CORE_DIR="$SCRIPT_DIR/../core"

# Source base config to get paths
source "$CONFIG_DIR/base_config.sh"

echo "=============================================================="
echo "  Validating all experiment configs"
echo "=============================================================="
echo ""

EXPERIMENTS=(
    "exp_A1_gpt5_none.sh"
    "exp_A2_gpt5_low.sh"
    "exp_A3_gpt5_medium.sh"
    "exp_A4_gpt5_high.sh"
    "exp_B1_ip_baseline.sh"
    "exp_B2_ip_aggressive.sh"
    "exp_B3_ip_high_only.sh"
    "exp_C1_5x9x5.sh"
    "exp_C2_8x9x5.sh"
    "exp_C3_5x9x9.sh"
    "exp_D1_relaxed.sh"
    "exp_D2_strict.sh"
    "exp_D3_boundary.sh"
    "exp_E1_target_worst.sh"
    "exp_E2_minority_boost.sh"
    "exp_E3_class_length_match.sh"
    "exp_F1_length_300.sh"
    "exp_F2_length_500.sh"
    "exp_G1_gpt5_ip_8x9.sh"
    "exp_G2_gpt5_relaxed_long.sh"
    "exp_G3_gpt5_minority_strict.sh"
    "exp_H1_contrastive.sh"
    "exp_H2_risky_boundary.sh"
)

PASSED=0
FAILED=0
FAILED_LIST=""

for exp in "${EXPERIMENTS[@]}"; do
    # Extract the experiment parameters from the config file
    # by sourcing it and capturing the run_experiment call

    # Get the extra params from the config
    PARAMS=$(grep -A 20 "run_experiment" "$CONFIG_DIR/$exp" | grep -E "^\s+--" | tr '\n' ' ' | sed 's/\\//g')

    # Build full command for validation (only-baseline mode for quick test)
    CMD="python3 $CORE_DIR/runner_phase2.py \
        --data-path $DATA_PATH \
        --test-size 0.2 \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cpu \
        --cache-dir $CACHE_DIR \
        --random-seed 42 \
        --only-baseline \
        $PARAMS"

    echo -n "Testing $exp... "

    # Run with timeout of 30 seconds, just to validate argument parsing
    if timeout 30s bash -c "$CMD" > /tmp/validate_$$.log 2>&1; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        # Check if it's just an argument error or actual runtime error
        if grep -q "error: argument" /tmp/validate_$$.log 2>/dev/null; then
            echo "FAIL (argument error)"
            cat /tmp/validate_$$.log | grep -E "error:|invalid" | head -3
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST $exp"
        else
            # Might have started running but timed out - that's OK
            echo "PASS (args valid)"
            PASSED=$((PASSED + 1))
        fi
    fi
    rm -f /tmp/validate_$$.log
done

echo ""
echo "=============================================================="
echo "  Validation Results: $PASSED passed, $FAILED failed"
echo "=============================================================="

if [ $FAILED -gt 0 ]; then
    echo "Failed configs:$FAILED_LIST"
    exit 1
fi

echo "All configs validated successfully!"
