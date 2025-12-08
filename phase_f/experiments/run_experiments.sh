#!/bin/bash
# =============================================================================
# Phase F - New Experiments Launcher
# =============================================================================
# 8 new configs combining best of Phase E and Phase F
# Evaluated with K-fold (k=5, repeats=3, total=15 folds)
# Uses GNU parallel for efficient execution
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_F_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PHASE_F_DIR/results"

# Configuration
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"  # Default 8 parallel jobs
SEED="${SEED:-42}"                    # Single seed, then K-fold for validation

# Config list
CONFIGS=(
    EXP1_phaseE_port
    EXP2_relaxed_plus
    EXP3_minority_focus
    EXP4_ultra_relaxed
    EXP5_quality_focus
    EXP6_knn_strict
    EXP7_hybrid_best
    EXP8_intj_protect
)

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Phase F - New Experiments (K-fold validated)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel not found."
    echo "Install with: sudo apt install parallel"
    exit 1
fi
echo "✓ GNU parallel found"

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi
echo "✓ OPENAI_API_KEY set"

# Check if base_config.sh exists
if [ ! -f "$PHASE_F_DIR/base_config.sh" ]; then
    echo "ERROR: base_config.sh not found at $PHASE_F_DIR/base_config.sh"
    exit 1
fi
echo "✓ base_config.sh found"

# Create results directory
mkdir -p "$RESULTS_DIR"
echo "✓ Results directory: $RESULTS_DIR"

echo ""
echo "  Configuration:"
echo "    Configs: ${#CONFIGS[@]}"
echo "    Seed: $SEED"
echo "    Parallel jobs: $PARALLEL_JOBS"
echo "    Evaluation: K-fold (k=5, repeats=3)"
echo ""
echo "  Experiments:"
for cfg in "${CONFIGS[@]}"; do
    echo "    - $cfg"
done
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

# Confirm
read -p "Proceed with experiments? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# Phase 1: Generate synthetics (parallel)
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  PHASE 1: Generating Synthetics (parallel)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

# Create job list
JOB_FILE=$(mktemp)
for cfg in "${CONFIGS[@]}"; do
    echo "cd $SCRIPT_DIR && SEED=$SEED bash configs/${cfg}.sh"
done > "$JOB_FILE"

echo "Running ${#CONFIGS[@]} experiments with $PARALLEL_JOBS parallel jobs..."
echo "Logs: $RESULTS_DIR/{config}_s${SEED}.log"
echo ""

# Run with parallel (line-buffer for real-time output)
cat "$JOB_FILE" | parallel -j "$PARALLEL_JOBS" --progress --line-buffer

PHASE1_END=$(date +%s)
PHASE1_ELAPSED=$((PHASE1_END - START_TIME))

echo ""
echo "Phase 1 complete in $((PHASE1_ELAPSED / 60))m $((PHASE1_ELAPSED % 60))s"

rm -f "$JOB_FILE"

# =============================================================================
# Phase 2: K-fold evaluation (sequential)
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  PHASE 2: K-fold Evaluation"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

for cfg in "${CONFIGS[@]}"; do
    SYNTH_FILE="$RESULTS_DIR/${cfg}_s${SEED}_synth.csv"

    if [ ! -f "$SYNTH_FILE" ]; then
        echo "⚠ Skipping $cfg: synthetic file not found"
        continue
    fi

    echo "Evaluating: $cfg"
    python3 -u "$PHASE_F_DIR/kfold_evaluator.py" \
        --config "$cfg" \
        --seed "$SEED" \
        --k 5 \
        --repeated 3 \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/${cfg}_s${SEED}_kfold.log"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Total time: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s"
echo ""

printf "  %-25s %10s %10s %10s\n" "Config" "Delta%" "Win Rate" "p-value"
printf "  %-25s %10s %10s %10s\n" "-------------------------" "----------" "----------" "----------"

for cfg in "${CONFIGS[@]}"; do
    KFOLD_FILE="$RESULTS_DIR/${cfg}_s${SEED}_kfold_k5.json"

    if [ -f "$KFOLD_FILE" ]; then
        # Extract metrics using python
        METRICS=$(python3 -c "
import json
with open('$KFOLD_FILE') as f:
    d = json.load(f)
delta = d['delta']['mean'] * 100
win_rate = d['delta']['win_rate'] * 100
p_value = d['delta']['p_value']
print(f'{delta:+.3f}% {win_rate:.0f}% {p_value:.6f}')
" 2>/dev/null || echo "ERR ERR ERR")

        IFS=' ' read -r DELTA WIN PVAL <<< "$METRICS"
        printf "  %-25s %10s %9s%% %10s\n" "$cfg" "$DELTA" "$WIN" "$PVAL"
    else
        printf "  %-25s %10s %10s %10s\n" "$cfg" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo ""

# Find best config
BEST_CFG=""
BEST_DELTA=-999

for cfg in "${CONFIGS[@]}"; do
    KFOLD_FILE="$RESULTS_DIR/${cfg}_s${SEED}_kfold_k5.json"

    if [ -f "$KFOLD_FILE" ]; then
        DELTA=$(python3 -c "
import json
with open('$KFOLD_FILE') as f:
    d = json.load(f)
print(d['delta']['mean'] * 100)
" 2>/dev/null || echo "-999")

        if (( $(echo "$DELTA > $BEST_DELTA" | bc -l) )); then
            BEST_DELTA=$DELTA
            BEST_CFG=$cfg
        fi
    fi
done

if [ -n "$BEST_CFG" ]; then
    echo "  🏆 Best config: $BEST_CFG with delta=${BEST_DELTA}%"

    # Compare to baseline
    if (( $(echo "$BEST_DELTA > 1.29" | bc -l) )); then
        echo "     ✅ BEATS ENS_Top3_G5 (+1.29%)!"
    elif (( $(echo "$BEST_DELTA > 1.22" | bc -l) )); then
        echo "     ⚠️ Better than ENS_Top3 (+1.22%) but not ENS_Top3_G5"
    else
        echo "     ❌ Below ENS_Top3 (+1.22%)"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
