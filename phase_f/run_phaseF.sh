#!/bin/bash
# Phase F - Calibrated Experiments Launcher
# 18 configs × 3 seeds = 54 runs
# Uses GNU parallel for efficient execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
SEEDS="${SEEDS:-42 100 123}"
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"  # 10 parallel OK when embeddings are pre-cached

# Original 12 experiments (gpt-4o-mini based)
CONFIGS_ORIGINAL=(V1_baseline V2_high_vol V3_low_vol CF1_conf_band CF2_knn_only CF3_relaxed IP1_2x IP2_3x IP3_skip_high CMB1_balanced CMB2_aggressive CMB3_skip)

# New GPT-5-mini experiments with varying K (samples-per-prompt) and reasoning
CONFIGS_GPT5=(G5_K5_none G5_K15_none G5_K25_none G5_K15_low G5_K25_medium G5_K100_medium)

# Combined array
CONFIGS=("${CONFIGS_ORIGINAL[@]}" "${CONFIGS_GPT5[@]}")

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel not found. Install with: sudo apt install parallel"
    exit 1
fi

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Phase F - Calibrated Experiments"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Configs: ${#CONFIGS[@]} (12 original + 6 GPT-5-mini)"
echo "  Seeds: $SEEDS"
echo "  Total runs: $((${#CONFIGS[@]} * $(echo $SEEDS | wc -w)))"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo ""
echo "  Experiments (gpt-4o-mini):"
echo "    WAVE 1 (Volume):     V1_baseline, V2_high_vol, V3_low_vol"
echo "    WAVE 2 (Confidence): CF1_conf_band, CF2_knn_only, CF3_relaxed"
echo "    WAVE 3 (IP Scaling): IP1_2x, IP2_3x, IP3_skip_high"
echo "    WAVE 4 (Combos):     CMB1_balanced, CMB2_aggressive, CMB3_skip"
echo ""
echo "  Experiments (gpt-5-mini):"
echo "    WAVE 5 (Context K):  G5_K5_none (baseline), G5_K15_none, G5_K25_none"
echo "    WAVE 5 (Reasoning):  G5_K15_low, G5_K25_medium, G5_K100_medium (TRYHARD)"
echo ""
echo "═══════════════════════════════════════════════════════════════"

# Confirm
read -p "Proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results

# Create job list
JOB_FILE=$(mktemp)
for cfg in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
        echo "SEED=$seed bash configs/${cfg}.sh"
    done
done > "$JOB_FILE"

echo ""
echo "Starting $(wc -l < "$JOB_FILE") experiments with $PARALLEL_JOBS parallel jobs..."
echo "Logs will be saved to results/"
echo ""

# Run with parallel
START_TIME=$(date +%s)
cat "$JOB_FILE" | parallel -j "$PARALLEL_JOBS" --progress --joblog results/parallel_joblog.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase F Complete!"
echo "  Time: ${HOURS}h ${MINUTES}m"
echo "  Results in: $SCRIPT_DIR/results/"
echo "═══════════════════════════════════════════════════════════════"

# Summary
echo ""
echo "Quick summary of results:"
echo ""
echo "  --- gpt-4o-mini experiments ---"
for cfg in "${CONFIGS_ORIGINAL[@]}"; do
    echo -n "  $cfg: "
    for seed in $SEEDS; do
        if [ -f "results/${cfg}_s${seed}_metrics.json" ]; then
            delta=$(python3 -c "import json; d=json.load(open('results/${cfg}_s${seed}_metrics.json')); print(f\"{d.get('improvement', {}).get('f1_delta_pct', 0):+.2f}%\")" 2>/dev/null || echo "ERR")
            echo -n "s$seed=$delta "
        else
            echo -n "s$seed=MISSING "
        fi
    done
    echo ""
done
echo ""
echo "  --- gpt-5-mini experiments ---"
for cfg in "${CONFIGS_GPT5[@]}"; do
    echo -n "  $cfg: "
    for seed in $SEEDS; do
        if [ -f "results/${cfg}_s${seed}_metrics.json" ]; then
            delta=$(python3 -c "import json; d=json.load(open('results/${cfg}_s${seed}_metrics.json')); print(f\"{d.get('improvement', {}).get('f1_delta_pct', 0):+.2f}%\")" 2>/dev/null || echo "ERR")
            echo -n "s$seed=$delta "
        else
            echo -n "s$seed=MISSING "
        fi
    done
    echo ""
done

rm -f "$JOB_FILE"
