#!/bin/bash
set -e
cd "$(dirname "$0")"
SEED=42

echo "═══════════════════════════════════════════════════════════"
echo "  Overnight Run - K-Fold Standard"
echo "  Seed: $SEED (single seed, K-Fold provides robustness)"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

# Ensure results directory exists
mkdir -p results/Variance_tests

# Part 1: K-Fold on existing synthetics (parallel, ~3 min)
echo ""
echo "=== PART 1: K-Fold on Existing Synthetics ==="
echo "Running 4 K-Fold evaluations in parallel..."

EXISTING_CONFIGS="G5_K25_medium CMB3_skip CF1_conf_band V1_baseline"
for cfg in $EXISTING_CONFIGS; do
    echo "python3 -u kfold_evaluator.py --config $cfg --seed $SEED --k 5 --repeated 3 --output results/Variance_tests/${cfg}_kfold.json 2>&1 | tee results/Variance_tests/${cfg}_kfold.log"
done | parallel -j 4

echo "Part 1 complete: $(date)"

# Part 2: New generation experiments (6 in parallel for ~45 min)
echo ""
echo "=== PART 2: New Generation Experiments (6 parallel) ==="

NEW_CONFIGS="V4_ultra V5_extreme G5V2_K25 G5V2_K25_med V2_presence V2_f1scaled"

# Create jobs file for parallel execution
for cfg in $NEW_CONFIGS; do
    echo "SEED=$SEED bash configs/${cfg}.sh && python3 -u kfold_evaluator.py --config $cfg --seed $SEED --k 5 --repeated 3 --output results/Variance_tests/${cfg}_kfold.json 2>&1 | tee results/Variance_tests/${cfg}_kfold.log"
done > /tmp/overnight_jobs.txt

echo "Launching 6 experiments: $NEW_CONFIGS"
cat /tmp/overnight_jobs.txt | parallel -j 6 --progress

echo "Part 2 complete: $(date)"

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RUN COMPLETE - Summary"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════"

echo ""
echo "K-Fold Results (sorted by delta):"
for f in results/Variance_tests/*_kfold.json; do
    cfg=$(basename "$f" _kfold.json)
    delta=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['delta']['mean']*100:+.2f}% (p={d['delta']['p_value']:.4f})\")" 2>/dev/null || echo "ERR")
    printf "  %-20s %s\n" "$cfg" "$delta"
done | sort -t'+' -k2 -rn

echo ""
echo "Reference: V2_high_vol = +1.88% (p=0.0016)"
