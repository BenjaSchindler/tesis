#!/bin/bash
# Compare multiple synthetic data sources with all classifiers
# Useful for comparing Phase F configs vs Phase G configs
set -e
cd "$(dirname "$0")"

SEED=${SEED:-42}
K=${K:-5}
REPEATS=${REPEATS:-3}

echo ""
echo "==============================================================="
echo "  Phase H: Compare Synthetic Data Sources"
echo "  K-Fold: k=$K, repeats=$REPEATS, seed=$SEED"
echo "==============================================================="

mkdir -p results

# List of synthetic files to compare
SYNTH_FILES=(
    # Phase F best configs
    "../phase_f/results/ENS_Top3_G5_s${SEED}_synth.csv"
    "../phase_f/results/CMB3_skip_s${SEED}_synth.csv"
    "../phase_f/results/V4_ultra_s${SEED}_synth.csv"
    # Phase G configs (when available)
    "../phase_g/results/G0_baseline_s${SEED}_synth.csv"
    "../phase_g/results/G5_combo_s${SEED}_synth.csv"
)

echo ""
for synth in "${SYNTH_FILES[@]}"; do
    if [ -f "$synth" ]; then
        name=$(basename "$synth" "_s${SEED}_synth.csv")
        echo "=== Evaluating: $name ==="

        python3 -u multi_classifier_evaluator.py \
            --synth-csv "$synth" \
            --classifier all \
            --k $K \
            --repeated $REPEATS \
            --seed $SEED \
            --output-dir "results" \
            2>&1 | tee "results/comparison_${name}.log"

        echo ""
    else
        echo "Skipping (not found): $synth"
    fi
done

# Summary
echo ""
echo "==============================================================="
echo "  COMPARISON COMPLETE"
echo "==============================================================="
echo ""
echo "Results saved in: results/"
ls -lh results/multiclf_*.json 2>/dev/null | tail -10
