#!/bin/bash
# Run single classifier evaluation
# Usage: ./run_single_classifier.sh <classifier> <synth_csv> [seed]
# Example: ./run_single_classifier.sh xgboost ../phase_f/results/ENS_Top3_G5_s42_synth.csv 42
set -e
cd "$(dirname "$0")"

CLASSIFIER=${1:-"logistic"}
SYNTH_CSV=${2:-"../phase_f/results/ENS_Top3_G5_s42_synth.csv"}
SEED=${3:-42}
K=${K:-5}
REPEATS=${REPEATS:-3}

echo ""
echo "==============================================================="
echo "  Phase H: Single Classifier Evaluation"
echo "  Classifier: $CLASSIFIER"
echo "  Synthetic: $SYNTH_CSV"
echo "==============================================================="

mkdir -p results

python3 -u multi_classifier_evaluator.py \
    --synth-csv "$SYNTH_CSV" \
    --classifier "$CLASSIFIER" \
    --k $K \
    --repeated $REPEATS \
    --seed $SEED \
    --output-dir results

echo ""
echo "Done!"
