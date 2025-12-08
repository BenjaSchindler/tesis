#!/bin/bash
# Phase H: Multi-Classifier Evaluation
# Tests best synthetic data with multiple classification models
set -e
cd "$(dirname "$0")"

SEED=${SEED:-42}
SYNTH_DIR=${SYNTH_DIR:-"../phase_f/results"}
K=${K:-5}
REPEATS=${REPEATS:-3}

echo ""
echo "==============================================================="
echo "  Phase H: Multi-Classifier Evaluation"
echo "  Seed: $SEED | Date: $(date)"
echo "==============================================================="
echo ""
echo "Available classifiers:"
echo "  - logistic (Logistic Regression)"
echo "  - svm (SVM RBF)"
echo "  - svm_linear (SVM Linear)"
echo "  - random_forest (Random Forest)"
echo "  - gradient_boosting (Gradient Boosting)"
echo "  - mlp (MLP Neural Network)"
echo "  - xgboost (XGBoost)*"
echo "  - lightgbm (LightGBM)*"
echo "  * requires pip install xgboost lightgbm"
echo ""

mkdir -p results

# Check for required packages
echo "Checking dependencies..."
python3 -c "import xgboost" 2>/dev/null || echo "  Warning: xgboost not installed"
python3 -c "import lightgbm" 2>/dev/null || echo "  Warning: lightgbm not installed"
echo ""

# Find best synthetic file (ENS_Top3_G5 from Phase F)
BEST_SYNTH="$SYNTH_DIR/ENS_Top3_G5_s${SEED}_synth.csv"
if [ ! -f "$BEST_SYNTH" ]; then
    echo "Looking for alternative synthetic files..."
    BEST_SYNTH=$(ls -t $SYNTH_DIR/*_synth.csv 2>/dev/null | head -1)
fi

if [ -z "$BEST_SYNTH" ] || [ ! -f "$BEST_SYNTH" ]; then
    echo "ERROR: No synthetic data found in $SYNTH_DIR"
    echo "Please run Phase F first or specify SYNTH_DIR"
    exit 1
fi

echo "Using synthetic data: $BEST_SYNTH"
echo "K-Fold: k=$K, repeats=$REPEATS"
echo ""

# Run all classifiers
echo "=== Running ALL classifiers ==="
python3 -u multi_classifier_evaluator.py \
    --synth-csv "$BEST_SYNTH" \
    --classifier all \
    --k $K \
    --repeated $REPEATS \
    --seed $SEED \
    --output-dir results \
    2>&1 | tee results/phaseH_all_classifiers.log

echo ""
echo "==============================================================="
echo "  PHASE H COMPLETE"
echo "  Results: results/multiclf_*.json"
echo "  Finished: $(date)"
echo "==============================================================="
