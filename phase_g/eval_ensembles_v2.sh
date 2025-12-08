#!/bin/bash
# Evaluate cross-phase ensembles with K-fold
# Usage: bash eval_ensembles_v2.sh

set -e
cd "$(dirname "$0")"

ENSEMBLE_DIR="results/ensembles_v2"
OUTPUT_FILE="results/ensembles_v2/kfold_ensembles_results.json"

echo "=============================================="
echo "Evaluating Cross-Phase Ensembles with K-fold"
echo "=============================================="
echo ""
echo "Ensemble directory: $ENSEMBLE_DIR"
echo "K=5, repeats=3 (15 total folds)"
echo ""

# List ensembles
echo "Ensembles to evaluate:"
for f in $ENSEMBLE_DIR/*_synth.csv; do
    name=$(basename "$f" _s42_synth.csv)
    lines=$(($(wc -l < "$f") - 1))
    echo "  - $name ($lines synthetics)"
done
echo ""

# Create temporary symlinks in results/ for the evaluator
echo "Creating temporary links..."
for f in $ENSEMBLE_DIR/*_s42_synth.csv; do
    name=$(basename "$f")
    ln -sf "ensembles_v2/$name" "results/$name" 2>/dev/null || true
done

# Run evaluation
echo ""
echo "Running K-fold evaluation (this may take ~30 minutes)..."
echo ""

python3 -u kfold_multimodel.py \
    --all \
    --k 5 \
    --repeats 3 \
    --models LogisticRegression \
    --output "$OUTPUT_FILE" \
    2>&1 | tee results/ensembles_v2/eval.log

# Cleanup symlinks
echo ""
echo "Cleaning up temporary links..."
for f in $ENSEMBLE_DIR/*_s42_synth.csv; do
    name=$(basename "$f")
    rm -f "results/$name" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
echo "Results: $OUTPUT_FILE"
