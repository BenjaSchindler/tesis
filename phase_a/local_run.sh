#!/bin/bash
#
# Phase A - Local Execution Script
# Runs a single experiment with Phase A configuration
#
# Usage: ./local_run.sh [SEED]
#

set -e

SEED=${1:-42}
DATASET="../MBTI_500.csv"

echo "════════════════════════════════════════════════════════"
echo "  Phase A - Local Execution"
echo "  Seed: $SEED"
echo "════════════════════════════════════════════════════════"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set"
    echo "Please run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Check dataset exists
if [ ! -f "$DATASET" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET"
    exit 1
fi

# Check if running in venv (recommended)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ WARNING: Not running in virtual environment"
    echo "Recommended: python3 -m venv .venv && source .venv/bin/activate"
    echo ""
fi

echo "Starting Phase A experiment..."
echo ""

python3 ../core/runner_phase2.py \
    --data-path "$DATASET" \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cpu \
    --embedding-batch-size 32 \
    --llm-model gpt-4o-mini \
    --max-clusters 3 \
    --prompts-per-cluster 3 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --anchor-quality-threshold 0.50 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --enable-adaptive-filters \
    --use-class-description \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 30 70 100 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-output "phaseA_seed${SEED}_synthetic.csv" \
    --augmented-train-output "phaseA_seed${SEED}_augmented.csv" \
    --metrics-output "phaseA_seed${SEED}_metrics.json"

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  ✓ Experiment completed successfully"
    echo "════════════════════════════════════════════════════════"
    echo ""
    echo "Results:"
    echo "  - phaseA_seed${SEED}_metrics.json"
    echo "  - phaseA_seed${SEED}_synthetic.csv"
    echo "  - phaseA_seed${SEED}_augmented.csv"
    echo ""
else
    echo ""
    echo "❌ Experiment failed"
    exit 1
fi
