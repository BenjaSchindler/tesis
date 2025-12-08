#!/bin/bash
# =============================================================================
# Best Configuration for Small Datasets (< 10K samples)
# =============================================================================
# This script runs the optimal configuration discovered in Phase E testing:
#   - Fixed auto-params (no cluster override)
#   - 5x9 clusters (45 prompts per class)
#   - Purity-aware clustering enabled
#
# Results: +4.15% F1 improvement with 48 synthetics
# =============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

cd "$(dirname "$0")/.."

DATA_PATH="${1:-../../mbti_1.csv}"
SEED="${2:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="small_datasets_tests/results/best_config_s${SEED}_${TIMESTAMP}"

echo "========================================"
echo "  BEST CONFIG FOR SMALL DATASETS"
echo "  Data: $DATA_PATH"
echo "  Seed: $SEED"
echo "  Output: $OUT"
echo "========================================"

python3 -u core/runner_phase2.py \
    --data-path "$DATA_PATH" \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 128 \
    --cache-dir embeddings_cache \
    --llm-model gpt-4o-mini \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --use-class-description \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --auto-params \
    --synthetic-output ${OUT}_synth.csv \
    --augmented-train-output ${OUT}_aug.csv \
    --metrics-output ${OUT}_metrics.json \
    2>&1 | tee ${OUT}.log

echo ""
echo "========================================"
echo "  RESULTS:"
python3 -c "
import json
with open('${OUT}_metrics.json') as f:
    d = json.load(f)
b = d['baseline']['macro_f1']
a = d['augmented']['macro_f1']
s = d.get('synthetic_data', {}).get('accepted_count', 0)
print(f'  Baseline F1:  {b:.4f}')
print(f'  Augmented F1: {a:.4f}')
print(f'  Delta:        {(a-b)/b*100:+.2f}%')
print(f'  Synthetics:   {s}')
"
echo "========================================"
