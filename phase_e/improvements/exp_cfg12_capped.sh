#!/bin/bash
# ============================================================================
# Experimento: cfg12_sim_095 + cap-class-ratio + más candidatos
# ============================================================================
# Hipótesis: cfg12 ganó con +0.82% pero solo generó 30 sintéticos.
#            Con más candidatos y cap-class-ratio puede mejorar más.
#
# Basado en cfg12_sim_095 (el ganador), agregando:
#   --similarity-threshold 0.95 (igual que cfg12)
#   --max-clusters 5 (antes: 3)
#   --prompts-per-cluster 9 (antes: 3)
#   --cap-class-ratio 0.15 (nuevo)
#   = 5 clusters × 9 prompts × 5 samples = 225 candidatos por clase
#
# Sin --use-f1-budget-scaling para evitar limitación artificial
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/cfg12_capped_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "========================================"
echo "  Experimento: CFG12_SIM095 + CAPPED"
echo "  Seed: $SEED"
echo "  Output: $OUT"
echo "  Best config (sim_095) + more synthetics"
echo "========================================"

python3 -u core/runner_phase2.py \
    --data-path ../MBTI_500.csv \
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
    --anchor-quality-threshold 0.30 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --use-class-description \
    --cap-class-ratio 0.15 \
    --similarity-threshold 0.95 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --synthetic-output ${OUT}_synth.csv \
    --augmented-train-output ${OUT}_aug.csv \
    --metrics-output ${OUT}_metrics.json \
    2>&1 | tee ${OUT}.log

echo ""
echo "========================================"
echo "  Resultados:"
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
print(f'  Sintéticos:   {s}')
"
echo "========================================"
