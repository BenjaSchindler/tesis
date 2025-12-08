#!/bin/bash
# ============================================================================
# Experimento: Más sintéticos por clase (CORREGIDO)
# ============================================================================
# Hipótesis: Con más sintéticos (hasta 15% del tamaño de clase) el impacto será mayor
#
# Cambios vs baseline:
#   --max-clusters 5 (antes: 3)
#   --prompts-per-cluster 9 (antes: 3)
#   = 5 clusters × 9 prompts × 5 samples = 225 candidatos por clase
#
# CORRECCIÓN: Ahora usa --cap-class-ratio 0.15 para permitir hasta 15% del tamaño
#   ISFJ (n=442):  hasta 66 sintéticos
#   ESFP (n=245):  hasta 36 sintéticos
#   ISTJ (n=845):  hasta 126 sintéticos
#   ENFJ (n=1043): hasta 156 sintéticos
#
# Nota: Se DESHABILITA --use-f1-budget-scaling para evitar el budget de 5
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/more_synthetics_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "========================================"
echo "  Experimento: MORE SYNTHETICS"
echo "  Seed: $SEED"
echo "  Output: $OUT"
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
    --similarity-threshold 0.90 \
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
