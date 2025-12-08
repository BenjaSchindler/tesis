#!/bin/bash
# ============================================================================
# Experimento: Length-Aware Generation (Phase E)
# ============================================================================
# Hipotesis: Los sinteticos son ~93% mas cortos que los textos reales
#   - ESFJ: Real=515 words, Synth=31 words (-94%)
#   - ESFP: Real=495 words, Synth=37 words (-92%)
#
# Solucion: Usar --length-aware para generar textos de ~500 palabras
#
# Token calculation:
#   - 500 words ~= 650-750 tokens
#   - 5 samples per prompt = ~3500-4000 tokens needed
#   - Default max_tokens=180 is WAY too low, usamos 4000
#
# Este script usa el runner_phase2.py principal con el flag --length-aware
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/length_aware_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "========================================"
echo "  Experimento: LENGTH-AWARE GENERATION"
echo "  Seed: $SEED"
echo "  Output: $OUT"
echo "  Target: ~500 words per synthetic text"
echo "  LLM max tokens: 4000"
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
    --max-clusters 3 \
    --prompts-per-cluster 3 \
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
    --cap-class-ratio 0.10 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.45 0.20 \
    --f1-budget-multipliers 0.0 0.5 1.0 \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --length-aware \
    --length-target-words 500 \
    --llm-max-tokens 4000 \
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
print(f'  Sinteticos:   {s}')

# Word count analysis
import pandas as pd
try:
    df = pd.read_csv('${OUT}_synth.csv')
    avg_words = df['text'].apply(lambda x: len(str(x).split())).mean()
    print(f'  Avg words:    {avg_words:.0f}')
except:
    pass
"
echo "========================================"
