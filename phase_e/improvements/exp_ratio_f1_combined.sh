#!/bin/bash
# ============================================================================
# Experimento: Budget combinado (Opción D)
# ============================================================================
# Hipótesis: Combinar cap-class-ratio + F1-based scaling da mejor control
#
# Lógica del budget:
#   budget_final = min(
#       cap_class_ratio × class_size,  # Límite proporcional al tamaño
#       f1_scaled_budget               # Límite basado en F1 (evita clases buenas)
#   )
#
# Configuración:
#   --cap-class-ratio 0.20          # Hasta 20% del tamaño de clase
#   --use-f1-budget-scaling         # Reduce budget para clases con alto F1
#   --f1-budget-multipliers 0.0 1.0 2.0  # Menos restrictivo que default
#
# Ejemplo de cálculo:
#   ISFJ (n=442, F1=0.252 MEDIUM):
#     ratio_budget = 442 × 0.20 = 88
#     f1_budget = base × 1.0 = 10  (ajustado)
#     final = min(88, 10) = 10
#
#   ENFJ (n=1043, F1=0.215 LOW):
#     ratio_budget = 1043 × 0.20 = 208
#     f1_budget = base × 2.0 = 20
#     final = min(208, 20) = 20
#
# Con esta config, las clases LOW (F1 < 0.20) reciben 2× más budget
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/ratio_f1_combined_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "========================================"
echo "  Experimento: RATIO + F1 COMBINED"
echo "  Seed: $SEED"
echo "  Output: $OUT"
echo "========================================"
echo ""
echo "Configuración de budget:"
echo "  --cap-class-ratio 0.20 (hasta 20% del tamaño)"
echo "  --f1-budget-multipliers 0.0 1.0 2.0"
echo "    HIGH (F1>0.45): skip"
echo "    MEDIUM (0.20<F1≤0.45): budget × 1.0"
echo "    LOW (F1≤0.20): budget × 2.0"
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
    --cap-class-ratio 0.20 \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.45 0.20 \
    --f1-budget-multipliers 0.0 1.0 2.0 \
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
