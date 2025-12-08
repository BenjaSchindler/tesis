#!/bin/bash
# Phase C v2.3 - Graduated Purity Threshold
# Small classes (n<150) get 0.7× threshold: 0.025 → 0.0175
# ESFJ (n=123) and ISFJ (n=442) will pass
# Expected: Both generate synthetics without overly blocking classes

set -e

# Verificar argumentos
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_path> <seed>"
    echo ""
    echo "Example:"
    echo "  ./local_run_phaseC.sh ../MBTI_500.csv 42"
    echo ""
    echo "Phase C Features:"
    echo "  ✅ Adaptive temperature (0.5 for MID-tier, 0.8 for LOW-tier)"
    echo "  ✅ All Phase A quality mechanisms"
    echo "  ✅ All Phase B adaptive weighting"
    exit 1
fi

DATASET=$1
SEED=$2

# Verificar que OPENAI_API_KEY esté configurada
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY no está configurada"
    echo "Ejecuta: export OPENAI_API_KEY='tu-api-key'"
    exit 1
fi

echo "════════════════════════════════════════════════════════"
echo "  Phase C v2.3 - Graduated Purity Threshold"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo ""
echo "Phase C v2.3 Changes:"
echo "  🎓 GRADUATED PURITY THRESHOLD:"
echo "     - Small classes (n<150): threshold × 0.7"
echo "     - ESFJ (n=123): 0.025 → 0.0175 (purity=0.009) → will PASS ✓"
echo "     - ISFJ (n=442): keeps 0.025 (purity=0.016) → will BLOCK ✗"
echo "     - Adapts to class size instead of fixed threshold"
echo ""
echo "  All v2.1 features:"
echo "  🎲 Deterministic probabilistic gate (seed=$SEED)"
echo "  📊 Quality Gate: 0.25"
echo "  💰 F1 Budget: thresholds 0.40 0.20"
echo "  🌡️  Adaptive Temperature: temp=0.5 for MID-tier"
echo ""
echo "Expected vs v2 (fixed threshold 0.025):"
echo "  • ESFJ: Generates ~10 synthetics (was 0 in v2)"
echo "  • Test if very small classes benefit from relaxed threshold"
echo "  • MID-tier: +0.10% to +0.25%"
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# Determinar dispositivo (GPU si está disponible, CPU sino)
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    BATCH_SIZE=128
    echo "✅ GPU detectada - usando CUDA"
else
    DEVICE="cpu"
    BATCH_SIZE=32
    echo "⚠️  GPU no detectada - usando CPU"
fi

# Timestamp para logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="phaseC_v2.3_seed${SEED}_${TIMESTAMP}.log"

echo ""
echo "Ejecutando Phase C con seed $SEED..."
echo "Logs: $LOG_FILE"
echo ""

# Ejecutar runner con configuración Phase C
python3 ../core/runner_phase2.py \
    --data-path "$DATASET" \
    --test-size 0.2 \
    --random-seed "$SEED" \
    \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device "$DEVICE" \
    --embedding-batch-size "$BATCH_SIZE" \
    \
    --llm-model gpt-4o-mini \
    --temperature 1.0 \
    --max-clusters 3 \
    --prompts-per-cluster 3 \
    --prompt-mode mix \
    \
    --use-ensemble-selection \
    \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    \
    --enable-anchor-gate \
    --anchor-quality-threshold 0.25 \
    --purity-gate-threshold 0.025 \
    --purity-gate-graduated \
    \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    \
    --enable-adaptive-filters \
    \
    --use-class-description \
    \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.40 0.20 \
    --f1-budget-multipliers 30 70 100 \
    \
    --enable-adaptive-weighting \
    --synthetic-weight 0.5 \
    \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    \
    --synthetic-output "phaseC_v2.3_seed${SEED}_synthetic.csv" \
    --augmented-train-output "phaseC_v2.3_seed${SEED}_augmented.csv" \
    --metrics-output "phaseC_v2.3_seed${SEED}_metrics.json" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase C v2.3 Completed!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  📊 phaseC_v2.3_seed${SEED}_metrics.json"
echo "  📝 phaseC_v2.3_seed${SEED}_synthetic.csv"
echo "  📈 phaseC_v2.3_seed${SEED}_augmented.csv"
echo "  📋 $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check graduated threshold messages in log"
echo "  2. Verify ESFJ generated synthetics (small class exception)"
echo "  3. Compare MID-tier vs v2/v2.2 to see best threshold approach"
echo ""
