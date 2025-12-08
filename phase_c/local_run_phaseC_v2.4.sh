#!/bin/bash
# Phase C v2.4 - Hardness-Aware Anchor Selection (Ensemble Method)
# Based on v2.1 (deterministic probabilistic gate) + ensemble anchor selection
# Expected: +0.2% to +0.5% additional improvement from boundary-aware anchors

set -e

# Verificar argumentos
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_path> <seed>"
    echo ""
    echo "Example:"
    echo "  ./local_run_phaseC_v2.4.sh ../MBTI_500.csv 42"
    echo ""
    echo "Phase C v2.4 Features:"
    echo "  ✅ All v2.1 features (deterministic gate, purity gate 0.025)"
    echo "  ✨ NEW: Hardness-aware anchor selection (ensemble method)"
    echo "  ✨ NEW: Selects anchors close to decision boundaries"
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
echo "  Phase C v2.4 - Hardness-Aware Anchor Selection"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo ""
echo "Phase C v2.4 NEW Features:"
echo "  🎯 HARDNESS-AWARE ANCHOR SELECTION:"
echo "     - Uses ensemble method (diversity + quality + hardness + stability)"
echo "     - Hardness score: proximity to decision boundary"
echo "     - Selects 'hard' samples near enemy classes"
echo "     - Expected gain: +0.2% to +0.5% MID-tier"
echo ""
echo "  All v2.1 features:"
echo "  🎲 Deterministic Probabilistic Gate (seeded RNG)"
echo "  🛡️  Purity Gate: threshold 0.025"
echo "  📊 Quality Gate: 0.25"
echo "  💰 F1 Budget: thresholds 0.40 0.20"
echo "  🌡️  Adaptive Temperature: temp=0.5 for MID-tier"
echo ""
echo "Expected vs v2.1:"
echo "  • v2.1: +1.72% MID-tier (with centroid anchors)"
echo "  • v2.4: +1.92% to +2.22% MID-tier (with hardness-aware anchors)"
echo "  • Improvement: Boundary samples should generate better synthetics"
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
LOG_FILE="phaseC_v2.4_seed${SEED}_${TIMESTAMP}.log"

echo ""
echo "Ejecutando Phase C v2.4 con seed $SEED..."
echo "Logs: $LOG_FILE"
echo ""

# Ejecutar runner con configuración Phase C v2.4
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
    \
    --enable-anchor-selection \
    --anchor-selection-method ensemble \
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
    --synthetic-output "phaseC_v2.4_seed${SEED}_synthetic.csv" \
    --augmented-train-output "phaseC_v2.4_seed${SEED}_augmented.csv" \
    --metrics-output "phaseC_v2.4_seed${SEED}_metrics.json" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase C v2.4 Completed!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  📊 phaseC_v2.4_seed${SEED}_metrics.json"
echo "  📝 phaseC_v2.4_seed${SEED}_synthetic.csv"
echo "  📈 phaseC_v2.4_seed${SEED}_augmented.csv"
echo "  📋 $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Compare with v2.1 (check if hardness-aware selection improved)"
echo "  2. Look for 'EnsembleAnchorSelector' in logs"
echo "  3. Check MID-tier improvement vs v2.1's +1.72%"
echo "  4. Analyze which classes benefited from boundary-aware anchors"
echo ""
