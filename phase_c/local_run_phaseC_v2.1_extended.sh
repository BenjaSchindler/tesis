#!/bin/bash
# Phase C v2.1 Extended - Target All MID + UPPER-MID Classes (F1 < 0.60)
# Based on v2.1 but extends coverage from 7/16 to 13/16 classes
# Expected: +0.8-1.0% overall macro F1 (vs v2.1's +0.377%)

set -e

# Verificar argumentos
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_path> <seed>"
    echo ""
    echo "Example:"
    echo "  ./local_run_phaseC_v2.1_extended.sh ../MBTI_500.csv 42"
    echo ""
    echo "Phase C v2.1 Extended Features:"
    echo "  ✨ Targets 13/16 classes (F1 < 0.60) - MID + UPPER-MID tiers"
    echo "  ✨ Skips only 3 HIGH-tier classes (F1 >= 0.60)"
    echo "  ✅ All v2.1 features (purity gate, deterministic RNG)"
    echo "  ✅ Expected: +0.8-1.0% overall macro F1"
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
echo "  Phase C v2.1 Extended - Wider Coverage"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo ""
echo "Phase C v2.1 Extended Changes:"
echo "  ✨ EXTENDED F1 BUDGET RANGE:"
echo "     - v2.1: Targets F1 0.20-0.40 (7 classes)"
echo "     - Extended: Targets F1 < 0.60 (13 classes)"
echo "     - Adds 6 UPPER-MID classes: INFP, INFJ, ENFP, INTJ, ENTP, ISTP"
echo ""
echo "  All v2.1 features:"
echo "  🎲 Deterministic Probabilistic Gate (seeded RNG)"
echo "  🛡️  Purity Gate: threshold 0.025"
echo "  📊 Quality Gate: 0.25"
echo "  💰 F1 Budget: thresholds 0.60 0.00 (extended range)"
echo "  🌡️  Adaptive Temperature: temp=0.5 for MID-tier"
echo ""
echo "Expected vs v2.1:"
echo "  • v2.1: +0.377% overall, +1.72% MID-tier (7/16 classes)"
echo "  • Extended: +0.8-1.0% overall (13/16 classes)"
echo "  • Goal: Beat Phase A target (+1.00% overall)"
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
LOG_FILE="phaseC_v2.1_extended_seed${SEED}_${TIMESTAMP}.log"

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
    --f1-budget-thresholds 0.60 0.00 \
    --f1-budget-multipliers 30 70 100 \
    \
    --enable-adaptive-weighting \
    --synthetic-weight 0.5 \
    \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    \
    --synthetic-output "phaseC_v2.1_extended_seed${SEED}_synthetic.csv" \
    --augmented-train-output "phaseC_v2.1_extended_seed${SEED}_augmented.csv" \
    --metrics-output "phaseC_v2.1_extended_seed${SEED}_metrics.json" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase C v2.1 Extended Completed!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  📊 phaseC_v2.1_extended_seed${SEED}_metrics.json"
echo "  📝 phaseC_v2.1_extended_seed${SEED}_synthetic.csv"
echo "  📈 phaseC_v2.1_extended_seed${SEED}_augmented.csv"
echo "  📋 $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Compare with v2.1 (check if wider coverage improved overall F1)"
echo "  2. Expected: +0.8-1.0% overall (vs v2.1's +0.377%)"
echo "  3. Check which UPPER-MID classes benefited (INFP, INFJ, ENFP, INTJ, ENTP, ISTP)"
echo ""
