#!/bin/bash
#
# Fase B - Adaptive Weighting Experiments
# Optimizado para RTX 3070 (8GB VRAM)
#
# Dataset options:
#   MBTI_10k.csv  - 10K samples (~10 min con GPU)
#   MBTI_500.csv  - 106K samples (~45-60 min con GPU)
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Fase B - Adaptive Weighting (RTX 3070)          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠ OPENAI_API_KEY no está configurada${NC}"
    echo "Por favor, ejecuta:"
    echo "  export OPENAI_API_KEY='tu-api-key'"
    exit 1
fi

# Configuration
DATASET="${1:-MBTI_10k.csv}"  # Default: 10K para test rápido
SEED="${2:-42}"
OUTPUT_PREFIX="batch5_phaseB_gpu_seed${SEED}"

echo -e "${GREEN}Configuración:${NC}"
echo "  Dataset: $DATASET"
echo "  Seed: $SEED"
echo "  Device: CUDA (RTX 3070)"
echo "  Embedding Model: all-mpnet-base-v2 (mejor calidad)"
echo "  Batch Size: 128 (optimizado para 8GB VRAM)"
echo ""

# Estimate time
if [[ "$DATASET" == *"10k"* ]]; then
    echo -e "${GREEN}⏱ Tiempo estimado: 10-15 minutos${NC}"
elif [[ "$DATASET" == *"500"* ]]; then
    echo -e "${YELLOW}⏱ Tiempo estimado: 45-60 minutos${NC}"
fi
echo ""

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ GPU no detectada, usando CPU (será más lento)${NC}"
    DEVICE="cpu"
    BATCH_SIZE=32
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
    echo -e "${GREEN}✓ GPU detectada: $GPU_NAME ($GPU_MEM)${NC}"
    DEVICE="cuda"
    BATCH_SIZE=128
fi
echo ""

# Countdown
echo -e "${BLUE}Iniciando en 3 segundos...${NC}"
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo ""

# Launch
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  LANZANDO EXPERIMENTO FASE B                           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

python3 ../core/runner_phase2.py \
  --data-path "$DATASET" \
  --test-size 0.2 \
  --random-seed "$SEED" \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --device "$DEVICE" \
  --embedding-batch-size "$BATCH_SIZE" \
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
  --enable-adaptive-filters \
  --use-class-description \
  --enable-adaptive-weighting \
  --synthetic-weight 0.5 \
  --synthetic-weight-mode flat \
  --similarity-threshold 0.90 \
  --min-classifier-confidence 0.10 \
  --contamination-threshold 0.95 \
  --synthetic-output "${OUTPUT_PREFIX}_synthetic.csv" \
  --augmented-train-output "${OUTPUT_PREFIX}_augmented.csv" \
  --metrics-output "${OUTPUT_PREFIX}_metrics.json"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ EXPERIMENTO COMPLETADO EXITOSAMENTE                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Archivos generados:${NC}"
    ls -lh "${OUTPUT_PREFIX}"* 2>/dev/null
    echo ""
    echo -e "${BLUE}Ver resultados:${NC}"
    echo "  cat ${OUTPUT_PREFIX}_metrics.json | jq"
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠ EXPERIMENTO FALLÓ (exit code: $EXIT_CODE)          ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════╝${NC}"
fi
