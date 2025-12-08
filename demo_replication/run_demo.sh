#!/bin/bash
# =============================================================================
# SMOTE-LLM Demo - Replication Runner
# =============================================================================
# Este script ejecuta N replicaciones del experimento SMOTE-LLM para medir
# la variabilidad por estocasticidad de los LLMs.
#
# Uso:
#   export OPENAI_API_KEY='sk-...'
#   ./run_demo.sh [num_replications]
#
# Ejemplo:
#   ./run_demo.sh 3    # Ejecuta 3 replicaciones (~2.5 horas, ~$17)
#   ./run_demo.sh 1    # Ejecuta 1 replicación (~45 min, ~$6)
#
# Requisitos:
#   - Python 3.10+
#   - GNU parallel
#   - CUDA (opcional, para GPU)
#   - OpenAI API key
#   - Dataset mbti_1.csv en el directorio padre
# =============================================================================

set -e

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEMO_DIR")"
NUM_REPLICATIONS="${1:-3}"
SEED=42
PARALLEL_JOBS=10

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           SMOTE-LLM Demo - Replication Runner                     ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# Verificaciones
# =============================================================================

echo "Verificando requisitos..."

# API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}ERROR: OPENAI_API_KEY no configurado${NC}"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} OPENAI_API_KEY configurado"

# GNU parallel
if ! command -v parallel &> /dev/null; then
    echo -e "${RED}ERROR: GNU parallel no encontrado${NC}"
    echo "  sudo apt install parallel"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} GNU parallel instalado"

# Dataset
if [ ! -f "$PROJECT_ROOT/mbti_1.csv" ]; then
    echo -e "${RED}ERROR: Dataset no encontrado: $PROJECT_ROOT/mbti_1.csv${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Dataset encontrado"

# Python dependencies
python3 -c "import pandas, numpy, sklearn, sentence_transformers" 2>/dev/null || {
    echo -e "${YELLOW}WARNING: Algunas dependencias de Python pueden faltar${NC}"
    echo "  pip install pandas numpy scikit-learn sentence-transformers openai"
}

echo ""
echo "Configuración:"
echo "  Replicaciones: $NUM_REPLICATIONS"
echo "  Seed: $SEED"
echo "  Configs: 9 (4 Phase F + 1 EXP + 4 Phase G)"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Costo estimado: ~\$$(echo "$NUM_REPLICATIONS * 5.60" | bc) USD"
echo "  Tiempo estimado: ~$(echo "$NUM_REPLICATIONS * 45" | bc) minutos"
echo ""

# =============================================================================
# Crear directorios de resultados
# =============================================================================

mkdir -p "$DEMO_DIR/results"

# =============================================================================
# Función para ejecutar una config
# =============================================================================

run_single_config() {
    local CONFIG_NAME="$1"
    local REPL_DIR="$2"
    local SEED="$3"

    echo "[$(date +%H:%M:%S)] Starting $CONFIG_NAME..."

    # Determinar paths según el tipo de config
    local CONFIG_PATH="$DEMO_DIR/configs/${CONFIG_NAME}.sh"
    local RESULTS_DIR="$DEMO_DIR/results"

    # Ejecutar config
    cd "$DEMO_DIR"
    SEED=$SEED BASE_DIR="$DEMO_DIR" PROJECT_ROOT="$PROJECT_ROOT" \
        bash "$CONFIG_PATH" > "$REPL_DIR/logs/${CONFIG_NAME}.log" 2>&1

    # Copiar resultado
    if [ -f "$RESULTS_DIR/${CONFIG_NAME}_s${SEED}_synth.csv" ]; then
        cp "$RESULTS_DIR/${CONFIG_NAME}_s${SEED}_synth.csv" "$REPL_DIR/results/"
        echo "[$(date +%H:%M:%S)] ✓ $CONFIG_NAME completado"
    else
        echo "[$(date +%H:%M:%S)] ✗ $CONFIG_NAME: synth.csv no encontrado"
    fi
}

export -f run_single_config
export DEMO_DIR PROJECT_ROOT

# =============================================================================
# Ejecutar replicaciones
# =============================================================================

START_TOTAL=$(date +%s)

for RUN_NUM in $(seq 1 $NUM_REPLICATIONS); do
    REPL_DIR="$DEMO_DIR/replication_run${RUN_NUM}"

    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║  REPLICACIÓN $RUN_NUM de $NUM_REPLICATIONS - $(date +%H:%M:%S)                              ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Crear directorios
    mkdir -p "$REPL_DIR/results"
    mkdir -p "$REPL_DIR/logs"

    # Lista de configs
    CONFIGS=(
        CMB3_skip
        CF1_conf_band
        V4_ultra
        G5_K25_medium
        EXP7_hybrid_best
        W9_contrastive
        W1_low_gate
        W1_force_problem
        W3_no_dedup
    )

    # Ejecutar en paralelo
    echo "Ejecutando 9 configs en paralelo..."
    printf '%s\n' "${CONFIGS[@]}" | parallel \
        --jobs "$PARALLEL_JOBS" \
        --line-buffer \
        --joblog "$REPL_DIR/logs/parallel_jobs.log" \
        "run_single_config {} '$REPL_DIR' '$SEED'"

    # Verificar resultados
    echo ""
    echo "Verificando resultados..."
    MISSING=0
    for cfg in "${CONFIGS[@]}"; do
        if [ -f "$REPL_DIR/results/${cfg}_s${SEED}_synth.csv" ]; then
            count=$(wc -l < "$REPL_DIR/results/${cfg}_s${SEED}_synth.csv")
            echo "  ✓ $cfg: $((count - 1)) sintéticos"
        else
            echo "  ✗ $cfg: FALTA"
            MISSING=$((MISSING + 1))
        fi
    done

    if [ $MISSING -gt 0 ]; then
        echo -e "${YELLOW}WARNING: $MISSING configs fallaron${NC}"
    fi

    # Crear ensembles
    echo ""
    echo "Creando ensembles..."
    python3 "$DEMO_DIR/scripts/create_ensembles_replication.py" \
        --repl-dir "$REPL_DIR/results" \
        --seed "$SEED" \
        2>&1 | tee "$REPL_DIR/logs/create_ensembles.log"

    # Evaluar con hold-out
    echo ""
    echo "Evaluando con hold-out..."
    for ens in ENS_Top3_G5 ENS_TopG5_Extended ENS_SUPER_G5_F7_v2; do
        if [ -f "$REPL_DIR/results/${ens}_synth.csv" ]; then
            echo "  Evaluando $ens..."
            python3 "$DEMO_DIR/scripts/eval_holdout_correct.py" \
                --synth "$REPL_DIR/results/${ens}_synth.csv" \
                --output "$REPL_DIR/results/${ens}_holdout.json" \
                2>&1 | tee "$REPL_DIR/logs/eval_${ens}.log"
        fi
    done

done

# =============================================================================
# Análisis de variabilidad
# =============================================================================

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Analizando variabilidad entre replicaciones                      ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$DEMO_DIR"
python3 scripts/analyze_replication_variance.py --runs "$NUM_REPLICATIONS" --base-dir "$DEMO_DIR"

# =============================================================================
# Resumen final
# =============================================================================

END_TOTAL=$(date +%s)
ELAPSED_TOTAL=$((END_TOTAL - START_TOTAL))
ELAPSED_HOURS=$((ELAPSED_TOTAL / 3600))
ELAPSED_MINS=$(((ELAPSED_TOTAL % 3600) / 60))

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  DEMO COMPLETADO                                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo "  Tiempo total: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "  Replicaciones: $NUM_REPLICATIONS"
echo ""
echo "  Resultados en:"
for i in $(seq 1 $NUM_REPLICATIONS); do
    echo "    - replication_run${i}/results/"
done
echo "    - replication_variance_analysis.json"
echo ""

# Mostrar resumen de resultados
if [ -f "$DEMO_DIR/replication_variance_analysis.json" ]; then
    echo "  Resumen de variabilidad:"
    python3 -c "
import json
with open('$DEMO_DIR/replication_variance_analysis.json') as f:
    data = json.load(f)
for name, info in data.items():
    print(f\"    {name}: {info['mean']:+.2f}% ± {info['std']:.2f}% (CV={info['cv']:.1f}%)\")
"
fi
