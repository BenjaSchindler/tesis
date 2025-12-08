#!/bin/bash
# Replicación completa desde cero
# Fecha: 2025-12-06
# Objetivo: Medir variabilidad por estocasticidad de LLMs
#
# Uso:
#   export OPENAI_API_KEY='...'
#   ./run_replication.sh [run_number]
#
# Ejemplo:
#   ./run_replication.sh 1   # Crea replication_run1
#   ./run_replication.sh 2   # Crea replication_run2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/benja/Desktop/Tesis/SMOTE-LLM"
RUN_NUM="${1:-1}"
REPL_DIR="$SCRIPT_DIR/replication_run${RUN_NUM}"
SEED=42  # Mismo seed para split, diferente generación LLM
PARALLEL_JOBS=10  # Número de jobs en paralelo

# Verificar API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY no configurado"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Verificar GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel no encontrado"
    echo "  sudo apt install parallel"
    exit 1
fi

# Crear directorios
mkdir -p "$REPL_DIR/results"
mkdir -p "$REPL_DIR/logs"

echo "═══════════════════════════════════════════════════════════════════"
echo "  REPLICACIÓN #${RUN_NUM} - Medición de Variabilidad LLM"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Directorio: $REPL_DIR"
echo "  Seed: $SEED"
echo "  Configs: 9 (4 Phase F + 1 EXP + 4 Phase G)"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Costo estimado: ~\$5.60"
echo "  Tiempo estimado: ~30-40 min (con paralelismo)"
echo ""

START_TIME=$(date +%s)

# ============================================================================
# Función para ejecutar una config
# ============================================================================
run_config() {
    local CONFIG_PATH="$1"
    local CONFIG_NAME="$2"
    local PHASE="$3"
    local DEST_DIR="$4"
    local SEED="$5"

    echo "[$(date +%H:%M:%S)] Starting $CONFIG_NAME..."

    cd "$PROJECT_ROOT/$PHASE"
    SEED=$SEED bash "$CONFIG_PATH" 2>&1 | tee "$DEST_DIR/logs/${CONFIG_NAME}.log"

    # Copiar resultado
    if [ -f "results/${CONFIG_NAME}_s${SEED}_synth.csv" ]; then
        cp "results/${CONFIG_NAME}_s${SEED}_synth.csv" "$DEST_DIR/results/"
        echo "[$(date +%H:%M:%S)] ✓ $CONFIG_NAME completado"
    else
        echo "[$(date +%H:%M:%S)] ✗ $CONFIG_NAME: synth.csv no encontrado"
    fi
}

export -f run_config
export PROJECT_ROOT REPL_DIR SEED

# ============================================================================
# Crear archivo de jobs para GNU parallel
# ============================================================================
JOB_FILE=$(mktemp)

cat > "$JOB_FILE" << 'JOBS'
phase_f|configs/CMB3_skip.sh|CMB3_skip
phase_f|configs/CF1_conf_band.sh|CF1_conf_band
phase_f|configs/V4_ultra.sh|V4_ultra
phase_f|configs/G5_K25_medium.sh|G5_K25_medium
phase_f|experiments/configs/EXP7_hybrid_best.sh|EXP7_hybrid_best
phase_g|configs/W9_contrastive.sh|W9_contrastive
phase_g|configs/W1_low_gate.sh|W1_low_gate
phase_g|configs/W1_force_problem.sh|W1_force_problem
phase_g|configs/W3_no_dedup.sh|W3_no_dedup
JOBS

echo "═══════════════════════════════════════════════════════════════════"
echo "  Ejecutando 9 configs en paralelo (max $PARALLEL_JOBS)..."
echo "═══════════════════════════════════════════════════════════════════"

# Ejecutar en paralelo con GNU parallel
# --line-buffer: output sin buffering
# --progress: mostrar barra de progreso
# --joblog: guardar log de jobs
cat "$JOB_FILE" | parallel \
    --jobs "$PARALLEL_JOBS" \
    --colsep '\|' \
    --line-buffer \
    --joblog "$REPL_DIR/logs/parallel_jobs.log" \
    --progress \
    "
    PHASE={1}
    CONFIG_PATH={2}
    CONFIG_NAME={3}

    echo '['\$(date +%H:%M:%S)'] Starting {3}...'

    cd '$PROJECT_ROOT'/\$PHASE
    SEED=$SEED bash \$CONFIG_PATH > '$REPL_DIR/logs/{3}.log' 2>&1

    if [ -f 'results/{3}_s${SEED}_synth.csv' ]; then
        cp 'results/{3}_s${SEED}_synth.csv' '$REPL_DIR/results/'
        echo '['\$(date +%H:%M:%S)'] ✓ {3} completado'
    else
        echo '['\$(date +%H:%M:%S)'] ✗ {3}: synth.csv no encontrado'
    fi
    "

rm -f "$JOB_FILE"

# ============================================================================
# Verificar resultados
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Verificando resultados..."
echo "═══════════════════════════════════════════════════════════════════"

EXPECTED_CONFIGS=(
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

MISSING=0
for cfg in "${EXPECTED_CONFIGS[@]}"; do
    if [ -f "$REPL_DIR/results/${cfg}_s${SEED}_synth.csv" ]; then
        count=$(wc -l < "$REPL_DIR/results/${cfg}_s${SEED}_synth.csv")
        echo "  ✓ $cfg: $((count - 1)) sintéticos"
    else
        echo "  ✗ $cfg: FALTA"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  ⚠️  $MISSING configs fallaron. Revisar logs en $REPL_DIR/logs/"
    echo "  Puedes re-ejecutar las fallidas manualmente."
fi

# ============================================================================
# Crear ensembles
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Creando ensembles..."
echo "═══════════════════════════════════════════════════════════════════"

cd "$SCRIPT_DIR"
python3 -u create_ensembles_replication.py --repl-dir "$REPL_DIR/results" 2>&1 | tee "$REPL_DIR/logs/create_ensembles.log"

# ============================================================================
# Evaluar con hold-out (usa cache de embeddings)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Evaluando con hold-out (usa cache de embeddings)..."
echo "═══════════════════════════════════════════════════════════════════"

# Evaluar los 3 ensembles
for ens in ENS_Top3_G5 ENS_TopG5_Extended ENS_SUPER_G5_F7_v2; do
    if [ -f "$REPL_DIR/results/${ens}_synth.csv" ]; then
        echo "  Evaluando $ens..."
        python3 -u eval_holdout_correct.py \
            --synth "$REPL_DIR/results/${ens}_synth.csv" \
            --output "$REPL_DIR/results/${ens}_holdout.json" \
            2>&1 | tee "$REPL_DIR/logs/eval_${ens}.log"
    fi
done

# ============================================================================
# Resumen
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  REPLICACIÓN #${RUN_NUM} COMPLETADA"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Tiempo total: ${ELAPSED_MIN} minutos"
echo "  Resultados en: $REPL_DIR/results/"
echo ""

# Mostrar resultados
echo "  Resultados de evaluación:"
for ens in ENS_Top3_G5 ENS_TopG5_Extended ENS_SUPER_G5_F7_v2; do
    json_file="$REPL_DIR/results/${ens}_holdout.json"
    if [ -f "$json_file" ]; then
        delta=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d['delta_percent']:+.2f}%\")")
        synth=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['n_synthetic'])")
        echo "    $ens: $delta ($synth sintéticos)"
    fi
done

echo ""
echo "  Para ejecutar otra replicación:"
echo "    ./run_replication.sh $((RUN_NUM + 1))"
echo ""
echo "  Para analizar variabilidad (después de 2+ runs):"
echo "    python3 analyze_replication_variance.py --runs $((RUN_NUM + 1))"
