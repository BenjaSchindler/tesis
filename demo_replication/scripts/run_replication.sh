#!/bin/bash
# Replicación completa desde cero (versión demo)
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

# Auto-detect paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEMO_DIR")"
RUN_NUM="${1:-1}"
REPL_DIR="$DEMO_DIR/replication_run${RUN_NUM}"
SEED=42
PARALLEL_JOBS=10

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
echo "  REPLICACIÓN #${RUN_NUM} - Demo"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Directorio: $REPL_DIR"
echo "  Seed: $SEED"
echo "  Configs: 9"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo ""

START_TIME=$(date +%s)

# Lista de configs (todos en configs/)
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

# Exportar variables para parallel
export DEMO_DIR PROJECT_ROOT REPL_DIR SEED

echo "═══════════════════════════════════════════════════════════════════"
echo "  Ejecutando 9 configs en paralelo..."
echo "═══════════════════════════════════════════════════════════════════"

# Ejecutar en paralelo
printf '%s\n' "${CONFIGS[@]}" | parallel \
    --jobs "$PARALLEL_JOBS" \
    --line-buffer \
    --joblog "$REPL_DIR/logs/parallel_jobs.log" \
    "
    CONFIG_NAME={}
    echo '['\$(date +%H:%M:%S)'] Starting {}...'

    cd '$DEMO_DIR'
    DEMO_DIR='$DEMO_DIR' PROJECT_ROOT='$PROJECT_ROOT' SEED=$SEED \
        bash configs/{}.sh > '$REPL_DIR/logs/{}.log' 2>&1

    if [ -f '$DEMO_DIR/results/{}_s${SEED}_synth.csv' ]; then
        cp '$DEMO_DIR/results/{}_s${SEED}_synth.csv' '$REPL_DIR/results/'
        echo '['\$(date +%H:%M:%S)'] ✓ {} completado'
    else
        echo '['\$(date +%H:%M:%S)'] ✗ {}: synth.csv no encontrado'
    fi
    "

# Verificar resultados
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Verificando resultados..."
echo "═══════════════════════════════════════════════════════════════════"

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
    echo ""
    echo "  ⚠️  $MISSING configs fallaron. Revisar logs en $REPL_DIR/logs/"
fi

# Crear ensembles
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Creando ensembles..."
echo "═══════════════════════════════════════════════════════════════════"

cd "$SCRIPT_DIR"
python3 -u create_ensembles_replication.py --repl-dir "$REPL_DIR/results" --seed "$SEED" 2>&1 | tee "$REPL_DIR/logs/create_ensembles.log"

# Evaluar con hold-out
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Evaluando con hold-out..."
echo "═══════════════════════════════════════════════════════════════════"

for ens in ENS_Top3_G5 ENS_TopG5_Extended ENS_SUPER_G5_F7_v2; do
    if [ -f "$REPL_DIR/results/${ens}_synth.csv" ]; then
        echo "  Evaluando $ens..."
        PROJECT_ROOT="$PROJECT_ROOT" python3 -u eval_holdout_correct.py \
            --synth "$REPL_DIR/results/${ens}_synth.csv" \
            --output "$REPL_DIR/results/${ens}_holdout.json" \
            2>&1 | tee "$REPL_DIR/logs/eval_${ens}.log"
    fi
done

# Resumen
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
