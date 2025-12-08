#!/bin/bash
# Master script para correr 3 replicaciones en secuencia
# Uso: nohup ./run_all_replications.sh > all_replications.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════════"
echo "  MASTER: 3 Replicaciones Completas"
echo "  Inicio: $(date)"
echo "═══════════════════════════════════════════════════════════════════"

# Verificar API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY no configurado"
    exit 1
fi

START_TOTAL=$(date +%s)

# Replicación 1
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  REPLICACIÓN 1 de 3 - Inicio: $(date +%H:%M:%S)                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
./run_replication.sh 1

# Replicación 2
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  REPLICACIÓN 2 de 3 - Inicio: $(date +%H:%M:%S)                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
./run_replication.sh 2

# Replicación 3
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  REPLICACIÓN 3 de 3 - Inicio: $(date +%H:%M:%S)                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
./run_replication.sh 3

END_TOTAL=$(date +%s)
ELAPSED_TOTAL=$((END_TOTAL - START_TOTAL))
ELAPSED_HOURS=$((ELAPSED_TOTAL / 3600))
ELAPSED_MINS=$(((ELAPSED_TOTAL % 3600) / 60))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  TODAS LAS REPLICACIONES COMPLETADAS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Tiempo total: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "  Fin: $(date)"
echo ""

# Análisis de variabilidad
echo "Ejecutando análisis de variabilidad..."
python3 -u analyze_replication_variance.py --runs 3

echo ""
echo "  Resultados en:"
echo "    - replication_run1/results/"
echo "    - replication_run2/results/"
echo "    - replication_run3/results/"
echo "    - replication_variance_analysis.json"
echo ""
