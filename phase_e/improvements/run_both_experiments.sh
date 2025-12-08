#!/bin/bash
# Ejecutar ambos experimentos de mejora
# Uso: ./run_both_experiments.sh [SEED]
#
# IMPORTANTE: Asegúrate de que no hay otros experimentos corriendo
#   ps aux | grep runner_phase2 | grep -v grep

set -e

SEED="${1:-42}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Check if other experiments are running
RUNNING=$(ps aux | grep runner_phase2 | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  Hay $RUNNING experimentos corriendo."
    echo "   GPU puede estar saturada."
    read -p "¿Continuar de todos modos? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd "$(dirname "$0")"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXPERIMENTOS DE MEJORA - Phase E                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Seed: $SEED"
echo ""

# Run experiments sequentially (safer for GPU)
echo "═══════════════════════════════════════════════════════════"
echo "  [1/2] Experimento: MORE SYNTHETICS"
echo "═══════════════════════════════════════════════════════════"
./exp_more_synthetics.sh $SEED

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  [2/2] Experimento: RELAXED GATE"
echo "═══════════════════════════════════════════════════════════"
./exp_relaxed_gate.sh $SEED

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   EXPERIMENTOS COMPLETADOS                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Resultados en: improvements/results/"
ls -la results/*.json 2>/dev/null || echo "No hay archivos JSON aún"
