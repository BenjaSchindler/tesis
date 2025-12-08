#!/bin/bash
# ============================================================================
# Auto-continuación: Espera Wave 1 y lanza Wave 2
# ============================================================================
# Este script:
#   1. Espera a que terminen los 4 procesos de Wave 1
#   2. Lanza Wave 2 (length_aware + more_synthetics) con seeds 42 y 100
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")"

# PIDs de Wave 1 (capturados al momento de crear este script)
WAVE1_PIDS="681360 681362 681368 681370"

echo "=============================================="
echo "  AUTO-CONTINUACIÓN: Wave 1 → Wave 2"
echo "=============================================="
echo ""
echo "Esperando a que terminen los procesos de Wave 1:"
echo "  PIDs: $WAVE1_PIDS"
echo ""
echo "Inicio: $(date)"
echo ""

# Esperar a que terminen todos los procesos de Wave 1
for PID in $WAVE1_PIDS; do
    if ps -p $PID > /dev/null 2>&1; then
        echo "  Esperando PID $PID..."
        while ps -p $PID > /dev/null 2>&1; do
            sleep 30
        done
        echo "  ✓ PID $PID terminó"
    else
        echo "  ✓ PID $PID ya terminó"
    fi
done

echo ""
echo "=============================================="
echo "  WAVE 1 COMPLETADA: $(date)"
echo "=============================================="
echo ""

# Pequeña pausa para liberar GPU memory
sleep 10

echo "=============================================="
echo "  LANZANDO WAVE 2: length_aware + more_synthetics"
echo "  Seeds: 42, 100"
echo "=============================================="
echo ""

mkdir -p logs

PIDS=()

for SEED in 42 100; do
    echo "=== Lanzando Seed $SEED ==="

    # Lanzar length_aware en background
    nohup ./improvements/exp_length_aware.sh $SEED \
        > logs/length_aware_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  length_aware seed $SEED: PID $!"

    # Lanzar more_synthetics en background
    nohup ./improvements/exp_more_synthetics.sh $SEED \
        > logs/more_synth_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  more_synthetics seed $SEED: PID $!"
done

echo ""
echo "=============================================="
echo "  WAVE 2 LANZADA: ${#PIDS[@]} procesos"
echo "  PIDs: ${PIDS[*]}"
echo "=============================================="
echo ""
echo "Para monitorear:"
echo "  tail -f logs/length_aware_s42.log"
echo "  tail -f logs/more_synth_s42.log"
echo ""

# Esperar a que termine Wave 2
echo "Esperando a que termine Wave 2..."
for PID in ${PIDS[*]}; do
    wait $PID 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "  WAVE 2 COMPLETADA: $(date)"
echo "=============================================="
echo ""
echo "Resumen de resultados:"
echo ""

for LOG in logs/length_aware_s*.log logs/more_synth_s*.log; do
    if [ -f "$LOG" ]; then
        echo "=== $LOG ==="
        grep -E "Delta|Baseline F1|Augmented F1|Sinteticos" "$LOG" 2>/dev/null | tail -5 || echo "  (sin resultados aún)"
        echo ""
    fi
done

echo "=============================================="
echo "  TODAS LAS OLEADAS COMPLETADAS"
echo "=============================================="
