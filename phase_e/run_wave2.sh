#!/bin/bash
# ============================================================================
# Oleada 2: length_aware + more_synthetics (3 seeds cada uno)
# ============================================================================
# Ejecuta 2 experimentos en paralelo con 3 seeds: 42, 100, 123
#
# Para monitorear:
#   tail -f logs/length_aware_s42.log
#   tail -f logs/more_synth_s42.log
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Ejecuta: export OPENAI_API_KEY='tu-key'"
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p logs

echo "=============================================="
echo "  OLEADA 2: length_aware + more_synthetics"
echo "  Seeds: 42, 100, 123"
echo "=============================================="
echo ""

PIDS=()

for SEED in 42 100 123; do
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
echo "  ${#PIDS[@]} procesos lanzados"
echo "=============================================="
echo ""
echo "Para monitorear (en terminales separadas):"
echo ""
echo "  # length_aware"
echo "  tail -f logs/length_aware_s42.log"
echo "  tail -f logs/length_aware_s100.log"
echo "  tail -f logs/length_aware_s123.log"
echo ""
echo "  # more_synthetics"
echo "  tail -f logs/more_synth_s42.log"
echo "  tail -f logs/more_synth_s100.log"
echo "  tail -f logs/more_synth_s123.log"
echo ""
echo "Para monitorear todos a la vez:"
echo "  watch -n 10 'grep -h \"Delta\\|COMPLETED\\|sinteticos acumulados\\|Avg words\" logs/*.log | tail -30'"
echo ""
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Para esperar a que terminen todos:"
echo "  wait ${PIDS[*]}"
