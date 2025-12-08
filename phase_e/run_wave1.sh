#!/bin/bash
# ============================================================================
# Oleada 1: ratio_f1_combined + ip_scaling (3 seeds cada uno)
# ============================================================================
# Ejecuta 2 experimentos en paralelo con 3 seeds: 42, 100, 123
#
# Para monitorear:
#   tail -f logs/ratio_f1_s42.log
#   tail -f logs/ip_scaling_s42.log
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
echo "  OLEADA 1: ratio_f1_combined + ip_scaling"
echo "  Seeds: 42, 100, 123"
echo "=============================================="
echo ""

PIDS=()

for SEED in 42 100 123; do
    echo "=== Lanzando Seed $SEED ==="

    # Lanzar ratio_f1_combined en background
    nohup ./improvements/exp_ratio_f1_combined.sh $SEED \
        > logs/ratio_f1_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  ratio_f1_combined seed $SEED: PID $!"

    # Lanzar ip_scaling en background
    nohup ./improvements/exp_ip_scaling.sh $SEED \
        > logs/ip_scaling_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  ip_scaling seed $SEED: PID $!"
done

echo ""
echo "=============================================="
echo "  ${#PIDS[@]} procesos lanzados"
echo "=============================================="
echo ""
echo "Para monitorear (en terminales separadas):"
echo ""
echo "  # ratio_f1_combined"
echo "  tail -f logs/ratio_f1_s42.log"
echo "  tail -f logs/ratio_f1_s100.log"
echo "  tail -f logs/ratio_f1_s123.log"
echo ""
echo "  # ip_scaling"
echo "  tail -f logs/ip_scaling_s42.log"
echo "  tail -f logs/ip_scaling_s100.log"
echo "  tail -f logs/ip_scaling_s123.log"
echo ""
echo "Para monitorear todos a la vez:"
echo "  watch -n 10 'grep -h \"Delta\\|COMPLETED\\|sinteticos acumulados\" logs/*.log | tail -30'"
echo ""
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Para esperar a que terminen todos:"
echo "  wait ${PIDS[*]}"
