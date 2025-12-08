#!/bin/bash
# Monitor dual GPU experiments in real-time

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  Dual GPU Monitor - $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""

    # GPU status
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name temp util_gpu util_mem mem_used mem_total; do
        printf "  GPU %s (%s): %s°C | GPU: %s%% | VRAM: %s/%sMB (%s%%)\n" \
            "$idx" "$name" "$temp" "$util_gpu" "$mem_used" "$mem_total" "$util_mem"
    done
    echo ""

    # Process count
    RUNNING=$(ps aux | grep -c '[p]ython3.*runner_phase2')
    echo "Running processes: $RUNNING"
    echo ""

    # Processes per GPU
    echo "Processes by GPU:"
    GPU0=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=0 2>/dev/null | wc -l)
    GPU1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=1 2>/dev/null | wc -l)
    echo "  RTX 3090 (GPU 0): $GPU0 processes"
    echo "  RTX 3070 (GPU 1): $GPU1 processes"
    echo ""

    # Completed experiments
    echo "Completed experiments:"
    COMPLETED=$(ls results_dual_gpu/*_metrics.json 2>/dev/null | wc -l)
    TOTAL=11  # 8 + 3
    echo "  $COMPLETED / $TOTAL seeds finished"

    if [ $COMPLETED -gt 0 ]; then
        echo ""
        echo "Recent completions:"
        ls -lt results_dual_gpu/*_metrics.json 2>/dev/null | head -3 | awk '{print "  - " $9}'
    fi

    echo ""
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo "Press Ctrl+C to exit | Refreshing every 10 seconds..."
    echo ""

    sleep 10
done
