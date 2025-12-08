#!/bin/bash
# Monitor 5-seed parallel validation progress

SEEDS=(42 100 123 456 789)

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════"
    echo "  Phase C v2.1 - Multi-Seed Validation Monitor"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    date
    echo ""

    # Count completed experiments
    COMPLETED=$(ls phaseC_v2.1_seed*_metrics.json 2>/dev/null | wc -l)
    # Subtract 1 because we already have seed 42 from earlier
    COMPLETED=$((COMPLETED - 1))
    if [ $COMPLETED -lt 0 ]; then
        COMPLETED=0
    fi

    echo "Progress: $COMPLETED/5 experiments completed"
    echo ""
    echo "───────────────────────────────────────────────────────────"

    # Check each seed
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "Seed $SEED:"

        # Check if metrics file exists
        if [ -f "phaseC_v2.1_seed${SEED}_metrics.json" ]; then
            echo "  ✅ COMPLETED"

            # Extract results
            OVERALL_DELTA=$(python3 << EOF
import json
with open('phaseC_v2.1_seed${SEED}_metrics.json') as f:
    data = json.load(f)
    delta = data['improvement']['f1_delta_pct']
    print(f"{delta:+.3f}%")
EOF
)
            SYNTHETICS=$(python3 << EOF
import json
with open('phaseC_v2.1_seed${SEED}_metrics.json') as f:
    data = json.load(f)
    count = data['synthetic_data']['accepted_count']
    print(count)
EOF
)
            echo "  Overall delta: $OVERALL_DELTA"
            echo "  Synthetics: $SYNTHETICS"

        elif [ -f "phaseC_v2.1_seed${SEED}_parallel.log" ]; then
            echo "  🔄 RUNNING"

            # Get last few log lines
            LAST_LINE=$(tail -3 phaseC_v2.1_seed${SEED}_parallel.log | grep -E "Clase|LLM generación|GATE" | tail -1)
            if [ -n "$LAST_LINE" ]; then
                echo "  Last: ${LAST_LINE:0:60}..."
            fi

            # Check for errors
            ERROR_COUNT=$(grep -c "Error\|Exception\|Failed" phaseC_v2.1_seed${SEED}_parallel.log 2>/dev/null || echo 0)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "  ⚠️  Errors detected: $ERROR_COUNT"
            fi

        else
            echo "  ⏸️  NOT STARTED or NO LOG"
        fi
    done

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo ""

    # GPU status
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "  GPU Util: %s%% | VRAM: %s/%s MB (%.1f%%)\n", $1, $2, $3, ($2/$3)*100}'
        echo ""

        # Count Python processes
        PYTHON_COUNT=$(nvidia-smi | grep -c "python3" || echo 0)
        echo "  Active Python processes: $PYTHON_COUNT"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"

    # Exit if all complete
    if [ $COMPLETED -eq 5 ]; then
        echo ""
        echo "🎉 All 5 experiments completed!"
        echo ""
        echo "Run analysis:"
        echo "  python3 analyze_5seeds.py"
        echo ""
        break
    fi

    echo ""
    echo "Press Ctrl+C to exit monitor (experiments will continue)"
    echo "Refreshing in 30 seconds..."
    sleep 30
done
