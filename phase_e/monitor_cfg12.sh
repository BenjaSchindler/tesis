#!/bin/bash
# ============================================================================
# Monitor cfg12_capped Progress (PID 642907)
# ============================================================================

PID=642907
RESULTS_DIR="improvements/results/live_20251201_132759"

while true; do
    clear
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  cfg12_capped Progress Monitor                               ║"
    echo "║  PID: $PID                                                    ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    # Check if process is still running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "❌ Process $PID is no longer running!"
        echo ""
        echo "Checking results..."
        if [ -f "$RESULTS_DIR/cfg12_capped_s42_metrics.json" ]; then
            echo "✅ Experiment completed! Results found:"
            python3 -c "
import json
with open('$RESULTS_DIR/cfg12_capped_s42_metrics.json') as f:
    d = json.load(f)
b = d['baseline']['macro_f1']
a = d['augmented']['macro_f1']
s = d.get('synthetic_data', {}).get('accepted_count', 0)
delta = (a-b)/b*100
print(f'  Baseline:  {b:.4f}')
print(f'  Augmented: {a:.4f}')
print(f'  Delta:     {delta:+.2f}%')
print(f'  Synth:     {s}')
"
        else
            echo "⚠️  No results found. Process may have crashed."
        fi
        break
    fi

    # Process status
    echo "📊 Process Status:"
    ps aux | grep "^[^ ]* *$PID " | awk '{printf "   CPU: %s%%  |  Mem: %s%%  |  Runtime: %s\n", $3, $4, $10}'
    echo ""

    # GPU status
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "   VRAM: %s/%s MiB (%s%% util)\n", $1, $2, $3}'
    echo ""

    # Check metrics file for current progress
    echo "📈 Experiment Progress:"

    # Count completed experiments
    COMPLETED=0
    if [ -f "$RESULTS_DIR/more_synthetics_s42_metrics.json" ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "   ✅ more_synthetics: DONE (+0.18%, 60 synth)"
    fi

    if [ -f "$RESULTS_DIR/relaxed_gate_s42_metrics.json" ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "   ✅ relaxed_gate: DONE (+0.10%, 64 synth)"
    fi

    # Check cfg12_capped status
    if [ -f "$RESULTS_DIR/cfg12_capped_s42_metrics.json" ]; then
        echo "   ✅ cfg12_capped: DONE"
        python3 -c "
import json
with open('$RESULTS_DIR/cfg12_capped_s42_metrics.json') as f:
    d = json.load(f)
b = d['baseline']['macro_f1']
a = d['augmented']['macro_f1']
s = d.get('synthetic_data', {}).get('accepted_count', 0)
delta = (a-b)/b*100
print(f'      Delta: {delta:+.2f}%, Synth: {s}')
" 2>/dev/null || echo "      (parsing results...)"
    else
        echo "   🔄 cfg12_capped: RUNNING"

        # Try to get synthetic count so far
        if [ -f "$RESULTS_DIR/cfg12_capped_s42_synth.csv" ]; then
            SYNTH_COUNT=$(wc -l < "$RESULTS_DIR/cfg12_capped_s42_synth.csv")
            SYNTH_COUNT=$((SYNTH_COUNT - 1))  # Subtract header
            echo "      Generated so far: $SYNTH_COUNT synthetics"
        fi
    fi

    echo ""
    echo "⏱️  Estimated Progress:"
    if [ $COMPLETED -eq 2 ]; then
        echo "   Experiments 1-2: COMPLETE (100%)"
        echo "   Experiment 3 (cfg12_capped): IN PROGRESS"
        echo ""
        echo "   ⚠️  Note: cfg12_capped uses 5x more prompts than relaxed_gate"
        echo "   Expected runtime: ~8-10 hours total"
    else
        echo "   Experiments completed: $COMPLETED/3"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Refreshing every 10 seconds... (Press Ctrl+C to exit)"
    echo "  Experiment continues in background"
    echo "═══════════════════════════════════════════════════════════════"

    sleep 10
done
