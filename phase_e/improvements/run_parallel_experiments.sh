#!/bin/bash
# ============================================================================
# Run multiple experiments IN PARALLEL
# ============================================================================
# Aprovecha:
#   - Cache de embeddings compartido (no duplica VRAM)
#   - Múltiples procesos en GPU (~800MB cada uno)
#   - API calls paralelos (aunque hay rate limit compartido)
#
# Uso: ./run_parallel_experiments.sh [SEED]
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="improvements/results/parallel_${TIMESTAMP}"
mkdir -p "$RESULTS_BASE"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  PARALLEL EXPERIMENTS - Seed $SEED                            ║"
echo "║  Results: $RESULTS_BASE                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Launching 3 experiments in parallel..."
echo ""

# Pre-warm: Ensure embeddings are cached (run once, shared by all)
echo "📦 Verifying embedding cache..."
python3 -c "
from core.embedding_cache import EmbeddingCache
cache = EmbeddingCache('embeddings_cache')
stats = cache.get_stats()
print(f'   Cache: {stats}')
"

# Function to run experiment in background
run_exp() {
    local NAME="$1"
    local EXTRA="$2"
    local LOG="${RESULTS_BASE}/${NAME}_s${SEED}.log"
    local OUT="${RESULTS_BASE}/${NAME}_s${SEED}"

    echo "🚀 Starting: $NAME"

    python3 -u core/runner_phase2.py \
        --data-path ../MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 128 \
        --cache-dir embeddings_cache \
        --llm-model gpt-4o-mini \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --enable-anchor-gate \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --cap-class-ratio 0.15 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --synthetic-output ${OUT}_synth.csv \
        --augmented-train-output ${OUT}_aug.csv \
        --metrics-output ${OUT}_metrics.json \
        $EXTRA \
        > "$LOG" 2>&1 &

    echo $!  # Return PID
}

# Launch experiments in parallel
# Each uses ~800MB VRAM, total ~2.4GB (well under 24GB limit)

PID1=$(run_exp "more_synth" "--max-clusters 5 --prompts-per-cluster 9 --val-tolerance 0.02 --anchor-quality-threshold 0.30 --similarity-threshold 0.90")
sleep 2  # Stagger slightly to avoid cache race

PID2=$(run_exp "relaxed" "--max-clusters 3 --prompts-per-cluster 3 --val-tolerance 0.05 --anchor-quality-threshold 0.15 --similarity-threshold 0.80 --min-classifier-confidence 0.05 --contamination-threshold 0.90")
sleep 2

PID3=$(run_exp "cfg12_cap" "--max-clusters 5 --prompts-per-cluster 9 --val-tolerance 0.02 --anchor-quality-threshold 0.30 --similarity-threshold 0.95")

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running in parallel:"
echo "    PID $PID1: more_synth"
echo "    PID $PID2: relaxed"
echo "    PID $PID3: cfg12_cap"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Monitor progress
monitor() {
    while true; do
        clear
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  PARALLEL EXPERIMENTS MONITOR - $(date +%H:%M:%S)              ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""

        # GPU status
        echo "🖥️  GPU Status:"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf "   VRAM: %s/%s MiB (%s%% util)\n", $1, $2, $3}'
        echo ""

        # Process status
        echo "📊 Experiment Status:"
        for NAME in more_synth relaxed cfg12_cap; do
            LOG="${RESULTS_BASE}/${NAME}_s${SEED}.log"
            if [ -f "$LOG" ]; then
                # Check if still running
                if pgrep -f "runner_phase2.*${NAME}" > /dev/null 2>&1; then
                    STATUS="🔄 Running"
                    # Get last class being processed
                    LAST=$(grep -E "^Clase |GATE.*for|Skipping" "$LOG" 2>/dev/null | tail -1 | head -c 60)
                else
                    if grep -q "Augmented macro-F1" "$LOG" 2>/dev/null; then
                        STATUS="✅ Complete"
                        LAST=$(grep "Augmented macro-F1" "$LOG" | tail -1)
                    else
                        STATUS="❌ Failed"
                        LAST=$(tail -1 "$LOG" 2>/dev/null | head -c 60)
                    fi
                fi
                printf "   %-12s %s\n" "$NAME:" "$STATUS"
                echo "      $LAST"
            else
                printf "   %-12s ⏳ Starting...\n" "$NAME:"
            fi
        done

        echo ""
        echo "Press Ctrl+C to stop monitoring (experiments continue in background)"
        sleep 10
    done
}

# Wait for all processes
echo "Waiting for experiments to complete..."
echo "(Press Ctrl+C to run in background)"
echo ""

wait $PID1 $PID2 $PID3

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE                                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Show results
echo "📊 Results Summary:"
for NAME in more_synth relaxed cfg12_cap; do
    METRICS="${RESULTS_BASE}/${NAME}_s${SEED}_metrics.json"
    if [ -f "$METRICS" ]; then
        echo ""
        echo "=== $NAME ==="
        python3 -c "
import json
with open('$METRICS') as f:
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
    fi
done

echo ""
echo "Results saved to: $RESULTS_BASE"
