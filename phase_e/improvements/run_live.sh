#!/bin/bash
# ============================================================================
# Run improvement experiments with LIVE OUTPUT
# Progress bars for embeddings, training, generation visible in real-time
# ============================================================================

set -e

cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="improvements/results/live_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase E - LIVE OUTPUT EXPERIMENTS                            ║"
echo "║  Results: $RESULTS_DIR                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Function to run single experiment with live output
run_live() {
    local NAME="$1"
    local SEED="$2"
    local EXTRA_ARGS="$3"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  EXPERIMENT: $NAME (seed $SEED)                              "
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"
    local T0=$(date +%s)

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
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --verbose-logging \
        --synthetic-output ${OUT}_synth.csv \
        --augmented-train-output ${OUT}_aug.csv \
        --metrics-output ${OUT}_metrics.json \
        $EXTRA_ARGS

    local T1=$(date +%s)
    local DUR=$((T1-T0))

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $NAME (seed $SEED) COMPLETED in ${DUR}s"

    if [ -f "${OUT}_metrics.json" ]; then
        python3 -c "
import json
with open('${OUT}_metrics.json') as f:
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
    echo "════════════════════════════════════════════════════════════════"
}

# ============================================================================
# EXPERIMENTS
# ============================================================================

SEED="${1:-42}"

echo "Running experiments for seed $SEED..."
echo ""

# 1. MORE SYNTHETICS (5 clusters × 9 prompts)
run_live "more_synthetics" $SEED \
    "--max-clusters 5 --prompts-per-cluster 9 --val-tolerance 0.02 --anchor-quality-threshold 0.30 --similarity-threshold 0.90"

# 2. RELAXED GATE (lower thresholds)
run_live "relaxed_gate" $SEED \
    "--max-clusters 3 --prompts-per-cluster 3 --val-tolerance 0.05 --anchor-quality-threshold 0.15 --similarity-threshold 0.80"

# 3. CFG12 CAPPED (best config + more synthetics)
run_live "cfg12_capped" $SEED \
    "--max-clusters 5 --prompts-per-cluster 9 --val-tolerance 0.02 --anchor-quality-threshold 0.30 --similarity-threshold 0.95"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE for seed $SEED                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*_metrics.json 2>/dev/null || echo "No results yet"
