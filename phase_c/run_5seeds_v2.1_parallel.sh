#!/bin/bash
# Multi-Seed Validation: Phase C v2.1
# Runs 5 seeds in parallel on RTX 3070 (8GB VRAM)
# Seeds: 42, 100, 123, 456, 789
# Expected: MID-tier mean +1.5% to +2.0% across seeds

set -e

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Check dataset exists
DATASET="../MBTI_500.csv"
if [ ! -f "$DATASET" ]; then
    echo "❌ Error: Dataset not found at $DATASET"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C v2.1 - Multi-Seed Validation (5 Seeds)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Seeds: 42, 100, 123, 456, 789"
echo "  Parallel processes: 5"
echo "  GPU: RTX 3070 (8GB VRAM)"
echo "  Batch size: 96 (reduced for parallel execution)"
echo "  Dataset: $DATASET"
echo ""
echo "Expected Results:"
echo "  ✅ MID-tier mean: +1.5% to +2.0%"
echo "  ✅ 95% CI excludes 0 (all seeds improve)"
echo "  ✅ Standard deviation < 0.5%"
echo "  ✅ Success rate: 80%+ of seeds improve MID-tier"
echo ""
echo "Time Estimate:"
echo "  Per experiment: ~90-120 min"
echo "  Total (parallel): ~90-120 min"
echo "  Completion: $(date -d '+2 hours' '+%H:%M %p')"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Confirm
read -p "Launch 5 parallel experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting 5 experiments in background..."
echo ""

# Launch all 5 seeds in parallel
SEEDS=(42 100 123 456 789)
PIDS=()

for SEED in "${SEEDS[@]}"; do
    echo "🚀 Launching seed $SEED..."

    # Run in background with reduced batch size for parallel execution
    # Use nohup to keep running if terminal closes
    nohup bash -c "
        export CUDA_VISIBLE_DEVICES=0

        # Reduced batch size for parallel execution (96 instead of 128)
        python3 ../core/runner_phase2.py \
            --data-path '$DATASET' \
            --test-size 0.2 \
            --random-seed $SEED \
            \
            --embedding-model sentence-transformers/all-mpnet-base-v2 \
            --device cuda \
            --embedding-batch-size 96 \
            \
            --llm-model gpt-4o-mini \
            --temperature 1.0 \
            --max-clusters 3 \
            --prompts-per-cluster 3 \
            --prompt-mode mix \
            \
            --use-ensemble-selection \
            \
            --use-val-gating \
            --val-size 0.15 \
            --val-tolerance 0.02 \
            \
            --enable-anchor-gate \
            --anchor-quality-threshold 0.25 \
            --purity-gate-threshold 0.025 \
            \
            --enable-anchor-selection \
            --anchor-selection-ratio 0.8 \
            --anchor-outlier-threshold 1.5 \
            \
            --enable-adaptive-filters \
            \
            --use-class-description \
            \
            --use-f1-budget-scaling \
            --f1-budget-thresholds 0.40 0.20 \
            --f1-budget-multipliers 30 70 100 \
            \
            --enable-adaptive-weighting \
            --synthetic-weight 0.5 \
            \
            --similarity-threshold 0.90 \
            --min-classifier-confidence 0.10 \
            --contamination-threshold 0.95 \
            \
            --synthetic-output 'phaseC_v2.1_seed${SEED}_synthetic.csv' \
            --augmented-train-output 'phaseC_v2.1_seed${SEED}_augmented.csv' \
            --metrics-output 'phaseC_v2.1_seed${SEED}_metrics.json'
    " > phaseC_v2.1_seed${SEED}_parallel.log 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "   PID: $PID"

    # Small delay to stagger GPU initialization
    sleep 3
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All 5 Experiments Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Process IDs: ${PIDS[@]}"
echo ""
echo "Monitor progress:"
echo "  ./monitor_5seeds.sh"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Check logs:"
echo "  tail -f phaseC_v2.1_seed42_parallel.log"
echo "  tail -f phaseC_v2.1_seed100_parallel.log"
echo ""
echo "Check completion status:"
echo "  ls -lh phaseC_v2.1_seed*_metrics.json | wc -l"
echo "  (Will show 5 when all complete, plus 1 from original seed 42 = 6 total)"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "⏳ Experiments running in background..."
echo "   Estimated completion: $(date -d '+2 hours' '+%H:%M %p')"
echo ""
echo "To wait for all to complete, run:"
echo "  wait ${PIDS[@]}"
echo ""
