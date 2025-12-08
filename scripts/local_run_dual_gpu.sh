#!/bin/bash
# Run SMOTE-LLM experiments in parallel on dual GPU setup
# RTX 3090 (24GB) + RTX 3070 (8GB)

set -e

# Configuration
SEEDS_3090=(42 100 123 456 789 1000 2000 3000)  # 8 seeds on RTX 3090
SEEDS_3070=(4000 5000 6000)                      # 3 seeds on RTX 3070

# Base arguments (ajustar según experimento)
BASE_ARGS="
    --data-path MBTI_500.csv
    --test-size 0.2
    --embedding-model sentence-transformers/all-mpnet-base-v2
    --embedding-batch-size 64
    --llm-model gpt-5-mini-2025-08-07
    --reasoning-effort minimal
    --output-verbosity high
    --device cuda
    --max-clusters 3
    --prompts-per-cluster 3
    --prompt-mode mix
    --use-ensemble-selection
    --use-val-gating
    --val-size 0.15
    --val-tolerance 0.02
    --enable-anchor-gate
    --anchor-quality-threshold 0.25
    --purity-gate-threshold 0.025
    --enable-anchor-selection
    --anchor-selection-ratio 0.8
    --anchor-outlier-threshold 1.5
    --use-class-description
    --use-f1-budget-scaling
    --f1-budget-thresholds 0.40 0.20
    --f1-budget-multipliers 30 70 100
    --synthetic-weight 0.5
    --similarity-threshold 0.90
    --min-classifier-confidence 0.10
    --contamination-threshold 0.95
"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Dual GPU Parallel Execution"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "GPUs detected:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "Configuration:"
echo "  RTX 3090 (GPU 0): ${#SEEDS_3090[@]} seeds"
echo "  RTX 3070 (GPU 1): ${#SEEDS_3070[@]} seeds"
echo "  Total: $((${#SEEDS_3090[@]} + ${#SEEDS_3070[@]})) seeds"
echo ""
read -p "Launch all experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

mkdir -p results_dual_gpu

echo ""
echo "Launching seeds on RTX 3090 (GPU 0)..."
for SEED in "${SEEDS_3090[@]}"; do
    echo "  - Seed $SEED"
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u core/runner_phase2.py \
        --random-seed $SEED \
        $BASE_ARGS \
        --synthetic-output results_dual_gpu/seed${SEED}_synthetic.csv \
        --metrics-output results_dual_gpu/seed${SEED}_metrics.json \
        > results_dual_gpu/seed${SEED}.log 2>&1 &

    sleep 2  # Stagger launches
done

echo ""
echo "Launching seeds on RTX 3070 (GPU 1)..."
for SEED in "${SEEDS_3070[@]}"; do
    echo "  - Seed $SEED"
    CUDA_VISIBLE_DEVICES=1 nohup python3 -u core/runner_phase2.py \
        --random-seed $SEED \
        $BASE_ARGS \
        --synthetic-output results_dual_gpu/seed${SEED}_synthetic.csv \
        --metrics-output results_dual_gpu/seed${SEED}_metrics.json \
        > results_dual_gpu/seed${SEED}.log 2>&1 &

    sleep 2
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All experiments launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress:"
echo "  watch -n 10 'ps aux | grep python3 | grep runner_phase2 | wc -l'"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "View logs:"
echo "  tail -f results_dual_gpu/seed42.log"
echo ""
echo "Kill all if needed:"
echo "  pkill -f runner_phase2.py"
echo ""
