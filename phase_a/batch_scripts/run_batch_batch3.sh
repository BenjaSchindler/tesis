#!/bin/bash
# VM batch3: Seeds 1000 2000 3000 4000 5000
# AUTO-SHUTDOWN enabled

# Activate venv
source .venv/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: venv not activated"
    exit 1
fi

# Small delay to ensure packages are settled
sleep 3

SEEDS=(1000 2000 3000 4000 5000)

echo "════════════════════════════════════════════════════════"
echo "  VM batch3 - Running 5 seeds sequentially"
echo "  Seeds: ${SEEDS[@]}"
echo "  AUTO-SHUTDOWN: Enabled (VM will shutdown after completion)"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo "────────────────────────────────────────────────────────"
    echo "Starting seed $seed at $(date)"
    echo "────────────────────────────────────────────────────────"
    
    python3 runner_phase2.py \
      --data-path MBTI_500.csv \
      --test-size 0.2 \
      --random-seed "$seed" \
      --embedding-model sentence-transformers/all-mpnet-base-v2 \
      --device cpu \
      --embedding-batch-size 64 \
      --llm-model gpt-4o-mini \
      --max-clusters 3 \
      --prompts-per-cluster 3 \
      --prompt-mode mix \
      --use-ensemble-selection \
      --use-val-gating \
      --val-size 0.15 \
      --val-tolerance 0.02 \
      --enable-anchor-gate \
      --anchor-quality-threshold 0.50 \
      --enable-anchor-selection \
      --anchor-selection-ratio 0.8 \
      --anchor-outlier-threshold 1.5 \
      --enable-adaptive-filters \
      --use-class-description \
      --use-f1-budget-scaling \
      --f1-budget-thresholds 0.35 0.20 \
      --f1-budget-multipliers 30 70 100 \
      --similarity-threshold 0.90 \
      --min-classifier-confidence 0.10 \
      --contamination-threshold 0.95 \
      --synthetic-weight 0.5 \
      --synthetic-weight-mode flat \
      --synthetic-output "phaseA_seed${seed}_synthetic.csv" \
      --augmented-train-output "phaseA_seed${seed}_augmented.csv" \
      --metrics-output "phaseA_seed${seed}_metrics.json"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $seed completed at $(date)"
    else
        echo "✗ Seed $seed FAILED with exit code $EXIT_CODE at $(date)"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  VM batch3 - All 5 seeds processed"
echo "  Completed: $((5 - FAILED_SEEDS))/5"
echo "  Failed: $FAILED_SEEDS/5"
echo "════════════════════════════════════════════════════════"
echo ""

# Auto-shutdown to save costs
echo "⚠ AUTO-SHUTDOWN: VM will shutdown in 30 seconds..."
echo "  (Results are safely stored on disk)"
echo ""
sleep 30

# Shutdown VM (keeps disk, costs ~\$0.01/hour when stopped)
echo "Shutting down VM batch3..."
sudo shutdown -h now
