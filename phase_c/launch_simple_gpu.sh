#!/bin/bash
# Simple GPU launch - direct execution without nested scripts

ZONE="us-central1-a"
SEEDS=(42 100 123 456 789)
VMS=(vm-batch1-gpu vm-batch2-gpu vm-batch3-gpu vm-batch4-gpu vm-batch4-gpu)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Launching 5 seeds on 4 GPU VMs"
echo "═══════════════════════════════════════════════════════════"
echo ""

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    VM=${VMS[$i]}

    echo "🔬 Launching seed $SEED on $VM..."

    gcloud compute ssh "$VM" --zone="$ZONE" --command="
        export OPENAI_API_KEY='$OPENAI_API_KEY'
        cd ~
        source venv/bin/activate

        nohup python3 -u runner_phase2.py \\
            --data-path MBTI_500.csv \\
            --test-size 0.2 \\
            --random-seed $SEED \\
            --embedding-model sentence-transformers/all-mpnet-base-v2 \\
            --device cuda \\
            --embedding-batch-size 64 \\
            --llm-model gpt-4o-mini \\
            --temperature 1.0 \\
            --max-clusters 3 \\
            --prompts-per-cluster 3 \\
            --prompt-mode mix \\
            --use-ensemble-selection \\
            --use-val-gating \\
            --val-size 0.15 \\
            --val-tolerance 0.02 \\
            --enable-anchor-gate \\
            --anchor-quality-threshold 0.25 \\
            --purity-gate-threshold 0.025 \\
            --enable-anchor-selection \\
            --anchor-selection-ratio 0.8 \\
            --anchor-outlier-threshold 1.5 \\
            --enable-adaptive-filters \\
            --use-class-description \\
            --use-f1-budget-scaling \\
            --f1-budget-thresholds 0.40 0.20 \\
            --f1-budget-multipliers 30 70 100 \\
            --enable-adaptive-weighting \\
            --synthetic-weight 0.5 \\
            --similarity-threshold 0.90 \\
            --min-classifier-confidence 0.10 \\
            --contamination-threshold 0.95 \\
            --synthetic-output phaseC_v2.1_seed${SEED}_synthetic.csv \\
            --augmented-train-output phaseC_v2.1_seed${SEED}_augmented.csv \\
            --metrics-output phaseC_v2.1_seed${SEED}_metrics.json \\
            > phaseC_seed${SEED}.log 2>&1 &

        sleep 2

        if pgrep -f 'python3.*runner_phase2.*--random-seed $SEED' > /dev/null; then
            echo \"✅ Seed $SEED launched successfully\"
        else
            echo \"❌ Seed $SEED failed to launch\"
        fi
    " &

    sleep 2
done

wait

echo ""
echo "✅ All seeds launched!"
echo ""
echo "Monitor: gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --command='tail phaseC_seed42.log'"
