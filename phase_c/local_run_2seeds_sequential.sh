#!/bin/bash
# Phase C v2.1 - 2 Seeds Sequential (Local Backup)
# Seeds: 111, 222 (different from GCP: 42, 100, 123, 456, 789)
# Expected time: ~4 hours total (2 hours per seed)

set -e

SEEDS=(111 222)

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C v2.1 - Local Sequential 2-Seed Backup"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Seeds: ${SEEDS[@]}"
echo "Mode: Sequential (one at a time)"
echo "GPU: RTX 3070 (cuda)"
echo "Expected time: ~4 hours total"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Starting Seed $SEED"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    START_TIME=$(date +%s)
    
    python3 ../core/runner_phase2.py \
        --data-path ../MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cpu \
        --embedding-batch-size 32 \
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
        --synthetic-output phaseC_v2.1_seed${SEED}_synthetic.csv \
        --augmented-train-output phaseC_v2.1_seed${SEED}_augmented.csv \
        --metrics-output phaseC_v2.1_seed${SEED}_metrics.json \
        2>&1 | tee local_seed${SEED}.log
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Seed $SEED completed in ${MINUTES} minutes"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    
    # Show results if available
    if [ -f "phaseC_v2.1_seed${SEED}_metrics.json" ]; then
        echo "Results:"
        python3 -c "
import json
with open('phaseC_v2.1_seed${SEED}_metrics.json') as f:
    d = json.load(f)
    print(f\"  Baseline F1: {d['baseline']['macro_f1']:.4f}\")
    print(f\"  Augmented F1: {d['augmented']['macro_f1']:.4f}\")
    print(f\"  Delta: +{d['improvement']['f1_delta_pct']:.3f}%\")
    print(f\"  Synthetics accepted: {d['synthetic_data']['accepted_count']}\")
"
        echo ""
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All 2 seeds completed!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Combined results (7 seeds total):"
echo "  GCP: 42, 100, 123, 456, 789 (running)"
echo "  Local: 111, 222 (completed)"
echo ""
echo "Next: Wait for GCP to complete, then analyze all 7 seeds"
echo ""
