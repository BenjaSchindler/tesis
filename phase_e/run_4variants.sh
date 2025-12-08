#!/bin/bash
# Phase E - 4-Variant Comparison Script
# Compares GPT-4o-mini vs GPT-5-mini with different reasoning_effort levels
#
# Variants:
#   A: GPT-4o-mini (baseline)
#   B: GPT-5-mini reasoning_effort=low
#   C: GPT-5-mini reasoning_effort=medium
#   D: GPT-5-mini reasoning_effort=high
#
# Usage: ./run_4variants.sh [SEED]
#   SEED: Random seed (default: 42)

set -e

SEED="${1:-42}"

# Verify API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Change to phase_e directory
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════"
echo "  Phase E - 4-Variant GPT-5-mini Comparison"
echo "  Seed: $SEED"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Variants:"
echo "  A: GPT-4o-mini (baseline)"
echo "  B: GPT-5-mini reasoning_effort=low"
echo "  C: GPT-5-mini reasoning_effort=medium"
echo "  D: GPT-5-mini reasoning_effort=high"
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

FAILED=0

# Common args (Phase A clean baseline configuration)
COMMON_ARGS="--data-path ../MBTI_500.csv \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cpu \
    --embedding-batch-size 64 \
    --max-clusters 3 \
    --prompts-per-cluster 3 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-gate \
    --anchor-quality-threshold 0.30 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --enable-adaptive-filters \
    --use-class-description \
    --use-f1-budget-scaling \
    --f1-budget-thresholds 0.45 0.20 \
    --f1-budget-multipliers 0.0 0.5 1.0 \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat"

# Variant A: GPT-4o-mini (baseline)
echo "────────────────────────────────────────────────────────"
echo "[A] GPT-4o-mini (baseline)"
echo "────────────────────────────────────────────────────────"
python3 core/runner_phase2.py $COMMON_ARGS \
    --llm-model gpt-4o-mini \
    --synthetic-output "results/phaseE_variantA_seed${SEED}_synthetic.csv" \
    --augmented-train-output "results/phaseE_variantA_seed${SEED}_augmented.csv" \
    --metrics-output "results/phaseE_variantA_seed${SEED}_metrics.json" || ((FAILED++))
echo ""

# Variant B: GPT-5-mini low
echo "────────────────────────────────────────────────────────"
echo "[B] GPT-5-mini reasoning_effort=low"
echo "────────────────────────────────────────────────────────"
python3 core/runner_phase2.py $COMMON_ARGS \
    --llm-model gpt-5-mini \
    --reasoning-effort low \
    --max-completion-tokens 1024 \
    --synthetic-output "results/phaseE_variantB_seed${SEED}_synthetic.csv" \
    --augmented-train-output "results/phaseE_variantB_seed${SEED}_augmented.csv" \
    --metrics-output "results/phaseE_variantB_seed${SEED}_metrics.json" || ((FAILED++))
echo ""

# Variant C: GPT-5-mini medium
echo "────────────────────────────────────────────────────────"
echo "[C] GPT-5-mini reasoning_effort=medium"
echo "────────────────────────────────────────────────────────"
python3 core/runner_phase2.py $COMMON_ARGS \
    --llm-model gpt-5-mini \
    --reasoning-effort medium \
    --max-completion-tokens 1024 \
    --synthetic-output "results/phaseE_variantC_seed${SEED}_synthetic.csv" \
    --augmented-train-output "results/phaseE_variantC_seed${SEED}_augmented.csv" \
    --metrics-output "results/phaseE_variantC_seed${SEED}_metrics.json" || ((FAILED++))
echo ""

# Variant D: GPT-5-mini high
echo "────────────────────────────────────────────────────────"
echo "[D] GPT-5-mini reasoning_effort=high"
echo "────────────────────────────────────────────────────────"
python3 core/runner_phase2.py $COMMON_ARGS \
    --llm-model gpt-5-mini \
    --reasoning-effort high \
    --max-completion-tokens 1024 \
    --synthetic-output "results/phaseE_variantD_seed${SEED}_synthetic.csv" \
    --augmented-train-output "results/phaseE_variantD_seed${SEED}_augmented.csv" \
    --metrics-output "results/phaseE_variantD_seed${SEED}_metrics.json" || ((FAILED++))
echo ""

# Summary
echo "════════════════════════════════════════════════════════"
echo "  Phase E - 4-Variant Comparison Complete"
echo "  Completed: $((4 - FAILED))/4"
echo "  Failed: $FAILED/4"
echo "════════════════════════════════════════════════════════"
echo ""

# Extract and display results
echo "Results Summary (seed $SEED):"
echo "─────────────────────────────────────────────────────────"
for variant in A B C D; do
    METRICS_FILE="results/phaseE_variant${variant}_seed${SEED}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        BASELINE=$(python3 -c "import json; m=json.load(open('$METRICS_FILE')); print(f'{m.get(\"baseline_macro_f1\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        AUGMENTED=$(python3 -c "import json; m=json.load(open('$METRICS_FILE')); print(f'{m.get(\"augmented_macro_f1\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        DELTA=$(python3 -c "import json; m=json.load(open('$METRICS_FILE')); d=m.get('augmented_macro_f1',0)-m.get('baseline_macro_f1',0); print(f'{d*100:+.3f}%')" 2>/dev/null || echo "N/A")
        SYNTH=$(python3 -c "import json; m=json.load(open('$METRICS_FILE')); print(m.get('total_synthetics_accepted', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "Variant $variant: Baseline=$BASELINE, Aug=$AUGMENTED, Delta=$DELTA, Synth=$SYNTH"
    else
        echo "Variant $variant: FAILED (no metrics file)"
    fi
done
echo ""

exit $FAILED
