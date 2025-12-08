#!/bin/bash
# ============================================================================
# Experimento: Budget PROPORCIONAL al tamaño de clase
# ============================================================================
# Hipótesis: Al usar cap-class-ratio como TARGET (no solo máximo),
#            generaremos más sintéticos para clases grandes
#
# SOLUCIÓN: Parchear runner_phase2.py para que:
#   1. Desactive el Enhanced Quality Gate budget (que limita a 10)
#   2. Use cap_class_ratio × n_samples como el budget real
#
# Con --cap-class-ratio 0.15:
#   ISFJ (n=442):  hasta 66 sintéticos (vs 10 actual)
#   ISTJ (n=845):  hasta 126 sintéticos
#   ENFJ (n=1043): hasta 156 sintéticos
#   ESFP (n=245):  hasta 36 sintéticos
# ============================================================================

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
RATIO="${2:-0.15}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/proportional_r${RATIO}_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "════════════════════════════════════════════════════════════════"
echo "  Experimento: PROPORTIONAL BUDGET"
echo "  Seed: $SEED"
echo "  Cap-class-ratio: $RATIO (=objetivo, no solo máximo)"
echo "  Output: $OUT"
echo "════════════════════════════════════════════════════════════════"

# Create patched runner that disables dynamic budget
python3 << 'PATCH_EOF'
import re

# Read original runner
with open('core/runner_phase2.py', 'r') as f:
    code = f.read()

# Patch: Make ENHANCED_GATE_AVAILABLE = False to disable dynamic budget
# This forces the "else" branch at line ~2733 which uses cap_class_abs directly
patched_code = code.replace(
    'ENHANCED_GATE_AVAILABLE = True',
    'ENHANCED_GATE_AVAILABLE = False  # PATCHED: Use proportional budget'
)

# Also disable the Phase 1 predictor to ensure we use cap_class_abs
# We do this by setting ANCHOR_QUALITY_AVAILABLE to False as well
patched_code = patched_code.replace(
    'ANCHOR_QUALITY_AVAILABLE = True',
    'ANCHOR_QUALITY_AVAILABLE = False  # PATCHED: Use proportional budget'
)

# Save patched version
with open('improvements/runner_proportional.py', 'w') as f:
    f.write(patched_code)

print("✅ Created patched runner: improvements/runner_proportional.py")
print("   - Disabled ENHANCED_GATE_AVAILABLE")
print("   - Disabled ANCHOR_QUALITY_AVAILABLE")
print("   - Will use cap_class_abs / cap_class_ratio directly")
PATCH_EOF

echo ""
echo "  Expected synthetics per class (${RATIO} of train size):"
echo "    ISFJ: ~$(python3 -c "print(int(442 * ${RATIO}))")    ISTJ: ~$(python3 -c "print(int(845 * ${RATIO}))")   ENFJ: ~$(python3 -c "print(int(1043 * ${RATIO}))")   ISFP: ~$(python3 -c "print(int(595 * ${RATIO}))")"
echo "    ESFP: ~$(python3 -c "print(int(245 * ${RATIO}))")    ESFJ: ~$(python3 -c "print(int(123 * ${RATIO}))")    ESTJ: ~$(python3 -c "print(int(328 * ${RATIO}))")    ESTP: skipped (F1>0.6)"
echo ""

python3 -u improvements/runner_proportional.py \
    --data-path ../MBTI_500.csv \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 128 \
    --cache-dir embeddings_cache \
    --llm-model gpt-4o-mini \
    --max-clusters 5 \
    --prompts-per-cluster 9 \
    --prompt-mode mix \
    --use-ensemble-selection \
    --use-val-gating \
    --val-size 0.15 \
    --val-tolerance 0.02 \
    --enable-anchor-selection \
    --anchor-selection-ratio 0.8 \
    --anchor-outlier-threshold 1.5 \
    --use-class-description \
    --cap-class-ratio $RATIO \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode flat \
    --synthetic-output ${OUT}_synth.csv \
    --augmented-train-output ${OUT}_aug.csv \
    --metrics-output ${OUT}_metrics.json \
    2>&1 | tee ${OUT}.log

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Resultados:"
python3 -c "
import json
with open('${OUT}_metrics.json') as f:
    d = json.load(f)
b = d['baseline']['macro_f1']
a = d['augmented']['macro_f1']
s = d.get('synthetic_data', {}).get('accepted_count', 0)
print(f'  Baseline F1:  {b:.4f}')
print(f'  Augmented F1: {a:.4f}')
print(f'  Delta:        {(a-b)/b*100:+.2f}%')
print(f'  Sintéticos:   {s}')
"
echo "════════════════════════════════════════════════════════════════"
