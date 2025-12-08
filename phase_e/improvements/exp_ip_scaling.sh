#!/bin/bash
# Experimento: IP-Based Budget Scaling
# Hipótesis: Dar más presupuesto a clases con alto IP (bajo F1 baseline)
#            mejorará el delta general
#
# Implementación:
#   1. Parchea calculate_enhanced_budget con IP scaling
#   2. Ejecuta runner_phase2.py normalmente
#   3. Restaura el código original

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")/.."

SEED="${1:-42}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="improvements/results/ip_scaling_s${SEED}_${TIMESTAMP}"
mkdir -p "improvements/results"

echo "========================================"
echo "  Experimento: IP-BASED SCALING"
echo "  Seed: $SEED"
echo "  Output: $OUT"
echo "========================================"

# Create patched runner
python3 << 'PATCH_EOF'
import sys
sys.path.insert(0, 'improvements')

# Read original runner
with open('core/runner_phase2.py', 'r') as f:
    original_code = f.read()

# Create IP-enhanced version of calculate_enhanced_budget
ip_function = '''
# === IP-ENHANCED BUDGET (Phase E Improvement) ===
def calculate_enhanced_budget(
    n_samples: int,
    quality_score: float,
    purity: float,
    baseline_f1: float,
    purity_low_threshold: float = 0.30,
    purity_low_multiplier: float = 0.3,
    f1_high_threshold: float = 0.45,
    f1_high_multiplier: float = 0.5,
    f1_low_threshold: float = 0.15,
    f1_low_multiplier: float = 1.5,
    target_ratio: float = 0.08
) -> Tuple[int, str, Dict[str, float]]:
    """IP-Enhanced budget calculator (Phase E improvement)."""
    base_budget = int(n_samples * target_ratio)

    # Quality multiplier
    if quality_score < 0.35:
        quality_mult = 0.1
        quality_reason = f"🔴 Very low quality ({quality_score:.3f})"
    elif quality_score < 0.40:
        quality_mult = 0.3
        quality_reason = f"⚠️  Low quality ({quality_score:.3f})"
    elif quality_score < 0.50:
        quality_mult = 0.7
        quality_reason = f"⚠️  Mediocre quality ({quality_score:.3f})"
    else:
        quality_mult = 1.0
        quality_reason = f"✅ Good quality ({quality_score:.3f})"

    # Purity multiplier
    if purity < purity_low_threshold:
        purity_mult = purity_low_multiplier
        purity_reason = f"🔴 Low purity ({purity:.3f}) → {int(purity_mult*100)}%"
    else:
        purity_mult = 1.0
        purity_reason = f"✅ Purity OK ({purity:.3f})"

    # IP multiplier (NEW - Phase E)
    ip = 1 - baseline_f1  # Improvement Potential
    ip_threshold = 0.7
    ip_boost_factor = 1.0

    if ip > ip_threshold:
        ip_mult = 1 + ip_boost_factor * ip
        ip_reason = f"📈 IP={ip:.3f} (high potential) → {int(ip_mult*100)}% boost"
    elif ip > 0.5:
        ip_mult = 1 + 0.5 * ip_boost_factor * ip
        ip_reason = f"📊 IP={ip:.3f} (moderate) → {int(ip_mult*100)}%"
    else:
        ip_mult = 1.0
        ip_reason = f"✅ IP={ip:.3f} (low potential)"

    # Combine
    total_mult = quality_mult * purity_mult * ip_mult
    budget = int(base_budget * total_mult)

    # IP RESCUE for high-IP classes
    if ip > ip_threshold:
        ip_minimum = int(10 + 20 * ip)
        if budget < ip_minimum:
            budget = ip_minimum
            ip_reason += f" → IP rescue: min {ip_minimum}"

    budget = max(10, budget)

    reason = (
        f"Base: {base_budget} (IP-enhanced)" + chr(10) +
        f"   {quality_reason}" + chr(10) +
        f"   {purity_reason}" + chr(10) +
        f"   {ip_reason}" + chr(10) +
        f"   -> Final: {budget} (x{total_mult:.2f})"
    )

    multipliers = {
        "quality": quality_mult,
        "purity": purity_mult,
        "ip": ip_mult,
        "improvement_potential": ip,
        "total": total_mult
    }

    return budget, reason, multipliers
# === END IP-ENHANCED BUDGET ===
'''

# Find and replace the function
import re
pattern = r'(# Phase 2: Enhanced Dynamic Budget Calculator.*?def calculate_enhanced_budget\(.*?\).*?return budget, reason, multipliers)'
replacement = ip_function

# Use dotall to match across lines
patched_code = re.sub(pattern, ip_function, original_code, flags=re.DOTALL)

# CRITICAL: Add sys.path to find modules in core/
path_fix = '''import sys
sys.path.insert(0, 'core')  # Phase E: Find modules in core/
'''
patched_code = path_fix + patched_code

# Save patched version
with open('improvements/runner_phase2_ip.py', 'w') as f:
    f.write(patched_code)

print("✅ Created IP-enhanced runner: improvements/runner_phase2_ip.py")
PATCH_EOF

# Run with patched version
echo ""
echo "Running with IP-enhanced budget calculator..."
echo ""

python3 -u improvements/runner_phase2_ip.py \
    --data-path ../MBTI_500.csv \
    --test-size 0.2 \
    --random-seed $SEED \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 128 \
    --cache-dir embeddings_cache \
    --llm-model gpt-4o-mini \
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
    --use-class-description \
    --cap-class-ratio 0.15 \
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
echo "========================================"
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
echo "========================================"
