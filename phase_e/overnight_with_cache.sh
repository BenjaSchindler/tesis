#!/bin/bash
# Phase E - Overnight Experiments with Embedding Cache
#
# Features:
# - Embedding caching: ~55% faster per experiment
# - Groups experiments by seed for maximum cache efficiency
# - Includes improvement experiments (IP scaling, more synthetics, relaxed gate)
#
# Expected duration: ~8-10 hours (vs ~20 hours without cache)

set -e

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")"

# Configuration
MAX_PARALLEL=2  # Reduced to 2 for better GPU efficiency
BATCH_SIZE=128
CACHE_DIR="embeddings_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cached_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"

LOG_FILE="$RESULTS_DIR/master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "experiment,seed,config_type,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s,cache_hits" > "$SUMMARY_FILE"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase E - Overnight with Embedding Cache                     ║"
echo "║  Experiments: Remaining configs + IP improvements             ║"
echo "║  Cache dir: $CACHE_DIR                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check what's already completed
COMPLETED_DIR="results/safe_20251129_215918"
if [ -d "$COMPLETED_DIR" ]; then
    COMPLETED=$(ls -1 "$COMPLETED_DIR"/*_metrics.json 2>/dev/null | wc -l)
    log "Found $COMPLETED completed experiments in $COMPLETED_DIR"
fi

# ============================================================================
# PRE-WARM CACHE: Compute embeddings for each seed once
# ============================================================================

log "=== PRE-WARMING EMBEDDING CACHE ==="

python3 << 'PREWARM_EOF'
import sys
sys.path.insert(0, 'core')

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from embedding_cache import EmbeddingCache

# Configuration
DATA_PATH = "../MBTI_500.csv"
SEEDS = [42, 100]
TEST_SIZE = 0.2
VAL_SIZE = 0.15
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 128
CACHE_DIR = "embeddings_cache"

cache = EmbeddingCache(CACHE_DIR)

# Load data once
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
# Rename columns to match runner expectations
df = df.rename(columns={"posts": "text", "type": "label"})
print(f"  Total samples: {len(df)}")

# Load embedder
print(f"\nLoading embedder: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)

def normalize_text(text):
    """Basic text normalization."""
    import re
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\|\|\|", " ", text)
    text = text.strip()
    return text

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Pre-warming cache for seed {seed}")
    print(f"{'='*60}")

    # Check if already cached
    cached_train = cache.load(DATA_PATH, seed, "train", MODEL_NAME, TEST_SIZE, VAL_SIZE)
    cached_test = cache.load(DATA_PATH, seed, "test", MODEL_NAME, TEST_SIZE, VAL_SIZE)
    cached_val = cache.load(DATA_PATH, seed, "val", MODEL_NAME, TEST_SIZE, VAL_SIZE)

    if all([cached_train is not None, cached_test is not None, cached_val is not None]):
        print(f"  All embeddings already cached for seed {seed}")
        continue

    # Split data exactly as runner_phase2.py does
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE, random_state=seed, stratify=train_val_df["label"]
    )

    print(f"  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Compute embeddings for each split
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cached = cache.load(DATA_PATH, seed, split_name, MODEL_NAME, TEST_SIZE, VAL_SIZE)
        if cached is not None:
            continue

        print(f"\n  Computing {split_name} embeddings for seed {seed}...")
        texts = [normalize_text(t) for t in split_df["text"]]
        embeddings = embedder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cache.save(embeddings, DATA_PATH, seed, split_name, MODEL_NAME, TEST_SIZE, VAL_SIZE)

print(f"\n{'='*60}")
print(f"  Cache pre-warming complete!")
print(f"  Stats: {cache.get_stats()}")
print(f"{'='*60}")
PREWARM_EOF

log "Cache pre-warming complete"

# ============================================================================
# RUN EXPERIMENTS (grouped by seed for cache efficiency)
# ============================================================================

run_exp() {
    local NAME="$1" SEED="$2" EXTRA_ARGS="$3" CONFIG_TYPE="$4"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME s$SEED ($CONFIG_TYPE)"
    local T0=$(date +%s)

    # Check if already completed
    if [ -f "${COMPLETED_DIR}/${NAME}_s${SEED}_metrics.json" ] 2>/dev/null; then
        log "SKIP: $NAME s$SEED (already completed)"
        return
    fi

    CMD="python3 -u core/runner_phase2.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size $BATCH_SIZE \
        --llm-model gpt-4o-mini --max-clusters 3 --prompts-per-cluster 3 --prompt-mode mix \
        --use-ensemble-selection --use-val-gating --val-size 0.15 --val-tolerance 0.02 \
        --enable-anchor-gate --enable-anchor-selection \
        --anchor-selection-ratio 0.8 --anchor-outlier-threshold 1.5 \
        --use-class-description --use-f1-budget-scaling --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --min-classifier-confidence 0.10 --contamination-threshold 0.95 \
        --synthetic-weight 0.5 --synthetic-weight-mode flat \
        --use-hard-anchors --deterministic-quality-gate \
        --synthetic-output ${OUT}_synth.csv \
        --augmented-train-output ${OUT}_aug.csv \
        --metrics-output ${OUT}_metrics.json \
        $EXTRA_ARGS"

    if eval $CMD > "${OUT}.log" 2>&1; then
        local T1=$(date +%s) DUR=$((T1-T0))
        if [ -f "${OUT}_metrics.json" ]; then
            local BF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('baseline',{}).get('macro_f1',0):.4f}\")")
            local AF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('augmented',{}).get('macro_f1',0):.4f}\")")
            local DELTA=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));b=d.get('baseline',{}).get('macro_f1',0);a=d.get('augmented',{}).get('macro_f1',0);print(f\"{((a-b)/b*100) if b>0 else 0:.2f}\")")
            local SYNTH=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(d.get('synthetic_data',{}).get('accepted_count',0))")
            log "DONE: $NAME s$SEED -> B=$BF1 A=$AF1 D=${DELTA}% S=$SYNTH (${DUR}s)"
            echo "$NAME,$SEED,$CONFIG_TYPE,$BF1,$AF1,$DELTA,$SYNTH,$DUR,cached" >> "$SUMMARY_FILE"
        fi
    else
        log "FAIL: $NAME s$SEED"
        echo "$NAME,$SEED,$CONFIG_TYPE,FAIL,FAIL,FAIL,FAIL,0,cached" >> "$SUMMARY_FILE"
    fi
}

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

log "=== STARTING EXPERIMENTS ==="

# Group 1: Seed 42 experiments
log "--- SEED 42 GROUP ---"

# Remaining standard configs (cfg07-cfg13)
run_exp "cfg07_anchor_035" 42 "--anchor-quality-threshold 0.35 --similarity-threshold 0.90" "anchor_sweep" &
run_exp "cfg08_anchor_040" 42 "--anchor-quality-threshold 0.40 --similarity-threshold 0.90" "anchor_sweep" &
wait

run_exp "cfg09_sim_085" 42 "--anchor-quality-threshold 0.30 --similarity-threshold 0.85" "sim_sweep" &
run_exp "cfg10_sim_088" 42 "--anchor-quality-threshold 0.30 --similarity-threshold 0.88" "sim_sweep" &
wait

run_exp "cfg11_sim_092" 42 "--anchor-quality-threshold 0.30 --similarity-threshold 0.92" "sim_sweep" &
run_exp "cfg12_sim_095" 42 "--anchor-quality-threshold 0.30 --similarity-threshold 0.95" "sim_sweep" &
wait

run_exp "cfg13_combined" 42 "--anchor-quality-threshold 0.25 --similarity-threshold 0.85" "combined" &
wait

# IMPROVEMENT EXPERIMENTS for seed 42
log "--- IMPROVEMENT EXPERIMENTS (Seed 42) ---"

# Exp: More synthetics (5 clusters x 9 prompts = 45 per class)
run_exp "imp_more_synth" 42 "--anchor-quality-threshold 0.30 --similarity-threshold 0.90 --max-clusters 5 --prompts-per-cluster 9" "improvement" &

# Exp: Relaxed gate (lower thresholds)
run_exp "imp_relaxed_gate" 42 "--anchor-quality-threshold 0.15 --similarity-threshold 0.80" "improvement" &
wait

# Group 2: Seed 100 experiments
log "--- SEED 100 GROUP ---"

run_exp "cfg07_anchor_035" 100 "--anchor-quality-threshold 0.35 --similarity-threshold 0.90" "anchor_sweep" &
run_exp "cfg08_anchor_040" 100 "--anchor-quality-threshold 0.40 --similarity-threshold 0.90" "anchor_sweep" &
wait

run_exp "cfg09_sim_085" 100 "--anchor-quality-threshold 0.30 --similarity-threshold 0.85" "sim_sweep" &
run_exp "cfg10_sim_088" 100 "--anchor-quality-threshold 0.30 --similarity-threshold 0.88" "sim_sweep" &
wait

run_exp "cfg11_sim_092" 100 "--anchor-quality-threshold 0.30 --similarity-threshold 0.92" "sim_sweep" &
run_exp "cfg12_sim_095" 100 "--anchor-quality-threshold 0.30 --similarity-threshold 0.95" "sim_sweep" &
wait

run_exp "cfg13_combined" 100 "--anchor-quality-threshold 0.25 --similarity-threshold 0.85" "combined" &
wait

# IMPROVEMENT EXPERIMENTS for seed 100
log "--- IMPROVEMENT EXPERIMENTS (Seed 100) ---"

run_exp "imp_more_synth" 100 "--anchor-quality-threshold 0.30 --similarity-threshold 0.90 --max-clusters 5 --prompts-per-cluster 9" "improvement" &
run_exp "imp_relaxed_gate" 100 "--anchor-quality-threshold 0.15 --similarity-threshold 0.80" "improvement" &
wait

# ============================================================================
# IP SCALING EXPERIMENTS (requires patched runner)
# ============================================================================

log "--- IP SCALING EXPERIMENTS ---"

# Create IP-enhanced runner
python3 << 'PATCH_EOF'
import re

# Read original runner
with open('core/runner_phase2.py', 'r') as f:
    original_code = f.read()

# IP-enhanced calculate_enhanced_budget function
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
        quality_reason = f"Very low quality ({quality_score:.3f})"
    elif quality_score < 0.40:
        quality_mult = 0.3
        quality_reason = f"Low quality ({quality_score:.3f})"
    elif quality_score < 0.50:
        quality_mult = 0.7
        quality_reason = f"Mediocre quality ({quality_score:.3f})"
    else:
        quality_mult = 1.0
        quality_reason = f"Good quality ({quality_score:.3f})"

    # Purity multiplier
    if purity < purity_low_threshold:
        purity_mult = purity_low_multiplier
        purity_reason = f"Low purity ({purity:.3f}) -> {int(purity_mult*100)}%"
    else:
        purity_mult = 1.0
        purity_reason = f"Purity OK ({purity:.3f})"

    # IP multiplier (NEW - Phase E)
    ip = 1 - baseline_f1  # Improvement Potential
    ip_threshold = 0.7
    ip_boost_factor = 1.0

    if ip > ip_threshold:
        ip_mult = 1 + ip_boost_factor * ip
        ip_reason = f"IP={ip:.3f} (high potential) -> {int(ip_mult*100)}% boost"
    elif ip > 0.5:
        ip_mult = 1 + 0.5 * ip_boost_factor * ip
        ip_reason = f"IP={ip:.3f} (moderate) -> {int(ip_mult*100)}%"
    else:
        ip_mult = 1.0
        ip_reason = f"IP={ip:.3f} (low potential)"

    # Combine
    total_mult = quality_mult * purity_mult * ip_mult
    budget = int(base_budget * total_mult)

    # IP RESCUE for high-IP classes
    if ip > ip_threshold:
        ip_minimum = int(10 + 20 * ip)
        if budget < ip_minimum:
            budget = ip_minimum
            ip_reason += f" -> IP rescue: min {ip_minimum}"

    budget = max(10, budget)

    reason = f"Base: {base_budget} (IP-enhanced)\\n"
    reason += f"   {quality_reason}\\n"
    reason += f"   {purity_reason}\\n"
    reason += f"   {ip_reason}\\n"
    reason += f"   -> Final: {budget} (x{total_mult:.2f})"

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
pattern = r'(# Phase 2: Enhanced Dynamic Budget Calculator.*?def calculate_enhanced_budget\(.*?\).*?return budget, reason, multipliers)'
patched_code = re.sub(pattern, ip_function, original_code, flags=re.DOTALL)

# Save patched version
with open('core/runner_phase2_ip.py', 'w') as f:
    f.write(patched_code)

print("Created IP-enhanced runner: core/runner_phase2_ip.py")
PATCH_EOF

# Run IP scaling experiments
run_ip_exp() {
    local NAME="$1" SEED="$2"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME s$SEED (IP scaling)"
    local T0=$(date +%s)

    CMD="python3 -u core/runner_phase2_ip.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size $BATCH_SIZE \
        --llm-model gpt-4o-mini --max-clusters 3 --prompts-per-cluster 3 --prompt-mode mix \
        --use-ensemble-selection --use-val-gating --val-size 0.15 --val-tolerance 0.02 \
        --enable-anchor-gate --anchor-quality-threshold 0.30 --enable-anchor-selection \
        --anchor-selection-ratio 0.8 --anchor-outlier-threshold 1.5 \
        --use-class-description --use-f1-budget-scaling --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 --contamination-threshold 0.95 \
        --synthetic-weight 0.5 --synthetic-weight-mode flat \
        --use-hard-anchors --deterministic-quality-gate \
        --synthetic-output ${OUT}_synth.csv \
        --augmented-train-output ${OUT}_aug.csv \
        --metrics-output ${OUT}_metrics.json"

    if eval $CMD > "${OUT}.log" 2>&1; then
        local T1=$(date +%s) DUR=$((T1-T0))
        if [ -f "${OUT}_metrics.json" ]; then
            local BF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('baseline',{}).get('macro_f1',0):.4f}\")")
            local AF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('augmented',{}).get('macro_f1',0):.4f}\")")
            local DELTA=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));b=d.get('baseline',{}).get('macro_f1',0);a=d.get('augmented',{}).get('macro_f1',0);print(f\"{((a-b)/b*100) if b>0 else 0:.2f}\")")
            local SYNTH=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(d.get('synthetic_data',{}).get('accepted_count',0))")
            log "DONE: $NAME s$SEED -> B=$BF1 A=$AF1 D=${DELTA}% S=$SYNTH (${DUR}s)"
            echo "$NAME,$SEED,ip_scaling,$BF1,$AF1,$DELTA,$SYNTH,$DUR,cached" >> "$SUMMARY_FILE"
        fi
    else
        log "FAIL: $NAME s$SEED"
        echo "$NAME,$SEED,ip_scaling,FAIL,FAIL,FAIL,FAIL,0,cached" >> "$SUMMARY_FILE"
    fi
}

run_ip_exp "imp_ip_scaling" 42 &
run_ip_exp "imp_ip_scaling" 100 &
wait

# ============================================================================
# SUMMARY
# ============================================================================

log "=== ALL EXPERIMENTS COMPLETE ==="

python3 << PYEOF
import pandas as pd
import os

summary_file = "$SUMMARY_FILE"
if os.path.exists(summary_file):
    df = pd.read_csv(summary_file)
    df_valid = df[df['baseline_f1'] != 'FAIL']

    print("="*70)
    print("  OVERNIGHT RESULTS SUMMARY")
    print("="*70)

    if len(df_valid) > 0:
        df_valid['delta_pct'] = df_valid['delta_pct'].astype(float)
        df_valid['synthetics'] = df_valid['synthetics'].astype(int)

        # Group by config type
        print("\n  BY CONFIG TYPE:")
        for ctype in df_valid['config_type'].unique():
            subset = df_valid[df_valid['config_type'] == ctype]
            mean_delta = subset['delta_pct'].mean()
            mean_synth = subset['synthetics'].mean()
            print(f"    {ctype}: delta={mean_delta:+.2f}%, synth={mean_synth:.0f} (n={len(subset)})")

        # Best experiments
        print("\n  TOP 5 BY DELTA:")
        top5 = df_valid.nlargest(5, 'delta_pct')[['experiment', 'seed', 'delta_pct', 'synthetics']]
        for _, row in top5.iterrows():
            print(f"    {row['experiment']} s{row['seed']}: {row['delta_pct']:+.2f}% ({row['synthetics']} synth)")

        # Improvement experiments
        imp = df_valid[df_valid['config_type'] == 'improvement']
        if len(imp) > 0:
            print("\n  IMPROVEMENT EXPERIMENTS:")
            for _, row in imp.iterrows():
                print(f"    {row['experiment']} s{row['seed']}: {row['delta_pct']:+.2f}% ({row['synthetics']} synth)")

        # IP scaling
        ip = df_valid[df_valid['config_type'] == 'ip_scaling']
        if len(ip) > 0:
            print("\n  IP SCALING EXPERIMENTS:")
            for _, row in ip.iterrows():
                print(f"    {row['experiment']} s{row['seed']}: {row['delta_pct']:+.2f}% ({row['synthetics']} synth)")

    print("="*70)
PYEOF

log "Results saved to: $SUMMARY_FILE"
echo ""
echo "View results: cat $SUMMARY_FILE"
echo "View logs: tail -f $LOG_FILE"
