#!/bin/bash
# Phase E - Run experiments with embedding cache
#
# Now that runner_phase2.py supports --cache-dir, experiments will:
# 1. Check cache for existing embeddings
# 2. If cache HIT: Skip embedding computation (~55% time savings)
# 3. If cache MISS: Compute and save to cache for future runs
#
# Usage:
#   ./run_with_cache.sh                    # Run all remaining experiments
#   ./run_with_cache.sh --parallel 3       # Run 3 experiments in parallel
#   ./run_with_cache.sh --quick-test       # Test with 2 experiments
#
# With cache hits, we can run more experiments in parallel since
# the GPU-heavy embedding step is skipped!

set -e

cd "$(dirname "$0")"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Configuration
CACHE_DIR="embeddings_cache"
PARALLEL=${1:-3}  # Default: 3 parallel (can be higher with cache!)
BATCH_SIZE=128

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cached_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CACHE_DIR"

LOG_FILE="$RESULTS_DIR/master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "experiment,seed,config_type,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s,cache_status" > "$SUMMARY_FILE"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase E - Experiments with Embedding Cache                   ║"
echo "║  Cache dir: $CACHE_DIR                                        ║"
echo "║  Parallel: $PARALLEL (safe with cache!)                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check existing cache
CACHED_FILES=$(ls -1 "$CACHE_DIR"/*.npz 2>/dev/null | wc -l || echo 0)
log "Found $CACHED_FILES cached embedding files"

# Pre-warm cache for seeds that will be used
log "=== PRE-WARMING EMBEDDING CACHE ==="

python3 << 'PREWARM_EOF'
import sys
sys.path.insert(0, 'core')

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from embedding_cache import EmbeddingCache
import re

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
df = df.rename(columns={"posts": "text", "type": "label"})
print(f"  Total samples: {len(df)}")

# Load embedder
print(f"\nLoading embedder: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)

def normalize_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\|\|\|", " ", text)
    return text.strip()

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Checking cache for seed {seed}")
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
# RUN EXPERIMENTS
# ============================================================================

run_exp() {
    local NAME="$1" SEED="$2" EXTRA_ARGS="$3" CONFIG_TYPE="$4"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME s$SEED ($CONFIG_TYPE)"
    local T0=$(date +%s)

    # Run with cache enabled!
    CMD="python3 -u core/runner_phase2.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size $BATCH_SIZE \
        --cache-dir $CACHE_DIR \
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
        log "FAIL: $NAME s$SEED (see ${OUT}.log)"
        echo "$NAME,$SEED,$CONFIG_TYPE,FAIL,FAIL,FAIL,FAIL,0,cached" >> "$SUMMARY_FILE"
    fi
}

export -f run_exp log
export RESULTS_DIR CACHE_DIR BATCH_SIZE SUMMARY_FILE LOG_FILE

log "=== STARTING EXPERIMENTS (parallel=$PARALLEL) ==="

# Define experiments
declare -a EXPERIMENTS=(
    # Standard configs - seed 42
    "cfg07_anchor_035|42|--anchor-quality-threshold 0.35 --similarity-threshold 0.90|anchor_sweep"
    "cfg08_anchor_040|42|--anchor-quality-threshold 0.40 --similarity-threshold 0.90|anchor_sweep"
    "cfg09_sim_085|42|--anchor-quality-threshold 0.30 --similarity-threshold 0.85|sim_sweep"
    "cfg10_sim_088|42|--anchor-quality-threshold 0.30 --similarity-threshold 0.88|sim_sweep"
    "cfg11_sim_092|42|--anchor-quality-threshold 0.30 --similarity-threshold 0.92|sim_sweep"
    "cfg12_sim_095|42|--anchor-quality-threshold 0.30 --similarity-threshold 0.95|sim_sweep"
    "cfg13_combined|42|--anchor-quality-threshold 0.25 --similarity-threshold 0.85|combined"

    # Improvement experiments - seed 42
    "imp_more_synth|42|--anchor-quality-threshold 0.30 --similarity-threshold 0.90 --max-clusters 5 --prompts-per-cluster 9|improvement"
    "imp_relaxed_gate|42|--anchor-quality-threshold 0.15 --similarity-threshold 0.80|improvement"

    # Standard configs - seed 100
    "cfg07_anchor_035|100|--anchor-quality-threshold 0.35 --similarity-threshold 0.90|anchor_sweep"
    "cfg08_anchor_040|100|--anchor-quality-threshold 0.40 --similarity-threshold 0.90|anchor_sweep"
    "cfg09_sim_085|100|--anchor-quality-threshold 0.30 --similarity-threshold 0.85|sim_sweep"
    "cfg10_sim_088|100|--anchor-quality-threshold 0.30 --similarity-threshold 0.88|sim_sweep"
    "cfg11_sim_092|100|--anchor-quality-threshold 0.30 --similarity-threshold 0.92|sim_sweep"
    "cfg12_sim_095|100|--anchor-quality-threshold 0.30 --similarity-threshold 0.95|sim_sweep"
    "cfg13_combined|100|--anchor-quality-threshold 0.25 --similarity-threshold 0.85|combined"

    # Improvement experiments - seed 100
    "imp_more_synth|100|--anchor-quality-threshold 0.30 --similarity-threshold 0.90 --max-clusters 5 --prompts-per-cluster 9|improvement"
    "imp_relaxed_gate|100|--anchor-quality-threshold 0.15 --similarity-threshold 0.80|improvement"
)

# Run experiments with parallel jobs
run_parallel() {
    local jobs=0
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME SEED ARGS CONFIG <<< "$exp"

        # Check if already completed
        if [ -f "${RESULTS_DIR}/${NAME}_s${SEED}_metrics.json" ]; then
            log "SKIP: $NAME s$SEED (already done)"
            continue
        fi

        run_exp "$NAME" "$SEED" "$ARGS" "$CONFIG" &
        ((jobs++))

        if [ $jobs -ge $PARALLEL ]; then
            wait -n
            ((jobs--))
        fi
    done
    wait
}

run_parallel

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
    print("  EXPERIMENT RESULTS SUMMARY")
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

        # Overall stats
        print(f"\n  OVERALL:")
        print(f"    Mean delta: {df_valid['delta_pct'].mean():+.2f}%")
        print(f"    Best: {df_valid['delta_pct'].max():+.2f}%")
        print(f"    Worst: {df_valid['delta_pct'].min():+.2f}%")

    print("="*70)
PYEOF

log "Results saved to: $SUMMARY_FILE"
echo ""
echo "View results: cat $SUMMARY_FILE"
echo "View logs: tail -f $LOG_FILE"
