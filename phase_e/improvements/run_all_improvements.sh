#!/bin/bash
# ============================================================================
# Run ALL Improvement Experiments with Cache Support
# ============================================================================
#
# Experiments (all corrected with cap-class-ratio 0.15):
#   1. more_synthetics - 5 clusters × 9 prompts = 225 candidatos/clase
#   2. relaxed_gate - Lower quality thresholds (0.15/0.80)
#   3. ip_scaling - IP-enhanced budget (sin doble escalado)
#   4. cfg12_capped - Best config (sim_095) + more synthetics
#
# Seeds: 42, 100 (for robustness)
# Total: 4 experiments × 2 seeds = 8 runs
#
# With cache: ~35 min/experiment
# Parallel 3: ~2-3 hours total
# ============================================================================

set -e

cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Configuration
PARALLEL=${1:-3}
SEEDS=(42 100)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="improvements/results/batch_${TIMESTAMP}"
LOG_FILE="$RESULTS_DIR/master.log"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase E - ALL IMPROVEMENT EXPERIMENTS                        ║"
echo "║  Experiments: more_synthetics, relaxed_gate, ip_scaling,      ║"
echo "║               cfg12_capped                                    ║"
echo "║  Seeds: ${SEEDS[*]}                                           ║"
echo "║  Parallel: $PARALLEL                                          ║"
echo "║  Cache: embeddings_cache (enabled)                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Pre-warm cache
log "=== PRE-WARMING EMBEDDING CACHE ==="

python3 << 'PREWARM_EOF'
import sys
sys.path.insert(0, 'core')

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

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"posts": "text", "type": "label"})
print(f"  Total samples: {len(df)}")

print(f"\nLoading embedder: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)

def normalize_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\|\|\|", " ", text)
    return text.strip()

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"  Checking cache for seed {seed}")
    print(f"{'='*50}")

    cached_train = cache.load(DATA_PATH, seed, "train", MODEL_NAME, TEST_SIZE, VAL_SIZE)
    cached_test = cache.load(DATA_PATH, seed, "test", MODEL_NAME, TEST_SIZE, VAL_SIZE)
    cached_val = cache.load(DATA_PATH, seed, "val", MODEL_NAME, TEST_SIZE, VAL_SIZE)

    if all([cached_train is not None, cached_test is not None, cached_val is not None]):
        print(f"  All embeddings already cached for seed {seed}")
        continue

    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE, random_state=seed, stratify=train_val_df["label"]
    )

    print(f"  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

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

print(f"\n{'='*50}")
print(f"  Cache ready! Stats: {cache.get_stats()}")
print(f"{'='*50}")
PREWARM_EOF

log "Cache ready"

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

run_exp() {
    local SCRIPT="$1"
    local SEED="$2"
    local NAME=$(basename "$SCRIPT" .sh)

    log "START: $NAME seed=$SEED"
    local T0=$(date +%s)

    if bash "$SCRIPT" "$SEED"; then
        local T1=$(date +%s)
        local DUR=$((T1-T0))
        log "DONE: $NAME seed=$SEED (${DUR}s)"
    else
        log "FAIL: $NAME seed=$SEED"
    fi
}

export -f run_exp log
export LOG_FILE

log "=== STARTING ALL EXPERIMENTS ==="

# Define experiments
EXPERIMENTS=(
    "improvements/exp_more_synthetics.sh"
    "improvements/exp_relaxed_gate.sh"
    "improvements/exp_ip_scaling.sh"
    "improvements/exp_cfg12_capped.sh"
)

# Run all combinations
JOBS=()
for SCRIPT in "${EXPERIMENTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        while [ $(jobs -r | wc -l) -ge $PARALLEL ]; do
            sleep 10
        done

        run_exp "$SCRIPT" "$SEED" &
        JOBS+=($!)
    done
done

# Wait for all
for JOB in "${JOBS[@]}"; do
    wait $JOB 2>/dev/null || true
done

log "=== ALL EXPERIMENTS COMPLETE ==="

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  RESULTS SUMMARY                                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python3 << PYEOF
import json
import glob
from pathlib import Path

results_dir = Path("improvements/results")
metrics_files = sorted(results_dir.glob("*_metrics.json"))

print(f"Found {len(metrics_files)} result files:\n")

results = []
for f in metrics_files:
    try:
        with open(f) as fp:
            d = json.load(fp)
        name = f.stem.replace("_metrics", "")
        b = d.get("baseline", {}).get("macro_f1", 0)
        a = d.get("augmented", {}).get("macro_f1", 0)
        s = d.get("synthetic_data", {}).get("accepted_count", 0)
        delta = ((a - b) / b * 100) if b > 0 else 0
        results.append((name, b, a, delta, s))
        print(f"  {name}:")
        print(f"    Baseline: {b:.4f}, Augmented: {a:.4f}")
        print(f"    Delta: {delta:+.2f}%, Synthetics: {s}")
        print()
    except Exception as e:
        print(f"  {f.name}: ERROR - {e}")

if results:
    print("=" * 60)
    print("  TOP RESULTS (by delta):")
    for name, b, a, delta, s in sorted(results, key=lambda x: -x[3])[:5]:
        print(f"    {delta:+.2f}%  {name} ({s} synth)")
PYEOF

log "Results summary saved to: $RESULTS_DIR"
echo ""
echo "View logs: tail -f $LOG_FILE"
