#!/bin/bash
# Pre-cache embeddings for all seeds before running experiments
# This allows running more parallel jobs since embedding computation is the GPU-heavy part

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEEDS="${SEEDS:-42 100 123}"
PROJECT_ROOT="/home/benja/Desktop/Tesis/SMOTE-LLM"

# IMPORTANT: Use same absolute paths as base_config.sh for cache key matching
python3 precache_embeddings.py \
    --data-path "$PROJECT_ROOT/mbti_1.csv" \
    --seeds $SEEDS \
    --cache-dir "$PROJECT_ROOT/phase_e/embeddings_cache" \
    --device cuda \
    --batch-size 128

echo ""
echo "Now run: ./run_phaseF.sh"
