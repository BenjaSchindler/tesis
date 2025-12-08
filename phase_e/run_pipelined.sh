#!/bin/bash
# Phase E - Pipelined Experiment Runner
#
# Optimized to maximize resource utilization:
# - Embeddings run sequentially (maximize GPU batch efficiency)
# - LLM generation can overlap (uses API, not GPU)
# - Training runs when GPU is free
#
# Pipeline architecture:
#   Exp 1: [EMBED]────────>[LLM GEN]────────>[TRAIN]
#   Exp 2:         [EMBED]────────>[LLM GEN]────────>[TRAIN]
#   Exp 3:                 [EMBED]────────>[LLM GEN]────────>...
#
# vs Sequential (current):
#   Exp 1: [EMBED]────────>[LLM GEN]────────>[TRAIN]
#   Exp 2:                                           [EMBED]────────>...
#
# Expected speedup: ~40% (LLM generation overlaps with next embedding)

set -e

cd "$(dirname "$0")"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Phase E - PIPELINED Experiment Runner                        ║"
echo "║  Optimized for GPU + API parallelization                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MAX_CONCURRENT_LLM=${1:-2}  # Default: 2 concurrent LLM generations

echo "Configuration:"
echo "  Max concurrent LLM generations: $MAX_CONCURRENT_LLM"
echo "  Data: ../MBTI_500.csv"
echo "  Seeds: 42, 100"
echo ""

# Run pipelined
python3 pipelined_runner.py --max-concurrent-llm "$MAX_CONCURRENT_LLM"
