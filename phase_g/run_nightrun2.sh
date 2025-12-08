#!/bin/bash
# NIGHTRUN2 - Phase G Extended
# 20 configs W1-W9, seed 42, K-Fold evaluation
# Ready to launch: nohup bash run_nightrun2.sh > results/nightrun2.log 2>&1 &

set -e
cd "$(dirname "$0")"

export OPENAI_API_KEY="${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set}"

PARALLEL_JOBS=10
SEED=42
RESULTS_DIR="results"

echo "=============================================="
echo "NIGHTRUN2 - Phase G Extended"
echo "=============================================="
echo "Start: $(date)"
echo "Configs: 20 (W1-W9)"
echo "Seed: $SEED"
echo "Parallel: $PARALLEL_JOBS"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Generate job list
cat > /tmp/nightrun2_jobs.txt << 'JOBS'
SEED=42 bash configs/W1_no_gate.sh
SEED=42 bash configs/W1_low_gate.sh
SEED=42 bash configs/W1_force_problem.sh
SEED=42 bash configs/W2_ultra_vol.sh
SEED=42 bash configs/W2_mega_vol.sh
SEED=42 bash configs/W3_permissive_filter.sh
SEED=42 bash configs/W3_no_dedup.sh
SEED=42 bash configs/W4_target_only.sh
SEED=42 bash configs/W5_zero_shot.sh
SEED=42 bash configs/W5_few_shot_3.sh
SEED=42 bash configs/W5_many_shot_10.sh
SEED=42 bash configs/W6_temp_low.sh
SEED=42 bash configs/W6_temp_high.sh
SEED=42 bash configs/W6_temp_extreme.sh
SEED=42 bash configs/W7_yolo.sh
SEED=42 bash configs/W7_yolo_force.sh
SEED=42 bash configs/W8_gpt5_high.sh
SEED=42 bash configs/W8_gpt5_reasoning.sh
SEED=42 bash configs/W9_contrastive.sh
SEED=42 bash configs/W9_best_combo.sh
JOBS

echo "=== PHASE 1: GENERATION (20 configs) ==="
cat /tmp/nightrun2_jobs.txt | parallel -j "$PARALLEL_JOBS" --joblog "$RESULTS_DIR/nightrun2_joblog.txt"

echo ""
echo "=== PHASE 2: K-FOLD EVALUATION ==="
python3 kfold_multimodel.py --all --k 5 --repeats 3 \
    --models LogisticRegression MLP_small \
    --output kfold_nightrun2_results.json

echo ""
echo "=============================================="
echo "NIGHTRUN2 COMPLETE"
echo "End: $(date)"
echo "=============================================="
