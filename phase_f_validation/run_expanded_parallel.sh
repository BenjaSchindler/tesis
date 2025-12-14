#!/bin/bash
# Run all expanded experiments in parallel
# Each experiment runs in background with separate log file

cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_f_validation

echo "=============================================="
echo "  Running Expanded Experiments in Parallel"
echo "=============================================="
echo "  Start time: $(date)"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    exit 1
fi

mkdir -p results/logs

# Launch all experiments in parallel
echo "Launching exp01_clustering_expanded..."
nohup python3 -u experiments/exp01_clustering_expanded.py > results/logs/exp01_expanded.log 2>&1 &
PID1=$!

echo "Launching exp03_k_neighbors_expanded..."
nohup python3 -u experiments/exp03_k_neighbors_expanded.py > results/logs/exp03_expanded.log 2>&1 &
PID2=$!

echo "Launching exp07c_budget_expanded..."
nohup python3 -u experiments/exp07c_budget_expanded.py > results/logs/exp07c_expanded.log 2>&1 &
PID3=$!

echo "Launching exp07a_weight_by_tier..."
nohup python3 -u experiments/exp07a_weight_by_tier.py > results/logs/exp07a_expanded.log 2>&1 &
PID4=$!

echo "Launching exp07b_temperature_diversity..."
nohup python3 -u experiments/exp07b_temperature_diversity.py > results/logs/exp07b_expanded.log 2>&1 &
PID5=$!

echo ""
echo "All experiments launched!"
echo "PIDs: $PID1, $PID2, $PID3, $PID4, $PID5"
echo ""
echo "Monitor with:"
echo "  tail -f results/logs/exp01_expanded.log"
echo "  tail -f results/logs/exp03_expanded.log"
echo "  tail -f results/logs/exp07a_expanded.log"
echo "  tail -f results/logs/exp07b_expanded.log"
echo "  tail -f results/logs/exp07c_expanded.log"
echo ""
echo "Or check all at once:"
echo "  for f in results/logs/exp*_expanded.log; do echo \"=== \$f ===\"; tail -3 \$f; done"
echo ""
echo "Wait for all to complete:"
echo "  wait $PID1 $PID2 $PID3 $PID4 $PID5"
