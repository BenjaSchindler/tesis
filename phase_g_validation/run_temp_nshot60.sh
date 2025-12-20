#!/bin/bash
# Run temperature experiments with optimal n_shot=60
# Expected duration: ~1 hour (5 configs x ~12 min each)

set -e

cd "$(dirname "$0")"

echo "=============================================="
echo "Temperature Experiments with n_shot=60"
echo "=============================================="
echo ""
echo "Configs: W5b_temp03_n60, W5b_temp06_n60, W5b_temp09_n60, W5b_temp12_n60, W5b_temp15_n60"
echo "Estimated time: ~1 hour"
echo ""

# Create results directory
mkdir -p results/wave5b_temp

# Run experiment
python3 -u experiments/exp_temp_with_nshot60.py 2>&1 | tee results/wave5b_temp/experiment.log

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Results in: results/wave5b_temp/"
echo "=============================================="
