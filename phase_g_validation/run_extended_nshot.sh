#!/bin/bash
# Extended n_shot experiments - Many-Shot Prompting
# Tests n_shot values: 20, 50, 100, 200

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Extended n_shot Experiments"
echo "=========================================="
echo "Testing n_shot values: 20, 50, 100, 200"
echo "Reference: n_shot=10 achieved +1.22 pp"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

echo "OPENAI_API_KEY is set"
echo ""

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/extended_nshot_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Results will be saved to: $LOG_DIR"
echo ""

# Run the experiment
python3 -u experiments/exp_extended_nshot.py 2>&1 | tee "$LOG_DIR/experiment.log"

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "Log saved to: $LOG_DIR/experiment.log"
echo "=========================================="
