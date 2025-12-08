#!/bin/bash
# Phase H - Multi-Embedder Evaluation
# Evaluates SMOTE-LLM robustness across 7 SOTA embedding models

set -e

cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_h

echo "=========================================="
echo "  Phase H - Multi-Embedder Evaluation"
echo "=========================================="
echo ""
echo "This will evaluate 7 embedders x 3 replications = 21 evaluations"
echo "Estimated time: ~45-50 minutes"
echo ""

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
echo ""

# Run evaluation
python3 -u eval_multi_embedder.py 2>&1 | tee results/multi_embedder.log

echo ""
echo "Done! Results saved to:"
echo "  - results/multi_embedder_results.json"
echo "  - results/multi_embedder.log"
