#!/bin/bash
# Run GPU evaluations for all replications

cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_h

SYNTHS=(
  "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/replication_run2/results/ENS_SUPER_G5_F7_v2_synth.csv"
  "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/replication_run3/results/ENS_SUPER_G5_F7_v2_synth.csv"
  "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/replication_run1/results/ENS_TopG5_Extended_synth.csv"
  "/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/replication_run1/results/ENS_Top3_G5_synth.csv"
)

for synth in "${SYNTHS[@]}"; do
  echo "=========================================="
  echo "Evaluating: $(basename $synth)"
  echo "=========================================="
  python3 -u eval_replication_gpu.py --synth "$synth" --k 5 2>&1
  echo ""
done
