#!/bin/bash
# Phase I: Run 4-config ensemble for a single LLM model
# Usage: ./run_model_ensemble.sh <model_prefix> [seed]
# Example: ./run_model_ensemble.sh gpt4o 42
#          ./run_model_ensemble.sh claude_opus 42

set -e
cd "$(dirname "$0")"

MODEL=$1
SEED=${2:-42}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_prefix> [seed]"
    echo "Example: $0 gpt4o 42"
    echo ""
    echo "Available model prefixes:"
    ls -1 configs/*.sh | sed 's|configs/||;s|_[A-Z].*||' | sort -u
    exit 1
fi

echo "==============================================================="
echo "  Phase I: Running 4-config ensemble for $MODEL"
echo "  Seed: $SEED | Date: $(date)"
echo "==============================================================="

mkdir -p results

# Check if all 4 configs exist
MISSING=0
for comp in CMB3 CF1 V4 G5; do
    if [ ! -f "configs/${MODEL}_${comp}.sh" ]; then
        echo "ERROR: Config not found: configs/${MODEL}_${comp}.sh"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Available configs:"
    ls -1 configs/*.sh
    exit 1
fi

# Make configs executable
chmod +x configs/${MODEL}_*.sh

# ===================================================================
# STEP 1: Run 4 component configs in parallel
# ===================================================================
echo ""
echo "=== STEP 1: Running 4 component configs in parallel ==="
echo ""

for comp in CMB3 CF1 V4 G5; do
    echo "SEED=$SEED bash configs/${MODEL}_${comp}.sh 2>&1 | tee results/${MODEL}_${comp}_s${SEED}.log"
done | parallel -j 4 --progress

# ===================================================================
# STEP 2: Create ensemble from 4 outputs
# ===================================================================
echo ""
echo "=== STEP 2: Creating ensemble from 4 outputs ==="
echo ""

python3 -c "
import pandas as pd
import os

results_dir = 'results'
model = '$MODEL'
seed = $SEED
components = ['CMB3', 'CF1', 'V4', 'G5']

dfs = []
for comp in components:
    path = f'{results_dir}/{model}_{comp}_s{seed}_synth.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'  {comp}: {len(df)} samples')
        dfs.append(df)
    else:
        print(f'  WARNING: {path} not found')

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    text_col = 'text' if 'text' in combined.columns else 'posts'
    initial = len(combined)
    combined = combined.drop_duplicates(subset=[text_col])
    print(f'  Combined: {initial} -> {len(combined)} (removed {initial-len(combined)} dups)')

    out_path = f'{results_dir}/ENS_{model}_s{seed}_synth.csv'
    combined.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
else:
    print('  ERROR: No synthetic data to combine!')
    exit(1)
"

# ===================================================================
# STEP 3: K-Fold evaluation (k=5, repeats=3)
# ===================================================================
echo ""
echo "=== STEP 3: K-Fold evaluation (k=5, repeats=3) ==="
echo ""

python3 kfold_evaluator.py \
    --config "ENS_${MODEL}" \
    --seed $SEED \
    --k 5 \
    --repeated 3 \
    2>&1 | tee "results/ENS_${MODEL}_s${SEED}_kfold.log"

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "==============================================================="
echo "  COMPLETE: ENS_${MODEL}_s${SEED}"
echo "==============================================================="
echo ""

# Print summary if metrics file exists
if [ -f "results/ENS_${MODEL}_s${SEED}_kfold_k5.json" ]; then
    python3 -c "
import json
with open('results/ENS_${MODEL}_s${SEED}_kfold_k5.json') as f:
    d = json.load(f)
    print('Results:')
    print(f\"  Synthetic samples: {d.get('n_synthetic', 'N/A')}\")
    print(f\"  Baseline F1: {d['baseline']['mean']:.4f} +/- {d['baseline']['std']:.4f}\")
    print(f\"  Augmented F1: {d['augmented']['mean']:.4f} +/- {d['augmented']['std']:.4f}\")
    print(f\"  Delta: {d['delta']['mean']*100:+.2f}% (p={d['delta']['p_value']:.4f})\")
    print(f\"  Win rate: {d['delta']['wins']}/{d['n_folds']} ({d['delta']['win_rate']*100:.1f}%)\")
"
fi
