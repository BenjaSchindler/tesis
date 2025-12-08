#!/bin/bash
# Phase G: Run 4-config ensemble (replicating ENS_Top3_G5)
# Focus on analyzing 5 problematic classes: ENFJ, ESFJ, ESFP, ESTJ, ISTJ
# Usage: ./run_phaseG_ensemble.sh [seed]

set -e
cd "$(dirname "$0")"

SEED=${1:-42}
PROBLEMATIC="ENFJ,ESFJ,ESFP,ESTJ,ISTJ"

echo "==============================================================="
echo "  Phase G: ENS_Top3_G5 Ensemble with Per-Class Analysis"
echo "  Seed: $SEED | Focus: $PROBLEMATIC"
echo "  Date: $(date)"
echo "==============================================================="

mkdir -p results

# Make configs executable
chmod +x configs/*.sh

# ===================================================================
# STEP 1: Run 4 component configs in parallel
# ===================================================================
echo ""
echo "=== STEP 1: Running 4 component configs in parallel ==="
echo ""

for cfg in CMB3_skip CF1_conf_band V4_ultra G5_K25_medium; do
    echo "SEED=$SEED bash configs/${cfg}.sh 2>&1 | tee results/${cfg}_s${SEED}.log"
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
seed = $SEED
components = ['CMB3_skip', 'CF1_conf_band', 'V4_ultra', 'G5_K25_medium']

dfs = []
for comp in components:
    path = f'{results_dir}/{comp}_s{seed}_synth.csv'
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

    # Show per-class breakdown
    label_col = 'type' if 'type' in combined.columns else 'label'
    print()
    print('  Per-class synthetic counts:')
    for cls in sorted(combined[label_col].unique()):
        count = len(combined[combined[label_col] == cls])
        marker = ' <-- PROBLEMATIC' if cls in ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISTJ'] else ''
        print(f'    {cls}: {count}{marker}')

    out_path = f'{results_dir}/ENS_Top3_G5_s{seed}_synth.csv'
    combined.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
else:
    print('  ERROR: No synthetic data to combine!')
    exit(1)
"

# ===================================================================
# STEP 3: K-Fold evaluation with per-class analysis
# ===================================================================
echo ""
echo "=== STEP 3: K-Fold evaluation (k=5, repeats=3) ==="
echo ""

python3 kfold_evaluator.py \
    --config "ENS_Top3_G5" \
    --seed $SEED \
    --k 5 \
    --repeated 3 \
    --report-per-class \
    2>&1 | tee "results/ENS_Top3_G5_s${SEED}_kfold.log"

# ===================================================================
# STEP 4: Analyze problematic classes
# ===================================================================
echo ""
echo "=== STEP 4: Analyzing 5 problematic classes ==="
echo ""

python3 -c "
import json
import os

seed = $SEED
problematic = ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISTJ']

# Load K-Fold results
kfold_path = f'results/ENS_Top3_G5_s{seed}_kfold_k5.json'
if os.path.exists(kfold_path):
    with open(kfold_path) as f:
        data = json.load(f)

    print('Global Metrics:')
    print(f\"  Baseline Macro F1:  {data['baseline']['mean']:.4f} +/- {data['baseline']['std']:.4f}\")
    print(f\"  Augmented Macro F1: {data['augmented']['mean']:.4f} +/- {data['augmented']['std']:.4f}\")
    print(f\"  Delta:              {data['delta']['mean']*100:+.2f}% (p={data['delta']['p_value']:.4f})\")
    print(f\"  Win rate:           {data['delta']['wins']}/{data['n_folds']} folds\")
    print()

    if 'per_class' in data:
        print('Problematic Classes Analysis:')
        print('-' * 60)
        for cls in problematic:
            if cls in data['per_class']:
                pc = data['per_class'][cls]
                delta_pct = (pc['augmented_mean'] - pc['baseline_mean']) / pc['baseline_mean'] * 100 if pc['baseline_mean'] > 0 else 0
                improved = 'IMPROVED' if delta_pct > 0 else 'WORSE'
                print(f\"  {cls}: Baseline={pc['baseline_mean']:.4f}, Aug={pc['augmented_mean']:.4f}, Delta={delta_pct:+.2f}% [{improved}]\")
        print()

        print('All Classes F1 Delta:')
        print('-' * 60)
        deltas = []
        for cls in sorted(data['per_class'].keys()):
            pc = data['per_class'][cls]
            delta_pct = (pc['augmented_mean'] - pc['baseline_mean']) / pc['baseline_mean'] * 100 if pc['baseline_mean'] > 0 else 0
            marker = ' <-- PROBLEMATIC' if cls in problematic else ''
            print(f\"  {cls}: {delta_pct:+.2f}%{marker}\")
            deltas.append((cls, delta_pct))

        # Summary
        prob_deltas = [d for c, d in deltas if c in problematic]
        other_deltas = [d for c, d in deltas if c not in problematic]
        print()
        print('Summary:')
        print(f\"  Problematic classes avg delta: {sum(prob_deltas)/len(prob_deltas):+.2f}%\")
        print(f\"  Other classes avg delta:       {sum(other_deltas)/len(other_deltas):+.2f}%\")
else:
    print(f'K-Fold results not found: {kfold_path}')
"

echo ""
echo "==============================================================="
echo "  COMPLETE: Phase G ENS_Top3_G5 seed=$SEED"
echo "==============================================================="
