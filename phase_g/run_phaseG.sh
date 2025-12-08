#!/bin/bash
set -e
cd "$(dirname "$0")"
SEED=${SEED:-42}

echo ""
echo "==============================================================="
echo "  Phase G: Improving Problematic Classes"
echo "  Seed: $SEED | Date: $(date)"
echo "  Target: ENFJ, ESFJ, ESFP, ESTJ, ISTJ"
echo "==============================================================="

mkdir -p results

# Wave 1: Baseline + Purity Gate + Volume (3 parallel)
echo ""
echo "=== WAVE 1: Generation Experiments (3 parallel) ==="
echo "    G0_baseline, G1_no_purity_gate, G2_ultra_volume"
echo ""

WAVE1="G0_baseline G1_no_purity_gate G2_ultra_volume"
for cfg in $WAVE1; do
    echo "SEED=$SEED bash configs/${cfg}.sh 2>&1 | tee results/${cfg}_s${SEED}.log"
done | parallel -j 3 --progress

echo ""
echo "Wave 1 complete: $(date)"

# Wave 2: Filter Relaxation (2 parallel)
echo ""
echo "=== WAVE 2: Filter Experiments (2 parallel) ==="
echo "    G3_low_filters, G4_no_f1_skip"
echo ""

WAVE2="G3_low_filters G4_no_f1_skip"
for cfg in $WAVE2; do
    echo "SEED=$SEED bash configs/${cfg}.sh 2>&1 | tee results/${cfg}_s${SEED}.log"
done | parallel -j 2 --progress

echo ""
echo "Wave 2 complete: $(date)"

# Wave 3: Combo (1 at a time - resource intensive)
echo ""
echo "=== WAVE 3: Combo Experiment ==="
echo "    G5_combo (all strategies combined)"
echo ""

SEED=$SEED bash configs/G5_combo.sh 2>&1 | tee results/G5_combo_s${SEED}.log

echo ""
echo "Wave 3 complete: $(date)"

# K-Fold evaluations (all in parallel)
echo ""
echo "=== K-FOLD EVALUATIONS (6 parallel) ==="
echo ""

ALL_CONFIGS="G0_baseline G1_no_purity_gate G2_ultra_volume G3_low_filters G4_no_f1_skip G5_combo"
for cfg in $ALL_CONFIGS; do
    echo "python3 -u kfold_evaluator.py --config $cfg --seed $SEED --k 5 --repeated 3 2>&1 | tee results/${cfg}_kfold.log"
done | parallel -j 6 --progress

# Summary
echo ""
echo "==============================================================="
echo "  PHASE G COMPLETE - Per-Class Analysis"
echo "  Finished: $(date)"
echo "==============================================================="

echo ""
echo "Config                 Synth   Delta    ESFP   ENFJ   ISTJ"
echo "-------------------------------------------------------------"
for cfg in $ALL_CONFIGS; do
    python3 -c "
import json
try:
    with open('results/${cfg}_s${SEED}_metrics.json') as f:
        d = json.load(f)
        synth = d.get('n_synthetic', 0)
        delta = d.get('f1_delta', 0) * 100

        # Per-class synthetics
        pc = d.get('per_class_quality', {})
        esfp = pc.get('ESFP', {}).get('total_accepted', 0)
        enfj = pc.get('ENFJ', {}).get('total_accepted', 0)
        istj = pc.get('ISTJ', {}).get('total_accepted', 0)

        print(f'${cfg:<20} {synth:>5}   {delta:+.2f}%   {esfp:>4}   {enfj:>4}   {istj:>4}')
except:
    print('${cfg:<20} (metrics not available)')
" 2>/dev/null || echo "${cfg} (error reading metrics)"
done

echo ""
echo "To monitor: tail -f results/*.log"
echo "To analyze: python3 -c \"import json; print(json.load(open('results/G5_combo_s42_metrics.json'))['per_class_quality'])\""
