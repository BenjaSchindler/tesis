#!/bin/bash
# Phase E - Overnight Experiments (2 seeds per config)
# Goal: Find best macro F1 configuration
# Expected duration: ~15 hours (13 configs x 2 seeds x ~35min each)
# Note: No 'set -e' so script continues even if experiments fail

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/overnight_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

# CSV header
echo "experiment,seed,hard_anchors,det_gate,anchor_thresh,sim_thresh,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s" > "$SUMMARY_FILE"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_exp() {
    local NAME="$1" SEED="$2" HA="$3" DG="$4" AT="$5" ST="$6"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME seed=$SEED (HA=$HA DG=$DG AT=$AT ST=$ST)"
    local T0=$(date +%s)

    CMD="python3 core/runner_phase2.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size 128 \
        --llm-model gpt-4o-mini --max-clusters 3 --prompts-per-cluster 3 --prompt-mode mix \
        --use-ensemble-selection --use-val-gating --val-size 0.15 --val-tolerance 0.02 \
        --enable-anchor-gate --anchor-quality-threshold $AT --enable-anchor-selection \
        --anchor-selection-ratio 0.8 --anchor-outlier-threshold 1.5 \
        --use-class-description --use-f1-budget-scaling --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 --similarity-threshold $ST \
        --min-classifier-confidence 0.10 --contamination-threshold 0.95 \
        --synthetic-weight 0.5 --synthetic-weight-mode flat \
        --synthetic-output ${OUT}_synth.csv --augmented-train-output ${OUT}_aug.csv \
        --metrics-output ${OUT}_metrics.json"

    [ "$HA" = "ON" ] && CMD="$CMD --use-hard-anchors" || CMD="$CMD --no-hard-anchors"
    [ "$DG" = "ON" ] && CMD="$CMD --deterministic-quality-gate" || CMD="$CMD --no-deterministic-quality-gate"

    if eval $CMD > "${OUT}.log" 2>&1; then
        local T1=$(date +%s) DUR=$((T1-T0))
        if [ -f "${OUT}_metrics.json" ]; then
            local BF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('baseline',{}).get('macro_f1',0):.4f}\")")
            local AF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('augmented',{}).get('macro_f1',0):.4f}\")")
            local DELTA=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));b=d.get('baseline',{}).get('macro_f1',0);a=d.get('augmented',{}).get('macro_f1',0);print(f\"{((a-b)/b*100) if b>0 else 0:.2f}\")")
            local SYNTH=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(d.get('synthetic_data',{}).get('total_accepted',0))")
            log "DONE: $NAME s$SEED -> B=$BF1 A=$AF1 D=${DELTA}% S=$SYNTH (${DUR}s)"
            echo "$NAME,$SEED,$HA,$DG,$AT,$ST,$BF1,$AF1,$DELTA,$SYNTH,$DUR" >> "$SUMMARY_FILE"
        fi
    else
        log "FAIL: $NAME seed=$SEED"
        echo "$NAME,$SEED,$HA,$DG,$AT,$ST,FAIL,FAIL,FAIL,FAIL,0" >> "$SUMMARY_FILE"
    fi
}

log "=========================================="
log "OVERNIGHT EXPERIMENTS STARTED"
log "Target: 13 configs x 2 seeds = 26 runs"
log "Results: $RESULTS_DIR"
log "=========================================="

SEEDS="42 100"

# ═══════════════════════════════════════════════════════════════
# GROUP 1: Phase A Improvement Combinations
# ═══════════════════════════════════════════════════════════════
for S in $SEEDS; do
    run_exp "cfg01_phaseA_default" $S "ON" "ON" 0.30 0.90
done

for S in $SEEDS; do
    run_exp "cfg02_no_hard_anchors" $S "OFF" "ON" 0.30 0.90
done

for S in $SEEDS; do
    run_exp "cfg03_no_det_gate" $S "ON" "OFF" 0.30 0.90
done

for S in $SEEDS; do
    run_exp "cfg04_original_pre_phaseA" $S "OFF" "OFF" 0.30 0.90
done

# ═══════════════════════════════════════════════════════════════
# GROUP 2: Anchor Quality Threshold Variations
# ═══════════════════════════════════════════════════════════════
for S in $SEEDS; do
    run_exp "cfg05_anchor_020" $S "ON" "ON" 0.20 0.90
done

for S in $SEEDS; do
    run_exp "cfg06_anchor_025" $S "ON" "ON" 0.25 0.90
done

for S in $SEEDS; do
    run_exp "cfg07_anchor_035" $S "ON" "ON" 0.35 0.90
done

for S in $SEEDS; do
    run_exp "cfg08_anchor_040" $S "ON" "ON" 0.40 0.90
done

# ═══════════════════════════════════════════════════════════════
# GROUP 3: Similarity Threshold Variations
# ═══════════════════════════════════════════════════════════════
for S in $SEEDS; do
    run_exp "cfg09_sim_085" $S "ON" "ON" 0.30 0.85
done

for S in $SEEDS; do
    run_exp "cfg10_sim_088" $S "ON" "ON" 0.30 0.88
done

for S in $SEEDS; do
    run_exp "cfg11_sim_092" $S "ON" "ON" 0.30 0.92
done

for S in $SEEDS; do
    run_exp "cfg12_sim_095" $S "ON" "ON" 0.30 0.95
done

# ═══════════════════════════════════════════════════════════════
# GROUP 4: Combined Best Candidates
# ═══════════════════════════════════════════════════════════════
for S in $SEEDS; do
    run_exp "cfg13_anchor025_sim088" $S "ON" "ON" 0.25 0.88
done

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
log ""
log "=========================================="
log "ALL EXPERIMENTS COMPLETED: $(date)"
log "=========================================="

python3 << 'PYEOF'
import csv
import os

results_dir = os.path.dirname(os.path.abspath("$SUMMARY_FILE")) or "."
# Find the most recent overnight directory
import glob
dirs = sorted(glob.glob("results/overnight_*"))
if dirs:
    summary_file = f"{dirs[-1]}/summary.csv"
else:
    summary_file = "$SUMMARY_FILE"

try:
    with open(summary_file) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['baseline_f1'] != 'FAIL']

    if not rows:
        print("No successful experiments found.")
        exit(0)

    # Calculate mean delta per config
    configs = {}
    for r in rows:
        cfg = r['experiment']
        if cfg not in configs:
            configs[cfg] = {'deltas': [], 'aug_f1s': [], 'params': r}
        configs[cfg]['deltas'].append(float(r['delta_pct']))
        configs[cfg]['aug_f1s'].append(float(r['augmented_f1']))

    print("\n" + "="*80)
    print("  RESULTS RANKED BY MEAN DELTA (2 seeds)")
    print("="*80)
    print(f"{'Config':<30} {'Mean Delta':>12} {'Mean Aug F1':>12} {'Params':<25}")
    print("-"*80)

    ranked = sorted(configs.items(), key=lambda x: sum(x[1]['deltas'])/len(x[1]['deltas']), reverse=True)

    for cfg, data in ranked:
        mean_delta = sum(data['deltas'])/len(data['deltas'])
        mean_aug = sum(data['aug_f1s'])/len(data['aug_f1s'])
        p = data['params']
        params = f"AT={p['anchor_thresh']} ST={p['sim_thresh']}"
        print(f"{cfg:<30} {mean_delta:>+11.2f}% {mean_aug:>12.4f} {params:<25}")

    best_cfg, best_data = ranked[0]
    print("\n" + "="*80)
    print(f"  BEST CONFIG: {best_cfg}")
    print(f"  Mean Delta: {sum(best_data['deltas'])/len(best_data['deltas']):+.2f}%")
    print(f"  Parameters: HA={best_data['params']['hard_anchors']} DG={best_data['params']['det_gate']} AT={best_data['params']['anchor_thresh']} ST={best_data['params']['sim_thresh']}")
    print("="*80)
except Exception as e:
    print(f"Error generating summary: {e}")
PYEOF

log "Summary saved to: $SUMMARY_FILE"
