#!/bin/bash
# Phase E - Safe Experiments (8 parallel max)
# Conservative approach to avoid VRAM crashes
# RTX 3090 (24GB): 8 experiments @ batch64 = ~16GB used (safe margin)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")"

MAX_PARALLEL=4
BATCH_SIZE=128

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/safe_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "experiment,seed,hard_anchors,det_gate,anchor_thresh,sim_thresh,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s" > "$SUMMARY_FILE"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_exp() {
    local NAME="$1" SEED="$2" HA="$3" DG="$4" AT="$5" ST="$6"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME s$SEED"
    local T0=$(date +%s)

    CMD="python3 -u core/runner_phase2.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size $BATCH_SIZE \
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
            local BF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('baseline',{}).get('macro_f1',0):.4f}\")" 2>/dev/null || echo "0.0000")
            local AF1=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(f\"{d.get('augmented',{}).get('macro_f1',0):.4f}\")" 2>/dev/null || echo "0.0000")
            local DELTA=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));b=d.get('baseline',{}).get('macro_f1',0);a=d.get('augmented',{}).get('macro_f1',0);print(f\"{((a-b)/b*100) if b>0 else 0:.2f}\")" 2>/dev/null || echo "0.00")
            local SYNTH=$(python3 -c "import json;d=json.load(open('${OUT}_metrics.json'));print(d.get('synthetic_data',{}).get('total_accepted',0))" 2>/dev/null || echo "0")
            log "DONE: $NAME s$SEED -> B=$BF1 A=$AF1 D=${DELTA}% (${DUR}s)"
            echo "$NAME,$SEED,$HA,$DG,$AT,$ST,$BF1,$AF1,$DELTA,$SYNTH,$DUR" >> "$SUMMARY_FILE"
        fi
    else
        log "FAIL: $NAME s$SEED"
        echo "$NAME,$SEED,$HA,$DG,$AT,$ST,FAIL,FAIL,FAIL,FAIL,0" >> "$SUMMARY_FILE"
    fi
}

log "=========================================="
log "SAFE EXPERIMENTS - $MAX_PARALLEL parallel @ batch$BATCH_SIZE"
log "Results: $RESULTS_DIR"
log "=========================================="

# All 26 experiments
EXPERIMENTS=(
    "cfg01_phaseA_default 42 ON ON 0.30 0.90"
    "cfg01_phaseA_default 100 ON ON 0.30 0.90"
    "cfg02_no_hard_anchors 42 OFF ON 0.30 0.90"
    "cfg02_no_hard_anchors 100 OFF ON 0.30 0.90"
    "cfg03_no_det_gate 42 ON OFF 0.30 0.90"
    "cfg03_no_det_gate 100 ON OFF 0.30 0.90"
    "cfg04_original_pre_phaseA 42 OFF OFF 0.30 0.90"
    "cfg04_original_pre_phaseA 100 OFF OFF 0.30 0.90"
    "cfg05_anchor_020 42 ON ON 0.20 0.90"
    "cfg05_anchor_020 100 ON ON 0.20 0.90"
    "cfg06_anchor_025 42 ON ON 0.25 0.90"
    "cfg06_anchor_025 100 ON ON 0.25 0.90"
    "cfg07_anchor_035 42 ON ON 0.35 0.90"
    "cfg07_anchor_035 100 ON ON 0.35 0.90"
    "cfg08_anchor_040 42 ON ON 0.40 0.90"
    "cfg08_anchor_040 100 ON ON 0.40 0.90"
    "cfg09_sim_085 42 ON ON 0.30 0.85"
    "cfg09_sim_085 100 ON ON 0.30 0.85"
    "cfg10_sim_088 42 ON ON 0.30 0.88"
    "cfg10_sim_088 100 ON ON 0.30 0.88"
    "cfg11_sim_092 42 ON ON 0.30 0.92"
    "cfg11_sim_092 100 ON ON 0.30 0.92"
    "cfg12_sim_095 42 ON ON 0.30 0.95"
    "cfg12_sim_095 100 ON ON 0.30 0.95"
    "cfg13_anchor025_sim088 42 ON ON 0.25 0.88"
    "cfg13_anchor025_sim088 100 ON ON 0.25 0.88"
)

TOTAL=${#EXPERIMENTS[@]}
log "Total experiments: $TOTAL"

# Run in batches
for ((i=0; i<TOTAL; i+=MAX_PARALLEL)); do
    BATCH_END=$((i + MAX_PARALLEL))
    [ $BATCH_END -gt $TOTAL ] && BATCH_END=$TOTAL

    log ""
    log "=== Batch $((i/MAX_PARALLEL + 1)): experiments $((i+1))-$BATCH_END ==="

    PIDS=()
    for ((j=i; j<BATCH_END; j++)); do
        read -r NAME SEED HA DG AT ST <<< "${EXPERIMENTS[$j]}"
        run_exp "$NAME" "$SEED" "$HA" "$DG" "$AT" "$ST" &
        PIDS+=($!)
        sleep 1
    done

    # Wait for batch to complete
    for pid in "${PIDS[@]}"; do
        wait $pid
    done

    log "Batch complete. Checking GPU..."
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG_FILE"
done

log ""
log "=========================================="
log "ALL EXPERIMENTS COMPLETED: $(date)"
log "=========================================="

# Summary
python3 << 'PYEOF'
import csv, glob

dirs = sorted(glob.glob("results/safe_*"))
if not dirs: exit(0)
summary_file = f"{dirs[-1]}/summary.csv"

try:
    with open(summary_file) as f:
        rows = [r for r in csv.DictReader(f) if r['baseline_f1'] != 'FAIL']

    if not rows: print("No results"); exit(0)

    configs = {}
    for r in rows:
        cfg = r['experiment']
        if cfg not in configs:
            configs[cfg] = {'deltas': [], 'aug_f1s': [], 'params': r}
        configs[cfg]['deltas'].append(float(r['delta_pct']))
        configs[cfg]['aug_f1s'].append(float(r['augmented_f1']))

    print("\n" + "="*70)
    print("  RESULTS RANKED BY MEAN DELTA")
    print("="*70)

    ranked = sorted(configs.items(), key=lambda x: sum(x[1]['deltas'])/len(x[1]['deltas']), reverse=True)

    for cfg, data in ranked:
        mean_delta = sum(data['deltas'])/len(data['deltas'])
        mean_aug = sum(data['aug_f1s'])/len(data['aug_f1s'])
        print(f"{cfg:<30} {mean_delta:>+8.2f}% {mean_aug:>10.4f}")

    best_cfg, best_data = ranked[0]
    print(f"\nBEST: {best_cfg} ({sum(best_data['deltas'])/len(best_data['deltas']):+.2f}%)")
except Exception as e:
    print(f"Error: {e}")
PYEOF
