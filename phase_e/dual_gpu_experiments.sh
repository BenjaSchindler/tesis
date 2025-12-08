#!/bin/bash
# Phase E - Dual GPU Hybrid Strategy
# Strategy: 21 experiments @ batch32 in parallel, then 5 @ batch128
# Expected: ~2.2 hours with dual GPUs (3090+3070)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

cd "$(dirname "$0")"

# Detect GPUs
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $N_GPUS GPU(s)"

if [ "$N_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs found"
    exit 1
fi

# GPU allocation
if [ "$N_GPUS" -eq 2 ]; then
    GPU0_PARALLEL=16  # RTX 3090 (24GB) @ batch32
    GPU1_PARALLEL=5   # RTX 3070 (8GB) @ batch32
    TOTAL_PARALLEL=$((GPU0_PARALLEL + GPU1_PARALLEL))
    echo "Dual GPU mode: $GPU0_PARALLEL on GPU0 + $GPU1_PARALLEL on GPU1 = $TOTAL_PARALLEL parallel"
elif [ "$N_GPUS" -eq 1 ]; then
    # Check VRAM to determine capacity
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM" -gt 20000 ]; then
        GPU0_PARALLEL=16  # RTX 3090
    else
        GPU0_PARALLEL=5   # RTX 3070
    fi
    GPU1_PARALLEL=0
    TOTAL_PARALLEL=$GPU0_PARALLEL
    echo "Single GPU mode: $GPU0_PARALLEL parallel on GPU0"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/dual_gpu_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

echo "experiment,seed,hard_anchors,det_gate,anchor_thresh,sim_thresh,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s,gpu_id" > "$SUMMARY_FILE"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Run experiment function
run_exp() {
    local NAME="$1" SEED="$2" HA="$3" DG="$4" AT="$5" ST="$6" BATCH="$7" GPU="$8"
    local OUT="${RESULTS_DIR}/${NAME}_s${SEED}"

    log "START: $NAME s$SEED (GPU=$GPU batch=$BATCH HA=$HA DG=$DG AT=$AT ST=$ST)"
    local T0=$(date +%s)

    CMD="CUDA_VISIBLE_DEVICES=$GPU python3 core/runner_phase2.py \
        --data-path ../MBTI_500.csv --test-size 0.2 --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size $BATCH \
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
            log "DONE: $NAME s$SEED GPU$GPU -> B=$BF1 A=$AF1 D=${DELTA}% S=$SYNTH (${DUR}s)"
            echo "$NAME,$SEED,$HA,$DG,$AT,$ST,$BF1,$AF1,$DELTA,$SYNTH,$DUR,$GPU" >> "$SUMMARY_FILE"
        fi
    else
        log "FAIL: $NAME s$SEED GPU$GPU"
        echo "$NAME,$SEED,$HA,$DG,$AT,$ST,FAIL,FAIL,FAIL,FAIL,0,$GPU" >> "$SUMMARY_FILE"
    fi
}

log "=========================================="
log "DUAL GPU HYBRID EXPERIMENTS"
log "Phase 1: 21 experiments @ batch32 parallel"
log "Phase 2: 5 experiments @ batch128"
log "Results: $RESULTS_DIR"
log "=========================================="

# Define all 26 experiments
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

# ═══════════════════════════════════════════════════════════════
# PHASE 1: First 21 experiments @ batch32 in parallel
# ═══════════════════════════════════════════════════════════════
log ""
log "PHASE 1: Launching 21 experiments @ batch32 in parallel"

PIDS=()
GPU_COUNTER=0

for i in {0..20}; do
    read -r NAME SEED HA DG AT ST <<< "${EXPERIMENTS[$i]}"

    # Assign to GPU (round-robin)
    if [ "$N_GPUS" -eq 2 ]; then
        if [ $GPU_COUNTER -lt $GPU0_PARALLEL ]; then
            GPU_ID=0
        else
            GPU_ID=1
        fi
    else
        GPU_ID=0
    fi

    run_exp "$NAME" "$SEED" "$HA" "$DG" "$AT" "$ST" 32 "$GPU_ID" &
    PIDS+=($!)

    GPU_COUNTER=$((GPU_COUNTER + 1))
    if [ "$GPU_COUNTER" -ge "$TOTAL_PARALLEL" ]; then
        GPU_COUNTER=0
    fi

    sleep 0.5  # Small delay to avoid startup race conditions
done

log "Waiting for Phase 1 (21 experiments) to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

log "Phase 1 complete!"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Last 5 experiments @ batch128
# ═══════════════════════════════════════════════════════════════
log ""
log "PHASE 2: Running last 5 experiments @ batch128"

PIDS=()
GPU_COUNTER=0

for i in {21..25}; do
    read -r NAME SEED HA DG AT ST <<< "${EXPERIMENTS[$i]}"

    # Distribute across available GPUs
    if [ "$N_GPUS" -eq 2 ]; then
        GPU_ID=$((GPU_COUNTER % 2))
    else
        GPU_ID=0
    fi

    run_exp "$NAME" "$SEED" "$HA" "$DG" "$AT" "$ST" 128 "$GPU_ID" &
    PIDS+=($!)
    GPU_COUNTER=$((GPU_COUNTER + 1))

    sleep 0.5
done

log "Waiting for Phase 2 (5 experiments) to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

log ""
log "=========================================="
log "ALL EXPERIMENTS COMPLETED: $(date)"
log "=========================================="

# Generate summary
python3 << 'PYEOF'
import csv
import glob

dirs = sorted(glob.glob("results/dual_gpu_*"))
if dirs:
    summary_file = f"{dirs[-1]}/summary.csv"
else:
    exit(0)

try:
    with open(summary_file) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['baseline_f1'] != 'FAIL']

    if not rows:
        print("No successful experiments found.")
        exit(0)

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
        params = f"HA={p['hard_anchors']} AT={p['anchor_thresh']} ST={p['sim_thresh']}"
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
