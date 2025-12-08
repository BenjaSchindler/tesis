#!/bin/bash
# Phase E - Overnight Experiments
# Runs multiple configurations to find best macro F1
# Expected duration: ~12-15 hours
# Started: $(date)

set -e

# Verify API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

cd "$(dirname "$0")"

# Create results directory
mkdir -p results/overnight_$(date +%Y%m%d)
RESULTS_DIR="results/overnight_$(date +%Y%m%d)"
LOG_FILE="$RESULTS_DIR/overnight_master.log"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# Initialize summary
echo "═══════════════════════════════════════════════════════════════" > "$SUMMARY_FILE"
echo "  OVERNIGHT EXPERIMENTS - Phase E" >> "$SUMMARY_FILE"
echo "  Started: $(date)" >> "$SUMMARY_FILE"
echo "═══════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_experiment() {
    local NAME="$1"
    local SEED="$2"
    local HARD_ANCHORS="$3"
    local DET_GATE="$4"
    local ANCHOR_THRESH="$5"
    local SIM_THRESH="$6"

    local OUTPUT_PREFIX="${RESULTS_DIR}/exp_${NAME}_seed${SEED}"

    log "════════════════════════════════════════════════════════════"
    log "Starting: $NAME (seed=$SEED)"
    log "  Hard anchors: $HARD_ANCHORS"
    log "  Deterministic gate: $DET_GATE"
    log "  Anchor threshold: $ANCHOR_THRESH"
    log "  Similarity threshold: $SIM_THRESH"
    log "════════════════════════════════════════════════════════════"

    START_TIME=$(date +%s)

    # Build command
    CMD="python3 core/runner_phase2.py \
        --data-path ../MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cpu \
        --embedding-batch-size 64 \
        --llm-model gpt-4o-mini \
        --max-clusters 3 \
        --prompts-per-cluster 3 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --anchor-quality-threshold $ANCHOR_THRESH \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --enable-adaptive-filters \
        --use-class-description \
        --use-f1-budget-scaling \
        --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --similarity-threshold $SIM_THRESH \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --synthetic-output ${OUTPUT_PREFIX}_synthetic.csv \
        --augmented-train-output ${OUTPUT_PREFIX}_augmented.csv \
        --metrics-output ${OUTPUT_PREFIX}_metrics.json"

    # Add Phase A improvement flags
    if [ "$HARD_ANCHORS" = "ON" ]; then
        CMD="$CMD --use-hard-anchors"
    else
        CMD="$CMD --no-hard-anchors"
    fi

    if [ "$DET_GATE" = "ON" ]; then
        CMD="$CMD --deterministic-quality-gate"
    else
        CMD="$CMD --no-deterministic-quality-gate"
    fi

    # Run experiment
    if eval $CMD > "${OUTPUT_PREFIX}.log" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Extract results from metrics file
        if [ -f "${OUTPUT_PREFIX}_metrics.json" ]; then
            BASELINE_F1=$(python3 -c "import json; d=json.load(open('${OUTPUT_PREFIX}_metrics.json')); print(f\"{d.get('baseline_macro_f1', 0):.4f}\")")
            AUG_F1=$(python3 -c "import json; d=json.load(open('${OUTPUT_PREFIX}_metrics.json')); print(f\"{d.get('augmented_macro_f1', 0):.4f}\")")
            DELTA=$(python3 -c "import json; d=json.load(open('${OUTPUT_PREFIX}_metrics.json')); b=d.get('baseline_macro_f1',0); a=d.get('augmented_macro_f1',0); print(f\"{((a-b)/b*100) if b>0 else 0:.2f}\")")
            SYNTH=$(python3 -c "import json; d=json.load(open('${OUTPUT_PREFIX}_metrics.json')); print(d.get('total_synthetics_added', 0))")

            log "COMPLETED: $NAME"
            log "  Baseline F1: $BASELINE_F1"
            log "  Augmented F1: $AUG_F1"
            log "  Delta: ${DELTA}%"
            log "  Synthetics: $SYNTH"
            log "  Duration: ${DURATION}s"

            # Append to summary
            echo "$NAME | seed=$SEED | HA=$HARD_ANCHORS DG=$DET_GATE AT=$ANCHOR_THRESH ST=$SIM_THRESH | B=$BASELINE_F1 A=$AUG_F1 D=${DELTA}% S=$SYNTH | ${DURATION}s" >> "$SUMMARY_FILE"
        else
            log "WARNING: No metrics file found for $NAME"
            echo "$NAME | seed=$SEED | FAILED - no metrics" >> "$SUMMARY_FILE"
        fi
    else
        log "FAILED: $NAME - check ${OUTPUT_PREFIX}.log"
        echo "$NAME | seed=$SEED | FAILED - see log" >> "$SUMMARY_FILE"
    fi

    echo "" >> "$SUMMARY_FILE"
}

log "Starting overnight experiments..."
log "Results will be saved to: $RESULTS_DIR"

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT MATRIX
# ═══════════════════════════════════════════════════════════════
# Each experiment ~60-90 minutes
# Target: ~12 experiments in 15 hours

# Group 1: Phase A Improvement Combinations (seed 42)
# Test all 4 combinations of hard anchors and deterministic gate
run_experiment "phaseA_default" 42 "ON" "ON" 0.30 0.90
run_experiment "no_hard_anchors" 42 "OFF" "ON" 0.30 0.90
run_experiment "no_det_gate" 42 "ON" "OFF" 0.30 0.90
run_experiment "original_baseline" 42 "OFF" "OFF" 0.30 0.90

# Group 2: Anchor Quality Threshold Variations (with best Phase A config)
run_experiment "anchor_thresh_025" 42 "ON" "ON" 0.25 0.90
run_experiment "anchor_thresh_035" 42 "ON" "ON" 0.35 0.90
run_experiment "anchor_thresh_020" 42 "ON" "ON" 0.20 0.90

# Group 3: Similarity Threshold Variations
run_experiment "sim_thresh_085" 42 "ON" "ON" 0.30 0.85
run_experiment "sim_thresh_095" 42 "ON" "ON" 0.30 0.95

# Group 4: Multi-seed validation of best config (seeds 100, 123)
run_experiment "phaseA_default" 100 "ON" "ON" 0.30 0.90
run_experiment "phaseA_default" 123 "ON" "ON" 0.30 0.90

# Group 5: Best parameter combos across different seeds
run_experiment "anchor_thresh_025" 100 "ON" "ON" 0.25 0.90
run_experiment "anchor_thresh_025" 123 "ON" "ON" 0.25 0.90

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════"
log "  ALL EXPERIMENTS COMPLETED"
log "  Finished: $(date)"
log "═══════════════════════════════════════════════════════════════"

echo "" >> "$SUMMARY_FILE"
echo "═══════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "  Completed: $(date)" >> "$SUMMARY_FILE"
echo "═══════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"

# Generate final analysis
python3 << 'EOF'
import json
import glob
import os

results_dir = os.environ.get('RESULTS_DIR', 'results/overnight_*')
pattern = f"{results_dir}/*_metrics.json" if '*' not in results_dir else "results/overnight_*/*_metrics.json"
files = sorted(glob.glob(pattern))

if not files:
    print("No results found yet.")
    exit(0)

print("\n" + "="*70)
print("  FINAL ANALYSIS - SORTED BY DELTA")
print("="*70)

results = []
for f in files:
    try:
        with open(f) as fp:
            d = json.load(fp)
        baseline = d.get('baseline_macro_f1', 0)
        augmented = d.get('augmented_macro_f1', 0)
        delta = ((augmented - baseline) / baseline * 100) if baseline > 0 else 0
        results.append({
            'file': os.path.basename(f),
            'baseline': baseline,
            'augmented': augmented,
            'delta': delta,
            'synthetics': d.get('total_synthetics_added', 0)
        })
    except:
        pass

# Sort by delta descending
results.sort(key=lambda x: x['delta'], reverse=True)

print(f"\n{'Experiment':<50} {'Baseline':>10} {'Augmented':>10} {'Delta':>10} {'Synth':>8}")
print("-"*90)
for r in results:
    name = r['file'].replace('_metrics.json', '')
    print(f"{name:<50} {r['baseline']:>10.4f} {r['augmented']:>10.4f} {r['delta']:>+9.2f}% {r['synthetics']:>8}")

if results:
    best = results[0]
    print("\n" + "="*70)
    print(f"  BEST RESULT: {best['file']}")
    print(f"  Delta: {best['delta']:+.2f}%")
    print(f"  Augmented F1: {best['augmented']:.4f}")
    print("="*70)
EOF

log "Check $SUMMARY_FILE for full results"
