#!/bin/bash
# Generic Results Collector for GCP Experiments
# ==============================================
#
# This template downloads results from VMs and analyzes them.
# Adapt the configuration section for your experiment.
#
# Usage: ./collect_results_template.sh

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFY FOR YOUR EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_NAME="my-experiment"
SEEDS=(42 100 123)

# VM configuration (must match launch script)
NUM_VMS=4
VM_PREFIX="vm-${EXPERIMENT_NAME}"
VMS=()
for i in $(seq 1 $NUM_VMS); do
    VMS+=("${VM_PREFIX}-${i}")
done

# VM prefixes and labels (must match launch script)
declare -A VM_PREFIXES
VM_PREFIXES[1]="exp_a"
VM_PREFIXES[2]="exp_b"
VM_PREFIXES[3]="exp_c"
VM_PREFIXES[4]="exp_d"

declare -A VM_LABELS
VM_LABELS[1]="Feature A Only"
VM_LABELS[2]="Feature B Only"
VM_LABELS[3]="Feature C Only"
VM_LABELS[4]="All Features"

# Results directory
RESULTS_DIR="results/${EXPERIMENT_NAME}"

# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

# Load toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/gcp_toolkit.sh"

echo "═══════════════════════════════════════════════════════════"
echo "  Collecting Results: $EXPERIMENT_NAME"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p "$PROJECT_ROOT/$RESULTS_DIR"

# Download results from each VM
for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    vm="${VMS[$i]}"
    prefix="${VM_PREFIXES[$vm_num]}"
    label="${VM_LABELS[$vm_num]}"

    echo "Downloading from $vm ($label)..."

    # Check if VM is running or stopped
    vm_status=$(gcloud compute instances describe "$vm" --zone="$GCP_ZONE" \
      --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "$vm_status" = "NOT_FOUND" ]; then
        echo "  Warning: VM $vm not found (may have been deleted)"
        continue
    fi

    # Download metrics, CSVs, and logs for each seed
    for seed in "${SEEDS[@]}"; do
        {
            gcloud compute scp "$vm":~/${prefix}_seed${seed}_metrics.json \
              "$PROJECT_ROOT/$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
            gcloud compute scp "$vm":~/${prefix}_seed${seed}_synthetic.csv \
              "$PROJECT_ROOT/$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
            gcloud compute scp "$vm":~/${prefix}_seed${seed}.log \
              "$PROJECT_ROOT/$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
        } || echo "  Warning: Some files missing for seed $seed"
    done

    # Download run.log (general log)
    gcloud compute scp "$vm":~/run.log \
      "$PROJECT_ROOT/$RESULTS_DIR/${prefix}_run.log" \
      --zone="$GCP_ZONE" --quiet 2>/dev/null || true

    echo "  ✓ Downloaded"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results Downloaded to: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Count downloaded metrics
METRICS_COUNT=$(ls "$PROJECT_ROOT/$RESULTS_DIR"/*_metrics.json 2>/dev/null | wc -l)
EXPECTED_COUNT=$((${#VMS[@]} * ${#SEEDS[@]}))

echo "Metrics files found: $METRICS_COUNT / $EXPECTED_COUNT expected"
echo ""

if [ "$METRICS_COUNT" -eq 0 ]; then
    echo "No metrics files found. Experiments may still be running."
    echo "Re-run this script later."
    exit 0
fi

# Analyze results with Python
echo "Analyzing results..."
echo ""

python3 << 'PYTHON_EOF'
import json
import os
import sys
from pathlib import Path
import statistics

# Configuration (passed via environment)
results_dir = os.environ.get('RESULTS_DIR', 'results')
project_root = os.environ.get('PROJECT_ROOT', '.')
seeds = [int(s) for s in os.environ.get('SEEDS', '42 100 123').split()]
num_vms = int(os.environ.get('NUM_VMS', '4'))

# Build variant info from environment
variants = {}
for i in range(1, num_vms + 1):
    prefix = os.environ.get(f'VM_PREFIX_{i}', f'exp_{i}')
    label = os.environ.get(f'VM_LABEL_{i}', f'Variant {i}')
    variants[label] = prefix

full_results_dir = os.path.join(project_root, results_dir)

# Collect data
variant_data = {}

for variant_name, prefix in variants.items():
    seed_results = []

    for seed in seeds:
        metrics_file = os.path.join(full_results_dir, f"{prefix}_seed{seed}_metrics.json")

        if not Path(metrics_file).exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)

            baseline_f1 = data["baseline"]["macro_f1"]
            augmented_f1 = data["augmented"]["macro_f1"]
            f1_delta_pct = data["improvement"]["f1_delta_pct"]
            synth_count = data["synthetic_data"]["accepted_count"]

            seed_results.append({
                "seed": seed,
                "baseline": baseline_f1,
                "augmented": augmented_f1,
                "delta": f1_delta_pct,
                "synth": synth_count
            })
        except Exception as e:
            print(f"Warning: Error reading {metrics_file}: {e}", file=sys.stderr)

    if seed_results:
        variant_data[variant_name] = seed_results

# Print per-seed results
print("┌" + "─" * 93 + "┐")
print("│" + " " * 33 + "PER-SEED RESULTS" + " " * 44 + "│")
print("├" + "─" * 93 + "┤")
print("│ Variant           │ Seed │ Baseline  │ Augmented │ F1 Delta  │ Synthetics │")
print("├" + "─" * 93 + "┤")

for variant_name, seed_results in variant_data.items():
    if not seed_results:
        continue

    for i, result in enumerate(seed_results):
        variant_display = variant_name if i == 0 else ""
        print(f"│ {variant_display:17} │ {result['seed']:4d} │ {result['baseline']:7.4f}   │ "
              f"{result['augmented']:7.4f}   │ {result['delta']:+7.3f}%  │ {result['synth']:10d} │")

print("└" + "─" * 93 + "┘")
print("")

# Compute aggregate statistics
if len(variant_data) > 0:
    print("┌" + "─" * 79 + "┐")
    print("│" + " " * 28 + "AGGREGATE STATISTICS" + " " * 31 + "│")
    print("├" + "─" * 79 + "┤")
    print("│ Variant           │ Mean Δ F1  │ Std Dev   │ Mean Synth │ N Seeds │")
    print("├" + "─" * 79 + "┤")

    aggregate_stats = {}

    for variant_name, seed_results in variant_data.items():
        if not seed_results:
            continue

        deltas = [r['delta'] for r in seed_results]
        synths = [r['synth'] for r in seed_results]

        mean_delta = statistics.mean(deltas)
        std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        mean_synth = statistics.mean(synths)
        n_seeds = len(seed_results)

        aggregate_stats[variant_name] = {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "mean_synth": mean_synth,
            "n_seeds": n_seeds
        }

        print(f"│ {variant_name:17} │ {mean_delta:+8.3f}%  │ ±{std_delta:7.3f}% │ "
              f"{mean_synth:8.1f}   │ {n_seeds}/{len(seeds):1d}     │")

    print("└" + "─" * 79 + "┘")
    print("")

    # Rank variants
    if len(aggregate_stats) >= 2:
        ranked = sorted(aggregate_stats.items(), key=lambda x: x[1]["mean_delta"], reverse=True)
        winner_name = ranked[0][0]
        winner_delta = ranked[0][1]["mean_delta"]

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"WINNER: {winner_name} ({winner_delta:+.3f}%)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("")

        print("Ranking by Mean F1 Improvement:")
        for i, (name, stats) in enumerate(ranked, 1):
            position = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i}th"
            print(f"  {position}. {name}: {stats['mean_delta']:+.3f}% "
                  f"(±{stats['std_delta']:.3f}%, {stats['mean_synth']:.0f} synthetics)")

PYTHON_EOF

# Pass environment variables to Python
export RESULTS_DIR="$RESULTS_DIR"
export PROJECT_ROOT="$PROJECT_ROOT"
export SEEDS="${SEEDS[*]}"
export NUM_VMS="$NUM_VMS"

for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    export "VM_PREFIX_${vm_num}"="${VM_PREFIXES[$vm_num]}"
    export "VM_LABEL_${vm_num}"="${VM_LABELS[$vm_num]}"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Delete VMs to save costs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting VMs..."
    gcp_delete_vms "${VMS[@]}"
else
    echo ""
    echo "VMs kept. Delete manually when done:"
    echo "  gcloud compute instances delete ${VMS[*]} --zone=$GCP_ZONE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
