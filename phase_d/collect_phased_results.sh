#!/bin/bash
# Collect Results from Phase D Experiment
#
# Downloads metrics, logs, and synthetic data from all 4 variants
# Analyzes results and ranks variants by F1 improvement
#
# Usage: ./collect_phased_results.sh

set -e

# Load GCP toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/gcp/gcp_toolkit.sh"

RESULTS_DIR="$SCRIPT_DIR/results/phased_variants"
SEEDS=(42 100 123)

# VM names
VM_A="vm-phased-a"
VM_B="vm-phased-b"
VM_C="vm-phased-c"
VM_D="vm-phased-d"

ALL_VMS=("$VM_A" "$VM_B" "$VM_C" "$VM_D")
VARIANTS=("a" "b" "c" "d")

echo "═══════════════════════════════════════════════════════════"
echo "  Collecting Results from Phase D Experiment"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p "$RESULTS_DIR"

# Download from each VM
for i in "${!ALL_VMS[@]}"; do
    vm="${ALL_VMS[$i]}"
    variant="${VARIANTS[$i]}"

    echo "Downloading from $vm (Variant ${variant^^})..."

    for seed in "${SEEDS[@]}"; do
        {
            gcloud compute scp "$vm":~/phased_${variant}_seed${seed}_metrics.json "$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
            gcloud compute scp "$vm":~/phased_${variant}_seed${seed}_synthetic.csv "$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
            gcloud compute scp "$vm":~/phased_${variant}_seed${seed}.log "$RESULTS_DIR/" --zone="$GCP_ZONE" --quiet 2>/dev/null
        } || echo "  Warning: Some files missing from $vm seed $seed"
    done

    echo "  ✓ Variant ${variant^^} downloaded"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results Downloaded to: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check available metrics
METRICS_COUNT=$(ls "$RESULTS_DIR"/*_metrics.json 2>/dev/null | wc -l)

if [ "$METRICS_COUNT" -ge 1 ]; then
    echo "Analyzing results..."
    echo ""

    python3 << PYTHON_EOF
import json
import os
from pathlib import Path
import statistics

results_dir = "$RESULTS_DIR"
seeds = [42, 100, 123]

variants = {
    "A (Contrastive)": {
        "prefix": "phased_a",
        "description": "Contrastive Prompting only"
    },
    "B (Focal Loss)": {
        "prefix": "phased_b",
        "description": "Focal Loss only"
    },
    "C (Two-Stage)": {
        "prefix": "phased_c",
        "description": "Two-Stage Training only"
    },
    "D (Full Stack)": {
        "prefix": "phased_d",
        "description": "ALL Phase D + GPT-5-mini"
    }
}

# Collect data for each variant across all seeds
variant_data = {}

for variant_name, variant_info in variants.items():
    prefix = variant_info["prefix"]
    seed_results = []

    for seed in seeds:
        metrics_file = f"{results_dir}/{prefix}_seed{seed}_metrics.json"

        if Path(metrics_file).exists():
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
                print(f"  Warning: Error reading {metrics_file}: {e}")

    variant_data[variant_name] = seed_results

# Print per-seed results
print("┌" + "─" * 93 + "┐")
print("│" + " " * 33 + "PER-SEED RESULTS" + " " * 44 + "│")
print("├" + "─" * 93 + "┤")
print("│ Variant           │ Seed │ Baseline  │ Augmented │ F1 Delta  │ Synthetics │")
print("├" + "─" * 93 + "┤")

for variant_name, seed_results in variant_data.items():
    if not seed_results:
        print(f"│ {variant_name:17} │ {'(no data)':^4} │ {'':^9} │ {'':^9} │ {'':^9} │ {'':^10} │")
        continue

    for i, result in enumerate(seed_results):
        variant_display = variant_name if i == 0 else ""
        print(f"│ {variant_display:17} │ {result['seed']:4d} │ {result['baseline']:7.4f}   │ {result['augmented']:7.4f}   │ {result['delta']:+7.3f}%  │ {result['synth']:10d} │")

    if len(seed_results) < len(seeds):
        for _ in range(len(seeds) - len(seed_results)):
            print(f"│ {'':17} │ {'(missing)':^4} │ {'':^9} │ {'':^9} │ {'':^9} │ {'':^10} │")

print("└" + "─" * 93 + "┘")
print("")

# Compute aggregate statistics
print("┌" + "─" * 79 + "┐")
print("│" + " " * 28 + "AGGREGATE STATISTICS" + " " * 31 + "│")
print("├" + "─" * 79 + "┤")
print("│ Variant           │ Mean Δ F1  │ Std Dev   │ Mean Synth │ N Seeds │")
print("├" + "─" * 79 + "┤")

aggregate_stats = {}

for variant_name, seed_results in variant_data.items():
    if not seed_results or len(seed_results) == 0:
        print(f"│ {variant_name:17} │ {'(no data)':^10} │ {'':^9} │ {'':^10} │ {'0/3':^7} │")
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

    print(f"│ {variant_name:17} │ {mean_delta:+8.3f}%  │ ±{std_delta:7.3f}% │ {mean_synth:8.1f}   │ {n_seeds}/3     │")

print("└" + "─" * 79 + "┘")
print("")

# Rank variants
if len(aggregate_stats) >= 2:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    ranked = sorted(aggregate_stats.items(), key=lambda x: x[1]["mean_delta"], reverse=True)
    winner_name = ranked[0][0]
    winner_delta = ranked[0][1]["mean_delta"]

    print(f"WINNER: {winner_name} ({winner_delta:+.3f}%)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

    print("Ranking by Mean F1 Improvement:")
    for i, (name, stats) in enumerate(ranked, 1):
        position = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i}th"
        print(f"  {position}. {name}: {stats['mean_delta']:+.3f}% (±{stats['std_delta']:.3f}%, {stats['mean_synth']:.0f} synthetics)")

    print("")

    # Recommendations
    print("Next Steps:")
    if winner_name == "D (Full Stack)":
        print("  ✓ Full Stack variant won! This confirms additive benefits of Phase D features.")
        print("  → Run with 5 seeds for statistical validation")
        print("  → Analyze per-class F1 to confirm LOW/MID tier improvements")
    elif winner_name == "A (Contrastive)":
        print("  ✓ Contrastive Prompting shows strongest individual effect")
        print("  → Consider combining with other features")
        print("  → Analyze confusion matrix for MBTI confuser pairs (ENFJ/INFJ, etc.)")
    elif winner_name == "B (Focal Loss)":
        print("  ✓ Focal Loss shows strongest individual effect")
        print("  → Verify LOW/MID tier F1 improvements")
        print("  → Consider tuning gamma and tier boost parameters")
    elif winner_name == "C (Two-Stage)":
        print("  ✓ Two-Stage Training shows strongest individual effect")
        print("  → Analyze Stage 1 vs Stage 2 F1 progression")
        print("  → Consider adjusting confidence threshold")

    print("")
    print("To analyze per-class F1 and LOW/MID tier performance:")
    print(f"  python3 -c 'import json; print(json.load(open(\"{results_dir}/{ranked[0][0].split()[0].lower()}_seed42_metrics.json\"))[\"baseline\"][\"per_class_f1\"])'")

else:
    print("Waiting for more results...")
    print("Re-run this script after experiments complete.")

print("")
PYTHON_EOF

else
    echo "No metrics files available yet."
    echo "Re-run this script after experiments complete."
    echo ""
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
read -p "Delete VMs to save costs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting VMs..."
    gcp_delete_vms "${ALL_VMS[@]}"
else
    echo ""
    echo "VMs kept. Delete manually when done:"
    echo "  gcloud compute instances delete ${ALL_VMS[*]} --zone=$GCP_ZONE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
