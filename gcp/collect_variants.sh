#!/bin/bash
# Collect Results from Robustness Variants
#
# Usage: ./collect_variants.sh <seed>

set -e

# Load toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gcp_toolkit.sh"

SEED="${1:-42}"
RESULTS_DIR="results/variants_seed${SEED}"

# VM names
VM_A="vm-variante-a-seed${SEED}"
VM_B="vm-variante-b-seed${SEED}"
VM_C1="vm-variante-c1-seed${SEED}"
VM_C3="vm-variante-c3-seed${SEED}"

ALL_VMS=("$VM_A" "$VM_B" "$VM_C1" "$VM_C3")

echo "═══════════════════════════════════════════════════════════"
echo "  Collecting Results from 4 Variants (Seed $SEED)"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p "$RESULTS_DIR"

# Download from each VM
for vm in "${ALL_VMS[@]}"; do
    variant=$(echo "$vm" | grep -oP 'variante-\K[^-]+')
    echo "Downloading from $vm ($variant)..."

    {
        gcloud compute scp "$vm":~/variante_${variant}_seed${SEED}_metrics.json "$RESULTS_DIR/" --zone="$GCP_ZONE" 2>/dev/null
        gcloud compute scp "$vm":~/variante_${variant}_seed${SEED}_synthetic.csv "$RESULTS_DIR/" --zone="$GCP_ZONE" 2>/dev/null
        gcloud compute scp "$vm":~/variante_${variant}_seed${SEED}.log "$RESULTS_DIR/" --zone="$GCP_ZONE" 2>/dev/null
    } || echo "  Warning: Some files missing from $vm"
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

results_dir = "$RESULTS_DIR"

variants = {
    "A (5 prompts)": f"{results_dir}/variante_a_seed${SEED}_metrics.json",
    "B (multi-temp)": f"{results_dir}/variante_b_seed${SEED}_metrics.json",
    "C1 (terse)": f"{results_dir}/variante_c1_seed${SEED}_metrics.json",
    "C3 (thorough)": f"{results_dir}/variante_c3_seed${SEED}_metrics.json",
}

print("┌─────────────────────┬───────────┬────────────┬───────────┬────────────┐")
print("│ Variant             │ Baseline  │ Augmented  │ F1 Delta  │ Synthetics │")
print("├─────────────────────┼───────────┼────────────┼───────────┼────────────┤")

results = {}
for variant_name, metrics_file in variants.items():
    if not Path(metrics_file).exists():
        print(f"│ {variant_name:19} │ {'(pending)':^9} │ {'(pending)':^10} │ {'(pending)':^9} │ {'(pending)':^10} │")
        continue

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        baseline_f1 = data["baseline"]["macro_f1"]
        augmented_f1 = data["augmented"]["macro_f1"]
        f1_delta_pct = data["improvement"]["f1_delta_pct"]
        synth_count = data["synthetic_data"]["accepted_count"]

        results[variant_name] = {
            "baseline": baseline_f1,
            "augmented": augmented_f1,
            "delta": f1_delta_pct,
            "synth": synth_count
        }

        print(f"│ {variant_name:19} │ {baseline_f1:7.4f}   │ {augmented_f1:8.4f}   │ {f1_delta_pct:+7.3f}%  │ {synth_count:10d} │")
    except Exception as e:
        print(f"│ {variant_name:19} │ ERROR: {str(e)[:40]:40} │")

print("└─────────────────────┴───────────┴────────────┴───────────┴────────────┘")
print("")

if len(results) >= 2:
    winner_name = max(results.keys(), key=lambda k: results[k]["delta"])
    winner_delta = results[winner_name]["delta"]

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"WINNER: {winner_name} ({winner_delta:+.3f}%)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

    print("Ranking by F1 improvement:")
    ranked = sorted(results.items(), key=lambda x: x[1]["delta"], reverse=True)
    for i, (name, data) in enumerate(ranked, 1):
        medal = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i}th"
        print(f"  {medal}. {name}: {data['delta']:+.3f}% ({data['synth']} synthetics)")

    print("")
    print("Next step: Run winner variant with 5 seeds for statistical validation")
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
