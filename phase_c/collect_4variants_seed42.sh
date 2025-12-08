#!/bin/bash
# Collect Results from 4 Robustness Variants (Seed 42)

set -e

ZONE="us-central1-a"
SEED=42

VM_VARIANTE_A="vm-variante-a-seed42"
VM_VARIANTE_B="vm-variante-b-seed42"
VM_VARIANTE_C1="vm-variante-c1-seed42"
VM_VARIANTE_C3="vm-variante-c3-seed42"

ALL_VMS="$VM_VARIANTE_A $VM_VARIANTE_B $VM_VARIANTE_C1 $VM_VARIANTE_C3"

RESULTS_DIR="results/4variants_seed42"

echo "═══════════════════════════════════════════════════════════"
echo "  Collecting Results from 4 Variants (Seed 42)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Downloading results from all 4 VMs..."
echo ""

# Download Variante A
echo "🅰️  Downloading Variante A (5 prompts/cluster)..."
{
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante A files missing"

# Download Variante B
echo "🅱️  Downloading Variante B (multi-temperature ensemble)..."
{
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante B files missing"

# Download Variante C1
echo "©️1 Downloading Variante C1 (GPT-5-mini terse)..."
{
    gcloud compute scp "$VM_VARIANTE_C1":~/variante_c1_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C1":~/variante_c1_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C1":~/variante_c1_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante C1 files missing"

# Download Variante C3
echo "©️3 Downloading Variante C3 (GPT-5-mini thorough)..."
{
    gcloud compute scp "$VM_VARIANTE_C3":~/variante_c3_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C3":~/variante_c3_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C3":~/variante_c3_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante C3 files missing"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results Downloaded to: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check which files exist
echo "Files downloaded:"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "  No metrics files yet"
echo ""

# Analyze results if metrics files exist
METRICS_COUNT=$(ls "$RESULTS_DIR"/*_metrics.json 2>/dev/null | wc -l)

if [ "$METRICS_COUNT" -ge 1 ]; then
    echo "📊 COMPARISON ANALYSIS (Seed 42)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python3 << 'PYTHON_EOF'
import json
import os
from pathlib import Path

results_dir = "results/4variants_seed42"

variants = {
    "A (5 prompts)": f"{results_dir}/variante_a_seed42_metrics.json",
    "B (multi-temp)": f"{results_dir}/variante_b_seed42_metrics.json",
    "C1 (terse)": f"{results_dir}/variante_c1_seed42_metrics.json",
    "C3 (thorough)": f"{results_dir}/variante_c3_seed42_metrics.json",
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
    # Determine winner
    winner_name = max(results.keys(), key=lambda k: results[k]["delta"])
    winner_delta = results[winner_name]["delta"]

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🏆 WINNER: {winner_name} ({winner_delta:+.3f}%)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

    # Ranking
    print("Ranking by F1 improvement:")
    ranked = sorted(results.items(), key=lambda x: x[1]["delta"], reverse=True)
    for i, (name, data) in enumerate(ranked, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {data['delta']:+.3f}% ({data['synth']} synthetics)")

    print("")
    print("Next step: Run winner variant with 5 seeds for statistical validation")
else:
    print("⏳ Waiting for more results...")
    print("   Re-run this script after experiments complete.")
print("")
PYTHON_EOF

else
    echo "⏳ No metrics files available yet."
    echo "   Re-run this script after experiments complete."
    echo ""
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
read -p "Delete VMs to save costs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting VMs..."
    gcloud compute instances delete $ALL_VMS --zone="$ZONE" --quiet
    echo "✅ VMs deleted!"
else
    echo ""
    echo "VMs kept. Delete manually when done:"
    echo "  gcloud compute instances delete $ALL_VMS --zone=$ZONE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
