#!/bin/bash
# Collect Results from 3 Robustness Variants (Seed 42)

set -e

ZONE="us-central1-a"
SEED=42

VM_VARIANTE_A="vm-variante-a-seed42"
VM_VARIANTE_B="vm-variante-b-seed42"
VM_VARIANTE_C="vm-variante-c-seed42"

RESULTS_DIR="results/3variants_seed42"

echo "═══════════════════════════════════════════════════════════"
echo "  Collecting Results from 3 Variants (Seed 42)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Downloading results from all 3 VMs..."
echo ""

# Download Variante A
echo "🅰️  Downloading Variante A (5 prompts/cluster)..."
{
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42_augmented.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_A":~/variante_a_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante A files missing (might still be running)"

# Download Variante B
echo "🅱️  Downloading Variante B (multi-temperature ensemble)..."
{
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42_augmented.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_B":~/variante_b_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante B files missing (might still be running)"

# Download Variante C
echo "©️  Downloading Variante C (GPT-5-mini reasoning)..."
{
    gcloud compute scp "$VM_VARIANTE_C":~/variante_c_seed42_metrics.json "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C":~/variante_c_seed42_synthetic.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C":~/variante_c_seed42_augmented.csv "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
    gcloud compute scp "$VM_VARIANTE_C":~/variante_c_seed42.log "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null
} || echo "  ⚠️  Some Variante C files missing (might still be running)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Results Downloaded to: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Analyze results if all metrics files exist
if [ -f "$RESULTS_DIR/variante_a_seed42_metrics.json" ] && \
   [ -f "$RESULTS_DIR/variante_b_seed42_metrics.json" ] && \
   [ -f "$RESULTS_DIR/variante_c_seed42_metrics.json" ]; then

    echo "📊 COMPARISON ANALYSIS (Seed 42)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Extract metrics using Python
    python3 << 'PYTHON_EOF'
import json
import sys

results_dir = "results/3variants_seed42"

variants = {
    "A (5 prompts/cluster)": f"{results_dir}/variante_a_seed42_metrics.json",
    "B (multi-temp ensemble)": f"{results_dir}/variante_b_seed42_metrics.json",
    "C (GPT-5-mini reasoning)": f"{results_dir}/variante_c_seed42_metrics.json",
}

print("┌─────────────────────────────┬───────────┬────────────┬───────────┬────────────┐")
print("│ Variant                     │ Baseline  │ Augmented  │ F1 Delta  │ Synthetics │")
print("├─────────────────────────────┼───────────┼────────────┼───────────┼────────────┤")

for variant_name, metrics_file in variants.items():
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        baseline_f1 = data["baseline"]["macro_f1"]
        augmented_f1 = data["augmented"]["macro_f1"]
        f1_delta_pct = data["improvement"]["f1_delta_pct"]
        synth_count = data["synthetic_data"]["accepted_count"]

        print(f"│ {variant_name:27} │ {baseline_f1:7.4f}   │ {augmented_f1:8.4f}   │ {f1_delta_pct:+7.3f}%  │ {synth_count:10d} │")
    except Exception as e:
        print(f"│ {variant_name:27} │ ERROR: {str(e):48} │")

print("└─────────────────────────────┴───────────┴────────────┴───────────┴────────────┘")
print("")

# Per-class breakdown
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("PER-CLASS F1 IMPROVEMENTS (Target Classes Only)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("")

all_classes = set()
variant_data = {}

for variant_name, metrics_file in variants.items():
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    variant_data[variant_name] = data
    all_classes.update(data["improvement"]["per_class"].keys())

all_classes = sorted(all_classes)

print(f"{'Class':<8} │ {'A (5 prompts)':<15} │ {'B (multi-temp)':<15} │ {'C (GPT-5-mini)':<15}")
print("─────────┼─────────────────┼─────────────────┼─────────────────")

for cls in all_classes:
    row = f"{cls:<8} │"
    for variant_name in ["A (5 prompts/cluster)", "B (multi-temp ensemble)", "C (GPT-5-mini reasoning)"]:
        per_class = variant_data[variant_name]["improvement"]["per_class"]
        if cls in per_class:
            delta_pct = per_class[cls]["delta_pct"]
            row += f" {delta_pct:+6.2f}%          │"
        else:
            row += f" {'N/A':<15} │"
    print(row)

print("")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("")

# Determine winner
winner_name = None
winner_f1_delta = float('-inf')

for variant_name, metrics_file in variants.items():
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    f1_delta_pct = data["improvement"]["f1_delta_pct"]
    if f1_delta_pct > winner_f1_delta:
        winner_f1_delta = f1_delta_pct
        winner_name = variant_name

print(f"🏆 WINNER: {winner_name} ({winner_f1_delta:+.3f}%)")
print("")
print("Recommendation:")
if "A (" in winner_name:
    print("  - Higher budget (5 prompts/cluster) provides most improvement")
    print("  - Suggests: More generation attempts overcome LLM stochasticity")
    print("  - Next step: Run 5-seed validation with Variante A")
elif "B (" in winner_name:
    print("  - Multi-temperature ensemble provides most improvement")
    print("  - Suggests: Diversity across temperatures increases robustness")
    print("  - Next step: Run 5-seed validation with Variante B")
else:
    print("  - GPT-5-mini reasoning provides most improvement")
    print("  - Suggests: Deterministic reasoning reduces variance")
    print("  - Next step: Run 5-seed validation with Variante C")
print("")
PYTHON_EOF

else
    echo "⚠️  Not all metrics files available yet. Missing:"
    [ ! -f "$RESULTS_DIR/variante_a_seed42_metrics.json" ] && echo "  - Variante A"
    [ ! -f "$RESULTS_DIR/variante_b_seed42_metrics.json" ] && echo "  - Variante B"
    [ ! -f "$RESULTS_DIR/variante_c_seed42_metrics.json" ] && echo "  - Variante C"
    echo ""
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
    gcloud compute instances delete "$VM_VARIANTE_A" "$VM_VARIANTE_B" "$VM_VARIANTE_C" --zone="$ZONE" --quiet
    echo "✅ VMs deleted!"
else
    echo ""
    echo "VMs kept running. Delete manually when done:"
    echo "  gcloud compute instances delete $VM_VARIANTE_A $VM_VARIANTE_B $VM_VARIANTE_C --zone=$ZONE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
