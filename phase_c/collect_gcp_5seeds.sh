#!/bin/bash
# Collect results from GCP Phase C 5-seed validation

set -e

ZONE="us-central1-a"
SEEDS=(42 100 123 456 789)

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C v2.1 - Collect GCP Results"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create results directory
mkdir -p gcp_results
cd gcp_results

echo "Downloading results from VMs..."
echo ""

for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "📥 Downloading from $VM_NAME (seed $SEED)..."

    # Check if VM exists and is running
    if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
        echo "  ⚠️  VM not found, skipping"
        continue
    fi

    # Download results
    gcloud compute scp "$VM_NAME:~/phaseC_v2.1_seed${SEED}_metrics.json" "./" --zone="$ZONE" --quiet 2>/dev/null || echo "  ⚠️  Metrics file not found"
    gcloud compute scp "$VM_NAME:~/phaseC_v2.1_seed${SEED}_synthetic.csv" "./" --zone="$ZONE" --quiet 2>/dev/null || echo "  ⚠️  Synthetic file not found"
    gcloud compute scp "$VM_NAME:~/phaseC_v2.1_seed${SEED}_augmented.csv" "./" --zone="$ZONE" --quiet 2>/dev/null || echo "  ⚠️  Augmented file not found"
    gcloud compute scp "$VM_NAME:~/phaseC_seed${SEED}.log" "./" --zone="$ZONE" --quiet 2>/dev/null || echo "  ⚠️  Log file not found"

    echo "  ✅ Downloaded"
done

echo ""
echo "Moving results to phase_c directory..."

# Move results to phase_c directory
mv phaseC_v2.1_seed*_metrics.json ../ 2>/dev/null || true
mv phaseC_v2.1_seed*_synthetic.csv ../ 2>/dev/null || true
mv phaseC_v2.1_seed*_augmented.csv ../ 2>/dev/null || true
mv phaseC_seed*.log ../ 2>/dev/null || true

cd ..
rmdir gcp_results 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Results Downloaded!"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Count downloaded results
METRICS_COUNT=$(ls phaseC_v2.1_seed*_metrics.json 2>/dev/null | wc -l)

echo "Results summary:"
echo "  Metrics files: $METRICS_COUNT/5"
echo ""

if [ $METRICS_COUNT -eq 5 ]; then
    echo "✅ All 5 seeds collected!"
    echo ""
    echo "Run analysis:"
    echo "  python3 analyze_5seeds.py"
    echo ""
else
    echo "⚠️  Only $METRICS_COUNT/5 seeds collected"
    echo ""
    echo "Missing seeds:"
    for SEED in "${SEEDS[@]}"; do
        if [ ! -f "phaseC_v2.1_seed${SEED}_metrics.json" ]; then
            echo "  - Seed $SEED"
        fi
    done
    echo ""
fi

echo "═══════════════════════════════════════════════════════════"
echo ""

# Ask to delete VMs
read -p "Delete all 5 VMs to save costs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting VMs..."

    for SEED in "${SEEDS[@]}"; do
        VM_NAME="vm-phasec-seed${SEED}"
        echo "  Deleting $VM_NAME..."
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet 2>/dev/null || echo "    Already deleted"
    done

    echo ""
    echo "✅ All VMs deleted!"
else
    echo ""
    echo "VMs kept running. To delete later:"
    for SEED in "${SEEDS[@]}"; do
        echo "  gcloud compute instances delete vm-phasec-seed${SEED} --zone=$ZONE"
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
