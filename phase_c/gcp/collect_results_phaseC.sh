#!/bin/bash
# Phase C Results Collector - Download and analyze results from GCP

VM_NAME="vm-phasec-test"
ZONE="us-west1-b"
RESULTS_DIR="phaseC_results"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════"
echo "  Phase C - Results Collector"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if VM exists
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${RED}❌ VM $VM_NAME not found in zone $ZONE${NC}"
    exit 1
fi

# Get VM status
VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)")
echo "VM Status: $VM_STATUS"
echo ""

# If VM is terminated, start it temporarily
if [ "$VM_STATUS" = "TERMINATED" ]; then
    echo -e "${YELLOW}VM is stopped. Starting temporarily to download results...${NC}"
    gcloud compute instances start "$VM_NAME" --zone="$ZONE"
    echo "Waiting 30 seconds for VM to boot..."
    sleep 30
    SHOULD_STOP=true
else
    SHOULD_STOP=false
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}[1/3] Downloading results files...${NC}"
echo ""

# Download all output files
FILES_TO_DOWNLOAD=(
    "phaseC_seed42_metrics.json"
    "phaseC_seed42_synthetic.csv"
    "phaseC_seed42_augmented.csv"
    "phaseC_output.log"
)

DOWNLOADED=0
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    if gcloud compute scp "$VM_NAME:~/PhaseC/$file" "$RESULTS_DIR/" --zone="$ZONE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $file"
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo -e "${YELLOW}⚠️${NC} $file (not found)"
    fi
done

echo ""
if [ $DOWNLOADED -eq 0 ]; then
    echo -e "${RED}❌ No files downloaded. Experiment may not have run or completed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Downloaded $DOWNLOADED/${#FILES_TO_DOWNLOAD[@]} files${NC}"
echo ""

# Analyze results if metrics file exists
if [ -f "$RESULTS_DIR/phaseC_seed42_metrics.json" ]; then
    echo -e "${BLUE}[2/3] Analyzing results...${NC}"
    echo ""

    python3 << 'PYTHON_ANALYSIS'
import json
import sys

try:
    with open('phaseC_results/phaseC_seed42_metrics.json') as f:
        data = json.load(f)

    print("═"*60)
    print("  PHASE C - RESULTS ANALYSIS (SEED 42)")
    print("═"*60)
    print()

    # Overall performance
    baseline = data['baseline_macro_f1']
    augmented = data['augmented_macro_f1']
    delta = (augmented - baseline) * 100

    print("Overall Performance:")
    print(f"  Baseline Macro F1:   {baseline:.4f}")
    print(f"  Augmented Macro F1:  {augmented:.4f}")
    print(f"  Delta:               {delta:+.2f}%")
    print()

    # MID-tier analysis
    print("MID-Tier Classes (F1 0.20-0.45):")
    print("─"*60)
    mid_deltas = []
    mid_classes = []

    for cls, m in sorted(data['per_class_metrics'].items()):
        f1_base = m['baseline_f1']
        f1_aug = m['augmented_f1']
        if 0.20 <= f1_base < 0.45:
            delta_cls = (f1_aug - f1_base) * 100
            mid_deltas.append(delta_cls)
            mid_classes.append(cls)

            status = "✅" if delta_cls > 0 else "⚠️" if delta_cls > -0.30 else "❌"
            print(f"  {status} {cls:6s}: {f1_base:.3f} → {f1_aug:.3f} ({delta_cls:+.2f}%)")

    print("─"*60)

    if mid_deltas:
        mid_mean = sum(mid_deltas) / len(mid_deltas)
        mid_positive = sum(1 for d in mid_deltas if d > 0)

        print(f"\nMID-Tier Summary:")
        print(f"  Mean Delta:     {mid_mean:+.2f}%")
        print(f"  Positive:       {mid_positive}/{len(mid_deltas)} classes")
        print(f"  Target:         ≥ +0.10%")
        print(f"  Phase B Baseline: -0.59%")
        print(f"  Improvement:    {mid_mean + 0.59:+.2f}pp")
        print()

        # Verdict
        if mid_mean >= 0.10:
            print("  ✅ SUCCESS: MID-tier target achieved!")
            print("     → Adaptive temperature works!")
            print("     → Proceed to 5-seed validation")
        elif mid_mean >= -0.30:
            print("  ⚠️  PARTIAL SUCCESS: Improvement but below target")
            print("     → Adaptive temperature helps but not enough")
            print("     → Proceed to Phase 2 (Hardness-Aware Anchors)")
            print("     → Expected combined impact: +0.30% to +0.50%")
        else:
            print("  ❌ NEEDS MORE WORK: Limited improvement")
            print("     → Temperature alone insufficient")
            print("     → Skip to Phase 2 (Hardness-Aware Anchors)")
            print("     → Consider Phase 3 (Multi-Stage Filtering)")

    print()

    # LOW-tier check (should be maintained)
    print("LOW-Tier Classes (F1 < 0.20) - Should Maintain Improvement:")
    print("─"*60)
    low_deltas = []

    for cls, m in sorted(data['per_class_metrics'].items()):
        f1_base = m['baseline_f1']
        f1_aug = m['augmented_f1']
        if f1_base < 0.20:
            delta_cls = (f1_aug - f1_base) * 100
            low_deltas.append(delta_cls)
            status = "✅" if delta_cls > 0 else "❌"
            print(f"  {status} {cls:6s}: {f1_base:.3f} → {f1_aug:.3f} ({delta_cls:+.2f}%)")

    if low_deltas:
        low_mean = sum(low_deltas) / len(low_deltas)
        print("─"*60)
        print(f"  Mean Delta:     {low_mean:+.2f}%")
        print(f"  Phase A Target: +12.17%")

        if low_mean >= 10.0:
            print("  ✅ LOW-tier improvement maintained")
        else:
            print("  ⚠️  LOW-tier improvement lower than Phase A")

    print()
    print("═"*60)

except FileNotFoundError:
    print("ERROR: Results file not found")
    sys.exit(1)
except json.JSONDecodeError:
    print("ERROR: Invalid JSON in results file")
    sys.exit(1)
PYTHON_ANALYSIS

else
    echo -e "${YELLOW}⚠️  Metrics file not found, skipping analysis${NC}"
fi

# Check for adaptive temperature messages in log
if [ -f "$RESULTS_DIR/phaseC_output.log" ]; then
    echo ""
    echo -e "${BLUE}[3/3] Checking adaptive temperature adjustments...${NC}"
    echo ""

    TEMP_MSGS=$(grep "🌡️  ADAPTIVE TEMP" "$RESULTS_DIR/phaseC_output.log" || echo "")

    if [ -n "$TEMP_MSGS" ]; then
        echo "Adaptive Temperature Adjustments Found:"
        echo "─────────────────────────────────────────────────────────"
        echo "$TEMP_MSGS"
        echo "─────────────────────────────────────────────────────────"
    else
        echo -e "${YELLOW}⚠️  No adaptive temperature messages found in log${NC}"
        echo "This might indicate:"
        echo "  • Experiment failed before reaching generation phase"
        echo "  • baseline_f1_scores not passed correctly"
        echo ""
        echo "Check full log: cat $RESULTS_DIR/phaseC_output.log"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo -e "${GREEN}  Results Downloaded Successfully!${NC}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Files saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/" | tail -n +2
echo ""

# Ask to delete VM
if [ "$SHOULD_STOP" = true ]; then
    echo "Stopping VM (was temporarily started for download)..."
    gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --quiet
    echo -e "${GREEN}✓ VM stopped${NC}"
    echo ""
fi

echo "Do you want to DELETE the VM to stop all charges?"
echo "(You can recreate it anytime with ./launch_phaseC.sh)"
echo ""
read -p "Delete VM? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting VM..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    echo -e "${GREEN}✓ VM deleted - no more charges${NC}"
else
    echo "VM kept (status: $VM_STATUS)"
    echo ""
    echo "Estimated costs while VM exists:"
    if [ "$VM_STATUS" = "RUNNING" ]; then
        echo "  RUNNING: ~\$0.55/hour (compute + GPU)"
    else
        echo "  STOPPED: ~\$0.01/hour (storage only)"
    fi
    echo ""
    echo "To delete later:"
    echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "Next Steps:"
echo "════════════════════════════════════════════════════════"
echo ""
echo "1. Review results in: $RESULTS_DIR/"
echo ""
echo "2. If MID-tier ≥ +0.10%:"
echo "   → SUCCESS! Run 5-seed validation"
echo "   → Modify launch_phaseC.sh for seeds: 42,100,123,456,789"
echo ""
echo "3. If MID-tier -0.30% to +0.10%:"
echo "   → PARTIAL. Implement Phase 2 (Hardness-Aware Anchors)"
echo "   → See: phase_c/README.md (Week 2 roadmap)"
echo ""
echo "4. If MID-tier < -0.30%:"
echo "   → Need stronger techniques"
echo "   → Skip to Hardness-Aware Anchors (90% success rate)"
echo ""
echo "════════════════════════════════════════════════════════"
