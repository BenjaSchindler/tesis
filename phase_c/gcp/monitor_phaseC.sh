#!/bin/bash
# Phase C Monitor - Check progress of GCP experiment

VM_NAME="vm-phasec-test"
ZONE="us-west1-b"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════"
echo "  Phase C - GCP Monitor"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if VM exists
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${RED}❌ VM $VM_NAME not found in zone $ZONE${NC}"
    echo ""
    echo "Did you run ./launch_phaseC.sh first?"
    exit 1
fi

# Get VM status
VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)")

echo "VM Status: $VM_STATUS"
echo ""

if [ "$VM_STATUS" != "RUNNING" ]; then
    echo -e "${YELLOW}⚠️  VM is not running (status: $VM_STATUS)${NC}"

    if [ "$VM_STATUS" = "TERMINATED" ]; then
        echo ""
        echo "VM has shut down (experiment likely completed)."
        echo "Download results with: ./collect_results_phaseC.sh"
    fi
    exit 0
fi

echo -e "${GREEN}✅ VM is RUNNING${NC}"
echo ""

# Check if Python process is running
echo "Checking experiment status..."
PYTHON_RUNNING=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ps aux | grep 'python3 runner_phase2.py' | grep -v grep" 2>/dev/null || echo "")

if [ -z "$PYTHON_RUNNING" ]; then
    echo -e "${YELLOW}⚠️  Python process not found${NC}"
    echo ""
    echo "Experiment may have:"
    echo "  • Completed successfully (check logs below)"
    echo "  • Failed during execution (check logs below)"
    echo "  • Not started yet (wait a moment)"
    echo ""
else
    echo -e "${GREEN}✅ Experiment is RUNNING${NC}"
    echo ""
    echo "Process info:"
    echo "$PYTHON_RUNNING" | awk '{print "  PID: "$2", CPU: "$3"%, MEM: "$4"%"}'
    echo ""
fi

# Check for results files
echo "Checking for output files..."
FILES_EXIST=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ls PhaseC/phaseC_seed42_metrics.json 2>/dev/null" || echo "")

if [ -n "$FILES_EXIST" ]; then
    echo -e "${GREEN}✅ Results file found (experiment completed!)${NC}"
    echo ""
else
    echo "  ⏳ Results file not yet created"
    echo ""
fi

# Show last 30 lines of log
echo "─────────────────────────────────────────────────────────"
echo -e "${CYAN}Last 30 lines of log:${NC}"
echo "─────────────────────────────────────────────────────────"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="tail -30 PhaseC/phaseC_output.log" 2>/dev/null || echo "  (log file not found yet)"

echo ""
echo "─────────────────────────────────────────────────────────"

# Check for adaptive temperature messages
echo ""
echo "Adaptive Temperature Adjustments:"
TEMP_MSGS=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="grep '🌡️' PhaseC/phaseC_output.log 2>/dev/null" || echo "")

if [ -n "$TEMP_MSGS" ]; then
    echo -e "${GREEN}$TEMP_MSGS${NC}"
else
    echo "  (not yet in log - experiment may be in early stages)"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "Commands:"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  View full log:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE --command='cat PhaseC/phaseC_output.log'"
echo ""
echo "  Follow log in real-time:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f PhaseC/phaseC_output.log'"
echo ""
echo "  SSH into VM:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "  Download results (when completed):"
echo "    ./collect_results_phaseC.sh"
echo ""
echo "  Re-run this monitor:"
echo "    ./monitor_phaseC.sh"
echo ""
echo "════════════════════════════════════════════════════════"
