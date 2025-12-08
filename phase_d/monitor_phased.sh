#!/bin/bash
# Monitor Phase D Experiment Progress
#
# Real-time monitoring of all 4 variants running on GCP
# Shows current seed being processed and last log lines
#
# Usage: ./monitor_phased.sh
#
# Press Ctrl+C to exit

# Load GCP toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/gcp/gcp_toolkit.sh"

# VM names
VM_A="vm-phased-a"
VM_B="vm-phased-b"
VM_C="vm-phased-c"
VM_D="vm-phased-d"

ALL_VMS=("$VM_A" "$VM_B" "$VM_C" "$VM_D")
VARIANT_NAMES=("A (Contrastive)" "B (Focal Loss)" "C (Two-Stage)" "D (Full Stack)")
VARIANT_PREFIXES=("phased_a" "phased_b" "phased_c" "phased_d")
SEEDS=(42 100 123)

# Function to get current seed progress for a VM
get_seed_progress() {
    local vm="$1"
    local prefix="$2"

    # Count completed metrics files
    local completed=0
    for seed in "${SEEDS[@]}"; do
        if gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="test -f ${prefix}_seed${seed}_metrics.json && echo 'exists'" 2>/dev/null | grep -q "exists"; then
            ((completed++))
        fi
    done

    echo "$completed"
}

# Function to get current running seed
get_running_seed() {
    local vm="$1"
    local prefix="$2"

    # Check which log files exist and are being written to
    for seed in "${SEEDS[@]}"; do
        local check=$(gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="pgrep -f 'runner_phase2.py' >/dev/null && echo 'RUNNING' || echo 'STOPPED'" 2>/dev/null)

        if [ "$check" = "RUNNING" ]; then
            # Find which log file is being actively written
            local active=$(gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="lsof ${prefix}_seed*.log 2>/dev/null | grep -oP '${prefix}_seed\K[0-9]+' | head -1" 2>/dev/null)
            if [ -n "$active" ]; then
                echo "$active"
                return
            fi

            # Fallback: check which seed doesn't have metrics yet
            if ! gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="test -f ${prefix}_seed${seed}_metrics.json" 2>/dev/null; then
                echo "$seed"
                return
            fi
        fi
    done

    echo "DONE"
}

# Function to get last log lines
get_last_logs() {
    local vm="$1"
    local seed="$2"
    local prefix="$3"

    if [ "$seed" = "DONE" ]; then
        echo "  All seeds completed"
        return
    fi

    local log_file="${prefix}_seed${seed}.log"
    gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="tail -5 $log_file 2>/dev/null || echo '  (log not available yet)'" 2>/dev/null | sed 's/^/  /'
}

# Main monitoring loop
echo "════════════════════════════════════════════════════════"
echo "  Phase D Experiment Monitor"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Monitoring 4 VMs × 3 seeds = 12 experiments"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════"
    echo "  Phase D Experiment Monitor"
    echo "════════════════════════════════════════════════════════"
    echo ""
    date
    echo ""

    for i in "${!ALL_VMS[@]}"; do
        vm="${ALL_VMS[$i]}"
        variant_name="${VARIANT_NAMES[$i]}"
        prefix="${VARIANT_PREFIXES[$i]}"

        echo "─────────────────────────────────────────────────────────"
        echo "[$variant_name] $vm"
        echo "─────────────────────────────────────────────────────────"

        # Check VM status
        vm_status=$(gcloud compute instances describe "$vm" --zone="$GCP_ZONE" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

        if [ "$vm_status" != "RUNNING" ]; then
            echo "Status: VM $vm_status"
            echo ""
            continue
        fi

        # Get progress
        completed=$(get_seed_progress "$vm" "$prefix")
        running_seed=$(get_running_seed "$vm" "$prefix")

        if [ "$running_seed" = "DONE" ]; then
            echo "Status: ✓ COMPLETED ($completed/3 seeds)"
        else
            echo "Status: ✓ RUNNING ($completed/3 completed, current seed: $running_seed)"
        fi

        echo ""
        echo "Last log lines:"
        get_last_logs "$vm" "$running_seed" "$prefix"
        echo ""
    done

    echo "════════════════════════════════════════════════════════"
    echo "Refreshing in 60 seconds... (Ctrl+C to exit)"
    echo ""

    sleep 60
done
