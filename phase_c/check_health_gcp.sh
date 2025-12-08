#!/bin/bash
# Check health monitor logs from all GCP VMs

ZONE="us-central1-a"
SEEDS=(42 100 123 456 789)

echo "═══════════════════════════════════════════════════════════"
echo "  Health Monitor Status - GCP Phase C"
echo "═══════════════════════════════════════════════════════════"
date
echo ""

for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "───────────────────────────────────────────────────────────"
    echo "Seed $SEED - $VM_NAME"
    echo "───────────────────────────────────────────────────────────"

    HEALTH_LOG=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        if [ -f health_monitor.log ]; then
            tail -10 health_monitor.log
        else
            echo 'Health monitor log not found'
        fi
    " 2>/dev/null)

    if [[ $HEALTH_LOG == *"CRITICAL"* ]]; then
        echo "🚨 CRITICAL ISSUE DETECTED!"
    elif [[ $HEALTH_LOG == *"WARNING"* ]]; then
        echo "⚠️  Warning detected"
    elif [[ $HEALTH_LOG == *"OK"* ]]; then
        echo "✅ Healthy"
    fi

    echo "$HEALTH_LOG"
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
