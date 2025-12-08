#!/bin/bash
# Monitor Phase C 5-seed validation on GCP

ZONE="us-central1-a"
SEEDS=(42 100 123 456 789)

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════"
    echo "  Phase C v2.1 - GCP Multi-Seed Validation Monitor"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    date
    echo ""

    # Count completed experiments
    COMPLETED=0

    for SEED in "${SEEDS[@]}"; do
        VM_NAME="vm-phasec-seed${SEED}"

        echo "───────────────────────────────────────────────────────────"
        echo "VM: $VM_NAME (Seed $SEED)"
        echo "───────────────────────────────────────────────────────────"

        # Check if VM exists
        if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
            echo "  ❌ VM does not exist"
            continue
        fi

        # Get VM status
        STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)")
        echo "  Status: $STATUS"

        if [ "$STATUS" != "RUNNING" ]; then
            echo "  ⚠️  VM not running"
            continue
        fi

        # Check if experiment completed
        HAS_RESULTS=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ls phaseC_v2.1_seed${SEED}_metrics.json 2>/dev/null" 2>/dev/null || echo "")

        if [ -n "$HAS_RESULTS" ]; then
            echo "  ✅ COMPLETED"
            COMPLETED=$((COMPLETED + 1))

            # Get results
            DELTA=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="python3 -c \"import json; f=open('phaseC_v2.1_seed${SEED}_metrics.json'); d=json.load(f); print(f\\\"{d['improvement']['f1_delta_pct']:.3f}%\\\")\"" 2>/dev/null || echo "N/A")
            SYNTH=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="python3 -c \"import json; f=open('phaseC_v2.1_seed${SEED}_metrics.json'); d=json.load(f); print(d['synthetic_data']['accepted_count'])\"" 2>/dev/null || echo "N/A")

            echo "  Overall delta: +$DELTA"
            echo "  Synthetics: $SYNTH"
        else
            echo "  🔄 RUNNING"

            # Get last log lines
            LAST_LOG=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="tail -3 phaseC_seed${SEED}.log 2>/dev/null | grep -E 'Clase|LLM|GATE' | tail -1" 2>/dev/null || echo "")

            if [ -n "$LAST_LOG" ]; then
                echo "  Last: ${LAST_LOG:0:60}..."
            fi
        fi

        echo ""
    done

    echo "───────────────────────────────────────────────────────────"
    echo ""
    echo "Progress: $COMPLETED/5 experiments completed"
    echo ""
    echo "═══════════════════════════════════════════════════════════"

    # Exit if all complete
    if [ $COMPLETED -eq 5 ]; then
        echo ""
        echo "🎉 All 5 experiments completed!"
        echo ""
        echo "Collect results:"
        echo "  ./collect_gcp_5seeds.sh"
        echo ""
        break
    fi

    echo ""
    echo "Press Ctrl+C to exit monitor (experiments will continue)"
    echo "Refreshing in 60 seconds..."
    sleep 60
done
