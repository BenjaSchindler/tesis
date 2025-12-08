#!/bin/bash
#
# Monitor 25-Seed Experiments (5 VMs × 5 seeds each)
#

ZONE="us-central1-a"
BATCHES=(batch1 batch2 batch3 batch4 batch5)

declare -A BATCH_SEEDS
BATCH_SEEDS[batch1]="42 100 123 456 789"
BATCH_SEEDS[batch2]="111 222 333 444 555"
BATCH_SEEDS[batch3]="1000 2000 3000 4000 5000"
BATCH_SEEDS[batch4]="7 13 21 37 101"
BATCH_SEEDS[batch5]="1234 2345 3456 4567 5678"

while true; do
    clear
    echo "════════════════════════════════════════════════════════"
    echo "  Fase A - 25 Seeds Monitor (5 VMs × 5 seeds)"
    echo "════════════════════════════════════════════════════════"
    echo ""
    date
    echo ""

    TOTAL_COMPLETED=0
    TOTAL_RUNNING=0
    TOTAL_FAILED=0

    for batch in "${BATCHES[@]}"; do
        VM_NAME="vm-${batch}"

        echo "─────────────────────────────────────────────────────────"
        echo "[$VM_NAME] Seeds: ${BATCH_SEEDS[$batch]}"
        echo "─────────────────────────────────────────────────────────"

        # Check if VM exists and is running
        VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(status)' 2>/dev/null || echo "NOT_FOUND")

        if [ "$VM_STATUS" != "RUNNING" ]; then
            echo "Status: $VM_STATUS"
            echo ""
            continue
        fi

        # Get detailed status
        timeout 15 gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="
            cd LaptopRuns 2>/dev/null || cd ~

            # Count completed seeds
            COMPLETED=\$(ls phaseA_seed*_metrics.json 2>/dev/null | wc -l)
            echo \"Completed: \$COMPLETED/5\"

            # Check if process is running
            if pgrep -f 'python3 runner_phase2' > /dev/null; then
                CURRENT_SEED=\$(ps aux | grep 'python3 runner_phase2' | grep -oP '(?<=--random-seed )[0-9]+' | head -1)
                echo \"Status: ✓ RUNNING (current seed: \$CURRENT_SEED)\"

                # Show last 3 log lines
                if [ -f nohup_${batch}.log ]; then
                    echo ''
                    echo 'Last log lines:'
                    tail -3 nohup_${batch}.log | sed 's/^/  /'
                fi
            else
                if [ \$COMPLETED -eq 5 ]; then
                    echo 'Status: ✓ ALL 5 SEEDS COMPLETED'
                elif [ \$COMPLETED -gt 0 ]; then
                    echo \"Status: ⚠ STOPPED (\$COMPLETED/5 completed)\"
                else
                    echo 'Status: ⏹ NOT STARTED or FAILED'
                fi
            fi
        " 2>/dev/null || echo "  ⚠ Cannot connect to VM"

        echo ""
    done

    echo "════════════════════════════════════════════════════════"
    echo "Actualizando cada 60 segundos... (Ctrl+C para salir)"
    sleep 60
done
