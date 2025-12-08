#!/bin/bash
# Phase C v2.1 - Multi-Seed Validation on GCP
# Run 5 seeds in parallel on 5 separate VMs
# Seeds: 42, 100, 123, 456, 789
# Each VM: n1-standard-4 (4 vCPUs, 15GB RAM, CPU-only)

set -e

# Configuration
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="30GB"

# Seeds to run (one per VM)
SEEDS=(42 100 123 456 789)

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C v2.1 - GCP Multi-Seed Validation (5 Seeds)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Total VMs: 5"
echo "  Seeds: 42, 100, 123, 456, 789"
echo "  Machine Type: $MACHINE_TYPE (4 vCPUs, 15GB RAM)"
echo "  Zone: $ZONE"
echo "  Dataset: MBTI_500.csv"
echo ""
echo "Estimated:"
echo "  Time: ~2 hours (parallel execution)"
echo "  Cost: ~\$2.00 total (\$0.40 per VM)"
echo "  API cost: ~\$0.15"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Confirm
read -p "Launch 5 GCP VMs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Checking for existing VMs..."

# Delete existing VMs if they exist
for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"
    if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
        echo "  Deleting existing VM: $VM_NAME"
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    fi
done

echo ""
echo "Creating 5 VMs in parallel..."
echo ""

# Create all VMs in parallel
for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "🚀 Creating VM: $VM_NAME (seed $SEED)"

    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-ssd \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv
' \
        --scopes=cloud-platform &
done

# Wait for all VM creations
wait

echo ""
echo "✅ All VMs created!"
echo ""
echo "Waiting 60 seconds for VMs to initialize..."
sleep 60

echo ""
echo "Uploading files to VMs..."
echo ""

# Upload files to each VM
for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "📤 Uploading to $VM_NAME (seed $SEED)..."

    # Upload core files
    gcloud compute scp ../core/runner_phase2.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/enhanced_quality_gate.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/ensemble_anchor_selector.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/contamination_aware_filter.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/anchor_quality_improvements.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/quality_gate_predictor.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/mbti_class_descriptions.py "$VM_NAME:~/" --zone="$ZONE" --quiet &

    # Upload dataset
    gcloud compute scp ../MBTI_500.csv "$VM_NAME:~/" --zone="$ZONE" --quiet &
done

# Wait for all uploads
wait

echo ""
echo "✅ Files uploaded!"
echo ""
echo "Setting up Python environments and installing dependencies..."
echo ""

# Setup Python and install dependencies on each VM
for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "⚙️  Setting up $VM_NAME..."

    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv
    " &
done

# Wait for all setups
wait

echo ""
echo "✅ Python environments ready!"
echo ""
echo "Launching Phase C v2.1 experiments..."
echo ""

# Launch experiments on each VM
for SEED in "${SEEDS[@]}"; do
    VM_NAME="vm-phasec-seed${SEED}"

    echo "🔬 Launching experiment on $VM_NAME (seed $SEED)..."

    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        export OPENAI_API_KEY='$OPENAI_API_KEY'
        source venv/bin/activate

        # Create health monitor script
        cat > health_monitor.sh << 'HEALTH_EOF'
#!/bin/bash
# Health monitor - checks if experiment is stuck
LOGFILE=phaseC_seed${SEED}.log
HEALTHLOG=health_monitor.log
STUCK_THRESHOLD=900  # 15 minutes without log growth = stuck

echo \"\$(date): Health monitor started\" > \$HEALTHLOG

LAST_SIZE=0
STUCK_COUNT=0

while true; do
    sleep 300  # Check every 5 minutes

    if [ ! -f \$LOGFILE ]; then
        echo \"\$(date): Log file not found yet, waiting...\" >> \$HEALTHLOG
        continue
    fi

    CURRENT_SIZE=\$(wc -c < \$LOGFILE 2>/dev/null || echo 0)

    if [ \$CURRENT_SIZE -eq \$LAST_SIZE ]; then
        STUCK_COUNT=\$((STUCK_COUNT + 300))
        echo \"\$(date): WARNING - Log hasn't grown in \${STUCK_COUNT}s (size: \${CURRENT_SIZE} bytes)\" >> \$HEALTHLOG

        # Check if process is still alive
        if ! pgrep -f 'python3.*runner_phase2' > /dev/null; then
            echo \"\$(date): ERROR - Process died!\" >> \$HEALTHLOG
            break
        fi

        # If stuck for more than threshold, report and exit
        if [ \$STUCK_COUNT -ge \$STUCK_THRESHOLD ]; then
            echo \"\$(date): CRITICAL - Experiment stuck for \${STUCK_COUNT}s, likely API timeout\" >> \$HEALTHLOG
            echo \"\$(date): Last 50 log lines:\" >> \$HEALTHLOG
            tail -50 \$LOGFILE >> \$HEALTHLOG 2>/dev/null
            # Don't kill - let it run, but log the issue
        fi
    else
        GROWTH=\$((CURRENT_SIZE - LAST_SIZE))
        STUCK_COUNT=0
        echo \"\$(date): OK - Log growing (+\${GROWTH} bytes, total: \${CURRENT_SIZE})\" >> \$HEALTHLOG
        LAST_SIZE=\$CURRENT_SIZE
    fi
done

echo \"\$(date): Health monitor stopped\" >> \$HEALTHLOG
HEALTH_EOF

        chmod +x health_monitor.sh
        nohup ./health_monitor.sh > /dev/null 2>&1 &

        # Launch experiment
        nohup python3 -u runner_phase2.py \
            --data-path MBTI_500.csv \
            --test-size 0.2 \
            --random-seed $SEED \
            \
            --embedding-model sentence-transformers/all-mpnet-base-v2 \
            --device cpu \
            --embedding-batch-size 32 \
            \
            --llm-model gpt-4o-mini \
            --temperature 1.0 \
            --max-clusters 3 \
            --prompts-per-cluster 3 \
            --prompt-mode mix \
            \
            --use-ensemble-selection \
            \
            --use-val-gating \
            --val-size 0.15 \
            --val-tolerance 0.02 \
            \
            --enable-anchor-gate \
            --anchor-quality-threshold 0.25 \
            --purity-gate-threshold 0.025 \
            \
            --enable-anchor-selection \
            --anchor-selection-ratio 0.8 \
            --anchor-outlier-threshold 1.5 \
            \
            --enable-adaptive-filters \
            \
            --use-class-description \
            \
            --use-f1-budget-scaling \
            --f1-budget-thresholds 0.40 0.20 \
            --f1-budget-multipliers 30 70 100 \
            \
            --enable-adaptive-weighting \
            --synthetic-weight 0.5 \
            \
            --similarity-threshold 0.90 \
            --min-classifier-confidence 0.10 \
            --contamination-threshold 0.95 \
            \
            --synthetic-output phaseC_v2.1_seed${SEED}_synthetic.csv \
            --augmented-train-output phaseC_v2.1_seed${SEED}_augmented.csv \
            --metrics-output phaseC_v2.1_seed${SEED}_metrics.json \
            > phaseC_seed${SEED}.log 2>&1 &

        echo 'Experiment launched in background'
    " &
done

# Wait for all launches
wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All 5 Experiments Launched on GCP!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running:"
for SEED in "${SEEDS[@]}"; do
    echo "  - vm-phasec-seed${SEED} (seed $SEED)"
done
echo ""
echo "Estimated completion: ~2 hours"
echo ""
echo "Monitor progress:"
echo "  ./monitor_gcp_5seeds.sh"
echo ""
echo "Collect results (after completion):"
echo "  ./collect_gcp_5seeds.sh"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
