#!/bin/bash
# Phase C v2.1 - Multi-Seed Validation on GCP with GPU
# Account: contacto@doctor911.cl (prexx-465515)
# Use existing GPU VMs + create 1 more if needed
# Seeds: 42, 100, 123, 456, 789

set -e

# Configuration
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="50GB"

# Seeds to run
SEEDS=(42 100 123 456 789)

# VMs to use (4 existing, vm-batch4 runs 2 seeds sequentially)
declare -A VM_SEEDS
VM_SEEDS[vm-batch1-gpu]="42"
VM_SEEDS[vm-batch2-gpu]="100"
VM_SEEDS[vm-batch3-gpu]="123"
VM_SEEDS[vm-batch4-gpu]="456 789"  # Sequential: 456 then 789

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C v2.1 - GCP GPU Multi-Seed (5 Seeds)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Account: contacto@doctor911.cl"
echo "  Project: prexx-465515"
echo "  Total VMs: 4 (GPU quota limit)"
echo "  Seeds: 42, 100, 123, 456+789 (sequential)"
echo "  Machine Type: $MACHINE_TYPE + $GPU_TYPE"
echo "  Zone: $ZONE"
echo ""
echo "Estimated:"
echo "  Time: ~45 min (3 VMs), ~90 min (vm-batch4-gpu with 2 seeds)"
echo "  Cost: ~$2.50 total"
echo "  Auto-shutdown: Enabled (2 min after completion)"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Launch experiments on GPU VMs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Starting existing VMs..."
echo ""

# Start existing VMs
for VM in vm-batch1-gpu vm-batch2-gpu vm-batch3-gpu vm-batch4-gpu; do
    echo "▶️  Starting $VM..."
    gcloud compute instances start "$VM" --zone="$ZONE" 2>/dev/null || echo "  (already running or doesn't exist)"
done

echo ""
echo "Waiting 60 seconds for VMs to initialize..."
sleep 60

echo ""
echo "Step 3: Uploading files to VMs..."
echo ""

# Upload files to each VM
for VM_NAME in "${!VM_SEEDS[@]}"; do
    SEED=${VM_SEEDS[$VM_NAME]}

    echo "📤 Uploading to $VM_NAME (seed $SEED)..."

    # Upload core files in parallel
    gcloud compute scp ../core/runner_phase2.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/enhanced_quality_gate.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/ensemble_anchor_selector.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/contamination_aware_filter.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/anchor_quality_improvements.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/quality_gate_predictor.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../core/mbti_class_descriptions.py "$VM_NAME:~/" --zone="$ZONE" --quiet &
    gcloud compute scp ../MBTI_500.csv "$VM_NAME:~/" --zone="$ZONE" --quiet &

    wait
done

echo ""
echo "✅ Files uploaded!"
echo ""
echo "Step 4: Setting up Python environments..."
echo ""

# Setup Python on each VM
for VM_NAME in "${!VM_SEEDS[@]}"; do
    echo "⚙️  Setting up $VM_NAME..."

    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        if [ ! -d venv ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install --upgrade pip -q
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
        pip install numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv -q
    " &
done

wait

echo ""
echo "✅ Python environments ready!"
echo ""
echo "Step 5: Launching experiments with GPU..."
echo ""

# Launch experiments
for VM_NAME in "${!VM_SEEDS[@]}"; do
    SEEDS_STR=${VM_SEEDS[$VM_NAME]}

    echo "🔬 Launching experiment(s) on $VM_NAME (seeds: $SEEDS_STR)..."

    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        export OPENAI_API_KEY='$OPENAI_API_KEY'
        source venv/bin/activate

        # Create sequential runner script
        cat > run_experiments.sh << 'RUN_EOF'
#!/bin/bash
SEEDS=($SEEDS_STR)

for SEED in \"\\\${SEEDS[@]}\"; do
    echo \"Starting experiment for seed \\\$SEED...\"

    python3 -u runner_phase2.py \\
        --data-path MBTI_500.csv \\
        --test-size 0.2 \\
        --random-seed \\\$SEED \\
        \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --device cuda \\
        --embedding-batch-size 64 \\
        \\
        --llm-model gpt-4o-mini \\
        --temperature 1.0 \\
        --max-clusters 3 \\
        --prompts-per-cluster 3 \\
        --prompt-mode mix \\
        \\
        --use-ensemble-selection \\
        \\
        --use-val-gating \\
        --val-size 0.15 \\
        --val-tolerance 0.02 \\
        \\
        --enable-anchor-gate \\
        --anchor-quality-threshold 0.25 \\
        --purity-gate-threshold 0.025 \\
        \\
        --enable-anchor-selection \\
        --anchor-selection-ratio 0.8 \\
        --anchor-outlier-threshold 1.5 \\
        \\
        --enable-adaptive-filters \\
        \\
        --use-class-description \\
        \\
        --use-f1-budget-scaling \\
        --f1-budget-thresholds 0.40 0.20 \\
        --f1-budget-multipliers 30 70 100 \\
        \\
        --enable-adaptive-weighting \\
        --synthetic-weight 0.5 \\
        \\
        --similarity-threshold 0.90 \\
        --min-classifier-confidence 0.10 \\
        --contamination-threshold 0.95 \\
        \\
        --synthetic-output phaseC_v2.1_seed\\\${SEED}_synthetic.csv \\
        --augmented-train-output phaseC_v2.1_seed\\\${SEED}_augmented.csv \\
        --metrics-output phaseC_v2.1_seed\\\${SEED}_metrics.json \\
        2>&1 | tee phaseC_seed\\\${SEED}.log

    if [ \\\$? -eq 0 ]; then
        echo \"Seed \\\$SEED completed successfully\"
    else
        echo \"Seed \\\$SEED failed with exit code \\\$?\"
    fi
done

# Auto-shutdown after all experiments complete
echo \"All experiments on this VM complete. Shutting down in 2 minutes...\"
sleep 120
sudo shutdown -h now
RUN_EOF

        chmod +x run_experiments.sh
        nohup ./run_experiments.sh > experiment_runner.log 2>&1 &

        echo 'Experiments launched with GPU acceleration + auto-shutdown'
    " &
done

wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All 5 Seeds Launched on 4 GPU VMs!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running (doctor911 account):"
echo "  - vm-batch1-gpu (seed 42)"
echo "  - vm-batch2-gpu (seed 100)"
echo "  - vm-batch3-gpu (seed 123)"
echo "  - vm-batch4-gpu (seeds 456, 789 sequential)"
echo ""
echo "Estimated completion:"
echo "  - VMs 1-3: ~45 minutes"
echo "  - VM 4: ~90 minutes (2 seeds sequential)"
echo ""
echo "Auto-shutdown: Enabled (2 min after completion)"
echo ""
echo "Monitor:"
echo "  # Check VM 1 (seed 42)"
echo "  gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --command='tail -20 phaseC_seed42.log'"
echo ""
echo "  # Check VM 4 (seeds 456+789)"
echo "  gcloud compute ssh vm-batch4-gpu --zone=us-central1-a --command='tail -20 experiment_runner.log'"
echo ""
echo "  # Check GPU usage"
echo "  gcloud compute ssh vm-batch1-gpu --zone=us-central1-a --command='nvidia-smi'"
echo ""
echo "═══════════════════════════════════════════════════════════"
