#!/bin/bash
# Launch 4 Robustness Variants in Parallel (Seed 42)
# Variante A: 5 prompts/cluster
# Variante B: Multi-temperature ensemble
# Variante C1: GPT-5-mini (no reasoning, low verbosity)
# Variante C3: GPT-5-mini (minimal reasoning, high verbosity)

set -e

# GCP Configuration
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="50GB"
SEED=42

# VM names for each variant
VM_VARIANTE_A="vm-variante-a-seed42"
VM_VARIANTE_B="vm-variante-b-seed42"
VM_VARIANTE_C1="vm-variante-c1-seed42"
VM_VARIANTE_C3="vm-variante-c3-seed42"

ALL_VMS="$VM_VARIANTE_A $VM_VARIANTE_B $VM_VARIANTE_C1 $VM_VARIANTE_C3"

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Phase C - 4 Robustness Variants (Seed 42)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Testing 4 approaches to reduce LLM stochasticity:"
echo ""
echo "  🅰️  Variante A: Higher Budget"
echo "      - 5 prompts/cluster (vs 3 baseline)"
echo "      - gpt-4o-mini, temperature=1.0"
echo "      - More generation attempts"
echo ""
echo "  🅱️  Variante B: Multi-Temperature Ensemble"
echo "      - 3 prompts/cluster"
echo "      - gpt-4o-mini, temperatures [0.7, 1.0, 1.3]"
echo "      - Combines outputs from all temps"
echo ""
echo "  ©️1 Variante C1: GPT-5-mini (Terse)"
echo "      - 3 prompts/cluster"
echo "      - gpt-5-mini, no reasoning, verbosity=low"
echo "      - Fastest, most concise output"
echo ""
echo "  ©️3 Variante C3: GPT-5-mini (Thorough)"
echo "      - 3 prompts/cluster"
echo "      - gpt-5-mini, reasoning=minimal, verbosity=high"
echo "      - Most detailed output"
echo ""
echo "Configuration:"
echo "  Seed: $SEED"
echo "  Machine Type: $MACHINE_TYPE + $GPU_TYPE"
echo "  Zone: $ZONE"
echo "  Estimated time: A~50min, B~80min, C1~45min, C3~50min"
echo "  Estimated cost: ~$4 total"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Launch all 4 variants in parallel? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Creating 4 GPU VMs..."
echo ""

# Create VMs in parallel
for VM_NAME in $ALL_VMS; do
    echo "🚀 Creating $VM_NAME..."
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --maintenance-policy=TERMINATE \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv
# Install CUDA drivers
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-2
' &
done

# Wait for all VM creations
wait

echo ""
echo "✅ VMs created! Waiting 90 seconds for CUDA driver installation..."
sleep 90

echo ""
echo "Step 2: Uploading files to VMs..."
echo ""

# Upload files to all VMs in parallel
for VM_NAME in $ALL_VMS; do
    echo "📤 Uploading to $VM_NAME..."
    {
        gcloud compute scp ../core/runner_phase2.py "$VM_NAME":~/ --zone="$ZONE"
        gcloud compute scp ../MBTI_500.csv "$VM_NAME":~/ --zone="$ZONE"
    } &
done

wait

echo ""
echo "✅ Files uploaded!"
echo ""
echo "Step 3: Setting up Python environments..."
echo ""

# Setup Python venv in parallel
for VM_NAME in $ALL_VMS; do
    echo "⚙️  Setting up $VM_NAME..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv
    " &
done

wait

echo ""
echo "✅ Python environments ready!"
echo ""
echo "Step 4: Launching experiments..."
echo ""

# Launch Variante A: 5 prompts/cluster
echo "🅰️  Launching Variante A (5 prompts/cluster)..."
gcloud compute ssh "$VM_VARIANTE_A" --zone="$ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 64 \
        \
        --llm-model gpt-4o-mini \
        --temperature 1.0 \
        --max-clusters 3 \
        --prompts-per-cluster 5 \
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
        --synthetic-output variante_a_seed42_synthetic.csv \
        --augmented-train-output variante_a_seed42_augmented.csv \
        --metrics-output variante_a_seed42_metrics.json \
        > variante_a_seed42.log 2>&1 &

    echo 'Variante A launched!'
    sleep 2
    sudo shutdown -h +120
" &

# Launch Variante B: Multi-temperature ensemble
echo "🅱️  Launching Variante B (multi-temperature ensemble)..."
gcloud compute ssh "$VM_VARIANTE_B" --zone="$ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 64 \
        \
        --llm-model gpt-4o-mini \
        --multi-temperature-ensemble \
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
        --synthetic-output variante_b_seed42_synthetic.csv \
        --augmented-train-output variante_b_seed42_augmented.csv \
        --metrics-output variante_b_seed42_metrics.json \
        > variante_b_seed42.log 2>&1 &

    echo 'Variante B launched!'
    sleep 2
    sudo shutdown -h +120
" &

# Launch Variante C1: GPT-5-mini (no reasoning, low verbosity)
echo "©️1 Launching Variante C1 (GPT-5-mini terse)..."
gcloud compute ssh "$VM_VARIANTE_C1" --zone="$ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 64 \
        \
        --llm-model gpt-5-mini-2025-08-07 \
        --reasoning-effort none \
        --output-verbosity low \
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
        --synthetic-output variante_c1_seed42_synthetic.csv \
        --augmented-train-output variante_c1_seed42_augmented.csv \
        --metrics-output variante_c1_seed42_metrics.json \
        > variante_c1_seed42.log 2>&1 &

    echo 'Variante C1 launched!'
    sleep 2
    sudo shutdown -h +120
" &

# Launch Variante C3: GPT-5-mini (minimal reasoning, high verbosity)
echo "©️3 Launching Variante C3 (GPT-5-mini thorough)..."
gcloud compute ssh "$VM_VARIANTE_C3" --zone="$ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $SEED \
        \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 64 \
        \
        --llm-model gpt-5-mini-2025-08-07 \
        --reasoning-effort minimal \
        --output-verbosity high \
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
        --synthetic-output variante_c3_seed42_synthetic.csv \
        --augmented-train-output variante_c3_seed42_augmented.csv \
        --metrics-output variante_c3_seed42_metrics.json \
        > variante_c3_seed42.log 2>&1 &

    echo 'Variante C3 launched!'
    sleep 2
    sudo shutdown -h +120
" &

wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All 4 Variants Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running:"
echo "  🅰️  $VM_VARIANTE_A (5 prompts/cluster)"
echo "  🅱️  $VM_VARIANTE_B (multi-temp ensemble)"
echo "  ©️1 $VM_VARIANTE_C1 (GPT-5-mini terse)"
echo "  ©️3 $VM_VARIANTE_C3 (GPT-5-mini thorough)"
echo ""
echo "Estimated completion:"
echo "  Variante A:  ~50 min"
echo "  Variante B:  ~80 min (slowest - 3x API calls)"
echo "  Variante C1: ~45 min"
echo "  Variante C3: ~50 min"
echo ""
echo "Auto-shutdown: 120 min after launch"
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $VM_VARIANTE_A --zone=$ZONE --command='tail -30 variante_a_seed42.log'"
echo "  gcloud compute ssh $VM_VARIANTE_B --zone=$ZONE --command='tail -30 variante_b_seed42.log'"
echo "  gcloud compute ssh $VM_VARIANTE_C1 --zone=$ZONE --command='tail -30 variante_c1_seed42.log'"
echo "  gcloud compute ssh $VM_VARIANTE_C3 --zone=$ZONE --command='tail -30 variante_c3_seed42.log'"
echo ""
echo "Download results (after completion):"
echo "  ./collect_4variants_seed42.sh"
echo ""
echo "═══════════════════════════════════════════════════════════"
