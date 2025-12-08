#!/bin/bash
# Phase C GCP Launcher - Single Seed Test with GPU
# Auto-shutdown after completion

set -e

# Configuration
VM_NAME="vm-phasec-test"
ZONE="us-west1-b"  # GPU available
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15GB RAM
ACCELERATOR="type=nvidia-tesla-t4,count=1"  # T4 GPU (~$0.35/hr)
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="50GB"  # Larger for GPU drivers
SEED=42

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║   Phase C - GCP GPU Test (1 Seed with Auto-Shutdown)  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: OPENAI_API_KEY not set${NC}"
    echo "Please run: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "Configuration:"
echo "  VM Name:      $VM_NAME"
echo "  Zone:         $ZONE"
echo "  Machine:      $MACHINE_TYPE"
echo "  GPU:          NVIDIA Tesla T4"
echo "  Seed:         $SEED"
echo "  Dataset:      MBTI_500.csv (331 MB)"
echo ""
echo "Estimated Cost:"
echo "  Compute:      ~\$0.20/hr × 1 hour = \$0.20"
echo "  GPU:          ~\$0.35/hr × 1 hour = \$0.35"
echo "  OpenAI API:   ~\$0.50"
echo "  Total:        ~\$1.05"
echo ""
echo "Runtime: ~45-60 minutes with GPU"
echo ""
echo -e "${YELLOW}⚠️  VM will auto-shutdown after completion${NC}"
echo ""

read -p "Continue with launch? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${YELLOW}⚠️  VM $VM_NAME already exists${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing VM..."
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
        echo -e "${GREEN}✓ VM deleted${NC}"
    else
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}[1/5] Creating GCP VM with GPU...${NC}"

# Create VM with GPU and startup script
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCELERATOR" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script='#!/bin/bash
# Install Python and basic dependencies
apt-get update
apt-get install -y python3-pip python3-venv git

# Install CUDA drivers for GPU (takes ~2-3 min)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing CUDA drivers..."
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update
    apt-get -y install cuda-toolkit-12-2 cuda-drivers
fi

echo "Startup script completed"
' \
    --labels=project=smote-llm,phase=c,experiment=adaptive-temp \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo -e "${GREEN}✓ VM created${NC}"

echo ""
echo -e "${BLUE}[2/5] Waiting for VM to be ready (30 seconds)...${NC}"
sleep 30

echo ""
echo -e "${BLUE}[3/5] Waiting for CUDA installation to complete...${NC}"
echo "This takes ~2-3 minutes. Checking every 30 seconds..."

CUDA_READY=false
MAX_WAIT=300  # 5 minutes
ELAPSED=0

while [ "$CUDA_READY" = false ] && [ $ELAPSED -lt $MAX_WAIT ]; do
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="nvidia-smi" &>/dev/null; then
        CUDA_READY=true
        echo -e "${GREEN}✓ CUDA drivers installed${NC}"
    else
        echo -n "."
        sleep 30
        ELAPSED=$((ELAPSED + 30))
    fi
done

if [ "$CUDA_READY" = false ]; then
    echo -e "${YELLOW}⚠️  CUDA installation took longer than expected${NC}"
    echo "Continuing anyway (will use CPU if CUDA not ready)"
fi

echo ""
echo -e "${BLUE}[4/5] Uploading files to VM...${NC}"

# Create remote directory
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p PhaseC"

# Upload core modules (all 9)
echo "  Uploading core modules..."
for module in runner_phase2.py ensemble_anchor_selector.py contamination_aware_filter.py \
              enhanced_quality_gate.py anchor_quality_improvements.py quality_gate_predictor.py \
              mbti_class_descriptions.py adversarial_discriminator.py multi_seed_ensemble.py; do
    gcloud compute scp "../../core/$module" "$VM_NAME:~/PhaseC/" --zone="$ZONE" &
done
wait

# Upload dataset
echo "  Uploading dataset (331 MB, ~30 seconds)..."
gcloud compute scp "../../MBTI_500.csv" "$VM_NAME:~/PhaseC/" --zone="$ZONE"

# Create and upload execution script
cat > /tmp/run_phaseC_remote.sh << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

cd ~/PhaseC

echo "════════════════════════════════════════════════════════"
echo "  Phase C - Adaptive Temperature Test (Seed 42)"
echo "════════════════════════════════════════════════════════"
echo ""
date
echo ""

# Create venv
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "[2/6] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scikit-learn sentence-transformers openai python-dotenv scipy torch

# Check GPU
echo ""
echo "[3/6] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
    BATCH_SIZE=128
    echo "✅ Using GPU (CUDA)"
else
    echo "⚠️  GPU not available, using CPU"
    DEVICE="cpu"
    BATCH_SIZE=32
fi

echo ""
echo "[4/6] Phase C configuration:"
echo "  • Adaptive Temperature: ENABLED"
echo "    - HIGH F1 (≥0.45): temp=0.3"
echo "    - MID F1 (0.20-0.45): temp=0.5 ← KEY FIX"
echo "    - LOW F1 (<0.20): temp=0.8"
echo "  • All Phase A quality mechanisms"
echo "  • All Phase B adaptive weighting"
echo ""

# Run experiment
echo "[5/6] Running Phase C experiment..."
echo ""

python3 runner_phase2.py \
    --data-path MBTI_500.csv \
    --test-size 0.2 \
    --random-seed 42 \
    \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device "$DEVICE" \
    --embedding-batch-size "$BATCH_SIZE" \
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
    --anchor-quality-threshold 0.50 \
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
    --f1-budget-thresholds 0.35 0.20 \
    --f1-budget-multipliers 30 70 100 \
    \
    --enable-adaptive-weighting \
    --synthetic-weight 0.5 \
    --synthetic-weight-mode adaptive \
    \
    --similarity-threshold 0.90 \
    --min-classifier-confidence 0.10 \
    --contamination-threshold 0.95 \
    \
    --output-prefix "phaseC_seed42"

echo ""
echo "[6/6] Phase C completed!"
date
echo ""

# Show results summary
if [ -f "phaseC_seed42_metrics.json" ]; then
    echo "════════════════════════════════════════════════════════"
    echo "  RESULTS SUMMARY"
    echo "════════════════════════════════════════════════════════"
    echo ""

    python3 << 'PYTHON_SUMMARY'
import json
with open('phaseC_seed42_metrics.json') as f:
    data = json.load(f)

    print("Overall Performance:")
    baseline = data['baseline_macro_f1']
    augmented = data['augmented_macro_f1']
    delta = (augmented - baseline) * 100
    print(f"  Baseline:   {baseline:.4f}")
    print(f"  Augmented:  {augmented:.4f}")
    print(f"  Delta:      {delta:+.2f}%")
    print()

    print("MID-Tier Classes (F1 0.20-0.45):")
    mid_deltas = []
    for cls, m in data['per_class_metrics'].items():
        f1_base = m['baseline_f1']
        f1_aug = m['augmented_f1']
        if 0.20 <= f1_base < 0.45:
            delta_cls = (f1_aug - f1_base) * 100
            mid_deltas.append(delta_cls)
            print(f"  {cls:6s}: {f1_base:.3f} → {f1_aug:.3f} ({delta_cls:+.2f}%)")

    if mid_deltas:
        print(f"\n  MID-tier Mean: {sum(mid_deltas)/len(mid_deltas):+.2f}%")
        print(f"  Target: ≥ +0.10% (vs Phase B: -0.59%)")

        if sum(mid_deltas)/len(mid_deltas) >= 0.10:
            print("\n  ✅ SUCCESS: MID-tier target achieved!")
        elif sum(mid_deltas)/len(mid_deltas) >= -0.30:
            print("\n  ⚠️  PARTIAL: Improvement but below target")
            print("  → Consider adding Hardness-Aware Anchors (Week 2)")
        else:
            print("\n  ❌ NEEDS WORK: Limited improvement")
            print("  → Proceed to Phase 2 (Hardness-Aware Anchors)")
PYTHON_SUMMARY

    echo ""
    echo "════════════════════════════════════════════════════════"
fi

echo ""
echo "Auto-shutdown in 60 seconds..."
echo "Press Ctrl+C to cancel shutdown and keep VM running"
sleep 60

echo "Shutting down VM..."
sudo shutdown -h now
REMOTE_SCRIPT

chmod +x /tmp/run_phaseC_remote.sh
gcloud compute scp /tmp/run_phaseC_remote.sh "$VM_NAME:~/PhaseC/" --zone="$ZONE"

echo -e "${GREEN}✓ All files uploaded${NC}"

echo ""
echo -e "${BLUE}[5/5] Starting Phase C experiment...${NC}"

# Set OpenAI API key and run experiment in background
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    cd PhaseC
    nohup ./run_phaseC_remote.sh > phaseC_output.log 2>&1 &
    echo \$! > phaseC.pid
"

echo -e "${GREEN}✓ Phase C experiment launched${NC}"

echo ""
echo "════════════════════════════════════════════════════════"
echo -e "${GREEN}  GCP VM Successfully Launched!${NC}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "VM Details:"
echo "  Name:    $VM_NAME"
echo "  Zone:    $ZONE"
echo "  Status:  RUNNING"
echo ""
echo "Experiment Details:"
echo "  Phase:   C (Adaptive Temperature)"
echo "  Seed:    42"
echo "  Runtime: ~45-60 minutes (GPU)"
echo "  Auto-shutdown: ENABLED (60s after completion)"
echo ""
echo "Monitoring Commands:"
echo ""
echo "  # Check if experiment is still running"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='ps aux | grep python3'"
echo ""
echo "  # View live log (last 50 lines)"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -50 PhaseC/phaseC_output.log'"
echo ""
echo "  # View live log (follow mode)"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f PhaseC/phaseC_output.log'"
echo ""
echo "  # Check for adaptive temperature messages"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='grep \"🌡️\" PhaseC/phaseC_output.log'"
echo ""
echo "  # SSH into VM"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "Or use the monitor script:"
echo "  ./monitor_phaseC.sh"
echo ""
echo "After completion (~1 hour), download results with:"
echo "  ./collect_results_phaseC.sh"
echo ""
echo "════════════════════════════════════════════════════════"
