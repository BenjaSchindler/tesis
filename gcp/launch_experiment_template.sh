#!/bin/bash
# Generic Experiment Launcher for GCP
# =====================================
#
# This template can be adapted for any experiment configuration.
# It handles:
# - VM creation with GPU
# - File uploads (runner + modules + dataset)
# - Dependency installation
# - Batch script generation and execution
# - Auto-shutdown when experiments complete
#
# Usage:
#   1. Copy this file to your experiment directory
#   2. Modify the configuration variables below
#   3. Run: ./launch_experiment_template.sh

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFY THIS SECTION FOR YOUR EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

# Experiment metadata
EXPERIMENT_NAME="my-experiment"
SEEDS=(42 100 123)  # Seeds to run on each VM

# VM configuration
NUM_VMS=4
VM_PREFIX="vm-${EXPERIMENT_NAME}"
VMS=()
for i in $(seq 1 $NUM_VMS); do
    VMS+=("${VM_PREFIX}-${i}")
done

# Files to upload (relative to project root)
FILES_TO_UPLOAD=(
    "core/runner_phase2.py"
    "core/module1.py"
    "core/module2.py"
    "MBTI_500.csv"
)

# Python dependencies (space-separated)
PYTHON_DEPS="numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv"

# Base command-line arguments (shared by all VMs)
BASE_ARGS="--data-path MBTI_500.csv \
    --test-size 0.2 \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --device cuda \
    --embedding-batch-size 256 \
    --llm-model gpt-4o-mini"

# VM-specific arguments (array indexed by VM number)
# Example: Different models, parameters, or features per VM
declare -A VM_ARGS
VM_ARGS[1]="--use-feature-a"
VM_ARGS[2]="--use-feature-b"
VM_ARGS[3]="--use-feature-c"
VM_ARGS[4]="--use-feature-a --use-feature-b --use-feature-c"

# VM names/labels (for logging)
declare -A VM_LABELS
VM_LABELS[1]="Feature A Only"
VM_LABELS[2]="Feature B Only"
VM_LABELS[3]="Feature C Only"
VM_LABELS[4]="All Features"

# Output file prefix per VM
declare -A VM_PREFIXES
VM_PREFIXES[1]="exp_a"
VM_PREFIXES[2]="exp_b"
VM_PREFIXES[3]="exp_c"
VM_PREFIXES[4]="exp_d"

# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT LOGIC - DO NOT MODIFY UNLESS YOU KNOW WHAT YOU'RE DOING
# ═══════════════════════════════════════════════════════════════════════════

# Detect project root and load toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/gcp_toolkit.sh"

# Check API key
check_api_key || exit 1

# Display configuration
echo "═══════════════════════════════════════════════════════════"
echo "  Launching Experiment: $EXPERIMENT_NAME"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  VMs: ${#VMS[@]}"
echo "  Seeds per VM: ${#SEEDS[@]} (${SEEDS[*]})"
echo "  Total runs: $((${#VMS[@]} * ${#SEEDS[@]}))"
echo "  Zone: $GCP_ZONE"
echo "  Machine: $GCP_MACHINE_TYPE + $GCP_GPU_TYPE"
echo ""
echo "Variants:"
for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    echo "  VM $vm_num: ${VM_LABELS[$vm_num]}"
done
echo ""
echo "Estimated cost: ~\$$(echo "scale=2; ${#VMS[@]} * 0.55 * 2.5" | bc) (compute only)"
echo "Estimated time: ~2.5 hours"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Launch all ${#VMS[@]} VMs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Create VMs
echo ""
echo "Step 1: Creating ${#VMS[@]} GPU VMs..."
gcp_create_gpu_vms "${VMS[@]}"

# Step 2: Upload files
echo ""
echo "Step 2: Uploading files to all VMs..."
for vm in "${VMS[@]}"; do
    echo "  Uploading to $vm..."
    for file in "${FILES_TO_UPLOAD[@]}"; do
        gcloud compute scp "$PROJECT_ROOT/$file" "$vm":~/ --zone="$GCP_ZONE" --quiet &
    done
done
wait
log_success "All files uploaded"

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing Python dependencies..."
for vm in "${VMS[@]}"; do
    echo "  Installing on $vm..."
    gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="
        sudo pip3 install --upgrade pip --quiet
        sudo pip3 install $PYTHON_DEPS --quiet
        nvidia-smi > /dev/null 2>&1 && echo 'GPU OK' || echo 'No GPU'
    " 2>&1 | tail -3 &
done
wait
log_success "Dependencies installed on all VMs"

# Step 4: Generate and upload batch scripts
echo ""
echo "Step 4: Generating batch scripts for each VM..."

for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    vm="${VMS[$i]}"
    vm_label="${VM_LABELS[$vm_num]}"
    vm_prefix="${VM_PREFIXES[$vm_num]}"
    vm_args="${VM_ARGS[$vm_num]}"

    # Generate batch script
    cat > "/tmp/run_${vm_prefix}_seeds.sh" << EOFBATCH
#!/bin/bash
SEEDS=(${SEEDS[*]})
export OPENAI_API_KEY='${OPENAI_API_KEY}'

echo "════════════════════════════════════════════════════════"
echo "  $vm_label"
echo "  Seeds: \${SEEDS[@]}"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "\${SEEDS[@]}"; do
    echo "Starting seed \$seed at \$(date)"

    python3 -u runner_phase2.py \\
        $BASE_ARGS \\
        $vm_args \\
        --random-seed \$seed \\
        --synthetic-output ${vm_prefix}_seed\${seed}_synthetic.csv \\
        --metrics-output ${vm_prefix}_seed\${seed}_metrics.json \\
        > ${vm_prefix}_seed\${seed}.log 2>&1

    EXIT_CODE=\$?
    if [ \$EXIT_CODE -eq 0 ]; then
        echo "✓ Seed \$seed completed at \$(date)"
    else
        echo "✗ Seed \$seed FAILED with exit code \$EXIT_CODE"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Completed: \$((${#SEEDS[@]} - FAILED_SEEDS))/${#SEEDS[@]}"
echo "  Failed: \$FAILED_SEEDS/${#SEEDS[@]}"
echo "════════════════════════════════════════════════════════"

# Auto-shutdown
if [ \$FAILED_SEEDS -eq ${#SEEDS[@]} ]; then
    echo "⚠ All seeds failed. Shutting down immediately..."
    sudo shutdown -h now
else
    echo "✓ Shutting down in 2 minutes..."
    sudo shutdown -h +2
fi
EOFBATCH

    chmod +x "/tmp/run_${vm_prefix}_seeds.sh"

    # Upload batch script
    gcloud compute scp "/tmp/run_${vm_prefix}_seeds.sh" "$vm":~/ --zone="$GCP_ZONE" --quiet &
done
wait
log_success "Batch scripts uploaded"

# Step 5: Launch experiments
echo ""
echo "Step 5: Launching experiments on all VMs..."
echo ""

for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    vm="${VMS[$i]}"
    vm_prefix="${VM_PREFIXES[$vm_num]}"

    log_info "Starting $vm (${VM_LABELS[$vm_num]})..."
    gcloud compute ssh "$vm" --zone="$GCP_ZONE" --command="
        nohup ./run_${vm_prefix}_seeds.sh > run.log 2>&1 &
        sleep 1
        pgrep -f run_${vm_prefix} > /dev/null && echo 'RUNNING' || echo 'FAILED TO START'
    " 2>&1 | tail -1 &
done
wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All Experiments Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running:"
for i in "${!VMS[@]}"; do
    vm_num=$((i + 1))
    echo "  ${VMS[$i]}: ${VM_LABELS[$vm_num]}"
done
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh ${VMS[0]} --zone=$GCP_ZONE \\"
echo "    --command='tail -20 ${VM_PREFIXES[1]}_seed${SEEDS[0]}.log'"
echo ""
echo "Check VM status:"
echo "  gcloud compute instances list --filter='name~$VM_PREFIX'"
echo ""
echo "═══════════════════════════════════════════════════════════"
