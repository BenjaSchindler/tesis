#!/bin/bash
# GCP VM Toolkit for SMOTE-LLM Experiments
# =========================================
#
# Reusable functions for creating, managing, and running experiments on GCP
# with GPU support and automatic shutdown when experiments finish.
#
# Usage: source gcp_toolkit.sh
#
# IMPORTANT LESSONS LEARNED:
# - Use Deep Learning VM image (has NVIDIA drivers pre-installed)
# - Use sudo pip3 install (not venv - avoids path issues)
# - Auto-shutdown should trigger when experiment FINISHES, not on fixed timer

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Override these before sourcing if needed
# ═══════════════════════════════════════════════════════════════════════════

: "${GCP_ZONE:=us-central1-a}"
: "${GCP_MACHINE_TYPE:=n1-standard-4}"
: "${GCP_GPU_TYPE:=nvidia-tesla-t4}"
: "${GCP_GPU_COUNT:=1}"

# Deep Learning VM with CUDA pre-installed (CRITICAL: use this, not ubuntu-2204-lts)
: "${GCP_IMAGE_FAMILY:=pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
: "${GCP_IMAGE_PROJECT:=deeplearning-platform-release}"

# 100GB minimum required by Deep Learning VM image
: "${GCP_BOOT_DISK_SIZE:=100GB}"

# Project paths (relative to repo root)
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${RUNNER_PATH:=$PROJECT_ROOT/core/runner_phase2.py}"
: "${DATA_PATH:=$PROJECT_ROOT/MBTI_500.csv}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_api_key() {
    if [ -z "$OPENAI_API_KEY" ]; then
        log_error "OPENAI_API_KEY not set"
        echo "Run: export OPENAI_API_KEY='your-key'"
        return 1
    fi
    return 0
}

# ═══════════════════════════════════════════════════════════════════════════
# VM CREATION (Deep Learning VM with CUDA pre-installed)
# ═══════════════════════════════════════════════════════════════════════════

# Create a single GPU VM
gcp_create_gpu_vm() {
    local vm_name="$1"

    if [ -z "$vm_name" ]; then
        log_error "Usage: gcp_create_gpu_vm <vm_name>"
        return 1
    fi

    log_info "Creating GPU VM: $vm_name (Deep Learning VM with CUDA)"

    gcloud compute instances create "$vm_name" \
        --zone="$GCP_ZONE" \
        --machine-type="$GCP_MACHINE_TYPE" \
        --accelerator="type=$GCP_GPU_TYPE,count=$GCP_GPU_COUNT" \
        --image-family="$GCP_IMAGE_FAMILY" \
        --image-project="$GCP_IMAGE_PROJECT" \
        --boot-disk-size="$GCP_BOOT_DISK_SIZE" \
        --maintenance-policy=TERMINATE

    log_success "VM $vm_name created"
    log_info "Waiting 30s for VM to be ready..."
    sleep 30
}

# Create multiple VMs in parallel
gcp_create_gpu_vms() {
    local vms=("$@")

    log_info "Creating ${#vms[@]} VMs in parallel..."

    for vm_name in "${vms[@]}"; do
        log_info "Creating $vm_name..."
        gcloud compute instances create "$vm_name" \
            --zone="$GCP_ZONE" \
            --machine-type="$GCP_MACHINE_TYPE" \
            --accelerator="type=$GCP_GPU_TYPE,count=$GCP_GPU_COUNT" \
            --image-family="$GCP_IMAGE_FAMILY" \
            --image-project="$GCP_IMAGE_PROJECT" \
            --boot-disk-size="$GCP_BOOT_DISK_SIZE" \
            --maintenance-policy=TERMINATE &
    done

    wait
    log_success "All VMs created"
    log_info "Waiting 30s for VMs to be ready..."
    sleep 30
}

# ═══════════════════════════════════════════════════════════════════════════
# VM SETUP (sudo pip3 install - no venv issues)
# ═══════════════════════════════════════════════════════════════════════════

# Install Python dependencies globally
gcp_setup_vm() {
    local vm_name="$1"

    if [ -z "$vm_name" ]; then
        log_error "Usage: gcp_setup_vm <vm_name>"
        return 1
    fi

    log_info "Installing Python dependencies on $vm_name..."

    gcloud compute ssh "$vm_name" --zone="$GCP_ZONE" --command='
        # Upgrade pip first (fixes dependency resolution bugs)
        sudo pip3 install --upgrade pip

        # Install to global site-packages (avoids venv path issues)
        sudo pip3 install numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv

        # Verify GPU is available
        echo ""
        echo "=== GPU Status ==="
        nvidia-smi --query-gpu=name,driver_version --format=csv

        echo ""
        echo "=== Python packages ==="
        pip3 show pandas | grep -E "Name|Version|Location"
    '

    log_success "VM $vm_name setup complete"
}

# Setup multiple VMs in parallel
gcp_setup_vms() {
    local vms=("$@")

    log_info "Setting up ${#vms[@]} VMs..."

    for vm_name in "${vms[@]}"; do
        gcloud compute ssh "$vm_name" --zone="$GCP_ZONE" --command='
            sudo pip3 install --upgrade pip
            sudo pip3 install numpy pandas scikit-learn scipy sentence-transformers openai tqdm python-dotenv
        ' &
    done

    wait
    log_success "All VMs setup complete"
}

# ═══════════════════════════════════════════════════════════════════════════
# FILE TRANSFER
# ═══════════════════════════════════════════════════════════════════════════

gcp_upload_files() {
    local vm_name="$1"
    local runner="${2:-$RUNNER_PATH}"
    local data="${3:-$DATA_PATH}"

    if [ -z "$vm_name" ]; then
        log_error "Usage: gcp_upload_files <vm_name> [runner_path] [data_path]"
        return 1
    fi

    log_info "Uploading files to $vm_name..."
    gcloud compute scp "$runner" "$vm_name":~/ --zone="$GCP_ZONE"
    gcloud compute scp "$data" "$vm_name":~/ --zone="$GCP_ZONE"
    log_success "Files uploaded to $vm_name"
}

gcp_upload_files_multi() {
    local vms=("$@")

    log_info "Uploading files to ${#vms[@]} VMs..."

    for vm_name in "${vms[@]}"; do
        {
            gcloud compute scp "$RUNNER_PATH" "$vm_name":~/ --zone="$GCP_ZONE"
            gcloud compute scp "$DATA_PATH" "$vm_name":~/ --zone="$GCP_ZONE"
        } &
    done

    wait
    log_success "Files uploaded to all VMs"
}

# Upload files for Phase D experiments (includes Phase D modules)
gcp_upload_phased_files() {
    local vms=("$@")

    log_info "Uploading Phase D files to ${#vms[@]} VMs..."

    # Detect project root (go up from gcp/)
    local UPLOAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    for vm_name in "${vms[@]}"; do
        {
            gcloud compute scp "$UPLOAD_ROOT/core/runner_phase2.py" "$vm_name":~/ --zone="$GCP_ZONE" --quiet
            gcloud compute scp "$UPLOAD_ROOT/core/mbti_confusers.py" "$vm_name":~/ --zone="$GCP_ZONE" --quiet
            gcloud compute scp "$UPLOAD_ROOT/core/focal_loss_training.py" "$vm_name":~/ --zone="$GCP_ZONE" --quiet
            gcloud compute scp "$UPLOAD_ROOT/core/two_stage_training.py" "$vm_name":~/ --zone="$GCP_ZONE" --quiet
            gcloud compute scp "$UPLOAD_ROOT/MBTI_500.csv" "$vm_name":~/ --zone="$GCP_ZONE" --quiet
        } &
    done

    wait
    log_success "Phase D files uploaded to all VMs"
}

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT EXECUTION (with proper auto-shutdown)
# ═══════════════════════════════════════════════════════════════════════════

# Launch experiment with auto-shutdown when finished
gcp_launch_experiment() {
    local vm_name="$1"
    local seed="$2"
    local prefix="$3"
    shift 3
    local extra_args="$*"

    if [ -z "$vm_name" ] || [ -z "$seed" ] || [ -z "$prefix" ]; then
        log_error "Usage: gcp_launch_experiment <vm_name> <seed> <output_prefix> [extra_args...]"
        return 1
    fi

    check_api_key || return 1

    log_info "Launching experiment on $vm_name (seed=$seed)..."

    gcloud compute ssh "$vm_name" --zone="$GCP_ZONE" --command="
        export OPENAI_API_KEY='$OPENAI_API_KEY'

        # Launch experiment in background
        nohup python3 -u runner_phase2.py \\
            --data-path MBTI_500.csv \\
            --test-size 0.2 \\
            --random-seed $seed \\
            --embedding-model sentence-transformers/all-mpnet-base-v2 \\
            --device cuda \\
            --embedding-batch-size 64 \\
            --llm-model gpt-4o-mini \\
            --temperature 1.0 \\
            --max-clusters 3 \\
            --prompts-per-cluster 3 \\
            --prompt-mode mix \\
            --use-ensemble-selection \\
            --use-val-gating \\
            --val-size 0.15 \\
            --val-tolerance 0.02 \\
            --enable-anchor-gate \\
            --anchor-quality-threshold 0.25 \\
            --purity-gate-threshold 0.025 \\
            --enable-anchor-selection \\
            --anchor-selection-ratio 0.8 \\
            --anchor-outlier-threshold 1.5 \\
            --use-class-description \\
            --use-f1-budget-scaling \\
            --f1-budget-thresholds 0.40 0.20 \\
            --f1-budget-multipliers 30 70 100 \\
            --synthetic-weight 0.5 \\
            --similarity-threshold 0.90 \\
            --min-classifier-confidence 0.10 \\
            --contamination-threshold 0.95 \\
            --synthetic-output ${prefix}_seed${seed}_synthetic.csv \\
            --metrics-output ${prefix}_seed${seed}_metrics.json \\
            $extra_args \\
            > ${prefix}_seed${seed}.log 2>&1 &

        sleep 2

        # Start auto-shutdown watcher (shuts down when python finishes)
        nohup bash -c 'while pgrep -f runner_phase2.py > /dev/null; do sleep 30; done; echo \"Experiment finished at \$(date)\" >> ${prefix}_seed${seed}.log; sudo shutdown -h now' > /dev/null 2>&1 &

        # Verify process started
        if pgrep -f 'python3.*runner_phase2' > /dev/null; then
            echo 'Experiment launched!'
            echo 'Auto-shutdown watcher active (VM will stop when experiment finishes)'
        else
            echo 'ERROR: Experiment failed to start'
            exit 1
        fi
    "

    log_success "Experiment launched on $vm_name"
}

# ═══════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════

gcp_check_status() {
    local vm_name="$1"

    if [ -z "$vm_name" ]; then
        log_error "Usage: gcp_check_status <vm_name>"
        return 1
    fi

    local count
    count=$(gcloud compute ssh "$vm_name" --zone="$GCP_ZONE" --command="pgrep -fc 'python3.*runner_phase2' || echo 0" 2>/dev/null)

    if [ "$count" -gt 0 ]; then
        echo -e "${GREEN}RUNNING${NC}"
        return 0
    else
        echo -e "${YELLOW}COMPLETED/STOPPED${NC}"
        return 1
    fi
}

gcp_get_logs() {
    local vm_name="$1"
    local log_file="$2"
    local lines="${3:-30}"

    gcloud compute ssh "$vm_name" --zone="$GCP_ZONE" --command="tail -$lines $log_file" 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS & CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

gcp_download_results() {
    local vm_name="$1"
    local local_dir="$2"

    if [ -z "$vm_name" ] || [ -z "$local_dir" ]; then
        log_error "Usage: gcp_download_results <vm_name> <local_dir>"
        return 1
    fi

    mkdir -p "$local_dir"

    log_info "Downloading results from $vm_name..."
    gcloud compute scp "$vm_name":~/*_metrics.json "$local_dir/" --zone="$GCP_ZONE" 2>/dev/null || true
    gcloud compute scp "$vm_name":~/*_synthetic.csv "$local_dir/" --zone="$GCP_ZONE" 2>/dev/null || true
    gcloud compute scp "$vm_name":~/*.log "$local_dir/" --zone="$GCP_ZONE" 2>/dev/null || true
    log_success "Results downloaded to $local_dir"
}

gcp_delete_vm() {
    local vm_name="$1"
    log_info "Deleting VM: $vm_name"
    gcloud compute instances delete "$vm_name" --zone="$GCP_ZONE" --quiet
    log_success "VM $vm_name deleted"
}

gcp_delete_vms() {
    local vms=("$@")
    log_info "Deleting ${#vms[@]} VMs..."
    gcloud compute instances delete "${vms[@]}" --zone="$GCP_ZONE" --quiet
    log_success "All VMs deleted"
}

gcp_list_vms() {
    gcloud compute instances list --format='table(name,zone,status,machineType)'
}

# ═══════════════════════════════════════════════════════════════════════════
# QUICK LAUNCH (all-in-one)
# ═══════════════════════════════════════════════════════════════════════════

gcp_quick_launch() {
    local vm_name="$1"
    local seed="$2"
    local prefix="$3"
    shift 3
    local extra_args="$*"

    check_api_key || return 1

    echo "═══════════════════════════════════════════════════════════"
    echo "  Quick Launch: $vm_name"
    echo "═══════════════════════════════════════════════════════════"

    gcp_create_gpu_vm "$vm_name"
    gcp_upload_files "$vm_name"
    gcp_setup_vm "$vm_name"
    gcp_launch_experiment "$vm_name" "$seed" "$prefix" "$extra_args"

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Experiment running!"
    echo "  VM will auto-shutdown when experiment finishes."
    echo "═══════════════════════════════════════════════════════════"
}

gcp_toolkit_help() {
    cat << 'EOF'
═══════════════════════════════════════════════════════════
  GCP VM Toolkit for SMOTE-LLM
═══════════════════════════════════════════════════════════

QUICK START:
  source gcp/gcp_toolkit.sh
  export OPENAI_API_KEY='sk-...'
  gcp_quick_launch my-vm 42 experiment

VM MANAGEMENT:
  gcp_create_gpu_vm <name>     Create GPU VM (Deep Learning image)
  gcp_create_gpu_vms <names>   Create multiple VMs in parallel
  gcp_setup_vm <name>          Install Python dependencies
  gcp_upload_files <name>      Upload runner and data
  gcp_delete_vm <name>         Delete VM
  gcp_list_vms                 List all VMs

EXPERIMENTS:
  gcp_launch_experiment <vm> <seed> <prefix> [args]
  gcp_quick_launch <vm> <seed> <prefix> [args]

MONITORING:
  gcp_check_status <name>      Check if running
  gcp_get_logs <name> <file>   Get log output

RESULTS:
  gcp_download_results <vm> <dir>

KEY FEATURES:
  - Uses Deep Learning VM (CUDA pre-installed)
  - Auto-shutdown when experiment finishes (not fixed timer)
  - Global pip install (no venv path issues)
EOF
}

# Export functions
export -f log_info log_success log_warning log_error check_api_key
export -f gcp_create_gpu_vm gcp_create_gpu_vms
export -f gcp_setup_vm gcp_setup_vms
export -f gcp_upload_files gcp_upload_files_multi gcp_upload_phased_files
export -f gcp_launch_experiment
export -f gcp_check_status gcp_get_logs
export -f gcp_download_results
export -f gcp_delete_vm gcp_delete_vms gcp_list_vms
export -f gcp_quick_launch gcp_toolkit_help

log_info "GCP Toolkit loaded. Run 'gcp_toolkit_help' for usage."
