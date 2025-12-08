#!/bin/bash
# Launch Multiple Robustness Variants in Parallel
#
# This script launches multiple experiments with different configurations
# to test robustness approaches for reducing LLM stochasticity.
#
# Usage:
#   ./launch_variants.sh <seed>
#
# Variants:
#   A: Higher budget (5 prompts/cluster)
#   B: Multi-temperature ensemble [0.7, 1.0, 1.3]
#   C1: GPT-5-mini (terse - no reasoning, low verbosity)
#   C3: GPT-5-mini (thorough - minimal reasoning, high verbosity)

set -e

# Load toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gcp_toolkit.sh"

# Configuration
SEED="${1:-42}"
AUTO_SHUTDOWN="${GCP_AUTO_SHUTDOWN_MINUTES:-150}"

# VM names
VM_A="vm-variante-a-seed${SEED}"
VM_B="vm-variante-b-seed${SEED}"
VM_C1="vm-variante-c1-seed${SEED}"
VM_C3="vm-variante-c3-seed${SEED}"

ALL_VMS=("$VM_A" "$VM_B" "$VM_C1" "$VM_C3")

# Check API key
check_api_key || exit 1

echo "═══════════════════════════════════════════════════════════"
echo "  Launching 4 Robustness Variants (Seed $SEED)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Variants:"
echo "  A:  5 prompts/cluster (gpt-4o-mini)"
echo "  B:  Multi-temp ensemble [0.7, 1.0, 1.3] (gpt-4o-mini)"
echo "  C1: GPT-5-mini (no reasoning, low verbosity)"
echo "  C3: GPT-5-mini (minimal reasoning, high verbosity)"
echo ""
echo "Configuration:"
echo "  Zone: $GCP_ZONE"
echo "  Machine: $GCP_MACHINE_TYPE + $GCP_GPU_TYPE"
echo "  Auto-shutdown: $AUTO_SHUTDOWN min"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

read -p "Launch all 4 variants? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Creating 4 GPU VMs..."
gcp_create_gpu_vms "${ALL_VMS[@]}"

echo ""
echo "Step 2: Uploading files..."
gcp_upload_files_multi "${ALL_VMS[@]}"

echo ""
echo "Step 3: Setting up Python environments..."
gcp_setup_vms "${ALL_VMS[@]}"

echo ""
echo "Step 4: Launching experiments..."
echo ""

# Launch Variante A: 5 prompts/cluster
echo "Launching Variante A (5 prompts/cluster)..."
gcloud compute ssh "$VM_A" --zone="$GCP_ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \\
        --data-path MBTI_500.csv \\
        --test-size 0.2 \\
        --random-seed $SEED \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --device cuda \\
        --embedding-batch-size 64 \\
        --llm-model gpt-4o-mini \\
        --temperature 1.0 \\
        --max-clusters 3 \\
        --prompts-per-cluster 5 \\
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
        --synthetic-output variante_a_seed${SEED}_synthetic.csv \\
        --metrics-output variante_a_seed${SEED}_metrics.json \\
        > variante_a_seed${SEED}.log 2>&1 &

    sleep 2
    sudo shutdown -h +$AUTO_SHUTDOWN
    echo 'Variante A launched!'
" &

# Launch Variante B: Multi-temperature ensemble
echo "Launching Variante B (multi-temp ensemble)..."
gcloud compute ssh "$VM_B" --zone="$GCP_ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \\
        --data-path MBTI_500.csv \\
        --test-size 0.2 \\
        --random-seed $SEED \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --device cuda \\
        --embedding-batch-size 64 \\
        --llm-model gpt-4o-mini \\
        --multi-temperature-ensemble \\
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
        --synthetic-output variante_b_seed${SEED}_synthetic.csv \\
        --metrics-output variante_b_seed${SEED}_metrics.json \\
        > variante_b_seed${SEED}.log 2>&1 &

    sleep 2
    sudo shutdown -h +$AUTO_SHUTDOWN
    echo 'Variante B launched!'
" &

# Launch Variante C1: GPT-5-mini (terse)
echo "Launching Variante C1 (GPT-5-mini terse)..."
gcloud compute ssh "$VM_C1" --zone="$GCP_ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \\
        --data-path MBTI_500.csv \\
        --test-size 0.2 \\
        --random-seed $SEED \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --device cuda \\
        --embedding-batch-size 64 \\
        --llm-model gpt-5-mini-2025-08-07 \\
        --reasoning-effort none \\
        --output-verbosity low \\
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
        --synthetic-output variante_c1_seed${SEED}_synthetic.csv \\
        --metrics-output variante_c1_seed${SEED}_metrics.json \\
        > variante_c1_seed${SEED}.log 2>&1 &

    sleep 2
    sudo shutdown -h +$AUTO_SHUTDOWN
    echo 'Variante C1 launched!'
" &

# Launch Variante C3: GPT-5-mini (thorough)
echo "Launching Variante C3 (GPT-5-mini thorough)..."
gcloud compute ssh "$VM_C3" --zone="$GCP_ZONE" --command="
    export OPENAI_API_KEY='$OPENAI_API_KEY'
    source venv/bin/activate

    nohup python3 -u runner_phase2.py \\
        --data-path MBTI_500.csv \\
        --test-size 0.2 \\
        --random-seed $SEED \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --device cuda \\
        --embedding-batch-size 64 \\
        --llm-model gpt-5-mini-2025-08-07 \\
        --reasoning-effort minimal \\
        --output-verbosity high \\
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
        --synthetic-output variante_c3_seed${SEED}_synthetic.csv \\
        --metrics-output variante_c3_seed${SEED}_metrics.json \\
        > variante_c3_seed${SEED}.log 2>&1 &

    sleep 2
    sudo shutdown -h +$AUTO_SHUTDOWN
    echo 'Variante C3 launched!'
" &

wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All 4 Variants Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running:"
echo "  A:  $VM_A (5 prompts/cluster)"
echo "  B:  $VM_B (multi-temp ensemble)"
echo "  C1: $VM_C1 (GPT-5-mini terse)"
echo "  C3: $VM_C3 (GPT-5-mini thorough)"
echo ""
echo "Estimated completion: 45-80 minutes"
echo "Auto-shutdown: $AUTO_SHUTDOWN minutes"
echo ""
echo "Monitor:"
for vm in "${ALL_VMS[@]}"; do
    variant=$(echo "$vm" | grep -oP 'variante-\K[^-]+')
    echo "  gcloud compute ssh $vm --zone=$GCP_ZONE --command='tail -20 variante_${variant}_seed${SEED}.log'"
done
echo ""
echo "Collect results:"
echo "  ./collect_variants.sh $SEED"
echo ""
