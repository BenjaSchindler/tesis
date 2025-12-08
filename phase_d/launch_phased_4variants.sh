#!/bin/bash
# Launch 4 Phase D Variants on GCP
# ===================================
#
# Variants tested:
#   A: Contrastive Prompting only (gpt-4o-mini)
#   B: Focal Loss only (gpt-4o-mini)
#   C: Two-Stage Training only (gpt-4o-mini)
#   D: ALL Phase D features + GPT-5-mini
#
# Each variant runs 3 seeds sequentially: 42, 100, 123
# Total: 12 experiments (4 variants × 3 seeds)
#
# Usage: ./launch_phased_4variants.sh
#

set -e

# Load GCP toolkit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/gcp/gcp_toolkit.sh"

# Configuration
SEEDS=(42 100 123)

# VM names
VM_A="vm-phased-a"
VM_B="vm-phased-b"
VM_C="vm-phased-c"
VM_D="vm-phased-d"

ALL_VMS=("$VM_A" "$VM_B" "$VM_C" "$VM_D")

# Check API key
check_api_key || exit 1

echo "═══════════════════════════════════════════════════════════"
echo "  Phase D Controlled Experiment - 4 Variants"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Variants to test:"
echo "  A: Contrastive Prompting only"
echo "  B: Focal Loss only"
echo "  C: Two-Stage Training only"
echo "  D: ALL Phase D features + GPT-5-mini"
echo ""
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: 12 (4 variants × 3 seeds)"
echo ""
echo "Configuration:"
echo "  Zone: $GCP_ZONE"
echo "  Machine: $GCP_MACHINE_TYPE + $GCP_GPU_TYPE"
echo "  Image: Deep Learning VM (CUDA pre-installed)"
echo "  Auto-shutdown: When all 3 seeds finish"
echo ""
echo "Estimated cost: ~\$13.00 total"
echo "Estimated time: ~2.5 hours per VM"
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
echo "Step 2: Uploading Phase D files..."
gcp_upload_phased_files "${ALL_VMS[@]}"

echo ""
echo "Step 3: Installing dependencies..."
gcp_setup_vms "${ALL_VMS[@]}"

echo ""
echo "Step 4: Generating and uploading batch scripts..."
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Generate batch scripts for each variant
# ═══════════════════════════════════════════════════════════════════════════

# Base arguments (Phase A configuration)
BASE_ARGS="--data-path MBTI_500.csv \\
    --test-size 0.2 \\
    --embedding-model sentence-transformers/all-mpnet-base-v2 \\
    --device cuda \\
    --embedding-batch-size 256 \\
    --llm-model gpt-4o-mini \\
    --max-clusters 3 \\
    --prompts-per-cluster 3 \\
    --prompt-mode mix \\
    --use-ensemble-selection \\
    --use-val-gating \\
    --val-size 0.15 \\
    --val-tolerance 0.02 \\
    --enable-anchor-gate \\
    --anchor-quality-threshold 0.30 \\
    --enable-anchor-selection \\
    --anchor-selection-ratio 0.8 \\
    --anchor-outlier-threshold 1.5 \\
    --use-class-description \\
    --use-f1-budget-scaling \\
    --f1-budget-thresholds 0.45 0.20 \\
    --f1-budget-multipliers 0.0 0.5 1.0 \\
    --synthetic-weight 0.5 \\
    --synthetic-weight-mode flat \\
    --similarity-threshold 0.90 \\
    --min-classifier-confidence 0.10 \\
    --contamination-threshold 0.95"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT A: Contrastive Prompting only
# ─────────────────────────────────────────────────────────────────────────────
log_info "Creating batch script for Variant A (Contrastive Prompting)..."
cat > /tmp/run_phased_a_seeds.sh << 'EOF'
#!/bin/bash
SEEDS=(42 100 123)
VARIANT="a"

echo "════════════════════════════════════════════════════════"
echo "  Variant A: Contrastive Prompting Only"
echo "  Seeds: ${SEEDS[@]}"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo "────────────────────────────────────────────────────────"
    echo "Starting seed $seed at $(date)"
    echo "────────────────────────────────────────────────────────"

    export OPENAI_API_KEY='__API_KEY_PLACEHOLDER__'

    python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $seed \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 256 \
        --llm-model gpt-4o-mini \
        --max-clusters 3 \
        --prompts-per-cluster 3 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --anchor-quality-threshold 0.30 \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --use-f1-budget-scaling \
        --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --use-contrastive-prompting \
        --use-contrastive-filter \
        --contrastive-top-k 2 \
        --synthetic-output phased_a_seed${seed}_synthetic.csv \
        --metrics-output phased_a_seed${seed}_metrics.json \
        > phased_a_seed${seed}.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $seed completed at $(date)"
    else
        echo "✗ Seed $seed FAILED with exit code $EXIT_CODE at $(date)"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Variant A - All seeds processed"
echo "  Completed: $((3 - FAILED_SEEDS))/3"
echo "  Failed: $FAILED_SEEDS/3"
echo "════════════════════════════════════════════════════════"

# Auto-shutdown after all seeds
if [ $FAILED_SEEDS -eq 3 ]; then
    echo "⚠ All seeds failed. Shutting down VM..."
    sudo shutdown -h now
else
    echo "✓ Shutting down VM in 2 minutes..."
    sudo shutdown -h +2
fi
EOF

# Replace API key placeholder
sed -i "s/__API_KEY_PLACEHOLDER__/$OPENAI_API_KEY/g" /tmp/run_phased_a_seeds.sh
chmod +x /tmp/run_phased_a_seeds.sh
gcloud compute scp /tmp/run_phased_a_seeds.sh "$VM_A":~/ --zone="$GCP_ZONE" --quiet
log_success "Variant A batch script uploaded"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT B: Focal Loss only
# ─────────────────────────────────────────────────────────────────────────────
log_info "Creating batch script for Variant B (Focal Loss)..."
cat > /tmp/run_phased_b_seeds.sh << 'EOF'
#!/bin/bash
SEEDS=(42 100 123)
VARIANT="b"

echo "════════════════════════════════════════════════════════"
echo "  Variant B: Focal Loss Only"
echo "  Seeds: ${SEEDS[@]}"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo "────────────────────────────────────────────────────────"
    echo "Starting seed $seed at $(date)"
    echo "────────────────────────────────────────────────────────"

    export OPENAI_API_KEY='__API_KEY_PLACEHOLDER__'

    python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $seed \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 256 \
        --llm-model gpt-4o-mini \
        --max-clusters 3 \
        --prompts-per-cluster 3 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --anchor-quality-threshold 0.30 \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --use-f1-budget-scaling \
        --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --use-focal-loss \
        --focal-gamma 2.0 \
        --focal-low-boost 2.0 \
        --focal-mid-boost 1.5 \
        --synthetic-output phased_b_seed${seed}_synthetic.csv \
        --metrics-output phased_b_seed${seed}_metrics.json \
        > phased_b_seed${seed}.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $seed completed at $(date)"
    else
        echo "✗ Seed $seed FAILED with exit code $EXIT_CODE at $(date)"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Variant B - All seeds processed"
echo "  Completed: $((3 - FAILED_SEEDS))/3"
echo "  Failed: $FAILED_SEEDS/3"
echo "════════════════════════════════════════════════════════"

# Auto-shutdown after all seeds
if [ $FAILED_SEEDS -eq 3 ]; then
    echo "⚠ All seeds failed. Shutting down VM..."
    sudo shutdown -h now
else
    echo "✓ Shutting down VM in 2 minutes..."
    sudo shutdown -h +2
fi
EOF

# Replace API key placeholder
sed -i "s/__API_KEY_PLACEHOLDER__/$OPENAI_API_KEY/g" /tmp/run_phased_b_seeds.sh
chmod +x /tmp/run_phased_b_seeds.sh
gcloud compute scp /tmp/run_phased_b_seeds.sh "$VM_B":~/ --zone="$GCP_ZONE" --quiet
log_success "Variant B batch script uploaded"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT C: Two-Stage Training only
# ─────────────────────────────────────────────────────────────────────────────
log_info "Creating batch script for Variant C (Two-Stage Training)..."
cat > /tmp/run_phased_c_seeds.sh << 'EOF'
#!/bin/bash
SEEDS=(42 100 123)
VARIANT="c"

echo "════════════════════════════════════════════════════════"
echo "  Variant C: Two-Stage Training Only"
echo "  Seeds: ${SEEDS[@]}"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo "────────────────────────────────────────────────────────"
    echo "Starting seed $seed at $(date)"
    echo "────────────────────────────────────────────────────────"

    export OPENAI_API_KEY='__API_KEY_PLACEHOLDER__'

    python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $seed \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 256 \
        --llm-model gpt-4o-mini \
        --max-clusters 3 \
        --prompts-per-cluster 3 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --anchor-quality-threshold 0.30 \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --use-f1-budget-scaling \
        --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --use-two-stage-training \
        --two-stage-confidence 0.7 \
        --two-stage-weight 0.5 \
        --synthetic-output phased_c_seed${seed}_synthetic.csv \
        --metrics-output phased_c_seed${seed}_metrics.json \
        > phased_c_seed${seed}.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $seed completed at $(date)"
    else
        echo "✗ Seed $seed FAILED with exit code $EXIT_CODE at $(date)"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Variant C - All seeds processed"
echo "  Completed: $((3 - FAILED_SEEDS))/3"
echo "  Failed: $FAILED_SEEDS/3"
echo "════════════════════════════════════════════════════════"

# Auto-shutdown after all seeds
if [ $FAILED_SEEDS -eq 3 ]; then
    echo "⚠ All seeds failed. Shutting down VM..."
    sudo shutdown -h now
else
    echo "✓ Shutting down VM in 2 minutes..."
    sudo shutdown -h +2
fi
EOF

# Replace API key placeholder
sed -i "s/__API_KEY_PLACEHOLDER__/$OPENAI_API_KEY/g" /tmp/run_phased_c_seeds.sh
chmod +x /tmp/run_phased_c_seeds.sh
gcloud compute scp /tmp/run_phased_c_seeds.sh "$VM_C":~/ --zone="$GCP_ZONE" --quiet
log_success "Variant C batch script uploaded"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT D: ALL Phase D features + GPT-5-mini
# ─────────────────────────────────────────────────────────────────────────────
log_info "Creating batch script for Variant D (Full Stack)..."
cat > /tmp/run_phased_d_seeds.sh << 'EOF'
#!/bin/bash
SEEDS=(42 100 123)
VARIANT="d"

echo "════════════════════════════════════════════════════════"
echo "  Variant D: Full Stack (ALL Phase D + GPT-5-mini)"
echo "  Seeds: ${SEEDS[@]}"
echo "════════════════════════════════════════════════════════"
echo ""

FAILED_SEEDS=0

for seed in "${SEEDS[@]}"; do
    echo "────────────────────────────────────────────────────────"
    echo "Starting seed $seed at $(date)"
    echo "────────────────────────────────────────────────────────"

    export OPENAI_API_KEY='__API_KEY_PLACEHOLDER__'

    python3 -u runner_phase2.py \
        --data-path MBTI_500.csv \
        --test-size 0.2 \
        --random-seed $seed \
        --embedding-model sentence-transformers/all-mpnet-base-v2 \
        --device cuda \
        --embedding-batch-size 256 \
        --llm-model gpt-5-mini-2025-08-07 \
        --reasoning-effort minimal \
        --output-verbosity high \
        --max-clusters 3 \
        --prompts-per-cluster 3 \
        --prompt-mode mix \
        --use-ensemble-selection \
        --use-val-gating \
        --val-size 0.15 \
        --val-tolerance 0.02 \
        --enable-anchor-gate \
        --anchor-quality-threshold 0.30 \
        --enable-anchor-selection \
        --anchor-selection-ratio 0.8 \
        --anchor-outlier-threshold 1.5 \
        --use-class-description \
        --use-f1-budget-scaling \
        --f1-budget-thresholds 0.45 0.20 \
        --f1-budget-multipliers 0.0 0.5 1.0 \
        --synthetic-weight 0.5 \
        --synthetic-weight-mode flat \
        --similarity-threshold 0.90 \
        --min-classifier-confidence 0.10 \
        --contamination-threshold 0.95 \
        --use-contrastive-prompting \
        --use-contrastive-filter \
        --contrastive-top-k 2 \
        --use-focal-loss \
        --focal-gamma 2.0 \
        --focal-low-boost 2.0 \
        --focal-mid-boost 1.5 \
        --use-two-stage-training \
        --two-stage-confidence 0.7 \
        --two-stage-weight 0.5 \
        --synthetic-output phased_d_seed${seed}_synthetic.csv \
        --metrics-output phased_d_seed${seed}_metrics.json \
        > phased_d_seed${seed}.log 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Seed $seed completed at $(date)"
    else
        echo "✗ Seed $seed FAILED with exit code $EXIT_CODE at $(date)"
        ((FAILED_SEEDS++))
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Variant D - All seeds processed"
echo "  Completed: $((3 - FAILED_SEEDS))/3"
echo "  Failed: $FAILED_SEEDS/3"
echo "════════════════════════════════════════════════════════"

# Auto-shutdown after all seeds
if [ $FAILED_SEEDS -eq 3 ]; then
    echo "⚠ All seeds failed. Shutting down VM..."
    sudo shutdown -h now
else
    echo "✓ Shutting down VM in 2 minutes..."
    sudo shutdown -h +2
fi
EOF

# Replace API key placeholder
sed -i "s/__API_KEY_PLACEHOLDER__/$OPENAI_API_KEY/g" /tmp/run_phased_d_seeds.sh
chmod +x /tmp/run_phased_d_seeds.sh
gcloud compute scp /tmp/run_phased_d_seeds.sh "$VM_D":~/ --zone="$GCP_ZONE" --quiet
log_success "Variant D batch script uploaded"

echo ""
echo "Step 5: Launching experiments on all 4 VMs..."
echo ""

# Launch experiments in parallel
log_info "Starting Variant A..."
gcloud compute ssh "$VM_A" --zone="$GCP_ZONE" --command="nohup ./run_phased_a_seeds.sh > run.log 2>&1 &" &

log_info "Starting Variant B..."
gcloud compute ssh "$VM_B" --zone="$GCP_ZONE" --command="nohup ./run_phased_b_seeds.sh > run.log 2>&1 &" &

log_info "Starting Variant C..."
gcloud compute ssh "$VM_C" --zone="$GCP_ZONE" --command="nohup ./run_phased_c_seeds.sh > run.log 2>&1 &" &

log_info "Starting Variant D..."
gcloud compute ssh "$VM_D" --zone="$GCP_ZONE" --command="nohup ./run_phased_d_seeds.sh > run.log 2>&1 &" &

wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All 4 Variants Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "VMs running:"
echo "  A: $VM_A (Contrastive Prompting)"
echo "  B: $VM_B (Focal Loss)"
echo "  C: $VM_C (Two-Stage Training)"
echo "  D: $VM_D (Full Stack)"
echo ""
echo "Each VM will run 3 seeds (42, 100, 123) sequentially."
echo "Auto-shutdown: VMs stop 2 minutes after all 3 seeds finish"
echo ""
echo "Monitor progress:"
echo "  cd phase_d && ./monitor_phased.sh"
echo ""
echo "Collect results (after ~2.5 hours):"
echo "  cd phase_d && ./collect_phased_results.sh"
echo ""
