# GCP Deployment Guide for SMOTE-LLM

This guide covers deploying SMOTE-LLM experiments on Google Cloud Platform (GCP) for scalable, parallel execution.

## Overview

**Why GCP?**
- Run 25 experiments in parallel (5 VMs × 5 seeds each)
- Complete in ~5 hours instead of ~125 hours locally
- Auto-shutdown saves costs (~$10 total vs $288/month if left running)
- Statistical robustness with multi-seed validation

## Prerequisites

### 1. GCP Account Setup

```bash
# Install gcloud CLI
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable compute.googleapis.com
```

### 3. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

## Phase A: 25-Seed Robustness Test

### Architecture

```
Local Machine
    │
    ├─ launch_25seeds.sh
    │
    ├──> GCP us-central1-a
         │
         ├─ vm-batch1 (n1-standard-4)
         │  └─ Seeds: 42, 100, 123, 456, 789
         │
         ├─ vm-batch2 (n1-standard-4)
         │  └─ Seeds: 111, 222, 333, 444, 555
         │
         ├─ vm-batch3 (n1-standard-4)
         │  └─ Seeds: 1000, 2000, 3000, 4000, 5000
         │
         ├─ vm-batch4 (n1-standard-4)
         │  └─ Seeds: 7, 13, 21, 37, 101
         │
         └─ vm-batch5 (n1-standard-4)
            └─ Seeds: 1234, 2345, 3456, 4567, 5678
```

### VM Specifications

- **Machine Type**: n1-standard-4
  - 4 vCPUs
  - 15 GB RAM
  - CPU-only (no GPU needed for Phase A)
- **OS**: Ubuntu 22.04 LTS
- **Boot Disk**: 30 GB SSD
- **Zone**: us-central1-a (Iowa, USA)

### Cost Breakdown

| Resource | Rate | Duration | Cost |
|----------|------|----------|------|
| 5x n1-standard-4 | $0.20/hr each | 5 hours | $5.00 |
| 5x Boot disks (30GB SSD) | $0.17/GB/month | ~5 hours | $0.04 |
| Network egress | ~$0.12/GB | ~2 GB | $0.24 |
| OpenAI API calls | ~$0.50/experiment | 25 runs | $12.50 |
| **Total** | | | **~$18** |

**Note**: If VMs auto-shutdown properly, ongoing storage cost is only ~$0.01/hour for stopped disks.

## Step-by-Step Deployment

### Step 1: Prepare Local Environment

```bash
cd phase_a/gcp

# Verify scripts exist
ls -l
# Should see:
# - launch_25seeds.sh
# - monitor.sh
# - collect_results.sh

# Verify API key is set
echo $OPENAI_API_KEY
```

### Step 2: Launch 25 Experiments

```bash
# Set API key
export OPENAI_API_KEY='your-key-here'

# Launch (will prompt for confirmation)
./launch_25seeds.sh
```

**What happens**:
1. ✓ Checks for existing VMs (deletes if found)
2. ✓ Creates 5 VMs in parallel
3. ✓ Waits for Python/venv installation (~2 min)
4. ✓ Uploads files to each VM:
   - runner_phase2.py
   - MBTI_500.csv
   - run_all_seeds.sh (batch script)
5. ✓ Installs Python dependencies in venv
6. ✓ Launches experiments in background
7. ✓ Verifies all 5 processes are running

**Expected output**:
```
╔════════════════════════════════════════════════════════╗
║   Fase A - 25 Seeds Robustness Test (5 VMs × 5 seeds) ║
╚════════════════════════════════════════════════════════╝

Configuration:
  Total VMs: 5
  Seeds per VM: 5
  Total Experiments: 25
  Machine Type: n1-standard-4 (4 vCPUs, 15GB RAM)
  Dataset: MBTI_500.csv (106K samples)
  Tiempo estimado: ~5 horas en paralelo
  Costo estimado: ~$10 USD

¿Continuar con el lanzamiento? (y/n)
```

### Step 3: Monitor Progress

In a new terminal window:

```bash
cd phase_a/gcp
./monitor.sh
```

**Output example**:
```
════════════════════════════════════════════════════════
  Fase A - 25 Seeds Monitor (5 VMs × 5 seeds)
════════════════════════════════════════════════════════

Thu Jan 15 14:32:11 UTC 2025

─────────────────────────────────────────────────────────
[vm-batch1] Seeds: 42 100 123 456 789
─────────────────────────────────────────────────────────
Completed: 2/5
Status: ✓ RUNNING (current seed: 123)

Last log lines:
  [Step 3/6] Training baseline classifier...
  Baseline macro F1: 0.5234
  [Step 4/6] Generating synthetic data for 16 classes...

─────────────────────────────────────────────────────────
[vm-batch2] Seeds: 111 222 333 444 555
─────────────────────────────────────────────────────────
Completed: 3/5
Status: ✓ RUNNING (current seed: 444)

Last log lines:
  [Step 5/6] Training augmented classifier...
  Augmented macro F1: 0.5286
  F1 improvement: +1.00%
```

**Monitoring features**:
- Real-time status of all 5 VMs
- Completion count per VM (X/5 seeds done)
- Current running seed
- Last 3 log lines
- Updates every 60 seconds

**To exit**: Press `Ctrl+C`

### Step 4: Check Individual VM

```bash
# SSH into a specific VM
gcloud compute ssh vm-batch1 --zone=us-central1-a

# Check process
ps aux | grep python3

# View log
tail -30 LaptopRuns/nohup_batch1.log

# Check completed seeds
ls -lh LaptopRuns/phaseA_seed*_metrics.json

# Exit VM
exit
```

### Step 5: Collect Results (~5 hours later)

```bash
cd phase_a/gcp
./collect_results.sh
```

**What happens**:
1. Downloads all results from each VM:
   - phaseA_seed{SEED}_metrics.json
   - phaseA_seed{SEED}_synthetic.csv
   - phaseA_seed{SEED}_augmented.csv
   - nohup logs
2. If VM is stopped, temporarily starts it for download, then re-stops it
3. Runs statistical analysis:
   - Mean/median/std of F1 deltas
   - 95% confidence interval
   - Statistical significance test (t-test)
   - Success rate (% seeds improved)
4. Saves summary to `summary_25seeds.json`
5. Prompts to delete VMs (saves money)

**Expected output**:
```
═══════════════════════════════════════════════════════════
RESULTADOS FASE A - 25 SEEDS ANALYSIS
═══════════════════════════════════════════════════════════

Seeds encontrados: 25/25

─── BATCH1 (seeds: [42, 100, 123, 456, 789]) ───
  Seed   42: Baseline=0.5234, Aug=0.5286, Delta=+1.00%, Synth=847 ✓
  Seed  100: Baseline=0.5198, Aug=0.5253, Delta=+1.06%, Synth=892 ✓
  Seed  123: Baseline=0.5256, Aug=0.5304, Delta=+0.91%, Synth=814 ✓
  Seed  456: Baseline=0.5212, Aug=0.5269, Delta=+1.09%, Synth=878 ✓
  Seed  789: Baseline=0.5243, Aug=0.5295, Delta=+0.99%, Synth=823 ✓
  Batch mean: +1.01% ± 0.07%

[... similar for batch2-5 ...]

═══════════════════════════════════════════════════════════
ESTADÍSTICAS AGREGADAS (25 seeds)
═══════════════════════════════════════════════════════════

  Mean Delta:        +1.0000% ± 0.2500%
  Median Delta:      +0.9800%
  Min Delta:         +0.6500%
  Max Delta:         +1.3500%
  25th percentile:   +0.8500%
  75th percentile:   +1.1500%

  Mean Synthetics:   847.2 ± 52.3

  Success Rate:      24/25 seeds improved

  95% CI for mean:   [+0.8966%, +1.1034%]

  ✅ META ALCANZADA: +1.0000% ≥ +1.00% target

  Significancia estadística:
    t-statistic: 12.5678
    p-value: 0.000001
    ✅ Mejora es estadísticamente significativa (p < 0.05)

  ✓ Summary saved to summary_25seeds.json

═══════════════════════════════════════════════════════════

¿Deseas eliminar las 5 VMs para ahorrar costos? (y/n)
```

### Step 6: Cleanup

After collecting results, you have two options:

**Option 1: Delete VMs (Recommended)**
```bash
# When prompted by collect_results.sh
y

# Or manually
gcloud compute instances delete vm-batch{1,2,3,4,5} --zone=us-central1-a
```

**Option 2: Keep VMs (for debugging)**
- Cost: ~$0.01/hour per disk (stopped)
- To restart: `gcloud compute instances start vm-batch1 --zone=us-central1-a`
- Delete later: Same as Option 1

## Troubleshooting

### VMs Not Starting

**Error**: `VM creation failed`

**Solutions**:
```bash
# Check quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Try different zone
# Edit launch_25seeds.sh, change ZONE="us-central1-a" to "us-west1-b"

# Check billing is enabled
gcloud beta billing accounts list
```

### Experiments Not Running

**Check 1**: SSH into VM and verify process
```bash
gcloud compute ssh vm-batch1 --zone=us-central1-a
ps aux | grep python3

# If not running, check logs
tail -50 LaptopRuns/nohup_batch1.log
```

**Check 2**: Verify venv was created
```bash
cd LaptopRuns
ls -la .venv/
source .venv/bin/activate
python3 --version
pip list
```

**Check 3**: Test manually
```bash
cd LaptopRuns
source .venv/bin/activate
export OPENAI_API_KEY='your-key'
python3 runner_phase2.py --help
```

### Files Not Uploading

**Error**: `SCP failed`

**Solution**:
```bash
# Test SSH connection
gcloud compute ssh vm-batch1 --zone=us-central1-a --command="echo test"

# Check firewall rules
gcloud compute firewall-rules list

# Try manual upload
gcloud compute scp runner_phase2.py vm-batch1:~/
```

### High Costs

**Check running instances**:
```bash
gcloud compute instances list --filter='status=RUNNING'

# Stop all
gcloud compute instances stop vm-batch{1,2,3,4,5} --zone=us-central1-a

# Delete all
gcloud compute instances delete vm-batch{1,2,3,4,5} --zone=us-central1-a
```

**Monitor costs**:
- GCP Console → Billing → Reports
- Set budget alerts: Billing → Budgets & Alerts

## Advanced Configurations

### Using Preemptible VMs (70% Cost Reduction)

Edit `launch_25seeds.sh`:
```bash
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --preemptible \  # Add this line
    --image-family="$IMAGE_FAMILY" \
    ...
```

**Trade-off**: VMs may be terminated early (need restart logic)

### Using Different Machine Types

| Type | vCPUs | RAM | Cost/hr | Best For |
|------|-------|-----|---------|----------|
| n1-standard-2 | 2 | 7.5GB | $0.10 | Small datasets |
| n1-standard-4 | 4 | 15GB | $0.20 | Recommended |
| n1-standard-8 | 8 | 30GB | $0.40 | Large datasets |
| n1-highmem-4 | 4 | 26GB | $0.24 | Memory-intensive |

### Running Fewer Seeds

Edit `launch_25seeds.sh`, change VM_SEEDS:
```bash
# 5 VMs × 2 seeds = 10 total
VM_SEEDS[batch1]="42 100"
VM_SEEDS[batch2]="111 222"
VM_SEEDS[batch3]="1000 2000"
VM_SEEDS[batch4]="7 13"
VM_SEEDS[batch5]="1234 2345"
```

Also update `/tmp/run_batch_batch1.sh` etc.:
```bash
for SEED in 42 100; do  # Remove extra seeds
    ...
done
```

## Phase B on GCP (Single Seed with GPU)

Phase B can benefit from GPU acceleration. Here's how to run it:

### 1. Create GPU VM

```bash
gcloud compute instances create vm-phaseb-gpu \
    --zone=us-west1-b \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv
# Install CUDA drivers
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-2
'
```

**Cost**: ~$0.50/hour (4x more than CPU-only)

### 2. Setup and Run

```bash
# Upload files
gcloud compute scp phase_b/local_run_gpu.sh vm-phaseb-gpu:~/ --zone=us-west1-b
gcloud compute scp core/runner_phase2.py vm-phaseb-gpu:~/ --zone=us-west1-b
gcloud compute scp MBTI_500.csv vm-phaseb-gpu:~/ --zone=us-west1-b

# SSH in
gcloud compute ssh vm-phaseb-gpu --zone=us-west1-b

# On VM
cd ~
chmod +x local_run_gpu.sh
export OPENAI_API_KEY='your-key'
./local_run_gpu.sh MBTI_500.csv 42
```

**Time**: ~45 minutes (vs 2-3 hours on CPU)

### 3. Download Results

```bash
gcloud compute scp vm-phaseb-gpu:~/phaseB_seed42_* ./ --zone=us-west1-b
```

### 4. Cleanup

```bash
gcloud compute instances delete vm-phaseb-gpu --zone=us-west1-b
```

## Best Practices

1. **Always set a budget alert**
   - Console → Billing → Budgets
   - Alert at 50%, 90%, 100% of monthly budget

2. **Tag your resources**
   ```bash
   --labels=project=smote-llm,phase=a,experiment=25seeds
   ```

3. **Use startup scripts** for automated setup

4. **Enable auto-shutdown** in batch scripts (already included)

5. **Monitor with Cloud Console**
   - Compute Engine → VM instances
   - Check CPU/Memory usage

6. **Keep logs**
   - Download nohup logs before deleting VMs
   - Store in results directory

7. **Version control scripts**
   - Commit changes to launch_25seeds.sh
   - Track which version produced which results

## FAQ

**Q: Can I pause experiments?**
A: Yes, stop the VM: `gcloud compute instances stop vm-batch1 --zone=us-central1-a`. Restart with `start`. Progress is saved.

**Q: How do I run only 1 seed?**
A: Use `phase_a/local_run.sh` locally instead of GCP.

**Q: Can I use a different cloud provider?**
A: Yes, adapt scripts for AWS EC2 or Azure VMs. Main changes: VM creation commands and SSH access.

**Q: What if an experiment fails mid-run?**
A: Check logs, fix issue, relaunch that specific seed. Results are independent.

**Q: How do I share results with team?**
A: Upload `phaseA_25seeds_results/` to Google Drive or GitHub releases.

## Summary Checklist

Before launching:
- [ ] OpenAI API key set
- [ ] gcloud CLI installed and authenticated
- [ ] Project set and billing enabled
- [ ] Budget alerts configured
- [ ] Scripts reviewed and paths verified

During execution:
- [ ] Launch script completed successfully
- [ ] Monitor shows all 5 VMs running
- [ ] Periodically check logs for errors

After completion:
- [ ] Results downloaded (summary_25seeds.json)
- [ ] Statistical analysis reviewed
- [ ] VMs deleted to stop charges
- [ ] Results backed up

## Resources

- [GCP Compute Pricing](https://cloud.google.com/compute/pricing)
- [gcloud CLI Reference](https://cloud.google.com/sdk/gcloud/reference)
- [Ubuntu Startup Scripts](https://cloud.google.com/compute/docs/instances/startup-scripts/linux)
- [OpenAI Pricing](https://openai.com/pricing)

For more help: See [README.md](README.md) or open an issue.
