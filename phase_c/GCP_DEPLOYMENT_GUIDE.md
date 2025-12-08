# Phase C v2.1 - GCP Multi-Seed Validation Guide

**Purpose:** Run 5-seed validation on GCP instead of local RTX 3070 (avoids GPU memory limitations)

**Why GCP?**
- ❌ Local RTX 3070: Only 8GB VRAM → Can't run 5 experiments in parallel
- ✅ GCP: Run 5 VMs in parallel (one seed each) → All complete in ~2 hours
- ✅ Cost: ~$2.00 total (vs struggling with local GPU)

---

## Quick Start

### 1. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

**Important:** This key will be passed to GCP VMs securely via environment variable.

---

### 2. Launch 5 VMs on GCP

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
./launch_5seeds_gcp.sh
```

**What it does:**
- Creates 5 VMs (vm-phasec-seed42, vm-phasec-seed100, etc.)
- Each VM: n1-standard-4 (4 vCPUs, 15GB RAM, CPU-only)
- Uploads Phase C v2.1 code and MBTI_500.csv dataset
- Installs Python dependencies
- Launches experiments in background

**Time:** ~5-10 minutes to setup, then ~2 hours for experiments

**Output:**
```
═══════════════════════════════════════════════════════════
  ✅ All 5 Experiments Launched on GCP!
═══════════════════════════════════════════════════════════

VMs running:
  - vm-phasec-seed42 (seed 42)
  - vm-phasec-seed100 (seed 100)
  - vm-phasec-seed123 (seed 123)
  - vm-phasec-seed456 (seed 456)
  - vm-phasec-seed789 (seed 789)

Estimated completion: ~2 hours
```

---

### 3. Monitor Progress

In a new terminal:

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
./monitor_gcp_5seeds.sh
```

**Shows:**
- Status of each VM (RUNNING/STOPPED)
- Completion status (COMPLETED/RUNNING)
- Per-seed results (overall delta, synthetics)
- Updates every 60 seconds

**Example output:**
```
═══════════════════════════════════════════════════════════
  Phase C v2.1 - GCP Multi-Seed Validation Monitor
═══════════════════════════════════════════════════════════

───────────────────────────────────────────────────────────
VM: vm-phasec-seed42 (Seed 42)
───────────────────────────────────────────────────────────
  Status: RUNNING
  ✅ COMPLETED
  Overall delta: +0.377%
  Synthetics: 87

───────────────────────────────────────────────────────────
VM: vm-phasec-seed100 (Seed 100)
───────────────────────────────────────────────────────────
  Status: RUNNING
  🔄 RUNNING
  Last: LLM generación ENFJ: 50%...

Progress: 1/5 experiments completed
```

**Exit:** Press `Ctrl+C` (experiments continue on GCP)

---

### 4. Collect Results (~2 hours later)

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
./collect_gcp_5seeds.sh
```

**What it does:**
- Downloads all results from 5 VMs:
  - phaseC_v2.1_seed{SEED}_metrics.json
  - phaseC_v2.1_seed{SEED}_synthetic.csv
  - phaseC_v2.1_seed{SEED}_augmented.csv
  - phaseC_seed{SEED}.log
- Prompts to delete VMs (save costs)

**Output:**
```
═══════════════════════════════════════════════════════════
  ✅ Results Downloaded!
═══════════════════════════════════════════════════════════

Results summary:
  Metrics files: 5/5

✅ All 5 seeds collected!

Run analysis:
  python3 analyze_5seeds.py

═══════════════════════════════════════════════════════════

Delete all 5 VMs to save costs? (y/n)
```

**Important:** Say 'y' to delete VMs and stop charges!

---

### 5. Analyze Results

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
python3 analyze_5seeds.py
```

**Output:** Statistical summary with mean, std, 95% CI, success rate

---

## Architecture

```
Local Machine
    │
    ├─ launch_5seeds_gcp.sh (creates VMs)
    │
    └──> GCP us-central1-a
         │
         ├─ vm-phasec-seed42 (n1-standard-4, CPU)
         │  └─ Phase C v2.1, seed 42
         │
         ├─ vm-phasec-seed100 (n1-standard-4, CPU)
         │  └─ Phase C v2.1, seed 100
         │
         ├─ vm-phasec-seed123 (n1-standard-4, CPU)
         │  └─ Phase C v2.1, seed 123
         │
         ├─ vm-phasec-seed456 (n1-standard-4, CPU)
         │  └─ Phase C v2.1, seed 456
         │
         └─ vm-phasec-seed789 (n1-standard-4, CPU)
            └─ Phase C v2.1, seed 789
```

**Each VM runs independently in parallel → All complete at same time (~2 hours)**

---

## Cost Breakdown

| Resource | Rate | Duration | Cost |
|----------|------|----------|------|
| 5× n1-standard-4 VMs | $0.20/hr each | 2 hours | $2.00 |
| 5× Boot disks (30GB SSD) | $0.17/GB/month | ~2 hours | $0.01 |
| Network egress | ~$0.12/GB | ~0.5 GB | $0.06 |
| OpenAI API calls (gpt-4o-mini) | ~$0.03/seed | 5 seeds | $0.15 |
| **Total** | | | **~$2.22** |

**Note:** If VMs are deleted immediately after collection, ongoing cost is $0.

---

## Prerequisites

### 1. GCP Setup

```bash
# Install gcloud CLI (if not already installed)
# macOS:
brew install google-cloud-sdk

# Linux:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable Compute Engine API
gcloud services enable compute.googleapis.com
```

### 2. OpenAI API Key

```bash
export OPENAI_API_KEY='your-key-here'
```

---

## Troubleshooting

### VM Creation Fails

**Error:** `Quota exceeded` or `Permission denied`

**Solution:**
```bash
# Check quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Try different zone
# Edit launch_5seeds_gcp.sh, change:
ZONE="us-central1-a"  # to
ZONE="us-west1-b"
```

---

### Experiments Not Starting

**Check 1:** SSH into VM and verify
```bash
gcloud compute ssh vm-phasec-seed42 --zone=us-central1-a

# On VM:
ps aux | grep python3
tail -50 phaseC_seed42.log
```

**Check 2:** Verify OpenAI API key
```bash
echo $OPENAI_API_KEY  # Should show your key
```

---

### One Seed Fails

**Check logs:**
```bash
gcloud compute ssh vm-phasec-seed100 --zone=us-central1-a
tail -100 phaseC_seed100.log | grep -i "error\|exception"
```

**Rerun failed seed:**
```bash
# SSH into VM
gcloud compute ssh vm-phasec-seed100 --zone=us-central1-a

# On VM:
source venv/bin/activate
export OPENAI_API_KEY='your-key'

# Rerun manually
python3 runner_phase2.py \
    --data-path MBTI_500.csv \
    --random-seed 100 \
    # ... (copy full command from launch script)
```

---

### Results Not Downloading

**Error:** `SCP failed` or `File not found`

**Check if file exists:**
```bash
gcloud compute ssh vm-phasec-seed42 --zone=us-central1-a \
    --command="ls -lh phaseC_v2.1_seed42_metrics.json"
```

**Manual download:**
```bash
gcloud compute scp vm-phasec-seed42:~/phaseC_v2.1_seed42_metrics.json ./ \
    --zone=us-central1-a
```

---

## Manual Operations

### Check VM Status

```bash
gcloud compute instances list --filter="name~vm-phasec"
```

### SSH into Specific VM

```bash
gcloud compute ssh vm-phasec-seed42 --zone=us-central1-a
```

### Stop VMs (Pause Charges)

```bash
gcloud compute instances stop vm-phasec-seed{42,100,123,456,789} \
    --zone=us-central1-a
```

### Start VMs (Resume)

```bash
gcloud compute instances start vm-phasec-seed{42,100,123,456,789} \
    --zone=us-central1-a
```

### Delete VMs (Cleanup)

```bash
gcloud compute instances delete vm-phasec-seed{42,100,123,456,789} \
    --zone=us-central1-a --quiet
```

---

## Expected Timeline

| Time | Event |
|------|-------|
| 0:00 | Launch script starts |
| 0:05 | All 5 VMs created |
| 0:06 | Files uploaded to VMs |
| 0:08 | Python dependencies installed |
| 0:10 | Experiments launched |
| 2:10 | All experiments complete |
| 2:15 | Results collected and VMs deleted |

**Total:** ~2.5 hours (mostly waiting)

---

## Comparison: Local vs GCP

| Aspect | Local (RTX 3070) | GCP (5 VMs) |
|--------|-----------------|-------------|
| **Parallel capacity** | 2 seeds max (OOM at 5) | 5 seeds ✅ |
| **Total time** | ~5 hours (sequential) | ~2 hours ✅ |
| **GPU needed** | Yes (8GB VRAM) | No (CPU-only) |
| **Cost** | $0 (own hardware) | ~$2.22 |
| **Risk** | OOM crashes | None ✅ |
| **Setup** | Already configured | Need gcloud CLI |

**GCP wins:** Faster, more reliable, minimal cost

---

## Scripts Summary

| Script | Purpose |
|--------|---------|
| **[launch_5seeds_gcp.sh](launch_5seeds_gcp.sh)** | Create VMs and launch experiments |
| **[monitor_gcp_5seeds.sh](monitor_gcp_5seeds.sh)** | Monitor progress (real-time) |
| **[collect_gcp_5seeds.sh](collect_gcp_5seeds.sh)** | Download results and delete VMs |
| **[analyze_5seeds.py](analyze_5seeds.py)** | Statistical analysis (same as local) |

---

## FAQ

**Q: Can I use a different GCP zone?**
A: Yes, edit `ZONE="us-central1-a"` in launch script to your preferred zone.

**Q: Can I use preemptible VMs to save 70% cost?**
A: Yes, but risk experiments terminating early. Add `--preemptible` flag in launch script.

**Q: How do I check billing?**
A: GCP Console → Billing → Reports

**Q: Can I run more than 5 seeds?**
A: Yes, edit `SEEDS` array in all 3 scripts and add more seeds.

**Q: What if an experiment takes longer than 2 hours?**
A: VMs continue running. Check with monitor script and collect when done.

**Q: Can I reuse existing VMs?**
A: Yes, but launch script deletes existing VMs first to ensure clean state.

---

## Next Steps After Validation

1. ✅ Run GCP multi-seed validation (this guide)
2. ✅ Analyze results with `python3 analyze_5seeds.py`
3. ✅ Update documentation with multi-seed statistics
4. ✅ Finish thesis with validated Phase C v2.1 configuration

---

**Last Updated:** 2025-11-19
**Status:** Ready to deploy
