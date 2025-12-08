# Phase C - GCP Deployment

Deploy Phase C (Adaptive Temperature) to Google Cloud Platform with GPU acceleration and auto-shutdown.

## Quick Start

```bash
cd phase_c/gcp

# 1. Set OpenAI API key
export OPENAI_API_KEY='your-openai-api-key'

# 2. Launch VM (creates, uploads files, starts experiment)
./launch_phaseC.sh

# 3. Monitor progress (in another terminal)
./monitor_phaseC.sh

# 4. Download results after ~1 hour
./collect_results_phaseC.sh
```

## What It Does

1. **Creates GPU VM** (n1-standard-4 + NVIDIA Tesla T4)
2. **Installs CUDA drivers** (~2-3 min)
3. **Uploads** all 9 core modules + dataset
4. **Runs Phase C** experiment (seed 42)
5. **Auto-shuts down** after completion (saves costs)

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **VM Name** | vm-phasec-test | Single test VM |
| **Zone** | us-west1-b | GPU available |
| **Machine** | n1-standard-4 | 4 vCPUs, 15GB RAM |
| **GPU** | NVIDIA Tesla T4 | 16GB VRAM |
| **Boot Disk** | 50GB SSD | Larger for CUDA |
| **Seed** | 42 | Fixed for reproducibility |
| **Runtime** | 45-60 min | GPU accelerated |
| **Cost** | ~$1.05 | Compute + GPU + API |

## Cost Breakdown

| Item | Rate | Duration | Cost |
|------|------|----------|------|
| Compute | $0.20/hr | 1 hour | $0.20 |
| GPU (T4) | $0.35/hr | 1 hour | $0.35 |
| OpenAI API | ~$0.50 | 1 run | $0.50 |
| **Total** | | | **~$1.05** |

**Auto-shutdown**: VM stops 60 seconds after completion, so you only pay for actual runtime.

## Scripts

### launch_phaseC.sh

Main deployment script:
- Checks for existing VM (offers to delete)
- Creates VM with GPU and startup script
- Waits for CUDA installation
- Uploads 9 core modules + dataset (331 MB)
- Launches experiment in background
- Provides monitoring commands

**Usage**:
```bash
export OPENAI_API_KEY='your-key'
./launch_phaseC.sh
```

### monitor_phaseC.sh

Real-time monitoring:
- VM status (RUNNING, TERMINATED, etc.)
- Python process status (running/completed)
- Last 30 lines of log
- Adaptive temperature messages (🌡️)
- Output files status

**Usage**:
```bash
./monitor_phaseC.sh
```

**Output**:
```
════════════════════════════════════════════════════════
  Phase C - GCP Monitor
════════════════════════════════════════════════════════

VM Status: RUNNING

✅ VM is RUNNING

✅ Experiment is RUNNING
  PID: 1234, CPU: 98.5%, MEM: 12.3%

─────────────────────────────────────────────────────────
Last 30 lines of log:
─────────────────────────────────────────────────────────
[Step 4/6] Generating synthetic data...
🌡️  ADAPTIVE TEMP: ENFP (F1=0.410) - temp=1.00 → 0.50
🌡️  ADAPTIVE TEMP: ENTP (F1=0.380) - temp=1.00 → 0.50
...
```

### collect_results_phaseC.sh

Results downloader and analyzer:
- Downloads all output files (metrics, synthetics, logs)
- Analyzes MID-tier performance
- Provides recommendations based on results
- Optionally deletes VM to stop charges

**Usage**:
```bash
./collect_results_phaseC.sh
```

**Output**:
```
════════════════════════════════════════════════════════
  Phase C - Results Collector
════════════════════════════════════════════════════════

[1/3] Downloading results files...

✓ phaseC_seed42_metrics.json
✓ phaseC_seed42_synthetic.csv
✓ phaseC_seed42_augmented.csv
✓ phaseC_output.log

[2/3] Analyzing results...

═══════════════════════════════════════════════════════
  PHASE C - RESULTS ANALYSIS (SEED 42)
═══════════════════════════════════════════════════════

Overall Performance:
  Baseline Macro F1:   0.4123
  Augmented Macro F1:  0.4245
  Delta:               +1.22%

MID-Tier Classes (F1 0.20-0.45):
──────────────────────────────────────────────────────
  ✅ ENFP  : 0.410 → 0.425 (+1.50%)
  ✅ ENTP  : 0.380 → 0.391 (+1.10%)
  ✅ ENTJ  : 0.310 → 0.318 (+0.80%)
  ⚠️  ESFJ  : 0.280 → 0.278 (-0.20%)
──────────────────────────────────────────────────────

MID-Tier Summary:
  Mean Delta:     +0.80%
  Positive:       3/4 classes
  Target:         ≥ +0.10%
  Phase B Baseline: -0.59%
  Improvement:    +1.39pp

  ✅ SUCCESS: MID-tier target achieved!
     → Adaptive temperature works!
     → Proceed to 5-seed validation
```

## Timeline

### Immediate (0-5 min)
- VM creation
- CUDA installation starts
- File uploads

### Early Phase (5-15 min)
- CUDA installation completes
- Dependencies install
- Data loading
- Baseline training

### Main Phase (15-60 min)
- Synthetic generation (with adaptive temp)
- Quality filtering
- Augmented training
- Evaluation

### Completion (60-65 min)
- Results summary displayed
- 60 second countdown
- Auto-shutdown

## Monitoring

### Check if Experiment is Running

```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='ps aux | grep python3'
```

### View Live Log (Last 50 Lines)

```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='tail -50 PhaseC/phaseC_output.log'
```

### Follow Log in Real-Time

```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='tail -f PhaseC/phaseC_output.log'
```

### Check Adaptive Temperature Messages

```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='grep "🌡️" PhaseC/phaseC_output.log'
```

Expected output:
```
🌡️  ADAPTIVE TEMP: ENFP (F1=0.410) - temp=1.00 → 0.50
🌡️  ADAPTIVE TEMP: ENTP (F1=0.380) - temp=1.00 → 0.50
🌡️  ADAPTIVE TEMP: ENTJ (F1=0.310) - temp=1.00 → 0.50
🌡️  ADAPTIVE TEMP: ESFJ (F1=0.280) - temp=1.00 → 0.50
```

### SSH into VM

```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b

# On VM:
cd PhaseC
tail -f phaseC_output.log
```

## Troubleshooting

### Issue: VM Creation Fails

**Error**: `Quota exceeded` or `Zone unavailable`

**Solutions**:
```bash
# Try different zone (no GPU)
# Edit launch_phaseC.sh, change:
ZONE="us-central1-a"
ACCELERATOR=""  # Remove GPU

# Or request quota increase in GCP Console
```

### Issue: CUDA Installation Takes Too Long

**Symptom**: Script waits >5 minutes for CUDA

**Solution**:
```bash
# SSH into VM manually
gcloud compute ssh vm-phasec-test --zone=us-west1-b

# Check installation logs
sudo journalctl -u google-startup-scripts.service

# If stuck, continue anyway (will use CPU)
```

### Issue: Experiment Fails to Start

**Check**:
```bash
# View full log
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='cat PhaseC/phaseC_output.log'

# Common causes:
# 1. Missing OpenAI API key
# 2. Module import errors
# 3. CUDA memory issues
```

### Issue: Results Not Found

**Error**: `collect_results_phaseC.sh` finds no files

**Solutions**:
```bash
# Check if experiment ran
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='ls -la PhaseC/'

# Check log for errors
gcloud compute ssh vm-phasec-test --zone=us-west1-b --command='cat PhaseC/phaseC_output.log'
```

### Issue: High Costs

**Check running instances**:
```bash
gcloud compute instances list --filter='status=RUNNING'
```

**Stop VM**:
```bash
gcloud compute instances stop vm-phasec-test --zone=us-west1-b
```

**Delete VM** (recommended after collecting results):
```bash
gcloud compute instances delete vm-phasec-test --zone=us-west1-b
```

## Expected Results

### Success Scenarios

**Best Case** (MID-tier ≥ +0.20%):
```
MID-tier: -0.59% → +0.30%
Overall: +1.00% → +1.30%

→ Adaptive temperature WORKS!
→ Run 5-seed validation
→ Possibly ready for 25-seed GCP deployment
```

**Good Case** (MID-tier +0.10% to +0.20%):
```
MID-tier: -0.59% → +0.15%
Overall: +1.00% → +1.15%

→ Adaptive temperature HELPS
→ Add Hardness-Aware Anchors (Phase 2)
→ Combined: +0.30% to +0.50% expected
```

**Acceptable Case** (MID-tier -0.30% to +0.10%):
```
MID-tier: -0.59% → -0.10%
Overall: +1.00% → +1.05%

→ Improvement but not enough
→ Proceed to Phase 2 (90% success rate)
→ May need Phase 3 as well
```

### If Results Are Poor (MID-tier < -0.30%)

Temperature alone is insufficient. Skip to:
1. **Phase 2: Hardness-Aware Anchors** (highest success rate: 90%)
2. **Phase 3: Multi-Stage Filtering** (80% success rate)
3. **Combined approach** (95% success rate)

## Next Steps After Success

### 1. Multi-Seed Validation (5 seeds)

Edit `launch_phaseC.sh` to run multiple seeds:
```bash
# Change SEED variable to loop:
for SEED in 42 100 123 456 789; do
    # Create VM per seed
    # Or modify to run sequentially on same VM
done
```

Cost: ~$5.00 (5 × $1.00)

### 2. Full Validation (25 seeds)

Create `launch_25seeds_phaseC.sh` based on `phase_a/gcp/launch_25seeds.sh`:
- 5 VMs × 5 seeds each
- Parallel execution
- Statistical analysis

Cost: ~$25 (25 × $1.00)

### 3. Production Deployment

If Phase C proves successful:
- Document in thesis
- Create production-ready script
- Add to project README
- Publish results

## Comparison with Phase A GCP

| Aspect | Phase A | Phase C |
|--------|---------|---------|
| **VMs** | 5 (parallel) | 1 (single test) |
| **Seeds** | 25 (5 per VM) | 1 |
| **Machine** | n1-standard-4 (CPU) | n1-standard-4 + T4 GPU |
| **Runtime** | ~5 hrs (per VM) | ~1 hr (GPU) |
| **Cost** | ~$10 (all 25) | ~$1 (single) |
| **Purpose** | Robustness validation | Quick technique test |
| **Auto-shutdown** | Yes | Yes |

Phase C GCP is designed for **quick validation** before committing to full multi-seed runs.

## Files Structure

```
phase_c/gcp/
├── README.md                  # This file
├── launch_phaseC.sh          # Main launcher (creates VM, runs experiment)
├── monitor_phaseC.sh         # Monitor script (check status)
└── collect_results_phaseC.sh # Download and analyze results

# After running:
phaseC_results/
├── phaseC_seed42_metrics.json
├── phaseC_seed42_synthetic.csv
├── phaseC_seed42_augmented.csv
└── phaseC_output.log
```

## FAQ

### Can I use CPU instead of GPU?

Yes, edit `launch_phaseC.sh`:
```bash
MACHINE_TYPE="n1-standard-4"
ACCELERATOR=""  # Remove this line

# In startup script, skip CUDA installation
```

Runtime: 2-3 hours (vs 45-60 min with GPU)
Cost: ~$0.70 (vs ~$1.05 with GPU)

### Can I run multiple seeds?

Yes, but requires script modification. Options:
1. Run sequentially on same VM (change auto-shutdown)
2. Create multiple VMs (1 per seed)
3. Wait for 5-seed launcher script (TODO)

### How do I cancel auto-shutdown?

SSH into VM before 60-second countdown:
```bash
gcloud compute ssh vm-phasec-test --zone=us-west1-b
# Press Ctrl+C during countdown
```

### What if I want to keep VM running?

Edit `run_phaseC_remote.sh` uploaded script, remove:
```bash
# Comment out these lines:
# echo "Auto-shutdown in 60 seconds..."
# sleep 60
# sudo shutdown -h now
```

## Support

For issues or questions:
1. Check logs: `./monitor_phaseC.sh`
2. Review troubleshooting section above
3. Check GCP Console → Compute Engine → VM instances
4. Check phase_c/README.md for Phase C documentation

---

**Last Updated**: 2025-11-15
**Status**: Ready for deployment
**Cost**: ~$1.05 per run with GPU
