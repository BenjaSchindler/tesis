# SMOTE-LLM Deployment Status

**Date:** 2025-11-15
**Status:** Phase A - 25 Seeds RUNNING

---

## Current Deployment

### VMs Created
- **vm-batch1** (seeds: 42, 100, 123, 456, 789) - RUNNING
- **vm-batch2** (seeds: 111, 222, 333, 444, 555) - RUNNING
- **vm-batch3** (seeds: 1000, 2000, 3000, 4000, 5000) - RUNNING
- **vm-batch4** (seeds: 7, 13, 21, 37, 101) - RUNNING
- **vm-batch5** (seeds: 1234, 2345, 3456, 4567, 5678) - RUNNING

### Configuration
- **Machine Type:** n1-standard-4 (4 vCPUs, 15GB RAM)
- **Zone:** us-central1-a
- **Dataset:** MBTI_500.csv (106K samples)
- **Estimated Runtime:** ~5 hours in parallel
- **Estimated Cost:** ~$10 USD

### Files Uploaded (Per VM)
**All 9 Core Modules (Phase 2 Complete):**
1. ✅ runner_phase2.py
2. ✅ ensemble_anchor_selector.py
3. ✅ contamination_aware_filter.py
4. ✅ enhanced_quality_gate.py
5. ✅ anchor_quality_improvements.py
6. ✅ quality_gate_predictor.py
7. ✅ mbti_class_descriptions.py
8. ✅ adversarial_discriminator.py
9. ✅ multi_seed_ensemble.py

**Additional Files:**
- ✅ MBTI_500.csv (331 MB dataset)
- ✅ run_all_seeds.sh (batch execution script)

---

## Phase A Configuration

### Enabled Features
```bash
--use-ensemble-selection       # Ensemble-based best prediction selection
--use-val-gating              # Validation-based quality gating
--enable-anchor-gate          # Quality gate for anchors
--enable-anchor-selection     # Top-k anchor selection
--enable-adaptive-filters     # Dynamic threshold adjustment
--use-class-description       # MBTI class descriptions
--use-f1-budget-scaling      # F1-based budget allocation
```

### Key Parameters
```python
max_clusters = 3
prompts_per_cluster = 3
prompt_mode = 'mix'
val_size = 0.15
val_tolerance = 0.02
anchor_quality_threshold = 0.50  # Updated from 0.35
anchor_selection_ratio = 0.8
anchor_outlier_threshold = 1.5
similarity_threshold = 0.90
min_classifier_confidence = 0.10
contamination_threshold = 0.95
synthetic_weight = 0.5
synthetic_weight_mode = 'flat'
f1_budget_thresholds = [0.35, 0.20]
f1_budget_multipliers = [30, 70, 100]
```

---

## Expected Results

### Targets
- **Macro F1 Delta:** +1.00% ± 0.25%
- **Seed Variance:** < 5pp range (vs 54pp baseline)
- **HIGH F1 Protection:** 100% (no degradation for F1 ≥ 45%)
- **LOW F1 Improvement:** +10-15% average

### Output Files (Per Seed)
For each seed, the following files will be generated:
- `phaseA_seed{SEED}_metrics.json` - Performance metrics
- `phaseA_seed{SEED}_synthetic.csv` - Generated synthetic samples
- `phaseA_seed{SEED}_augmented.csv` - Augmented training data

**Total Expected:** 75 files (25 seeds × 3 files)

---

## Improvements from Previous Runs

### Fixed Issues
1. ❌ **Previous:** Missing Phase 2 modules → Running Phase 1 fallback
   ✅ **Now:** All 9 modules uploaded → Full Phase A capability

2. ❌ **Previous:** Only runner_phase2.py uploaded
   ✅ **Now:** Complete core/ directory (10 files total)

3. ❌ **Previous:** Warning messages about missing modules
   ✅ **Now:** Clean execution with all features enabled

### Configuration Updates
- Updated `anchor_quality_threshold` from 0.35 to 0.50
- Added F1-budget scaling arguments
- Added synthetic weighting arguments
- Fixed all argument name mismatches

---

## Repository Status

### SMOTE-LLM Project
**Location:** `/home/benja/Desktop/Tesis/SMOTE-LLM/`

**Structure:**
```
SMOTE-LLM/ (331 MB)
├── core/          (9 modules, 239 KB)
├── phase_a/       (local + GCP scripts)
├── phase_b/       (adaptive weighting)
├── docs/          (18 documents, 233 KB)
├── scripts/       (analysis tools)
├── MBTI_500.csv   (331 MB)
├── README.md      (9.3 KB)
├── CLAUDE.md      (15 KB - GCP guide)
└── requirements.txt
```

**Features:**
- ✅ Self-contained (relative paths)
- ✅ Complete documentation (200+ pages)
- ✅ Local execution tested (working)
- ✅ GCP deployment scripts updated
- ✅ All Phase 2 modules included
- ✅ Analysis and visualization tools

---

## Next Steps

### Immediate (Automated)
1. ⏳ File upload to VMs (in progress)
2. ⏳ Dependency installation in venv
3. ⏳ Launch 25 experiments in parallel
4. ⏳ Monitor progress (~5 hours runtime)

### After Completion (~5 hours)
1. Collect results from all 5 VMs
2. Generate statistical analysis
3. Verify all 25 seeds completed successfully
4. Calculate aggregate metrics:
   - Mean/median/std of F1 deltas
   - 95% confidence interval
   - Success rate
5. Delete VMs to stop charges

### Monitoring Commands
```bash
# Check VM status
gcloud compute instances list --filter='name~vm-batch'

# SSH into a VM
gcloud compute ssh vm-batch1 --zone=us-central1-a

# Check experiment progress (on VM)
ps aux | grep python3
tail -f LaptopRuns/nohup_batch1.log

# Collect results (when done)
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/gcp
./collect_results.sh
```

---

## Cost Breakdown

### Compute
- 5 VMs × n1-standard-4 @ $0.20/hr × 5 hours = **$5.00**
- Boot disks (5 × 30GB) @ $0.17/GB/month × 5 hours = **$0.04**
- Network egress ~2GB @ $0.12/GB = **$0.24**

### API Calls
- OpenAI gpt-4o-mini @ ~$0.50 per experiment × 25 = **$12.50**

### Total Estimated Cost
**$17.78** (if auto-shutdown works)
**$288/month** (if VMs left running - DON'T DO THIS!)

---

## Troubleshooting

### If Experiments Fail
```bash
# Check logs on VM
gcloud compute ssh vm-batch1 --zone=us-central1-a
cd LaptopRuns
tail -100 nohup_batch1.log

# Check for module errors
grep "module not found" nohup_batch1.log

# Verify all modules present
ls -la *.py
```

### If VMs Run Out of Memory
```bash
# Check memory usage
free -h

# Check process
ps aux | grep python3 | awk '{print $6}'
```

### If Need to Restart
```bash
# Delete VMs
gcloud compute instances delete vm-batch{1,2,3,4,5} --zone=us-central1-a

# Relaunch
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/gcp
export OPENAI_API_KEY='your-key'
./launch_25seeds.sh
```

---

## Success Criteria

### Phase A Complete When:
- ✅ All 25 experiments finished successfully
- ✅ 75 output files generated (25 seeds × 3 files)
- ✅ Statistical analysis shows:
  - Mean delta ≥ +0.70%
  - Seed variance < 10pp
  - Success rate ≥ 80%
  - 95% CI does not include 0
- ✅ Results backed up locally
- ✅ VMs deleted (costs stopped)

---

**Last Updated:** 2025-11-15 05:50 UTC
**Deployment Started (Fixed):** 2025-11-15 05:47 UTC
**Expected Completion:** 2025-11-15 ~10:47 UTC (~5 hours)

## Recent Issues and Fixes

### Issue 1: Missing venv Activation (FIXED ✅)
**Problem:** Seed 42 failed with `ModuleNotFoundError: No module named 'sentence_transformers'`
**Root Cause:** Batch scripts didn't activate venv - when launched via nohup, parent shell's venv activation didn't carry over
**Fix Applied:** Added `source .venv/bin/activate` to all 5 batch scripts
**Status:** VMs deleted and relaunched with fixed scripts (05:47 UTC)
