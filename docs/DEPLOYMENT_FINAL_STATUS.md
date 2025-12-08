# Phase A - 25 Seeds Deployment - Final Status

**Deployment Date:** 2025-11-15
**Status:** ✅ RUNNING (Fixed and Relaunched)
**Start Time:** 05:47 UTC
**Expected Completion:** ~10:47 UTC (5 hours runtime)

---

## Deployment Configuration

### VMs Created (5 total)
```
vm-batch1: seeds 42, 100, 123, 456, 789       (IP: 34.61.143.230)
vm-batch2: seeds 111, 222, 333, 444, 555      (IP: 34.68.245.212)
vm-batch3: seeds 1000, 2000, 3000, 4000, 5000 (IP: 34.29.218.223)
vm-batch4: seeds 7, 13, 21, 37, 101           (IP: 34.16.11.90)
vm-batch5: seeds 1234, 2345, 3456, 4567, 5678 (IP: 34.9.194.89)
```

### Machine Specs
- Type: n1-standard-4 (4 vCPUs, 15GB RAM)
- Zone: us-central1-a
- Boot disk: 30GB pd-standard

---

## Files Uploaded (Per VM)

### Core Modules (9 files - Phase 2 Complete)
1. ✅ runner_phase2.py (119 KB)
2. ✅ ensemble_anchor_selector.py (16 KB)
3. ✅ contamination_aware_filter.py (14 KB)
4. ✅ enhanced_quality_gate.py (13 KB)
5. ✅ anchor_quality_improvements.py (28 KB)
6. ✅ quality_gate_predictor.py (7.9 KB)
7. ✅ mbti_class_descriptions.py (7.5 KB)
8. ✅ adversarial_discriminator.py (12 KB)
9. ✅ multi_seed_ensemble.py (14 KB)

### Additional Files
- ✅ MBTI_500.csv (331 MB dataset)
- ✅ run_all_seeds.sh (batch execution script - **FIXED with venv activation**)

---

## Phase A Configuration

### Enabled Features
```python
--use-ensemble-selection        # Ensemble-based best prediction selection
--use-val-gating               # Validation-based quality gating
--enable-anchor-gate           # Quality gate for anchors
--enable-anchor-selection      # Top-k anchor selection
--enable-adaptive-filters      # Dynamic threshold adjustment
--use-class-description        # MBTI class descriptions
--use-f1-budget-scaling       # F1-based budget allocation
```

### Key Parameters
```python
max_clusters = 3
prompts_per_cluster = 3
prompt_mode = 'mix'
val_size = 0.15
val_tolerance = 0.02
anchor_quality_threshold = 0.50
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

## Issue Resolution

### ❌ Issue 1: Missing venv Activation (FIXED ✅)

**Symptom:**
```
Seed 42 failed with: ModuleNotFoundError: No module named 'sentence_transformers'
Seed 100 started successfully (indicated venv issue, not missing modules)
```

**Root Cause:**
Batch scripts used `python3` directly without activating venv. When launched via `nohup bash run_all_seeds.sh`, the parent shell's venv activation didn't carry over to the subshell.

**Fix Applied:**
Added `source .venv/bin/activate` to all 5 batch scripts:
```bash
#!/bin/bash
# VM batch1: Seeds 42 100 123 456 789
# AUTO-SHUTDOWN enabled

# Activate venv (ADDED)
source .venv/bin/activate

SEEDS=(42 100 123 456 789)
...
```

**Verification:**
- VMs deleted (05:47 UTC)
- Fixed batch scripts uploaded to all VMs
- Relaunched deployment (05:47 UTC)
- **Status: ✅ FIXED**

---

## Expected Results

### Output Files (25 seeds × 3 files = 75 total)
For each seed, the following files will be generated:
```
phaseA_seed{SEED}_metrics.json        - Performance metrics
phaseA_seed{SEED}_synthetic.csv       - Generated synthetic samples
phaseA_seed{SEED}_augmented.csv       - Augmented training data
```

### Target Metrics
- **Macro F1 Delta:** +1.00% ± 0.25%
- **Seed Variance:** < 5pp range (vs 54pp baseline)
- **HIGH F1 Protection:** 100% (no degradation for F1 ≥ 45%)
- **LOW F1 Improvement:** +10-15% average

---

## Cost Breakdown

### Compute
- 5 VMs × n1-standard-4 @ $0.20/hr × 5 hours = **$5.00**
- Boot disks (5 × 30GB) @ $0.17/GB/month × 5 hours = **$0.04**
- Network egress ~2GB @ $0.12/GB = **$0.24**

### API Calls
- OpenAI gpt-4o-mini @ ~$0.50 per experiment × 25 = **$12.50**

### Total Estimated Cost
**$17.78** (with auto-shutdown)

---

## Monitoring Commands

### Check VM Status
```bash
gcloud compute instances list --filter='name~vm-batch'
```

### SSH into a VM
```bash
gcloud compute ssh vm-batch1 --zone=us-central1-a
```

### Check Experiment Progress (on VM)
```bash
cd LaptopRuns
tail -f nohup_batch1.log
```

### Check for Module Warnings (verify Phase 2 active)
```bash
gcloud compute ssh vm-batch1 --zone=us-central1-a --command="cd LaptopRuns && grep -i 'module not found' nohup_batch1.log"
# Should return empty (no warnings)
```

### Verify Python Processes Running
```bash
gcloud compute ssh vm-batch1 --zone=us-central1-a --command="ps aux | grep python3"
```

---

## Collection Commands (After ~5 hours)

### 1. Verify All Experiments Completed
```bash
for batch in batch1 batch2 batch3 batch4 batch5; do
  echo "[$batch]"
  gcloud compute ssh "vm-$batch" --zone=us-central1-a --command="cd LaptopRuns && ls -lh phaseA_*.json 2>/dev/null | wc -l"
done
# Should show 5 metrics files per VM
```

### 2. Collect Results from All VMs
```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_a/results
mkdir -p batch1 batch2 batch3 batch4 batch5

for batch in batch1 batch2 batch3 batch4 batch5; do
  gcloud compute scp "vm-$batch:LaptopRuns/phaseA_*.json" "$batch/" --zone=us-central1-a
  gcloud compute scp "vm-$batch:LaptopRuns/phaseA_*.csv" "$batch/" --zone=us-central1-a
done
```

### 3. Delete VMs to Stop Charges
```bash
gcloud compute instances delete vm-batch1 vm-batch2 vm-batch3 vm-batch4 vm-batch5 --zone=us-central1-a
```

---

## Success Criteria

✅ **Deployment Successful If:**
1. All 25 experiments completed without errors
2. 75 output files generated (25 × 3)
3. No "module not found" warnings in logs
4. Statistical analysis shows:
   - Mean delta ≥ +0.70%
   - Seed variance < 10pp
   - Success rate ≥ 80%
   - 95% CI does not include 0

---

## Next Steps After Completion

1. **Collect Results** (~5 hours from 05:47 UTC)
2. **Generate Statistical Analysis**
3. **Verify All 25 Seeds Completed Successfully**
4. **Calculate Aggregate Metrics:**
   - Mean/median/std of F1 deltas
   - 95% confidence interval
   - Success rate
5. **Delete VMs** to stop charges
6. **Backup Results** to local storage
7. **Update DEPLOYMENT_STATUS.md** with final results

---

**Last Updated:** 2025-11-15 05:52 UTC
**Deployment Log:** `/tmp/launch_phaseA_fixed.log`
**Repository:** `/home/benja/Desktop/Tesis/SMOTE-LLM/`
