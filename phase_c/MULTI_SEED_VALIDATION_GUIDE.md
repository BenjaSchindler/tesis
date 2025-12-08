# Multi-Seed Validation Guide - Phase C v2.1

**Purpose:** Validate that Phase C v2.1's +1.72% MID-tier improvement is robust across multiple random seeds.

**Goal:** Prove v2.1 is not "lucky" with seed 42 but consistently improves MID-tier classes.

---

## Quick Start

### 1. Launch All 5 Seeds in Parallel

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
./run_5seeds_v2.1_parallel.sh
```

**What it does:**
- Runs Phase C v2.1 with seeds: 42, 100, 123, 456, 789
- All 5 run in parallel on your RTX 3070
- Reduced batch size (96) to fit 5 experiments in 8GB VRAM
- Estimated time: ~90-120 minutes total

**Requirements:**
- RTX 3070 with 8GB VRAM
- OpenAI API key set: `export OPENAI_API_KEY='your-key'`
- ~6-7GB available VRAM
- ~$0.10-0.25 total API cost (very cheap!)

---

### 2. Monitor Progress

In a separate terminal:

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
./monitor_5seeds.sh
```

**Shows:**
- Completion status (X/5 done)
- Per-seed progress
- GPU utilization
- Errors (if any)
- Updates every 30 seconds

**Exit:** Press `Ctrl+C` (experiments continue in background)

---

### 3. Analyze Results (After All Complete)

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c
python3 analyze_5seeds.py
```

**Computes:**
- Mean/median/std of MID-tier improvements
- 95% confidence interval
- Statistical significance (t-test vs target +0.10%)
- Per-class improvements across seeds
- Success rate

**Output:**
- Terminal summary with statistics
- `phaseC_v2.1_5seeds_summary.json` (for documentation)

---

## Expected Results

Based on v2.1's single-seed performance (+1.72% MID-tier), we expect:

### Excellent Outcome (Most Likely)
- **MID-tier mean:** +1.5% to +2.0%
- **95% CI:** Excludes 0 (all seeds improve)
- **Std:** < 0.5%
- **Success rate:** 100% (5/5 seeds improve)
- **Verdict:** ✅ v2.1 is robust and production-ready

### Good Outcome
- **MID-tier mean:** +1.0% to +1.5%
- **95% CI:** Excludes 0
- **Std:** 0.5% to 0.8%
- **Success rate:** 80% (4/5 seeds improve)
- **Verdict:** ✅ v2.1 is robust (some variance but acceptable)

### Acceptable Outcome
- **MID-tier mean:** +0.5% to +1.0%
- **95% CI:** May include 0
- **Std:** 0.8% to 1.2%
- **Success rate:** 60% (3/5 seeds improve)
- **Verdict:** ⚠️ Target achieved but high variance (investigate outliers)

### Failure (Unlikely)
- **MID-tier mean:** < +0.5%
- **Verdict:** ❌ v2.1 was lucky with seed 42 (re-evaluate)

**Most likely:** Excellent or Good outcome (v2.1 has strong signal from single seed).

---

## Troubleshooting

### GPU Out of Memory (OOM)

**Symptom:** Experiment crashes with "CUDA out of memory"

**Solution 1:** Run fewer seeds in parallel
```bash
# Edit run_5seeds_v2.1_parallel.sh
# Change SEEDS to only 3-4 seeds:
SEEDS=(42 100 123)  # Run 3 first
# Then run remaining 2 separately:
SEEDS=(456 789)
```

**Solution 2:** Reduce batch size further
```bash
# In run_5seeds_v2.1_parallel.sh, change:
--embedding-batch-size 96
# to:
--embedding-batch-size 64
```

---

### OpenAI Rate Limit Errors

**Symptom:** Logs show "Rate limit exceeded"

**Solution:** Experiments auto-retry after delay. Just wait - they'll complete.

**Prevention:** Stagger launches (already done with 3-second delays)

---

### One Seed Fails

**Symptom:** Monitor shows 4/5 complete, one stuck

**Diagnosis:**
```bash
# Check failed seed's log
tail -50 phaseC_v2.1_seed{SEED}_parallel.log

# Look for errors
grep -i "error\|exception\|failed" phaseC_v2.1_seed{SEED}_parallel.log
```

**Solution:** Re-run failed seed individually
```bash
./local_run_phaseC_v2.1.sh ../MBTI_500.csv {SEED}
```

---

### Experiments Taking Too Long

**Expected time:** 90-120 minutes

**If > 3 hours:** Check GPU utilization
```bash
watch -n 5 nvidia-smi
```

**Should see:**
- GPU util: 40-60% (embedding generation bursts)
- VRAM: ~6-7GB total (5 experiments × 1.2-1.5GB each)
- 5 Python processes

**If GPU util is 0%:** Experiments stalled, check logs for API errors

---

## Advanced: Run Sequentially (Safer)

If parallel execution causes issues, run one at a time:

```bash
cd /home/benja/Desktop/Tesis/SMOTE-LLM/phase_c

for SEED in 100 123 456 789; do
    echo "Running seed $SEED..."
    ./local_run_phaseC_v2.1.sh ../MBTI_500.csv $SEED
done
```

**Time:** ~6-8 hours total (90-120 min per seed)

**Pros:** No memory issues, easier to debug

**Cons:** Takes much longer

---

## Files Generated

After completion, you'll have:

### Per-Seed Results (5 files each × 5 seeds = 25 files)
```
phaseC_v2.1_seed42_metrics.json         (already exists from earlier)
phaseC_v2.1_seed42_synthetic.csv        (already exists from earlier)
phaseC_v2.1_seed42_augmented.csv        (already exists from earlier)

phaseC_v2.1_seed100_metrics.json
phaseC_v2.1_seed100_synthetic.csv
phaseC_v2.1_seed100_augmented.csv
phaseC_v2.1_seed100_parallel.log

phaseC_v2.1_seed123_metrics.json
...
(similar for seeds 456, 789)
```

### Summary Files
```
phaseC_v2.1_5seeds_summary.json         (statistical summary)
```

### Logs
```
phaseC_v2.1_seed{SEED}_parallel.log     (per-seed execution logs)
```

---

## Interpreting Results

### Key Metrics to Check

**1. MID-Tier Mean Across Seeds**
- **Target:** +0.10%
- **v2.1 single seed:** +1.72%
- **Expected:** +1.5% to +2.0%
- **Minimum acceptable:** +0.50%

**2. 95% Confidence Interval**
- **Should exclude 0** (lower bound > 0)
- Shows we're confident all seeds improve

**3. Standard Deviation**
- **< 0.5%:** Excellent (very consistent)
- **0.5-1.0%:** Good (some variance)
- **> 1.0%:** High variance (investigate why)

**4. Success Rate**
- **100% (5/5):** Excellent
- **80% (4/5):** Good
- **< 60%:** Concerning (outlier investigation needed)

---

## Example Output (What "Success" Looks Like)

```
════════════════════════════════════════════════════════════════════════════════
  Phase C v2.1 - Multi-Seed Validation Analysis (5 Seeds)
════════════════════════════════════════════════════════════════════════════════

Seeds analyzed: 5/5

────────────────────────────────────────────────────────────────────────────────

MID-TIER CLASSES ANALYSIS
────────────────────────────────────────────────────────────────────────────────

  Seed  42: MID-tier mean = +1.720%
  Seed 100: MID-tier mean = +1.650%
  Seed 123: MID-tier mean = +1.830%
  Seed 456: MID-tier mean = +1.590%
  Seed 789: MID-tier mean = +1.710%

  Mean across seeds:   +1.700%
  Median:              +1.710%
  Std:                 0.089%
  Min:                 +1.590%
  Max:                 +1.830%
  95% CI:              [+1.589%, +1.811%]

  t-test vs target (+0.10%): t=40.123, p=0.000001
  ✅ Significantly better than target (p < 0.05)

────────────────────────────────────────────────────────────────────────────────

SUCCESS RATE
────────────────────────────────────────────────────────────────────────────────

  Overall improved: 5/5 seeds (100%)
  MID-tier improved: 5/5 seeds (100%)
  Target achieved (+0.10% MID-tier): 5/5 seeds (100%)

────────────────────────────────────────────────────────────────────────────────

FINAL VERDICT
────────────────────────────────────────────────────────────────────────────────

  ✅ EXCELLENT - v2.1 is robust and exceeds target!
     MID-tier mean: +1.700% (target: +0.10%)
     Improvement factor: 17.0×

════════════════════════════════════════════════════════════════════════════════
```

**This shows:**
- Low variance (std = 0.089%)
- All seeds improve (5/5 = 100%)
- 95% CI excludes 0 ([+1.589%, +1.811%])
- 17× better than target
- **Conclusion: v2.1 is production-ready!**

---

## After Validation

### If Results Are Excellent/Good:
1. ✅ Accept v2.1 as final Phase C configuration
2. ✅ Document in thesis with multi-seed statistics
3. ✅ Proceed to Phase D (if applicable)
4. ✅ Publish results with confidence

### If Results Are Acceptable (High Variance):
1. Investigate outlier seeds (why did some fail?)
2. Check if outliers have common patterns
3. Consider running 2-3 more seeds to increase statistical power
4. May still proceed but note variance in thesis

### If Results Fail:
1. ❌ Re-evaluate v2.1 configuration
2. Check if seed 42 was an outlier
3. Investigate what makes some seeds succeed/fail
4. May need to adjust purity threshold or other hyperparameters

---

## Next Steps After Validation

1. **Document findings:**
   - Add multi-seed statistics to [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)
   - Update [FINAL_COMPARISON_v2.1_v2.2_v2.3.md](FINAL_COMPARISON_v2.1_v2.2_v2.3.md)

2. **Include in thesis:**
   - Mean ± std for MID-tier improvement
   - 95% confidence interval
   - Statistical significance (p-value)
   - Success rate

3. **Production deployment:**
   - Use v2.1 configuration for final results
   - Consider ensemble across multiple seeds for maximum robustness

4. **Publication:**
   - Multi-seed validation makes results more credible
   - Reviewers expect statistical rigor (this provides it)

---

## Cost Estimate

| Item | Amount | Cost |
|------|--------|------|
| LLM API calls (gpt-4o-mini) | ~150-250K tokens | $0.10-0.25 |
| Electricity (RTX 3070, 2 hours) | ~400W × 2h | $0.02-0.05 |
| **Total** | | **~$0.12-0.30** |

**Less than a coffee!** ☕

---

## FAQ

**Q: Can I run this on CPU instead of GPU?**
A: Yes, but it will be MUCH slower (~10× longer, 15-20 hours total). Not recommended.

**Q: Can I run more than 5 seeds?**
A: Yes! Edit `SEEDS` array in `run_5seeds_v2.1_parallel.sh` and `analyze_5seeds.py`.
   Recommended: 5-10 seeds for good statistical power.

**Q: What if I only have 4GB VRAM?**
A: Run 2 seeds in parallel max, or run sequentially (one at a time).

**Q: Can I pause and resume?**
A: Not directly. If you kill processes, results are lost. Let them complete.

**Q: How do I know when all are done?**
A: Monitor will show "5/5 completed" or run:
   ```bash
   ls phaseC_v2.1_seed*_metrics.json | wc -l
   # Should show 6 (5 new + 1 from earlier seed 42)
   ```

**Q: Do I need to re-run seed 42?**
A: No! We already have seed 42 results from earlier. Just run the other 4 seeds (100, 123, 456, 789).

---

## Contact

For issues or questions, see [phase_c/README.md](README.md) or review logs.

**Git Commit After Completion:**
```bash
git add phase_c/phaseC_v2.1_seed*_metrics.json
git add phase_c/phaseC_v2.1_5seeds_summary.json
git commit -m "Phase C v2.1: Multi-seed validation (5 seeds) - Statistical validation complete"
```

---

**Last Updated:** 2025-01-16
**Status:** Ready to run
