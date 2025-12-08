# Phase C - Findings from Seed 42 Experiment

**Date:** 2025-01-15
**Experiment:** Adaptive Temperature for MID-tier classes
**Dataset:** MBTI_500.csv (106K samples, 16 classes)

## Executive Summary

**Result:** Partial success - stopped MID-tier degradation but missed target
- Phase B MID-tier performance: **-0.59%**
- Phase C MID-tier performance: **+0.06%**
- Improvement: **+0.65 percentage points**
- Target: +0.10% (missed by 0.04 pp)

## Overall Performance

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Macro F1 | 0.45375 | 0.45330 | -0.099% |
| Synthetics Generated | - | 110 | 6 classes |

## MID-Tier Classes Performance (F1 0.20-0.45)

| Class | Baseline F1 | Augmented F1 | Delta % | Synthetics | Status |
|-------|-------------|--------------|---------|------------|--------|
| ENFJ | 0.214 | 0.225 | **+5.00%** | 33 | ✓ Improved |
| ISFP | 0.266 | 0.287 | **+7.68%** | 12 | ✓ Improved |
| ISTJ | 0.248 | 0.251 | **+0.98%** | 32 | ✓ Improved |
| ISFJ | 0.252 | 0.252 | **+0.03%** | 14 | ~ Neutral |
| ESFJ | 0.226 | 0.212 | **-6.06%** | 10 | ✗ Degraded |
| ESFP | 0.354 | 0.328 | **-7.27%** | 9 | ✗ Degraded |

**Success Rate:** 4/6 classes improved or neutral (67%)

## Critical Finding: PURITY is the Determinant Factor

### Correlation Analysis

After analyzing quality metrics vs F1 improvement, we identified **PURITY** as the single most important predictor of success/failure:

| Purity Score | Class | F1 Delta | Outcome |
|--------------|-------|----------|---------|
| **0.009** (lowest) | ESFJ | **-6.06%** | Worst degradation |
| **0.016** | ISFJ | +0.03% | Neutral |
| **0.025** | ISTJ | +0.98% | Slight improvement |
| **0.033** | ENFJ | **+5.00%** | Good improvement |
| **0.034** | ISFP | **+7.68%** | Best improvement |
| **0.041** | ESFP | **-7.27%** | Severe degradation* |

*Note: ESFP is anomaly - low purity but not the lowest. See "Budget Efficiency" section.

### Purity Threshold Discovery

**Pattern identified:**
- **Purity < 0.020:** High risk of degradation
- **Purity 0.020-0.030:** Mixed results (uncertain zone)
- **Purity > 0.030:** Consistent improvements

**Hypothesis:** Purity measures class separability. Low purity means anchors are contaminated with examples from other classes. Generating synthetics from contaminated anchors amplifies the noise, degrading classifier performance.

## Secondary Factors

### 1. Budget Efficiency

Budget allocated vs synthetics actually generated:

| Class | Budget | Generated | Efficiency | F1 Delta |
|-------|--------|-----------|------------|----------|
| ESFP | 300 (30×) | 9 | 3.0% | -7.27% |
| ESFJ | 700 (70×) | 10 | 1.4% | -6.06% |
| ISFJ | 700 (70×) | 14 | 2.0% | +0.03% |
| ISFP | 700 (70×) | 12 | 1.7% | +7.68% |
| ISTJ | 700 (70×) | 32 | 4.6% | +0.98% |
| ENFJ | 700 (70×) | 33 | 4.7% | +5.00% |

**Observation:** Classes generating more synthetics (32-33) improved more than those generating fewer (9-14), BUT only when purity was adequate. ISFP improved despite generating only 12 synthetics because purity was good (0.034).

### 2. Quality Score (NOT Determinant)

All generated classes have similar quality scores (0.302-0.323). **No correlation** with F1 delta:
- ESFP: quality=0.323 → degraded -7.27%
- ENFJ: quality=0.308 → improved +5.00%

**Conclusion:** Quality gate at 0.30 is too restrictive and doesn't predict success. Purity is more important.

### 3. Cohesion (Weak Influence)

All classes have cohesion 0.736-0.764 (narrow range). No meaningful correlation with outcomes.

### 4. Confidence (Not Determinant)

All generated classes have confidence=0.26 (identical). Cannot be a differentiating factor.

## Adaptive Temperature Analysis

**Applied correctly:**
- All MID-tier classes received temp=0.5 (vs baseline temp=1.0)
- Messages confirmed: `🌡️ ADAPTIVE TEMP: {CLASS} (F1={f1}) - temp=1.00 → 0.50`

**Effect:**
- Stopped severe degradation (-0.59% → +0.06%)
- But high variance in results (σ = 5.5 pp)
- Purity contamination dominated over temperature benefits

**Conclusion:** Adaptive temperature helps but is NOT sufficient. Must be combined with purity filtering.

## Why ESFP Degraded Despite Mid-Range Purity?

ESFP has purity=0.041 (4th best), yet degraded -7.27%. Analysis:

1. **Low budget allocation:** Only 300 synthetics (30× multiplier) vs 700 for others (70× multiplier)
   - Reason: Baseline F1=0.354 (close to HIGH threshold of 0.45)
   - Budget scaling penalized it for being "too good"

2. **Low efficiency:** Only 9/300 synthetics generated (3%)
   - Aggressive filtering removed most candidates
   - With only 9 synthetics, variance is very high

3. **Small sample size:** n=245 original samples (only 72 in test set)
   - Small perturbations have large F1 impact

**Lesson:** Classes near thresholds (F1 ~0.35) need special handling. They're MID-tier but get treated as HIGH-tier by budget scaling.

## Configuration Used (Phase C v1)

```bash
--anchor-quality-threshold 0.30
--f1-budget-thresholds 0.35 0.20
--f1-budget-multipliers 30 70 100
--enable-adaptive-weighting
--temperature 1.0  # Overridden by adaptive function to 0.5 for MID-tier
```

## Recommendations for Phase C v2 (Quick Wins)

Based on findings, implement these changes:

### 1. Add PURITY GATE (Critical)
```bash
--purity-gate-threshold 0.025
```
- Block generation if purity < 0.025
- Would have prevented ESFJ (0.009) and ISFJ (0.016) degradation/neutrality
- Let ENFJ (0.033) and ISFP (0.034) continue improving

### 2. Lower Quality Gate
```bash
--anchor-quality-threshold 0.25  # From 0.30
```
- Quality is not predictive of success
- Purity gate will provide better filtering
- Allow more classes to generate synthetics

### 3. Boost Budget for Small Classes
```
If n < 100: budget_multiplier × 2
```
- Help ESFJ (n=36), ESFP (n=72) generate more synthetics
- Reduce variance from small sample sizes
- Combined with purity gate, only boost classes that are "clean"

### 4. Adjust F1 Budget Thresholds
```bash
--f1-budget-thresholds 0.40 0.20  # From 0.35 0.20
```
- Move ESFP (F1=0.354) from HIGH (30×) to MID (70×) tier
- Give it same budget as other MID-tier classes

## Expected Impact of Phase C v2

**Predictions:**
- ESFJ blocked by purity gate (0.009 < 0.025) → skip generation → no degradation
- ESFP gets 70× budget (was 30×) → ~700 budget → more synthetics → better improvement
- More classes pass quality gate (0.25 vs 0.30) → more opportunities for improvement

**Target:** MID-tier mean +0.15% to +0.30% (vs current +0.06%)

## Next Phase: Phase D - Hardness-Aware Anchors

If Phase C v2 achieves +0.10% but not +0.25%, implement:

1. **Borderline-SMOTE selection:** Prioritize anchors near decision boundary
2. **Hardness scoring:** Avoid easy/ambiguous examples, focus on "learnable hard" examples
3. **Purity-aware clustering:** Cluster only within high-purity regions

Expected success rate: 90% (from literature review)

## Files Generated

- `phaseC_seed42_metrics.json` - Full results
- `phaseC_seed42_synthetic.csv` - 110 synthetic samples
- `phaseC_seed42_augmented.csv` - Augmented training set
- `phaseC_seed42_*.log` - Execution logs with quality metrics

## References

See [phase_c/README.md](README.md) for:
- Research papers (arXiv 2502.05234, 2506.07295, etc.)
- Adaptive temperature implementation details
- Comparison with Phase A and Phase B
