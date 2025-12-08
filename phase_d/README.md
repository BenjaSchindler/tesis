# Phase D: Hardness-Aware Anchor Selection + Purity Filtering

**Status:** Planning / Early Implementation
**Timeline:** 1-2 weeks
**Goal:** Achieve MID-tier improvement ≥ +0.25% (vs Phase C v1: +0.06%)

## Overview

Phase D combines multiple advanced techniques to address the root causes identified in Phase C:

1. **Purity Gate** (Quick Win - Implemented in Phase C v2)
2. **Hardness-Aware Anchor Selection** (Main Phase D technique)
3. **Adaptive Temperature** (Carried over from Phase C)
4. **Budget Optimization** (Enhanced from Phase C v2)

## Motivation

### Phase C v1 Results Summary

| Metric | Phase B | Phase C v1 | Target | Status |
|--------|---------|------------|--------|--------|
| MID-tier mean delta | -0.59% | +0.06% | +0.10% | ⚠️ Missed by 0.04pp |
| Overall macro F1 | +1.00% | -0.10% | +1.00% | ⚠️ Degraded |
| Success rate (MID-tier) | 0/8 | 4/6 | 5/6+ | Partial |

### Critical Finding: PURITY Determines Success

See [phase_c/FINDINGS_SEED42.md](../phase_c/FINDINGS_SEED42.md) for full analysis.

**Key insight:** Classes with purity < 0.025 degraded (-6% to -7%), while purity > 0.030 improved (+5% to +8%).

**Problem:** Low purity means anchors are contaminated with examples from other classes. Generating synthetics from bad anchors amplifies noise.

**Solution:**
1. **Phase C v2:** Add purity gate to block contaminated classes
2. **Phase D:** Select only high-quality anchors within each class (hardness-aware)

## Phase D Roadmap

### Week 1: Purity Gate + Hardness Basics

**Days 1-2: Phase C v2 (Quick Wins)**
- [x] Add purity gate (threshold: 0.025)
- [x] Lower quality gate (0.30 → 0.25)
- [x] Boost budget for small classes (n < 100)
- [ ] Run seed 42 experiment
- [ ] Analyze results vs Phase C v1

**Days 3-4: Hardness Scoring Implementation**
- [ ] Implement hardness calculation (distance to decision boundary)
- [ ] Add `--enable-hardness-selection` flag
- [ ] Classify anchors: EASY / LEARNABLE-HARD / AMBIGUOUS
- [ ] Test on single class (ENFJ)

**Days 5-7: Integration & Testing**
- [ ] Integrate with purity gate
- [ ] Run full experiment seed 42
- [ ] Compare: Phase C v1 vs C v2 vs D v1
- [ ] Statistical analysis (if promising, run 5 seeds)

### Week 2: Advanced Techniques (if needed)

**Days 8-10: Borderline-SMOTE**
- [ ] Identify borderline samples (k-NN with other classes)
- [ ] Prioritize borderline anchors for generation
- [ ] Expected: +90% success rate (from research)

**Days 11-12: Purity-Aware Clustering**
- [ ] Cluster within high-purity regions only
- [ ] Separate "pure core" from "contaminated boundary"
- [ ] Generate only from pure clusters

**Days 13-14: Final Validation**
- [ ] Run 25-seed experiment (if resources allow)
- [ ] Statistical significance test
- [ ] Write thesis section

## Techniques Overview

### 1. Purity Gate (Phase C v2)

**What:** Block classes with purity < threshold from generating synthetics

**Why:** Low purity = contaminated anchors = noisy synthetics = degradation

**Implementation:**
```python
if class_purity < args.purity_gate_threshold:
    print(f"⚠️ PURITY GATE: Skipping {class_name} - purity={class_purity:.3f} < {threshold}")
    continue  # Skip generation
```

**Parameters:**
- `--purity-gate-threshold`: Default 0.025 (based on Phase C analysis)

**Expected Impact:**
- Block ESFJ (purity=0.009) → prevent -6.06% degradation
- Block ISFJ (purity=0.016) → prevent neutral/slight degradation
- Allow ENFJ (0.033), ISFP (0.034) → maintain +5% to +8% improvements

See [PURITY_GATE.md](PURITY_GATE.md) for detailed documentation.

### 2. Hardness-Aware Anchor Selection (Phase D Main)

**What:** Select anchors based on their "hardness" (difficulty for classifier)

**Hardness Categories:**
1. **EASY:** Far from decision boundary, correctly classified with high confidence
   - Action: Skip or low priority
   - Reason: Don't teach anything new

2. **LEARNABLE-HARD:** Near boundary, challenging but learnable
   - Action: **High priority** for synthetic generation
   - Reason: These teach the most useful patterns

3. **AMBIGUOUS:** Very close to boundary, may be mislabeled or truly ambiguous
   - Action: Skip
   - Reason: Noise, may degrade performance

**Implementation:**
```python
def compute_hardness(sample, model, X_train, y_train):
    """
    Compute hardness score for anchor selection.

    Returns:
        - hardness: float in [0, 1], 0=easy, 1=ambiguous
        - category: "easy" | "learnable_hard" | "ambiguous"
    """
    # Get prediction probability
    prob = model.predict_proba([sample])[0]
    confidence = prob.max()

    # Distance to nearest other-class sample
    distances_other_class = compute_distances_to_other_classes(sample, X_train, y_train)
    min_dist_other = min(distances_other_class)

    # Hardness score
    if confidence > 0.8 and min_dist_other > threshold_far:
        return 0.1, "easy"
    elif confidence > 0.5 and min_dist_other > threshold_near:
        return 0.5, "learnable_hard"  # TARGET
    else:
        return 0.9, "ambiguous"
```

**Parameters:**
- `--enable-hardness-selection`: Enable hardness-aware selection
- `--hardness-easy-threshold`: Confidence threshold for "easy" (default: 0.8)
- `--hardness-ambiguous-threshold`: Confidence threshold for "ambiguous" (default: 0.5)
- `--hardness-distance-far`: Distance to other class for "easy" (default: 0.5)
- `--hardness-distance-near`: Distance to other class for "learnable" (default: 0.3)

**Expected Impact:**
- Reduce variance in MID-tier results (currently ±6%)
- Increase success rate from 67% to 90%+
- Better utilization of budget (generate from informative anchors)

### 3. Borderline-SMOTE (Phase D Week 2)

**What:** Variation of SMOTE focusing on borderline samples

**Original Paper:** "Borderline-SMOTE: A New Over-Sampling Method" (2005), updated in 2024 research

**Implementation:**
1. For each anchor, find k-nearest neighbors
2. Count how many neighbors are from different class
3. If m neighbors (where m/k ≈ 0.5), anchor is "borderline"
4. Prioritize borderline anchors for generation

**Why it works:** Borderline samples define the decision boundary. Augmenting near the boundary improves generalization.

**Expected success rate:** 85-90% (from Phase C research review)

### 4. Purity-Aware Clustering (Phase D Week 2)

**What:** Cluster only within high-purity regions of each class

**Problem:** Current clustering includes contaminated samples, creating bad prototypes

**Solution:**
1. Compute local purity for each sample (purity of its k-NN neighborhood)
2. Keep only samples with local_purity > threshold (e.g., 0.05)
3. Cluster on the "pure core" only
4. Generate synthetics from pure clusters

**Expected impact:** Higher quality cluster centroids → better prompts → better synthetics

## Experimental Plan

### Phase C v2 (Current)

**Configuration:**
```bash
--anchor-quality-threshold 0.25       # Lowered from 0.30
--purity-gate-threshold 0.025         # NEW
--f1-budget-thresholds 0.40 0.20      # Adjusted from 0.35 0.20
--small-class-budget-boost 2.0        # NEW: 2× for n < 100
```

**Expected Results (Seed 42):**
- ESFJ: Blocked by purity gate → no degradation (vs -6.06%)
- ESFP: 70× budget (vs 30×) + temp=0.5 → +1% to +3% (vs -7.27%)
- ENFJ: Continue improving → +5% to +7%
- ISFP: Continue improving → +7% to +9%
- MID-tier mean: **+0.15% to +0.25%** (vs +0.06%)

### Phase D v1 (Week 1)

**Additional Configuration:**
```bash
--enable-hardness-selection
--hardness-easy-threshold 0.8
--hardness-ambiguous-threshold 0.5
```

**Expected Results (Seed 42):**
- Lower variance in MID-tier (±3% vs ±6%)
- Success rate: 5/6 or 6/6 (vs 4/6)
- MID-tier mean: **+0.20% to +0.35%**
- Overall: +1.05% to +1.15% (vs Phase A: +1.00%)

### Phase D v2 (Week 2, if needed)

**Additional Configuration:**
```bash
--enable-borderline-smote
--borderline-k 10
--borderline-m-ratio 0.5
```

**Expected Results:**
- MID-tier mean: **+0.30% to +0.50%**
- Success rate: 90%+
- Overall: +1.15% to +1.30%

## Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| MID-tier mean delta | +0.10% | +0.25% | +0.40% |
| MID-tier success rate | 4/6 (67%) | 5/6 (83%) | 6/6 (100%) |
| Overall macro F1 | +1.00% | +1.10% | +1.20% |
| Statistical significance (25 seeds) | p < 0.10 | p < 0.05 | p < 0.01 |

## Implementation Checklist

### Phase C v2 (Days 1-2)
- [x] Document Phase C v1 findings
- [x] Create Phase D directory and docs
- [ ] Add purity gate to runner_phase2.py
- [ ] Lower quality gate to 0.25
- [ ] Add small class budget boost
- [ ] Create local_run_phaseC_v2.sh
- [ ] Run experiment seed 42
- [ ] Analyze results

### Phase D v1 (Days 3-7)
- [ ] Implement hardness scoring function
- [ ] Add hardness-based anchor filtering
- [ ] Test on ENFJ (single class)
- [ ] Integrate with full pipeline
- [ ] Run experiment seed 42
- [ ] Compare C v1 vs C v2 vs D v1
- [ ] If promising: Run 5-seed validation

### Phase D v2 (Days 8-14, optional)
- [ ] Implement borderline-SMOTE selection
- [ ] Implement purity-aware clustering
- [ ] Run experiments
- [ ] 25-seed validation (if budget allows)
- [ ] Write thesis section

## Cost Estimates

### Local Execution (with GPU)
- Phase C v2 (seed 42): ~45-60 min, ~$0.50 in API calls
- Phase D v1 (seed 42): ~60-90 min, ~$0.70 in API calls
- 5-seed validation: ~5-7 hours, ~$3.50 in API calls

### GCP Execution (if needed)
- Single seed (with GPU): ~45 min, ~$0.60 (compute + API)
- 25 seeds (5 VMs): ~5 hours, ~$20 total

## Related Documentation

- [phase_c/FINDINGS_SEED42.md](../phase_c/FINDINGS_SEED42.md) - Detailed analysis of Phase C results
- [phase_c/README.md](../phase_c/README.md) - Adaptive temperature technique
- [PURITY_GATE.md](PURITY_GATE.md) - Purity gate implementation details
- [HARDNESS_SELECTION.md](HARDNESS_SELECTION.md) - Hardness-aware selection (to be created)

## References

1. **Purity-based filtering:** Discovered through Phase C empirical analysis
2. **Hardness-aware selection:** arXiv 2502.05234 (2024)
3. **Borderline-SMOTE:** Han et al. 2005, updated in arXiv 2506.07295 (2024)
4. **Adaptive temperature:** arXiv 2502.05234, 2506.07295 (2024-2025)

## Notes

- Phase D builds on all Phase A, B, and C mechanisms
- Each technique is incremental - can be enabled/disabled independently
- Priority: Phase C v2 Quick Wins first (highest ROI), then Phase D if needed
