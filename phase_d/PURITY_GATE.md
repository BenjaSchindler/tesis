# Purity Gate: Quality-Based Class Filtering

**Technique:** Purity Gate
**Phase:** C v2 / D
**Type:** Pre-generation filtering
**Discovery:** Empirical analysis of Phase C v1 results (Seed 42)

## Problem Statement

In Phase C v1, we discovered that **class purity** is the single strongest predictor of whether synthetic data augmentation will improve or degrade performance:

| Purity | Class | F1 Delta | Outcome |
|--------|-------|----------|---------|
| 0.009 | ESFJ | -6.06% | Severe degradation |
| 0.016 | ISFJ | +0.03% | Neutral (wasted effort) |
| 0.025 | ISTJ | +0.98% | Slight improvement |
| 0.033 | ENFJ | +5.00% | Good improvement |
| 0.034 | ISFP | +7.68% | Excellent improvement |
| 0.041 | ESFP | -7.27% | Anomaly* |

*ESFP degradation due to low budget allocation (separate issue)

**Pattern:** Classes with purity < 0.025 consistently degraded or showed no improvement.

## What is Purity?

**Definition:** Purity measures how "pure" or separable a class is from other classes.

**Formula (in our pipeline):**
```python
def compute_purity(class_samples, all_samples, labels):
    """
    Compute class purity as the ratio of intra-class variance to total variance.

    Higher purity = class is well-separated from others
    Lower purity = class overlaps with other classes (contaminated)
    """
    # Simplified version - actual implementation may vary
    # See core/runner_phase2.py for exact calculation

    intra_class_distances = compute_intra_class_distances(class_samples)
    inter_class_distances = compute_inter_class_distances(class_samples, all_samples, labels)

    purity = inter_class_distances.mean() / (inter_class_distances.mean() + intra_class_distances.mean())
    return purity
```

**Interpretation:**
- **Purity ≈ 1.0:** Class is perfectly separated (very rare)
- **Purity 0.3-0.5:** Reasonably well-separated
- **Purity 0.1-0.3:** Some overlap with other classes
- **Purity < 0.1:** Heavily contaminated, poor separability

## Why Low Purity Causes Degradation

### Contamination Amplification Effect

1. **Low purity** → Anchors (representative samples) include contaminated examples
2. **Contaminated anchors** → LLM generates synthetics that mix characteristics of multiple classes
3. **Mixed synthetics** → Classifier learns blurred decision boundaries
4. **Blurred boundaries** → Performance degrades on test set

### Example: ESFJ (Purity = 0.009)

**Scenario:**
- ESFJ has very low purity (0.009)
- Anchors include samples that are partially ISFJ, ISFP, or other similar types
- LLM prompt: "Generate more samples like these..." (but "these" are contaminated)
- Generated synthetics: Mix of ESFJ + ISFJ + ISFP characteristics
- Classifier: Confused about true ESFJ boundary
- Result: F1 degrades from 0.226 → 0.212 (-6.06%)

## Solution: Purity Gate

**Idea:** Block classes with purity below a threshold from generating synthetics.

**Rationale:**
- If a class has low purity, generating synthetics will likely harm rather than help
- Better to skip generation and rely on original samples only
- Preserve classifier performance rather than risk degradation

### Implementation

**Step 1: Add argument to runner_phase2.py**
```python
parser.add_argument(
    '--purity-gate-threshold',
    type=float,
    default=0.025,
    help='Minimum purity required to generate synthetics (default: 0.025)'
)
```

**Step 2: Check purity before generation**
```python
def should_generate_class(class_name, class_quality_metrics, args):
    """
    Determine if a class should generate synthetics based on purity gate.
    """
    purity = class_quality_metrics['purity']

    if purity < args.purity_gate_threshold:
        print(f"⚠️ PURITY GATE: Skipping {class_name}")
        print(f"   Purity {purity:.3f} < threshold {args.purity_gate_threshold:.3f}")
        print(f"   Reason: High contamination risk - synthetic generation likely to degrade performance")
        return False

    # Also check existing quality gate
    quality = class_quality_metrics['quality']
    if quality < args.anchor_quality_threshold:
        print(f"⚠️ QUALITY GATE: Skipping {class_name} - quality={quality:.3f} < {args.anchor_quality_threshold}")
        return False

    return True
```

**Step 3: Apply in augmentation loop**
```python
for class_name in target_classes:
    # Compute quality metrics (purity, cohesion, separation)
    quality_metrics = compute_quality_metrics(class_name, X_train, y_train)

    # Check gates
    if not should_generate_class(class_name, quality_metrics, args):
        continue  # Skip this class

    # Proceed with generation
    generate_synthetics(class_name, ...)
```

## Threshold Selection: Why 0.025?

Based on Phase C v1 empirical analysis:

| Threshold | Classes Blocked | Classes Allowed | Impact |
|-----------|-----------------|-----------------|--------|
| 0.010 | ESFJ (0.009) | ISFJ (0.016), all others | Prevents worst degradation only |
| 0.020 | ESFJ, ISFJ | ISTJ (0.025), all others | Prevents degradation + neutral |
| **0.025** | **ESFJ, ISFJ** | **ISTJ, ENFJ, ISFP, ESFP** | **Optimal: blocks bad, allows good** |
| 0.030 | ESFJ, ISFJ, ISTJ | ENFJ (0.033), ISFP, ESFP | Too restrictive - blocks slight improvement |
| 0.035 | All except ESFP | ESFP (0.041) only | Far too restrictive |

**Chosen threshold: 0.025**
- Blocks: ESFJ (0.009), ISFJ (0.016)
- Allows: ISTJ (+0.98%), ENFJ (+5.00%), ISFP (+7.68%)
- Prevents: -6.06% and neutral results
- Enables: All positive improvements

## Expected Impact (Phase C v2)

### With Purity Gate at 0.025

**Predicted outcomes for Seed 42:**

| Class | Purity | Gate Decision | Phase C v1 Result | Phase C v2 Prediction |
|-------|--------|---------------|-------------------|----------------------|
| ESFJ | 0.009 | ❌ BLOCKED | -6.06% | **0.00% (no change)** |
| ISFJ | 0.016 | ❌ BLOCKED | +0.03% | **0.00% (no change)** |
| ISTJ | 0.025 | ✅ PASS | +0.98% | +0.98% (maintained) |
| ENFJ | 0.033 | ✅ PASS | +5.00% | +5.00% (maintained) |
| ISFP | 0.034 | ✅ PASS | +7.68% | +7.68% (maintained) |
| ESFP | 0.041 | ✅ PASS | -7.27% | +1% to +3% (budget fix) |

**MID-tier Mean:**
- Phase C v1: (+5.00 +7.68 +0.98 +0.03 -6.06 -7.27) / 6 = **+0.06%**
- Phase C v2 (purity gate): (+5.00 +7.68 +0.98 +0.00 +0.00 +2.00) / 6 = **+2.61%**

**Improvement:** +2.55 percentage points (43× better!)

*Note: ESFP improvement assumes budget fix (separate from purity gate)*

## Combination with Other Gates

The purity gate works **in combination** with existing quality mechanisms:

### 1. Quality Gate (existing)
- **Metric:** Overall quality score (combination of cohesion, purity, separation)
- **Threshold:** 0.25 (Phase C v2) or 0.30 (Phase C v1)
- **Purpose:** Block classes with overall poor quality

### 2. Purity Gate (new)
- **Metric:** Purity only
- **Threshold:** 0.025
- **Purpose:** Block classes with high contamination risk

### 3. Val-Gating (existing)
- **Metric:** F1 improvement on validation set
- **Threshold:** -0.02 tolerance
- **Purpose:** Reject synthetics that degrade validation performance

**Sequence:**
```
For each class:
  1. Compute quality metrics (purity, cohesion, separation)
  2. Check quality gate (quality >= 0.25) → Skip if fail
  3. Check purity gate (purity >= 0.025) → Skip if fail
  4. Generate synthetics
  5. Check val-gating (val_delta >= -0.02) → Reject if fail
  6. Accept synthetics
```

## Limitations and Edge Cases

### 1. ESFP Anomaly

ESFP has purity=0.041 (good) but degraded -7.27%. Why?
- **Root cause:** Low budget allocation (30× vs 70× for others)
- **Lesson:** Purity gate is necessary but not sufficient
- **Solution:** Also fix budget allocation in Phase C v2

### 2. Threshold Sensitivity

Purity threshold of 0.025 is based on **one seed (42)**.
- **Risk:** May not generalize to other seeds
- **Mitigation:** Run 5-seed validation to confirm threshold
- **Alternative:** Make threshold adaptive based on dataset characteristics

### 3. Small Sample Size Classes

Classes with very few samples (n < 50) may have unstable purity estimates.
- **Example:** ESFJ has only n=36 samples
- **Risk:** Purity calculation has high variance
- **Solution:** Use confidence intervals or bootstrap for purity estimation

### 4. Domain Shift

Purity is dataset-specific. A class may have low purity in MBTI_500 but high purity in another dataset.
- **Example:** ESFJ may be inherently ambiguous in personality typing
- **Generalization:** Purity gate should be recalibrated for each new dataset

## Comparison with Alternative Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Purity Gate (ours)** | Simple, effective, interpretable | Requires threshold tuning per dataset |
| **Post-hoc filtering** | Can fix mistakes after generation | Wastes API calls on bad synthetics |
| **Confidence-based** | Uses model predictions | Model may be overconfident on contaminated data |
| **Cluster-based** | More granular than class-level | Complex, computationally expensive |

## Implementation Checklist

- [ ] Add `--purity-gate-threshold` argument to runner_phase2.py
- [ ] Modify quality metrics computation to extract purity separately
- [ ] Add purity gate check before generation loop
- [ ] Add logging messages for blocked classes
- [ ] Update documentation
- [ ] Test on Phase C v1 dataset (should block ESFJ, ISFJ)
- [ ] Run experiment seed 42
- [ ] Analyze results vs Phase C v1
- [ ] If successful: Run 5-seed validation
- [ ] If robust: Add to Phase D baseline configuration

## Expected Timeline

- **Implementation:** 30 minutes
- **Testing:** 1 hour (run seed 42 experiment)
- **Analysis:** 30 minutes
- **Validation (5 seeds):** 4-5 hours (if promising)

## References

- **Discovery:** Phase C v1 empirical analysis (see [FINDINGS_SEED42.md](../phase_c/FINDINGS_SEED42.md))
- **Related work:** Class separability in imbalanced learning (Napierała & Stefanowski, 2016)
- **Contamination detection:** "Identifying Mislabeled Data" (Brodley & Friedl, 1999)

## Next Steps

1. **Implement purity gate** in runner_phase2.py
2. **Run Phase C v2** with purity gate + other quick wins
3. **Analyze results** - confirm purity threshold of 0.025 is optimal
4. **If successful:** Move to Phase D (hardness-aware anchors)
5. **If threshold needs tuning:** Run grid search over [0.015, 0.020, 0.025, 0.030]
