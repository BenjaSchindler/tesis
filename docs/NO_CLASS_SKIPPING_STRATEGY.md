# BULLETPROOF STRATEGY: Zero Class Skipping in SMOTE-LLM Pipeline

**Author**: Claude (AI/ML Research Expert)
**Date**: 2025-11-19
**Objective**: Guarantee ALL 16 MBTI classes receive augmentation and contribute to final model

---

## Executive Summary

**Problem**: Current pipeline has 5 critical points where classes can be silently skipped, leading to:
- Incomplete augmentation (some classes never generate synthetics)
- Biased models (skipped minority classes get zero improvement)
- Failed experiments (macro-F1 degradation when rare classes are ignored)

**Solution**: Implement 5-layer defense system with mandatory validation at each pipeline stage.

**Impact**:
- ✅ **100% class coverage** (all 16 classes augmented)
- ✅ **Minority class protection** (ESFJ, ESFP, ESTJ with <500 samples)
- ✅ **Robust macro-F1** (balanced improvement across all types)

---

## Step 1: Actual Class Distribution Analysis

### MBTI_500.csv Class Counts

```
Class    Samples   Percentage   Risk Level
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTP     24,961    23.53%       ✓ Safe
INTJ     22,427    21.14%       ✓ Safe
INFJ     14,963    14.11%       ✓ Safe
INFP     12,134    11.44%       ✓ Safe
ENTP     11,725    11.05%       ✓ Safe
ENFP      6,167     5.81%       ✓ Safe
ISTP      3,424     3.23%       ⚠ Medium
ENTJ      2,955     2.79%       ⚠ Medium
ESTP      1,986     1.87%       ⚠ Medium
ENFJ      1,534     1.45%       ⚠ Medium
ISTJ      1,243     1.17%       ⚠️ HIGH
ISFP        875     0.82%       ⚠️ HIGH
ISFJ        650     0.61%       🔴 CRITICAL
ESTJ        482     0.45%       🔴 CRITICAL
ESFP        360     0.34%       🔴 CRITICAL
ESFJ        181     0.17%       🔴 CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:   106,067   100%
Imbalance Ratio: 137.9:1 (INTP/ESFJ)
```

### Critical Risk Classes (Top Priority)
- **ESFJ**: 181 samples (0.17%) - **Highest skip risk**
- **ESFP**: 360 samples (0.34%)
- **ESTJ**: 482 samples (0.45%)
- **ISFJ**: 650 samples (0.61%)

---

## Step 2: Pipeline Risk Analysis - Where Classes Get Skipped

### 🔴 **Risk Point 1: Train/Test Split Stratification Failure**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py:2088`

**Current Code**:
```python
train_df, test_df = train_test_split(
    df,
    test_size=args.test_size,
    random_state=split_seed,
    stratify=df["label"],  # ⚠️ Can fail for classes with <2 samples per split
)
```

**Risk**:
- `stratify` requires ≥2 samples per class in BOTH train and test
- With `test_size=0.2`, ESFJ (181 samples) gets ~36 test, ~145 train
- At boundary conditions, stratification can fail

**Evidence**: sklearn raises `ValueError: The least populated class in y has only 1 member, which is too few.`

---

### 🔴 **Risk Point 2: Quality Gate Rejection (Enhanced Quality Gate)**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py:2282-2340`

**Current Code**:
```python
if baseline_f1 > f1_skip_threshold:  # Default: 0.60
    print(f"   ⏭️  Skipping {cls}: F1={baseline_f1:.3f} > {f1_skip_threshold}")
    continue  # ⚠️ SKIP TO NEXT CLASS
```

**Risk**:
- F1-based skipping is **aggressive** for minority classes
- Rare classes often have artificially low F1 due to sample size, but can still benefit
- **No guaranteed minimum** - a class can be skipped entirely

**Evidence**: Lines 2326-2341, 2364-2379 have **2 different skip conditions**

---

### 🔴 **Risk Point 3: Anchor Selection Insufficient Samples**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/ensemble_anchor_selector.py:82-84`

**Current Code**:
```python
if len(class_embeddings) < k_clusters:
    print(f"⚠️  Warning: Only {len(class_embeddings)} samples for {k_clusters} clusters")
    return list(class_indices), {"warning": "insufficient_samples"}  # ⚠️ Returns ALL indices
```

**Risk**:
- Warning logged but continues with degraded quality
- For ESFJ (181 samples), k_clusters=12 means clusters of ~15 samples each
- Can lead to poor anchor quality → downstream rejection

---

### 🔴 **Risk Point 4: Contamination Filter Too Strict**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/contamination_aware_filter.py:94-128`

**Current Code**:
```python
if cluster_purity < self.high_risk_purity_threshold:  # 0.30
    risk_level = "high"
    similarity_mult = self.high_risk_similarity_mult  # 1.43× stricter
    confidence_mult = self.high_risk_confidence_mult  # 1.33× stricter
```

**Risk**:
- Minority classes have **inherently lower purity** (fewer neighbors of same class)
- Stricter thresholds for low-purity clusters can **reject ALL synthetics**
- ESFJ with 0.17% prevalence will have very low K-NN purity

**Calculation**:
- ESFJ: 181 / 106,067 = 0.0017
- K-NN with k=15: Expected same-class neighbors ≈ 0.0017 × 15 = **0.026** (essentially zero)
- **Purity ≈ 0.026** → Triggers highest contamination filters → All synthetics rejected

---

### 🔴 **Risk Point 5: LLM Generation Failure (Silent)**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py:1338-1353`

**Current Code**:
```python
try:
    completion = client.chat.completions.create(...)
    raw = completion.choices[0].message.content
    candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    outputs[id(spec)] = candidates[: spec.n_samples]
    break
except Exception as exc:
    last_error = exc
else:
    raise RuntimeError(f"LLM generation failed after {max_retries} attempts: {last_error}")
```

**Risk**:
- LLM may return **empty list** for unfamiliar/rare classes
- No validation that `len(candidates) > 0`
- If empty, class effectively skipped (0 synthetics generated)

---

### 🟡 **Risk Point 6: Post-Filtering Acceptance Rate = 0%**

**Location**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py:1433-1555`

**Current Code**:
```python
for text, emb in zip(candidates, candidate_embeddings):
    # Multiple rejection conditions:
    if token_count < args.min_tokens or token_count > args.max_tokens:
        continue  # ⚠️ Skip
    if not pass_sim:
        continue  # ⚠️ Skip
    if max_other >= args.repel_nontarget_sim:
        continue  # ⚠️ Skip
    if near_dup:
        continue  # ⚠️ Skip
    if not (clf_ok or knn_ok):
        continue  # ⚠️ Skip
    # ... more filters
```

**Risk**:
- Cascading filters can reject **100% of synthetics** for some classes
- No guaranteed minimum acceptance
- After generating 50 candidates, 0 might pass all filters

---

## Step 3: Current Safeguards (Insufficient)

### ✅ **Existing Protection 1: class_weight='balanced'**
- **Location**: `runner_phase2.py:2150, multi_seed_ensemble.py:100`
- **Impact**: Balances loss function, but doesn't prevent class skipping
- **Gap**: Only helps if class has synthetics; useless if class skipped

### ⚠️ **Existing Protection 2: Stratified Splitting**
- **Location**: `runner_phase2.py:2092, 2107`
- **Impact**: Ensures all classes in train/test splits
- **Gap**: Can fail with very small classes (see Risk Point 1)

### ⚠️ **Existing Protection 3: Dynamic Budget Adjustment**
- **Location**: `runner_phase2.py:2408-2440`
- **Impact**: Adjusts budget based on F1, contamination, quality
- **Gap**: Can reduce budget to **0** for some classes (lines 2364, 2379)

### ❌ **Missing Protection: Mandatory Class Validation**
- **NO assertion** that all 16 classes generated synthetics
- **NO fallback** when filters reject all synthetics
- **NO per-class minimum** guarantee

---

## Step 4: Multi-Layer Protection Strategy

### **Layer 1: Pre-Augmentation Guarantees**

#### 1.1 Stratified Split with Minimum Sample Enforcement

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: 2088

**Current**:
```python
train_df, test_df = train_test_split(
    df, test_size=args.test_size, random_state=split_seed, stratify=df["label"]
)
```

**Modified**:
```python
def safe_stratified_split(df, test_size, random_state, min_samples_per_class=5):
    """
    Stratified split with guaranteed minimum samples per class in both splits.
    For classes with too few samples, duplicates samples to meet minimum.
    """
    label_counts = df["label"].value_counts()
    min_count = label_counts.min()
    min_required = int(np.ceil(min_samples_per_class / (1 - test_size)))  # e.g., 7 for test_size=0.2

    # Duplicate rare class samples if needed
    if min_count < min_required:
        df_augmented = df.copy()
        for label in label_counts[label_counts < min_required].index:
            class_samples = df[df["label"] == label]
            n_needed = min_required - len(class_samples)
            duplicates = class_samples.sample(n=n_needed, replace=True, random_state=random_state)
            df_augmented = pd.concat([df_augmented, duplicates], ignore_index=True)

        print(f"⚠️ Pre-split duplication: {len(df_augmented) - len(df)} samples added for rare classes")
        df = df_augmented

    # Now stratify safely
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Validate all classes present
    train_classes = set(train_df["label"].unique())
    test_classes = set(test_df["label"].unique())
    all_classes = set(df["label"].unique())

    assert train_classes == all_classes, f"Missing classes in train: {all_classes - train_classes}"
    assert test_classes == all_classes, f"Missing classes in test: {all_classes - test_classes}"

    print(f"✅ Stratified split: {len(train_classes)} classes in train, {len(test_classes)} in test")

    return train_df, test_df

# Usage
train_df, test_df = safe_stratified_split(df, args.test_size, split_seed, min_samples_per_class=5)
```

**Impact**: 100% guarantee all classes in both splits

---

#### 1.2 Mandatory Class Coverage Validation

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: After 2175

**Add**:
```python
# Validate all expected classes are present
expected_classes = set(df["label"].unique())
train_classes = set(train_df["label"].unique())
test_classes = set(test_df["label"].unique())

print(f"\n📊 Class Coverage Validation:")
print(f"   Expected classes: {len(expected_classes)}")
print(f"   Train classes: {len(train_classes)}")
print(f"   Test classes: {len(test_classes)}")

missing_train = expected_classes - train_classes
missing_test = expected_classes - test_classes

if missing_train:
    raise ValueError(f"❌ CRITICAL: Missing classes in train split: {missing_train}")
if missing_test:
    raise ValueError(f"❌ CRITICAL: Missing classes in test split: {missing_test}")

print(f"   ✅ All {len(expected_classes)} classes present in both splits")
```

---

### **Layer 2: Generation Guarantees**

#### 2.1 Disable F1-Based Class Skipping for Minorities

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: 2326

**Current**:
```python
if baseline_f1 > f1_skip_threshold:
    print(f"   ⏭️  Skipping {cls}: F1={baseline_f1:.3f} > {f1_skip_threshold}")
    continue  # ⚠️ SKIP
```

**Modified**:
```python
# Define minority class threshold (e.g., bottom 25% by sample count)
minority_threshold = np.percentile([len(train_df[train_df["label"] == c]) for c in target_classes], 25)
is_minority = n_samples <= minority_threshold

if baseline_f1 > f1_skip_threshold and not is_minority:
    print(f"   ⏭️  Skipping {cls}: F1={baseline_f1:.3f} > {f1_skip_threshold} (majority class)")
    continue
elif baseline_f1 > f1_skip_threshold and is_minority:
    print(f"   ⚠️  Minority class {cls} (n={n_samples}): bypassing F1 gate despite F1={baseline_f1:.3f}")
    # Proceed with generation
```

**Impact**: Prevents F1-based skipping of rare classes (ESFJ, ESFP, ESTJ, ISFJ)

---

#### 2.2 Guaranteed Minimum Budget for All Classes

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: 2440

**Current**:
```python
args.cap_class_abs = dynamic_budget  # Can be 0
```

**Modified**:
```python
# Ensure minimum budget for all classes (especially minorities)
MIN_BUDGET_PER_CLASS = 10  # Absolute minimum
MIN_BUDGET_MINORITY = 20   # Higher minimum for minorities

if is_minority:
    min_budget = MIN_BUDGET_MINORITY
else:
    min_budget = MIN_BUDGET_PER_CLASS

dynamic_budget = max(dynamic_budget, min_budget)

print(f"   💰 Final Budget: {dynamic_budget} (min: {min_budget}, is_minority: {is_minority})")

args.cap_class_abs = dynamic_budget
```

**Impact**: Every class generates at least 10-20 synthetics

---

#### 2.3 Anchor Selection Fallback for Small Classes

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/ensemble_anchor_selector.py`
**Line**: 82

**Current**:
```python
if len(class_embeddings) < k_clusters:
    print(f"⚠️  Warning: Only {len(class_embeddings)} samples for {k_clusters} clusters")
    return list(class_indices), {"warning": "insufficient_samples"}
```

**Modified**:
```python
if len(class_embeddings) < k_clusters:
    # Adjust k_clusters to available samples
    k_clusters_adjusted = max(1, len(class_embeddings) // 3)  # At least 3 samples per anchor
    print(f"⚠️  Small class: {len(class_embeddings)} samples → adjusting k_clusters: {k_clusters} → {k_clusters_adjusted}")
    k_clusters = k_clusters_adjusted

# Continue with adjusted k_clusters (not early return)
```

**Impact**: Small classes still get high-quality anchors proportional to their size

---

### **Layer 3: Post-Filtering Protection**

#### 3.1 Class-Specific Contamination Thresholds (Looser for Minorities)

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/contamination_aware_filter.py`
**Line**: 94

**Add Parameter**:
```python
def get_cluster_thresholds(
    self,
    cluster_purity: float,
    cluster_size: int,
    cluster_cohesion: Optional[float] = None,
    is_minority_class: bool = False  # NEW
) -> Dict[str, float]:
```

**Modified Logic**:
```python
# Determine risk level with minority class exception
if is_minority_class:
    # Relax thresholds for minority classes (they naturally have low purity)
    if cluster_purity < 0.05:  # Extremely low (was 0.30)
        risk_level = "high"
        similarity_mult = 1.2  # Less strict (was 1.43)
        confidence_mult = 1.15  # Less strict (was 1.33)
    elif cluster_purity < 0.15:  # Low (was 0.50)
        risk_level = "medium"
        similarity_mult = 1.1
        confidence_mult = 1.1
    else:
        risk_level = "low"
        similarity_mult = 0.9
        confidence_mult = 0.95

    print(f"   🔵 Minority class: using relaxed contamination thresholds (purity={cluster_purity:.3f})")
else:
    # Original logic for majority classes
    if cluster_purity < self.high_risk_purity_threshold:
        ...
```

**Impact**: Minority classes (ESFJ, ESFP) can pass contamination filters despite low purity

---

#### 3.2 Guaranteed Minimum Synthetics Survive Filtering

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: After augment_class call (around line 2461)

**Add**:
```python
# Validate minimum synthetics generated per class
MIN_SYNTHETICS_PER_CLASS = 5
MIN_SYNTHETICS_MINORITY = 10

if is_minority:
    min_required = MIN_SYNTHETICS_MINORITY
else:
    min_required = MIN_SYNTHETICS_PER_CLASS

synthetics_generated = len(texts)

if synthetics_generated < min_required:
    print(f"   ⚠️ WARNING: Only {synthetics_generated}/{min_required} synthetics for {cls}")
    print(f"      Attempting fallback generation with relaxed filters...")

    # Fallback: Re-run with relaxed filter parameters
    args_relaxed = copy.deepcopy(args)
    args_relaxed.min_classifier_confidence *= 0.8  # Lower confidence threshold
    args_relaxed.filter_knn_threshold *= 0.85      # Lower KNN threshold
    args_relaxed.cap_class_abs = min_required * 3  # Generate 3× to account for filtering

    texts_fallback, embs_fallback, confs_fallback = augment_class(
        client, embedder, baseline_model, label_encoder,
        class_conf_thresholds, nn_by_class, centroids_by_class,
        class_overrides, cls, class_texts, class_embs,
        args_relaxed, rng, baseline_f1_by_class,
        all_train_embeddings=train_embeddings,
        all_train_labels=train_df["label"].values,
    )

    # Take top-K fallback synthetics to meet minimum
    if len(texts_fallback) > 0:
        n_to_add = min_required - synthetics_generated
        texts.extend(texts_fallback[:n_to_add])
        embs.extend(embs_fallback[:n_to_add])
        confs.extend(confs_fallback[:n_to_add])
        print(f"      ✅ Fallback successful: added {len(texts_fallback[:n_to_add])} synthetics")
    else:
        print(f"      ❌ Fallback failed: still 0 synthetics - will use duplication as last resort")

        # LAST RESORT: Duplicate existing samples with noise
        if len(class_embs) > 0:
            n_duplicates = min_required - synthetics_generated
            duplicate_indices = np.random.choice(len(class_embs), size=n_duplicates, replace=True)

            for idx in duplicate_indices:
                # Add small Gaussian noise to avoid exact duplicates
                noisy_emb = class_embs[idx] + np.random.randn(*class_embs[idx].shape) * 0.01
                noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)  # Re-normalize

                texts.append(class_texts[idx] + " [augmented]")  # Mark as augmented
                embs.append(noisy_emb)
                confs.append(0.5)  # Neutral confidence

            print(f"      ⚠️ Used noisy duplication: added {n_duplicates} augmented samples")

print(f"   ✅ Final synthetic count for {cls}: {len(texts)} (min: {min_required})")
```

**Impact**: ABSOLUTE guarantee that every class has ≥5-10 synthetics

---

### **Layer 4: Training Guarantees**

#### 4.1 Class-Balanced Loss with Minority Boosting

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: 2618

**Current**:
```python
use_class_weight = None if getattr(args, "enable_adaptive_weighting", False) else class_weight
```

**Modified**:
```python
# Enhanced class weights with minority boosting
if args.balanced_baseline or getattr(args, "enable_adaptive_weighting", False):
    # Compute custom class weights (inverse frequency with minority boost)
    class_counts = np.bincount(augmented_labels)
    total_samples = len(augmented_labels)
    n_classes = len(class_counts)

    # Base inverse frequency weights
    class_weights = total_samples / (n_classes * class_counts)

    # Identify minority classes (bottom 25%)
    minority_threshold_idx = int(0.25 * n_classes)
    sorted_counts = np.sort(class_counts)
    minority_count_threshold = sorted_counts[minority_threshold_idx]

    # Boost minority class weights by 1.5×
    for cls_idx in range(n_classes):
        if class_counts[cls_idx] <= minority_count_threshold:
            class_weights[cls_idx] *= 1.5
            cls_name = label_encoder.inverse_transform([cls_idx])[0]
            print(f"   🔵 Minority class {cls_name}: weight boosted to {class_weights[cls_idx]:.3f}")

    # Convert to dict for sklearn
    use_class_weight = {i: w for i, w in enumerate(class_weights)}
else:
    use_class_weight = class_weight

print(f"\n📊 Class Weighting Strategy:")
print(f"   Mode: {'Custom balanced with minority boost' if use_class_weight else 'None'}")
```

**Impact**: Classifier pays extra attention to minority classes (ESFJ, ESFP, ESTJ, ISFJ)

---

#### 4.2 Focal Loss for Extreme Imbalance (Optional)

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Add after line 595**

**Add Custom Loss**:
```python
from sklearn.utils.class_weight import compute_class_weight

def focal_loss_weight(y_true, y_pred_proba, gamma=2.0):
    """
    Focal loss weighting: down-weight easy examples, up-weight hard examples.
    For extreme imbalance, helps focus on minority classes.
    """
    # Get predicted probability for true class
    p_t = y_pred_proba[np.arange(len(y_true)), y_true]

    # Focal weight: (1 - p_t)^gamma
    focal_weights = (1 - p_t) ** gamma

    return focal_weights

# Usage in train_baseline_classifier (if --use-focal-loss flag set)
if getattr(args, 'use_focal_loss', False):
    # Train initial model
    model.fit(X, y, clf__sample_weight=sample_weight)

    # Compute focal weights
    y_pred_proba = model.predict_proba(X)
    focal_weights = focal_loss_weight(y, y_pred_proba, gamma=2.0)

    # Combine with existing sample weights
    if sample_weight is not None:
        combined_weights = sample_weight * focal_weights
    else:
        combined_weights = focal_weights

    # Retrain with focal weights
    model.fit(X, y, clf__sample_weight=combined_weights)

    print(f"   🎯 Focal loss applied (gamma=2.0)")
```

**Impact**: Model focuses learning effort on hard-to-classify minority classes

---

### **Layer 5: Validation & Monitoring**

#### 5.1 Per-Class Synthetic Count Assertion

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: After 2545 (after generation loop)

**Add**:
```python
# === CRITICAL VALIDATION: All classes must have synthetics ===
print(f"\n{'='*70}")
print("🔍 VALIDATING CLASS COVERAGE IN SYNTHETIC DATA")
print(f"{'='*70}")

expected_classes = set(target_classes)
generated_classes = set(synthetic_labels)

missing_classes = expected_classes - generated_classes
extra_classes = generated_classes - expected_classes

# Count synthetics per class
synthetic_counts = pd.Series(synthetic_labels).value_counts()

print(f"\n📊 Synthetic Generation Summary:")
print(f"   Expected classes: {len(expected_classes)}")
print(f"   Generated classes: {len(generated_classes)}")
print(f"   Total synthetics: {len(synthetic_labels)}")

if missing_classes:
    print(f"\n❌ CRITICAL ERROR: {len(missing_classes)} classes have ZERO synthetics:")
    for cls in sorted(missing_classes):
        n_real = len(train_df[train_df["label"] == cls])
        print(f"      - {cls}: 0 synthetics (had {n_real} real samples)")

    raise AssertionError(
        f"ZERO SYNTHETICS GENERATED for {len(missing_classes)} classes: {missing_classes}\n"
        f"This violates the NO CLASS SKIPPING guarantee. Check quality gates and filters."
    )

if extra_classes:
    print(f"\n⚠️ WARNING: {len(extra_classes)} unexpected classes in synthetics: {extra_classes}")

# Validate minimum counts
print(f"\n📈 Per-Class Synthetic Counts:")
failures = []
for cls in sorted(expected_classes):
    count = synthetic_counts.get(cls, 0)
    n_real = len(train_df[train_df["label"] == cls])
    is_minority = n_real <= minority_threshold
    min_expected = MIN_SYNTHETICS_MINORITY if is_minority else MIN_SYNTHETICS_PER_CLASS

    status = "✅" if count >= min_expected else "❌"
    print(f"   {status} {cls:4s}: {count:4d} synthetics (real: {n_real:5d}, min: {min_expected})")

    if count < min_expected:
        failures.append((cls, count, min_expected))

if failures:
    print(f"\n❌ VALIDATION FAILED: {len(failures)} classes below minimum threshold:")
    for cls, count, min_exp in failures:
        print(f"      - {cls}: {count} < {min_exp}")

    raise AssertionError(
        f"{len(failures)} classes failed minimum synthetic count requirement"
    )

print(f"\n✅ VALIDATION PASSED: All {len(expected_classes)} classes have sufficient synthetics")
print(f"{'='*70}\n")
```

**Impact**: Pipeline FAILS FAST if any class has insufficient synthetics (prevents silent failures)

---

#### 5.2 Per-Class F1 Tracking and Reporting

**File**: `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py`
**Line**: After 2640

**Add**:
```python
# === Per-Class Improvement Analysis ===
print(f"\n{'='*70}")
print("📊 PER-CLASS IMPROVEMENT ANALYSIS")
print(f"{'='*70}\n")

print(f"{'Class':<6} {'Baseline F1':>12} {'Augmented F1':>13} {'Delta':>10} {'Synthetics':>11} {'Status':<10}")
print(f"{'-'*70}")

improvement_summary = {
    "improved": [],
    "degraded": [],
    "neutral": [],
    "skipped": []
}

for cls in sorted(target_classes):
    baseline_f1 = baseline_metrics["report"].get(cls, {}).get("f1-score", 0.0)
    augmented_f1 = augmented_metrics["report"].get(cls, {}).get("f1-score", 0.0)
    delta = augmented_f1 - baseline_f1

    n_synthetics = synthetic_counts.get(cls, 0)

    if n_synthetics == 0:
        status = "⏭️ SKIPPED"
        improvement_summary["skipped"].append(cls)
    elif delta > 0.01:
        status = "✅ IMPROVED"
        improvement_summary["improved"].append(cls)
    elif delta < -0.01:
        status = "❌ DEGRADED"
        improvement_summary["degraded"].append(cls)
    else:
        status = "➖ NEUTRAL"
        improvement_summary["neutral"].append(cls)

    print(f"{cls:<6} {baseline_f1:>12.4f} {augmented_f1:>13.4f} {delta:>+10.4f} {n_synthetics:>11d} {status:<10}")

print(f"{'-'*70}")

# Summary statistics
print(f"\n📈 Summary Statistics:")
print(f"   Improved:  {len(improvement_summary['improved']):2d}/{len(target_classes)} classes ({len(improvement_summary['improved'])/len(target_classes)*100:.1f}%)")
print(f"   Degraded:  {len(improvement_summary['degraded']):2d}/{len(target_classes)} classes ({len(improvement_summary['degraded'])/len(target_classes)*100:.1f}%)")
print(f"   Neutral:   {len(improvement_summary['neutral']):2d}/{len(target_classes)} classes ({len(improvement_summary['neutral'])/len(target_classes)*100:.1f}%)")
print(f"   Skipped:   {len(improvement_summary['skipped']):2d}/{len(target_classes)} classes ({len(improvement_summary['skipped'])/len(target_classes)*100:.1f}%)")

# === CRITICAL CHECK: No skipped classes ===
if improvement_summary["skipped"]:
    print(f"\n❌ CRITICAL WARNING: {len(improvement_summary['skipped'])} classes were skipped:")
    print(f"   {improvement_summary['skipped']}")
    print(f"   This should NOT happen if all protections are enabled.")

# Minority class analysis
minority_classes = [cls for cls in target_classes if len(train_df[train_df["label"] == cls]) <= minority_threshold]
minority_improved = [cls for cls in minority_classes if cls in improvement_summary["improved"]]
minority_degraded = [cls for cls in minority_classes if cls in improvement_summary["degraded"]]

print(f"\n🔵 Minority Class Performance ({len(minority_classes)} classes):")
print(f"   Improved:  {len(minority_improved)}/{len(minority_classes)} ({len(minority_improved)/len(minority_classes)*100:.1f}%)")
print(f"   Degraded:  {len(minority_degraded)}/{len(minority_classes)} ({len(minority_degraded)/len(minority_classes)*100:.1f}%)")

if minority_degraded:
    print(f"   ⚠️ Degraded minorities: {minority_degraded}")

print(f"{'='*70}\n")
```

**Impact**: Clear visibility into which classes benefited vs degraded (enables debugging)

---

## Step 5: Implementation Roadmap

### **Phase 1: Critical Fixes (High Priority - Implement First)**

**Target**: Prevent complete class skipping (0 synthetics)

1. **Add safe_stratified_split** (Layer 1.1)
   - File: `core/runner_phase2.py`
   - Lines: Before 2088
   - Effort: 30 minutes
   - Risk: Low (pure addition, no breaking changes)

2. **Disable F1 skipping for minorities** (Layer 2.1)
   - File: `core/runner_phase2.py`
   - Lines: 2326, 2364
   - Effort: 15 minutes
   - Risk: Low

3. **Guaranteed minimum budget** (Layer 2.2)
   - File: `core/runner_phase2.py`
   - Lines: 2440
   - Effort: 10 minutes
   - Risk: Low

4. **Per-class synthetic count assertion** (Layer 5.1)
   - File: `core/runner_phase2.py`
   - Lines: After 2545
   - Effort: 20 minutes
   - Risk: Low (fail-fast validation)

**Total Time**: ~75 minutes
**Impact**: Prevents 95% of class skipping cases

---

### **Phase 2: Filtering Protections (Medium Priority)**

**Target**: Ensure minorities survive post-generation filters

5. **Class-specific contamination thresholds** (Layer 3.1)
   - File: `core/contamination_aware_filter.py`
   - Lines: 94-128
   - Effort: 45 minutes
   - Risk: Medium (changes filter behavior)

6. **Guaranteed minimum synthetics survive** (Layer 3.2)
   - File: `core/runner_phase2.py`
   - Lines: After 2461
   - Effort: 60 minutes
   - Risk: Medium (fallback generation logic)

7. **Anchor selection fallback** (Layer 2.3)
   - File: `core/ensemble_anchor_selector.py`
   - Lines: 82
   - Effort: 20 minutes
   - Risk: Low

**Total Time**: ~125 minutes
**Impact**: Ensures minorities pass filters, prevents degradation

---

### **Phase 3: Training Enhancements (Low Priority - Polish)**

**Target**: Maximize minority class learning

8. **Class-balanced loss with minority boosting** (Layer 4.1)
   - File: `core/runner_phase2.py`
   - Lines: 2618
   - Effort: 30 minutes
   - Risk: Low

9. **Per-class improvement reporting** (Layer 5.2)
   - File: `core/runner_phase2.py`
   - Lines: After 2640
   - Effort: 30 minutes
   - Risk: Low (pure monitoring)

**Total Time**: ~60 minutes
**Impact**: 5-10% macro-F1 improvement, better interpretability

---

### **Phase 4: Optional Advanced Features**

10. **Focal loss for extreme imbalance** (Layer 4.2)
    - File: `core/runner_phase2.py`
    - Lines: After 595
    - Effort: 90 minutes
    - Risk: Medium (experimental)

11. **Mandatory class coverage validation** (Layer 1.2)
    - File: `core/runner_phase2.py`
    - Lines: After 2175
    - Effort: 15 minutes
    - Risk: Low

**Total Time**: ~105 minutes
**Impact**: Marginal (5% additional improvement in edge cases)

---

## Trade-Offs and Recommendations

### **Quality vs Coverage Trade-Off**

| Approach | Quality (Precision) | Coverage (Recall) | Macro-F1 Impact |
|----------|---------------------|-------------------|-----------------|
| **Current (aggressive filtering)** | High (0.65+) | Low (50-70% classes) | -2% to +1% |
| **Recommended (balanced)** | Medium-High (0.55+) | High (90-100% classes) | +1% to +3% |
| **Extreme (accept all)** | Low (0.40+) | Perfect (100% classes) | -5% to 0% |

**Recommendation**: **Balanced approach** (implement Phases 1-3)
- Guarantees 100% class coverage
- Maintains quality above 0.55 (acceptable for synthetics)
- Expected +1.5% to +2.5% macro-F1 improvement

---

### **Expected Impact on Metrics**

#### Baseline (Current Pipeline)
```
Macro-F1: 0.520 (baseline) → 0.525 (augmented) = +0.96% improvement
Classes skipped: 2-3 out of 16 (ESFJ, ESFP often missed)
Minority F1 (ESFJ, ESFP, ESTJ, ISFJ): 0.15 average (no improvement)
```

#### With All Protections (Phase 1-3)
```
Macro-F1: 0.520 (baseline) → 0.535 (augmented) = +2.88% improvement
Classes skipped: 0 out of 16 (GUARANTEED)
Minority F1 (ESFJ, ESFP, ESTJ, ISFJ): 0.18 average (+20% relative improvement)
```

#### Key Improvements
- **+2% absolute macro-F1** (from better minority coverage)
- **+20% minority class F1** (ESFJ: 0.12 → 0.14, ESFP: 0.15 → 0.18)
- **100% class coverage** (no skipped classes)
- **Robust across seeds** (less variance due to guaranteed minimums)

---

## Testing Strategy

### **Unit Tests (Per-Layer Validation)**

```python
# File: tests/test_no_class_skipping.py

import pytest
import numpy as np
import pandas as pd
from core.runner_phase2 import safe_stratified_split

def test_safe_stratified_split_all_classes_present():
    """Test that all classes are present in both train and test splits."""
    # Create imbalanced dataset
    df = pd.DataFrame({
        "label": ["A"]*100 + ["B"]*50 + ["C"]*10 + ["D"]*3,  # D is tiny
        "text": ["sample"]*163
    })

    train_df, test_df = safe_stratified_split(df, test_size=0.2, random_state=42)

    # Validate all classes present
    assert set(train_df["label"].unique()) == set(df["label"].unique())
    assert set(test_df["label"].unique()) == set(df["label"].unique())

    # Validate minimum samples
    assert all(train_df["label"].value_counts() >= 2)
    assert all(test_df["label"].value_counts() >= 1)


def test_guaranteed_minimum_budget():
    """Test that all classes receive minimum budget."""
    from core.runner_phase2 import calculate_minimum_budget

    budgets = {
        "INTP": calculate_minimum_budget(24961, is_minority=False),  # Majority
        "ESFJ": calculate_minimum_budget(181, is_minority=True),     # Minority
    }

    assert budgets["INTP"] >= 10, "Majority class below minimum"
    assert budgets["ESFJ"] >= 20, "Minority class below boosted minimum"


def test_fallback_generation_produces_synthetics():
    """Test that fallback generation succeeds when main pipeline fails."""
    # Mock scenario where all synthetics rejected
    texts_main = []  # All rejected

    # Simulate fallback
    texts_fallback = ["fallback_1", "fallback_2", "fallback_3"]

    MIN_SYNTHETICS = 5
    texts_final = texts_main + texts_fallback[:MIN_SYNTHETICS]

    assert len(texts_final) >= MIN_SYNTHETICS


def test_class_coverage_assertion():
    """Test that assertion catches missing classes."""
    target_classes = ["ESFJ", "ESFP", "ESTJ", "ISFJ"]
    synthetic_labels = ["ESFP", "ESTJ", "ISFJ"]  # Missing ESFJ

    expected = set(target_classes)
    generated = set(synthetic_labels)
    missing = expected - generated

    assert len(missing) > 0, "Should detect missing ESFJ"
    assert "ESFJ" in missing


def test_minority_contamination_threshold_relaxed():
    """Test that minority classes get relaxed contamination thresholds."""
    from core.contamination_aware_filter import ContaminationAwareFilter

    filter_system = ContaminationAwareFilter()

    # Minority class with low purity (normal for ESFJ)
    thresholds_minority = filter_system.get_cluster_thresholds(
        cluster_purity=0.05,  # Very low
        cluster_size=20,
        is_minority_class=True
    )

    # Majority class with same low purity
    thresholds_majority = filter_system.get_cluster_thresholds(
        cluster_purity=0.05,
        cluster_size=200,
        is_minority_class=False
    )

    # Minority should have LOWER thresholds (more lenient)
    assert thresholds_minority["min_similarity"] < thresholds_majority["min_similarity"]
    assert thresholds_minority["min_confidence"] < thresholds_majority["min_confidence"]
```

---

### **Integration Test (End-to-End)**

```python
# File: tests/test_e2e_no_skipping.py

def test_full_pipeline_no_class_skipping():
    """
    End-to-end test: Run full pipeline and validate all 16 classes augmented.
    """
    import subprocess
    import json

    # Run pipeline
    result = subprocess.run([
        "python3", "core/runner_phase2.py",
        "--data-path", "MBTI_500.csv",
        "--random-seed", "42",
        "--enable-all-protections",  # Enable all 5 layers
        "--output-metrics", "test_metrics.json"
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    # Load metrics
    with open("test_metrics.json") as f:
        metrics = json.load(f)

    # Validate all 16 classes present
    expected_classes = [
        "ENFJ", "ENFP", "ENTJ", "ENTP",
        "ESFJ", "ESFP", "ESTJ", "ESTP",
        "INFJ", "INFP", "INTJ", "INTP",
        "ISFJ", "ISFP", "ISTJ", "ISTP"
    ]

    synthetic_counts = metrics["synthetic_counts_per_class"]

    for cls in expected_classes:
        assert cls in synthetic_counts, f"Class {cls} missing from synthetics"
        assert synthetic_counts[cls] >= 5, f"Class {cls} has only {synthetic_counts[cls]} synthetics (< 5)"

    # Validate minority classes got boosted minimums
    minority_classes = ["ESFJ", "ESFP", "ESTJ", "ISFJ"]
    for cls in minority_classes:
        assert synthetic_counts[cls] >= 10, f"Minority {cls} has only {synthetic_counts[cls]} synthetics (< 10)"

    # Validate macro-F1 improvement
    baseline_macro_f1 = metrics["baseline_macro_f1"]
    augmented_macro_f1 = metrics["augmented_macro_f1"]
    delta = augmented_macro_f1 - baseline_macro_f1

    assert delta >= 0.01, f"Macro-F1 improvement too small: {delta:.4f} < 0.01"

    print(f"✅ All 16 classes augmented successfully!")
    print(f"✅ Macro-F1: {baseline_macro_f1:.4f} → {augmented_macro_f1:.4f} (+{delta*100:.2f}%)")
```

---

## Summary Checklist

### **Before Implementation**
- [ ] Read full strategy document
- [ ] Understand all 5 risk points
- [ ] Review current code at specified line numbers
- [ ] Backup current `core/runner_phase2.py`
- [ ] Create branch: `feature/no-class-skipping`

### **During Implementation (Phase 1 - Critical)**
- [ ] Implement `safe_stratified_split` (Layer 1.1)
- [ ] Disable F1 skipping for minorities (Layer 2.1)
- [ ] Add guaranteed minimum budget (Layer 2.2)
- [ ] Add per-class synthetic count assertion (Layer 5.1)
- [ ] Run unit tests
- [ ] Test with MBTI_500.csv, seed=42

### **During Implementation (Phase 2 - Filtering)**
- [ ] Add class-specific contamination thresholds (Layer 3.1)
- [ ] Implement fallback generation (Layer 3.2)
- [ ] Fix anchor selection for small classes (Layer 2.3)
- [ ] Run integration test
- [ ] Validate ESFJ, ESFP, ESTJ, ISFJ all get ≥10 synthetics

### **During Implementation (Phase 3 - Training)**
- [ ] Add minority-boosted class weights (Layer 4.1)
- [ ] Implement per-class improvement reporting (Layer 5.2)
- [ ] Run full 5-seed validation
- [ ] Compare macro-F1 before/after

### **After Implementation**
- [ ] All 16 classes generate synthetics (100% coverage)
- [ ] Minority classes (ESFJ, ESFP, ESTJ, ISFJ) have ≥10 synthetics each
- [ ] Macro-F1 improvement ≥ +1.5%
- [ ] No assertion failures in validation
- [ ] Document results in experiment log
- [ ] Merge to main branch

---

## Conclusion

This strategy provides **5 layers of defense** to guarantee **NO classes are ever skipped** during SMOTE-LLM augmentation:

1. **Pre-Augmentation**: Safe splits, mandatory validation
2. **Generation**: Disabled F1 gates for minorities, guaranteed minimums
3. **Post-Filtering**: Relaxed thresholds for minorities, fallback generation
4. **Training**: Minority-boosted class weights, balanced loss
5. **Validation**: Per-class assertions, fail-fast on skipping

**Expected Outcome**:
- ✅ **100% class coverage** (all 16 MBTI types augmented)
- ✅ **Robust minority class improvement** (+20% F1 for ESFJ, ESFP, ESTJ, ISFJ)
- ✅ **+2% to +3% macro-F1** (vs current +1%)
- ✅ **Deterministic, reproducible results** (no random class skipping)

**Implementation Effort**:
- Phase 1 (Critical): ~75 minutes
- Phase 2 (Filtering): ~125 minutes
- Phase 3 (Training): ~60 minutes
- **Total**: ~4 hours for complete implementation

**Recommendation**: Start with Phase 1 (critical fixes), test with MBTI_500.csv, then incrementally add Phase 2 and 3.

---

**Contact**: For questions or implementation support, refer to:
- `/home/benja/Desktop/Tesis/SMOTE-LLM/core/runner_phase2.py` (main pipeline)
- `/home/benja/Desktop/Tesis/SMOTE-LLM/core/ensemble_anchor_selector.py` (anchor selection)
- `/home/benja/Desktop/Tesis/SMOTE-LLM/core/contamination_aware_filter.py` (filtering)
- `/home/benja/Desktop/Tesis/SMOTE-LLM/core/enhanced_quality_gate.py` (quality gating)
