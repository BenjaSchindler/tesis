#!/usr/bin/env python3
"""
SMOTE + Geometric Filter Experiment

Tests whether applying cascade_l1 filter to SMOTE-generated samples improves quality.

Hypothesis: SMOTE can generate outliers; filtering them geometrically may help.

Pipeline:
1. Generate 2x target samples with SMOTE
2. Apply cascade_l1 filter
3. Select top-N by quality score
4. Compare against unfiltered SMOTE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "smote_filtered"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Datasets
DATASETS = ["sms_spam_10shot", "sms_spam_25shot",
            "20newsgroups_10shot", "20newsgroups_25shot",
            "hate_speech_davidson_10shot", "hate_speech_davidson_25shot"]

# SMOTE config
SYNTHETIC_PER_CLASS = 50
SMOTE_OVERSAMPLE_RATIO = 2.0  # Generate 2x, filter to 1x

# Filter config
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}

# Methods to compare
METHODS = ["baseline", "smote", "smote_filtered"]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SMOTEFilterResult:
    """Result for one method on one dataset."""
    dataset: str
    method: str
    baseline_f1: float
    augmented_f1: float
    delta: float
    n_generated: int
    n_filtered: int


# ============================================================================
# SMOTE GENERATION
# ============================================================================

def generate_smote_samples(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    n_per_class: int,
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """Generate SMOTE samples for all classes."""
    classes = list(set(train_labels))
    all_synth_emb = []
    all_synth_labels = []

    for cls in classes:
        class_mask = train_labels == cls
        n_class = class_mask.sum()

        k_neighbors = min(5, n_class - 1)
        if k_neighbors < 1:
            continue

        binary_labels = class_mask.astype(int)
        target_count = n_class + n_per_class

        try:
            smote = SMOTE(
                k_neighbors=k_neighbors,
                sampling_strategy={1: target_count},
                random_state=seed
            )
            X_resampled, y_resampled = smote.fit_resample(train_emb, binary_labels)

            new_samples = X_resampled[len(train_emb):]
            if len(new_samples) > 0:
                all_synth_emb.append(new_samples[:n_per_class])
                all_synth_labels.extend([cls] * min(len(new_samples), n_per_class))

        except Exception as e:
            print(f"    SMOTE error for {cls}: {e}")
            continue

    if all_synth_emb:
        return np.vstack(all_synth_emb), all_synth_labels
    return np.array([]).reshape(0, train_emb.shape[1]), []


def filter_smote_samples(
    smote_emb: np.ndarray,
    smote_labels: List[str],
    real_emb: np.ndarray,
    real_labels: np.ndarray,
    target_per_class: int,
    filter_obj: FilterCascade
) -> Tuple[np.ndarray, List[str]]:
    """Filter SMOTE samples using geometric cascade filter."""
    classes = list(set(smote_labels))
    smote_labels_arr = np.array(smote_labels)

    filtered_emb = []
    filtered_labels = []

    for cls in classes:
        # Get SMOTE samples for this class
        class_mask = smote_labels_arr == cls
        cls_smote_emb = smote_emb[class_mask]

        if len(cls_smote_emb) == 0:
            continue

        # Filter using cascade
        real_class_mask = real_labels == cls
        if not real_class_mask.any():
            # No real samples, take first N
            filtered_emb.append(cls_smote_emb[:target_per_class])
            filtered_labels.extend([cls] * min(len(cls_smote_emb), target_per_class))
            continue

        # Compute quality scores
        class_anchor = real_emb[real_class_mask].mean(axis=0)
        scores, _ = filter_obj.compute_quality_scores(
            cls_smote_emb, class_anchor, real_emb, real_labels, cls
        )

        # Select top-N by score
        n_select = min(target_per_class, len(cls_smote_emb))
        top_idx = np.argsort(scores)[-n_select:]

        filtered_emb.append(cls_smote_emb[top_idx])
        filtered_labels.extend([cls] * n_select)

    if filtered_emb:
        return np.vstack(filtered_emb), filtered_labels
    return np.array([]).reshape(0, real_emb.shape[1]), []


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


def evaluate_augmented(
    train_emb, train_labels, test_emb, test_labels,
    synth_emb, synth_labels
) -> float:
    """Evaluate with augmented data."""
    if len(synth_emb) == 0:
        return evaluate_baseline(train_emb, train_labels, test_emb, test_labels)

    aug_emb = np.vstack([train_emb, synth_emb])
    aug_labels = list(train_labels) + list(synth_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(
    dataset_name: str,
    embed_model: SentenceTransformer,
    filter_obj: FilterCascade
) -> List[SMOTEFilterResult]:
    """Run experiment for one dataset."""
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]

    # Embed
    print("  Embedding...")
    train_emb = embed_model.encode(train_texts, show_progress_bar=False)
    test_emb = embed_model.encode(data["test_texts"], show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    results = []

    # Baseline
    baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, test_labels)
    print(f"  Baseline F1: {baseline_f1:.4f}")

    results.append(SMOTEFilterResult(
        dataset=dataset_name,
        method="baseline",
        baseline_f1=baseline_f1,
        augmented_f1=baseline_f1,
        delta=0.0,
        n_generated=0,
        n_filtered=0
    ))

    # SMOTE (standard)
    print("  Running SMOTE...")
    smote_emb, smote_labels = generate_smote_samples(
        train_emb, train_labels_arr, SYNTHETIC_PER_CLASS
    )
    smote_f1 = evaluate_augmented(
        train_emb, train_labels_arr, test_emb, test_labels,
        smote_emb, smote_labels
    )
    smote_delta = (smote_f1 - baseline_f1) * 100
    print(f"  SMOTE F1: {smote_f1:.4f} ({smote_delta:+.2f}pp)")

    results.append(SMOTEFilterResult(
        dataset=dataset_name,
        method="smote",
        baseline_f1=baseline_f1,
        augmented_f1=smote_f1,
        delta=smote_delta,
        n_generated=len(smote_emb),
        n_filtered=len(smote_emb)
    ))

    # SMOTE + Filter
    print("  Running SMOTE + Filter...")

    # Generate 2x samples
    oversample_n = int(SYNTHETIC_PER_CLASS * SMOTE_OVERSAMPLE_RATIO)
    smote_over_emb, smote_over_labels = generate_smote_samples(
        train_emb, train_labels_arr, oversample_n
    )

    if len(smote_over_emb) > 0:
        # Filter to target
        filtered_emb, filtered_labels = filter_smote_samples(
            smote_over_emb, smote_over_labels,
            train_emb, train_labels_arr,
            SYNTHETIC_PER_CLASS, filter_obj
        )

        filtered_f1 = evaluate_augmented(
            train_emb, train_labels_arr, test_emb, test_labels,
            filtered_emb, filtered_labels
        )
    else:
        filtered_emb = np.array([]).reshape(0, train_emb.shape[1])
        filtered_labels = []
        filtered_f1 = baseline_f1

    filtered_delta = (filtered_f1 - baseline_f1) * 100
    print(f"  SMOTE+Filter F1: {filtered_f1:.4f} ({filtered_delta:+.2f}pp)")
    print(f"    Generated: {len(smote_over_emb)}, Filtered to: {len(filtered_emb)}")

    results.append(SMOTEFilterResult(
        dataset=dataset_name,
        method="smote_filtered",
        baseline_f1=baseline_f1,
        augmented_f1=filtered_f1,
        delta=filtered_delta,
        n_generated=len(smote_over_emb),
        n_filtered=len(filtered_emb)
    ))

    # Compare
    improvement = filtered_delta - smote_delta
    print(f"  Filter improvement vs SMOTE: {improvement:+.2f}pp")

    return results


def main():
    print("=" * 70)
    print("SMOTE + GEOMETRIC FILTER EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Synthetic per class: {SYNTHETIC_PER_CLASS}")
    print(f"  Oversample ratio: {SMOTE_OVERSAMPLE_RATIO}x")
    print(f"  Filter: cascade level={FILTER_CONFIG['filter_level']}")

    # Initialize
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    filter_obj = FilterCascade(**FILTER_CONFIG)

    all_results = []

    for dataset_name in DATASETS:
        try:
            results = run_experiment(dataset_name, embed_model, filter_obj)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[SMOTEFilterResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: SMOTE + FILTER")
    print("=" * 70)

    print(f"\n{'Dataset':<30} {'SMOTE':<15} {'SMOTE+Filter':<15} {'Improvement':<12}")
    print("-" * 75)

    for dataset in DATASETS:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        if not dataset_results:
            continue

        smote_r = next((r for r in dataset_results if r.method == "smote"), None)
        filtered_r = next((r for r in dataset_results if r.method == "smote_filtered"), None)

        if smote_r and filtered_r:
            improvement = filtered_r.delta - smote_r.delta
            print(f"{dataset:<30} {smote_r.delta:+12.2f}pp {filtered_r.delta:+12.2f}pp "
                  f"{improvement:+10.2f}pp")

    # Overall stats
    print("\n" + "-" * 75)

    smote_results = [r for r in all_results if r.method == "smote"]
    filtered_results = [r for r in all_results if r.method == "smote_filtered"]

    if smote_results and filtered_results:
        avg_smote = np.mean([r.delta for r in smote_results])
        avg_filtered = np.mean([r.delta for r in filtered_results])
        avg_improvement = avg_filtered - avg_smote

        n_better = sum(1 for s, f in zip(smote_results, filtered_results)
                      if f.delta > s.delta)

        print(f"Average SMOTE delta: {avg_smote:+.2f}pp")
        print(f"Average SMOTE+Filter delta: {avg_filtered:+.2f}pp")
        print(f"Average improvement: {avg_improvement:+.2f}pp")
        print(f"Filter wins: {n_better}/{len(smote_results)} datasets")


def save_results(all_results: List[SMOTEFilterResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"smote_filtered_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "datasets": DATASETS,
            "synthetic_per_class": SYNTHETIC_PER_CLASS,
            "oversample_ratio": SMOTE_OVERSAMPLE_RATIO,
            "filter_config": FILTER_CONFIG,
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Latest summary
    summary_file = RESULTS_DIR / "latest_summary.json"
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
