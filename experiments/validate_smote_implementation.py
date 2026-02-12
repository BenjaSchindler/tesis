#!/usr/bin/env python3
"""
SMOTE Implementation Validation (MUST-RUN A3)

Compares two SMOTE implementations:
  1. Dummy-binary: Creates random Gaussian noise as fake class (used in thesis_final)
  2. Binary-mask: Target class=1 vs rest=0 (used in exp_statistical_validation)

If both produce equivalent F1 scores (<0.1pp difference), the dummy-binary
approach is validated. If not, the main experiment needs correction.

Usage:
    python experiments/validate_smote_implementation.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

# Test 3 representative datasets × 3 seeds × 3 classifiers
DATASETS = ["sms_spam_10shot", "20newsgroups_10shot", "hate_speech_davidson_10shot"]
SEEDS = [42, 123, 456]
SYNTHETIC_PER_CLASS = 50
CLASSIFIERS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "ridge": lambda seed: RidgeClassifier(random_state=seed),
    "svc_linear": lambda seed: SVC(kernel='linear', random_state=seed),
}


def load_dataset(name):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        print(f"  Dataset not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def smote_dummy_binary(class_embeddings, n_generate, seed, k_neighbors=5):
    """Thesis_final implementation: random Gaussian noise as dummy class."""
    if len(class_embeddings) < 2:
        return np.array([]).reshape(0, class_embeddings.shape[1])
    k = min(k_neighbors, len(class_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, class_embeddings.shape[1])

    n_base = len(class_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    rng = np.random.RandomState(seed)

    X = np.vstack([class_embeddings, rng.randn(n_dummy, class_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)

    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res[np.where(y_res == 0)[0][n_base:]][:n_generate]
    except Exception as e:
        print(f"    Dummy-binary SMOTE failed: {e}")
        return np.array([]).reshape(0, class_embeddings.shape[1])


def smote_binary_mask(all_embeddings, all_labels, target_class, n_generate, seed, k_neighbors=5):
    """Statistical_validation implementation: target class=1, rest=0."""
    class_mask = (np.array(all_labels) == target_class).astype(int)
    n_class = class_mask.sum()

    k = min(k_neighbors, n_class - 1)
    if k < 1:
        return np.array([]).reshape(0, all_embeddings.shape[1])

    target_count = n_class + n_generate

    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={1: target_count},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(all_embeddings, class_mask)
        new_samples = X_res[len(all_embeddings):]
        return new_samples[:n_generate]
    except Exception as e:
        print(f"    Binary-mask SMOTE failed: {e}")
        return np.array([]).reshape(0, all_embeddings.shape[1])


def run_comparison():
    print("=" * 70)
    print("SMOTE IMPLEMENTATION VALIDATION")
    print("=" * 70)

    print("\nLoading embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    results = []
    all_diffs = []

    for ds_name in DATASETS:
        print(f"\n{'─' * 50}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 50}")

        data = load_dataset(ds_name)
        if data is None:
            continue

        train_texts = data["train_texts"]
        train_labels = data["train_labels"]
        test_texts = data["test_texts"]
        test_labels = data["test_labels"]
        classes = sorted(set(train_labels))

        print(f"  Classes: {classes}, Train: {len(train_texts)}, Test: {len(test_texts)}")

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)

        for seed in SEEDS:
            # Generate SMOTE samples with both methods
            dummy_synth_emb = []
            dummy_synth_labels = []
            mask_synth_emb = []
            mask_synth_labels = []

            for cls in classes:
                cls_mask = np.array(train_labels) == cls
                cls_emb = train_emb[cls_mask]

                # Method 1: Dummy binary
                d_samples = smote_dummy_binary(cls_emb, SYNTHETIC_PER_CLASS, seed)
                if len(d_samples) > 0:
                    dummy_synth_emb.append(d_samples)
                    dummy_synth_labels.extend([cls] * len(d_samples))

                # Method 2: Binary mask
                m_samples = smote_binary_mask(train_emb, train_labels, cls, SYNTHETIC_PER_CLASS, seed)
                if len(m_samples) > 0:
                    mask_synth_emb.append(m_samples)
                    mask_synth_labels.extend([cls] * len(m_samples))

            if not dummy_synth_emb or not mask_synth_emb:
                print(f"  Seed {seed}: SMOTE generation failed, skipping")
                continue

            dummy_all_emb = np.vstack([train_emb] + dummy_synth_emb)
            dummy_all_labels = list(train_labels) + dummy_synth_labels
            mask_all_emb = np.vstack([train_emb] + mask_synth_emb)
            mask_all_labels = list(train_labels) + mask_synth_labels

            for clf_name, clf_factory in CLASSIFIERS.items():
                # Dummy binary
                clf_d = clf_factory(seed)
                clf_d.fit(dummy_all_emb, dummy_all_labels)
                pred_d = clf_d.predict(test_emb)
                f1_d = f1_score(test_labels, pred_d, average='macro')

                # Binary mask
                clf_m = clf_factory(seed)
                clf_m.fit(mask_all_emb, mask_all_labels)
                pred_m = clf_m.predict(test_emb)
                f1_m = f1_score(test_labels, pred_m, average='macro')

                diff_pp = (f1_d - f1_m) * 100
                all_diffs.append(diff_pp)

                results.append({
                    "dataset": ds_name,
                    "seed": seed,
                    "classifier": clf_name,
                    "f1_dummy_binary": f1_d,
                    "f1_binary_mask": f1_m,
                    "diff_pp": diff_pp,
                })

                print(f"  Seed {seed}, {clf_name}: dummy={f1_d:.4f} mask={f1_m:.4f} "
                      f"diff={diff_pp:+.2f}pp")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if not all_diffs:
        print("No results to summarize.")
        return

    diffs = np.array(all_diffs)
    mean_diff = np.mean(np.abs(diffs))
    max_diff = np.max(np.abs(diffs))
    print(f"\n  N comparisons: {len(diffs)}")
    print(f"  Mean |difference|: {mean_diff:.4f}pp")
    print(f"  Max |difference|:  {max_diff:.4f}pp")
    print(f"  Std of differences: {np.std(diffs):.4f}pp")

    THRESHOLD = 0.1  # pp
    if max_diff < THRESHOLD:
        print(f"\n  RESULT: VALIDATED")
        print(f"  All differences < {THRESHOLD}pp threshold.")
        print(f"  The dummy-binary SMOTE implementation is equivalent to standard binary-mask SMOTE.")
        print(f"  Add footnote to thesis: 'We verified that our per-class SMOTE implementation")
        print(f"  produces results equivalent to standard multi-class SMOTE (max difference <0.1pp).'")
    elif mean_diff < THRESHOLD:
        print(f"\n  RESULT: MOSTLY VALIDATED (with caveats)")
        print(f"  Mean difference < {THRESHOLD}pp, but max difference = {max_diff:.4f}pp.")
        print(f"  Review outlier cases above.")
    else:
        print(f"\n  RESULT: NOT VALIDATED")
        print(f"  Mean difference = {mean_diff:.4f}pp exceeds threshold.")
        print(f"  Consider rerunning main experiment with binary-mask SMOTE.")

    # Per-dataset summary
    print(f"\n  Per-dataset breakdown:")
    for ds in DATASETS:
        ds_diffs = [r["diff_pp"] for r in results if r["dataset"] == ds]
        if ds_diffs:
            print(f"    {ds}: mean={np.mean(ds_diffs):+.4f}pp, "
                  f"max|diff|={np.max(np.abs(ds_diffs)):.4f}pp")


if __name__ == "__main__":
    run_comparison()
