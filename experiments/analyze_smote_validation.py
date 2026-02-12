#!/usr/bin/env python3
"""
SMOTE Validation: Dummy-Binary vs Standard Multi-Class

Validates that the "dummy binary" SMOTE trick (used throughout this project)
produces equivalent results to standard multi-class SMOTE on the same data.

Tests on 3 representative datasets x 3 seeds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

N_SYNTHETIC_PER_CLASS = 50
SEEDS = [42, 123, 456]
DATASETS = ["sms_spam_10shot", "20newsgroups_10shot", "emotion_10shot"]


def generate_smote_dummy_binary(real_embeddings, n_generate, seed=42, k_neighbors=5):
    """The project's dummy-binary SMOTE approach (per-class)."""
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    rng = np.random.RandomState(seed)
    X = np.vstack([real_embeddings, rng.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res[np.where(y_res == 0)[0][n_base:]][:n_generate]
    except Exception:
        return np.array([]).reshape(0, real_embeddings.shape[1])


def generate_smote_standard(train_embeddings, train_labels, n_per_class, seed=42, k_neighbors=5):
    """Standard multi-class SMOTE applied to the full dataset at once."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    class_counts = {cls: np.sum(labels_arr == cls) for cls in unique_classes}
    target_counts = {cls: count + n_per_class for cls, count in class_counts.items()}
    k = min(k_neighbors, min(class_counts.values()) - 1)
    if k < 1:
        return train_embeddings.copy(), list(train_labels)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy=target_counts, random_state=seed)
        X_res, y_res = smote.fit_resample(train_embeddings, labels_arr)
        return X_res, list(y_res)
    except Exception as e:
        print(f"  Standard SMOTE failed: {e}")
        return train_embeddings.copy(), list(train_labels)


def main():
    print("=" * 70)
    print("SMOTE VALIDATION: Dummy-Binary vs Standard Multi-Class")
    print("=" * 70)

    model = SentenceTransformer("all-mpnet-base-v2")

    results = []

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        with open(dataset_path) as f:
            data = json.load(f)

        train_texts = data["train_texts"]
        train_labels = data["train_labels"]
        test_texts = data["test_texts"]
        test_labels = data["test_labels"]
        unique_classes = sorted(set(train_labels))

        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name} ({len(unique_classes)} classes)")
        print(f"{'=' * 70}")

        train_emb = model.encode(train_texts, show_progress_bar=False)
        test_emb = model.encode(test_texts, show_progress_bar=False)
        labels_arr = np.array(train_labels)

        for seed in SEEDS:
            # --- Method 1: Dummy-binary (per-class) ---
            dummy_embs, dummy_labels = [], []
            for cls in unique_classes:
                cls_emb = train_emb[labels_arr == cls]
                s = generate_smote_dummy_binary(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                if len(s) > 0:
                    dummy_embs.append(s)
                    dummy_labels.extend([cls] * len(s))

            if dummy_embs:
                aug_emb_dummy = np.vstack([train_emb, np.vstack(dummy_embs)])
                aug_labels_dummy = list(train_labels) + dummy_labels
            else:
                aug_emb_dummy = train_emb
                aug_labels_dummy = list(train_labels)

            clf_dummy = RidgeClassifier(alpha=1.0)
            clf_dummy.fit(aug_emb_dummy, aug_labels_dummy)
            f1_dummy = f1_score(test_labels, clf_dummy.predict(test_emb), average="macro")

            # --- Method 2: Standard multi-class SMOTE ---
            X_std, y_std = generate_smote_standard(
                train_emb, train_labels, N_SYNTHETIC_PER_CLASS, seed=seed
            )
            clf_std = RidgeClassifier(alpha=1.0)
            clf_std.fit(X_std, y_std)
            f1_std = f1_score(test_labels, clf_std.predict(test_emb), average="macro")

            delta = (f1_dummy - f1_std) * 100
            n_dummy_synth = len(aug_labels_dummy) - len(train_labels)
            n_std_synth = len(y_std) - len(train_labels)

            print(f"\n  Seed {seed}:")
            print(f"    Dummy-binary SMOTE: F1={f1_dummy:.4f} ({n_dummy_synth} synthetic)")
            print(f"    Standard SMOTE:     F1={f1_std:.4f} ({n_std_synth} synthetic)")
            print(f"    Delta:              {delta:+.2f}pp")

            # Check embedding distance distributions
            if dummy_embs:
                dummy_all = np.vstack(dummy_embs)
                std_synth = X_std[len(train_emb):]
                if len(std_synth) > 0 and len(dummy_all) > 0:
                    dummy_dists = np.linalg.norm(
                        dummy_all - train_emb.mean(axis=0), axis=1
                    )
                    std_dists = np.linalg.norm(
                        std_synth - train_emb.mean(axis=0), axis=1
                    )
                    print(f"    Dummy dist to centroid: mean={dummy_dists.mean():.4f}, std={dummy_dists.std():.4f}")
                    print(f"    Std dist to centroid:   mean={std_dists.mean():.4f}, std={std_dists.std():.4f}")

            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "f1_dummy_binary": float(f1_dummy),
                "f1_standard": float(f1_std),
                "delta_pp": float(delta),
                "n_synthetic_dummy": n_dummy_synth,
                "n_synthetic_standard": n_std_synth,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    deltas = [r["delta_pp"] for r in results]
    print(f"\nMean delta (dummy - standard): {np.mean(deltas):+.4f}pp")
    print(f"Std delta: {np.std(deltas):.4f}pp")
    print(f"Max abs delta: {max(abs(d) for d in deltas):.4f}pp")
    print(f"Conclusion: {'EQUIVALENT' if max(abs(d) for d in deltas) < 1.0 else 'DIVERGENT'}")

    # Save
    output_path = PROJECT_ROOT / "results" / "smote_validation.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "summary": {
            "mean_delta_pp": float(np.mean(deltas)),
            "std_delta_pp": float(np.std(deltas)),
            "max_abs_delta_pp": float(max(abs(d) for d in deltas)),
        }}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
