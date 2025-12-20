#!/usr/bin/env python3
"""
Experiment: RARE_MLP Original (Replication)

Recreates the original RARE_MLP_arch_512 experiment to verify:
- Baseline: 0.2075
- Augmented: 0.2492
- Delta: +20.11%

Uses RARE_massive_oversample with DEFAULT n_shot (not 60).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from base_config import RESULTS_DIR

# MLP architecture (same as RARE_MLP)
MLP_CONFIG = {
    "hidden_layer_sizes": (512, 256, 128),
    "max_iter": 300,
    "early_stopping": True,
    "random_state": 42,
    "verbose": False
}

RARE_CLASSES = ["ESFJ", "ESFP", "ESTJ"]


def get_rare_massive_original_params():
    """
    Get RARE_massive_oversample params - ORIGINAL (no n_shot override).
    """
    params = {
        "force_generation_classes": ["ESFJ", "ESFP", "ESTJ"],
        "min_synthetic_per_class": 100,
        "rare_class_boost": 5.0,
        "samples_per_prompt": 15,
        "prompts_per_cluster": 15,
        "max_clusters": 3,
        "disable_quality_gate": True,
        "similarity_threshold": 0.99,
        # NO n_shot override - use default
    }
    return params


def generate_synthetics(cache, texts, labels, embeddings, params):
    """Generate synthetics for rare classes with given params."""
    generator = SyntheticGenerator(cache, params)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    all_synth_emb = []
    all_synth_labels = []

    target_classes = params.get("force_generation_classes", RARE_CLASSES)

    for label in unique_labels:
        if label not in target_classes:
            continue

        class_mask = np.array(labels) == label
        class_texts = [t for t, m in zip(texts, class_mask) if m]
        class_emb = embeddings[class_mask]

        n_original = len(class_texts)

        try:
            synth_texts, synth_labels = generator.generate_for_class(
                np.array(class_texts), class_emb, label
            )

            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                all_synth_emb.append(synth_emb)
                all_synth_labels.extend([label_to_idx[label]] * len(synth_emb))
                print(f"    {label}: {n_original} -> +{len(synth_emb)} synthetic")
        except Exception as e:
            print(f"    {label}: Error - {e}")

    if all_synth_emb:
        return np.vstack(all_synth_emb), np.array(all_synth_labels)
    return np.array([]).reshape(0, embeddings.shape[1]), np.array([])


def run_kfold_mlp(X_orig, y_orig, X_synth, y_synth, unique_labels,
                  n_splits=5, n_repeats=3, seed=42):
    """
    Run K-fold CV with MLP - SAME setup as RARE_MLP experiments.
    """
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    baseline_f1s, augmented_f1s, deltas = [], [], []
    per_class_base = {l: [] for l in unique_labels}
    per_class_aug = {l: [] for l in unique_labels}

    total_folds = n_splits * n_repeats

    print(f"\n  Running {total_folds}-fold CV with MLP (512-256-128)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_orig)):
        X_train, y_train = X_orig[train_idx], y_orig[train_idx]
        X_test, y_test = X_orig[test_idx], y_orig[test_idx]

        # === BASELINE ===
        clf_base = MLPClassifier(**MLP_CONFIG)
        scaler_base = StandardScaler()
        X_tr_base = scaler_base.fit_transform(X_train)
        X_te_base = scaler_base.transform(X_test)

        clf_base.fit(X_tr_base, y_train)
        y_pred_base = clf_base.predict(X_te_base)
        base_f1 = f1_score(y_test, y_pred_base, average="macro")
        baseline_f1s.append(base_f1)

        base_pc = f1_score(y_test, y_pred_base, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_base[l].append(base_pc[i])

        # === AUGMENTED ===
        X_train_aug = np.vstack([X_train, X_synth]) if len(X_synth) > 0 else X_train
        y_train_aug = np.concatenate([y_train, y_synth]) if len(y_synth) > 0 else y_train

        scaler_aug = StandardScaler()
        X_tr_aug = scaler_aug.fit_transform(X_train_aug)
        X_te_aug = scaler_aug.transform(X_test)

        clf_aug = MLPClassifier(**MLP_CONFIG)
        clf_aug.fit(X_tr_aug, y_train_aug)
        y_pred_aug = clf_aug.predict(X_te_aug)
        aug_f1 = f1_score(y_test, y_pred_aug, average="macro")
        augmented_f1s.append(aug_f1)

        aug_pc = f1_score(y_test, y_pred_aug, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_aug[l].append(aug_pc[i])

        delta = aug_f1 - base_f1
        deltas.append(delta)

        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{total_folds}: base={base_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}")

    # Statistics
    base_mean = np.mean(baseline_f1s)
    base_std = np.std(baseline_f1s, ddof=1)
    aug_mean = np.mean(augmented_f1s)
    aug_std = np.std(augmented_f1s, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    delta_pct = (delta_mean / base_mean) * 100 if base_mean > 0 else 0

    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    win_rate = sum(1 for d in deltas if d > 0) / n

    per_class_delta = {l: np.mean(per_class_aug[l]) - np.mean(per_class_base[l]) for l in unique_labels}

    return {
        "n_folds": total_folds,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "augmented_mean": aug_mean,
        "augmented_std": aug_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_pct": delta_pct,
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": win_rate,
        "per_class_delta": per_class_delta,
        "n_synthetic": len(X_synth),
    }


def main():
    print("="*70)
    print("RARE_MLP Original (Replication)")
    print("="*70)
    print("\nReplicating original experiment:")
    print("  - RARE_massive_oversample config")
    print("  - DEFAULT n_shot (no override)")
    print("  - MLP 512-256-128 classifier")
    print("  - K-Fold setup (seed=42)")
    print("\nExpected results:")
    print("  - Baseline:  0.2075")
    print("  - Augmented: 0.2492")
    print("  - Delta:     +20.11%")

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[l] for l in labels])

    print(f"Classes: {len(unique_labels)}")
    print(f"Samples: {len(texts)}")

    # Rare class counts
    print("\nRare class counts:")
    labels_list = list(labels)
    for cls in RARE_CLASSES:
        count = labels_list.count(cls)
        print(f"  {cls}: {count} samples")

    # Get original params (NO n_shot override)
    params = get_rare_massive_original_params()
    print(f"\nConfig params (ORIGINAL):")
    print(f"  n_shot: DEFAULT (not overridden)")
    print(f"  rare_class_boost: {params.get('rare_class_boost')}")
    print(f"  force_generation_classes: {params.get('force_generation_classes')}")

    # Generate synthetics
    print("\nGenerating synthetics...")
    X_synth, y_synth = generate_synthetics(cache, texts, labels, embeddings, params)
    print(f"\nTotal synthetic samples: {len(X_synth)}")

    # Run K-fold
    results = run_kfold_mlp(embeddings, y_encoded, X_synth, y_synth, unique_labels)

    results["config_name"] = "RARE_MLP_original_replication"
    results["description"] = "RARE_massive_oversample (DEFAULT n_shot) + MLP 512-256-128"
    results["mlp_architecture"] = "MLP_512_256_128"
    results["n_shot"] = "default"
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "="*70)
    print("RESULTS: RARE_MLP Original Replication")
    print("="*70)
    print(f"\n  Baseline MLP:    {results['baseline_mean']:.4f} +/- {results['baseline_std']:.4f}")
    print(f"  Augmented MLP:   {results['augmented_mean']:.4f} +/- {results['augmented_std']:.4f}")
    print(f"  Delta:           {results['delta_mean']*100:+.2f} pp ({results['delta_pct']:+.2f}%)")
    print(f"  p-value:         {results['p_value']:.6f}")
    print(f"  Significant:     {'Yes' if results['significant'] else 'No'}")
    print(f"  Synthetics:      {results['n_synthetic']}")

    print("\n  Rare class deltas:")
    for cls in RARE_CLASSES:
        delta = results['per_class_delta'].get(cls, 0)
        print(f"    {cls}: {delta:+.4f} ({delta*100:+.2f}%)")

    # Compare with expected
    print("\n" + "="*70)
    print("COMPARISON WITH EXPECTED")
    print("="*70)
    print(f"\n  {'Metric':<20} {'Expected':<12} {'Actual':<12} {'Match':<10}")
    print(f"  {'-'*54}")

    base_match = abs(results['baseline_mean'] - 0.2075) < 0.005
    aug_match = abs(results['augmented_mean'] - 0.2492) < 0.01
    delta_match = abs(results['delta_pct'] - 20.11) < 2.0

    print(f"  {'Baseline':<20} {'0.2075':<12} {results['baseline_mean']:.4f}{'':>6} {'✓' if base_match else '✗'}")
    print(f"  {'Augmented':<20} {'0.2492':<12} {results['augmented_mean']:.4f}{'':>6} {'✓' if aug_match else '✗'}")
    print(f"  {'Delta %':<20} {'+20.11%':<12} {results['delta_pct']:+.2f}%{'':>5} {'✓' if delta_match else '✗'}")

    # Save results
    results_dir = RESULTS_DIR / "rare_mlp_original"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / "RARE_MLP_original_replication_kfold.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {result_file}")

    print("\n" + "="*70)
    print("Replication Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
