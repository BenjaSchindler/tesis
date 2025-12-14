#!/usr/bin/env python3
"""
Experiment 14: Multi-Classifier Evaluation for Rare Classes

Test if more powerful classifiers (XGBoost, MLP, RandomForest) can
better leverage synthetic data for rare classes ESFJ, ESFP, ESTJ.

Uses the synthetic data from RARE_massive_oversample config.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from config_definitions import RARE_CLASS_EXPERIMENTS, get_config_params
from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from base_config import RESULTS_DIR

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

# Try importing LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available")


# ============================================================================
# Classifier definitions
# ============================================================================

CLASSIFIERS = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {"max_iter": 2000, "solver": "lbfgs", "n_jobs": -1},
        "needs_scaling": True,
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5,
                   "n_jobs": -1, "random_state": 42},
        "needs_scaling": False,
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
                   "random_state": 42},
        "needs_scaling": False,
    },
    "MLP_small": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (256, 128), "max_iter": 200,
                   "early_stopping": True, "random_state": 42},
        "needs_scaling": True,
    },
    "MLP_large": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300,
                   "early_stopping": True, "random_state": 42},
        "needs_scaling": True,
    },
}

if HAS_XGBOOST:
    CLASSIFIERS["XGBoost"] = {
        "class": xgb.XGBClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        },
        "needs_scaling": False,
    }

if HAS_LIGHTGBM:
    CLASSIFIERS["LightGBM"] = {
        "class": lgb.LGBMClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
        "needs_scaling": False,
    }


# ============================================================================
# K-Fold Evaluation
# ============================================================================

def run_kfold_with_classifier(
    clf_name: str,
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
    unique_labels: List[str],
    synthetic_weight: float = 1.0,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42
) -> Dict:
    """Run K-fold evaluation with specified classifier."""

    clf_config = CLASSIFIERS[clf_name]
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    baseline_f1s = []
    augmented_f1s = []
    deltas = []
    per_class_baselines = {label: [] for label in unique_labels}
    per_class_augmented = {label: [] for label in unique_labels}

    total_folds = n_splits * n_repeats

    print(f"\n    Running K-Fold with {clf_name}...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
        X_train = X_original[train_idx]
        y_train = y_original[train_idx]
        X_test = X_original[test_idx]
        y_test = y_original[test_idx]

        # Baseline
        clf_base = clf_config["class"](**clf_config["params"])

        if clf_config["needs_scaling"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        clf_base.fit(X_train_scaled, y_train)
        y_pred_base = clf_base.predict(X_test_scaled)
        base_f1 = f1_score(y_test, y_pred_base, average="macro")
        baseline_f1s.append(base_f1)

        # Per-class baseline
        base_per_class = f1_score(y_test, y_pred_base, average=None, labels=range(len(unique_labels)))
        for i, label in enumerate(unique_labels):
            per_class_baselines[label].append(base_per_class[i])

        # Augmented
        clf_aug = clf_config["class"](**clf_config["params"])

        X_train_aug = np.vstack([X_train, X_synthetic])
        y_train_aug = np.concatenate([y_train, y_synthetic])

        sample_weights = np.concatenate([
            np.ones(len(y_train)),
            np.full(len(y_synthetic), synthetic_weight)
        ])

        if clf_config["needs_scaling"]:
            scaler_aug = StandardScaler()
            X_train_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_test_aug_scaled = scaler_aug.transform(X_test)
        else:
            X_train_aug_scaled = X_train_aug
            X_test_aug_scaled = X_test_scaled

        # Some classifiers support sample_weight
        try:
            clf_aug.fit(X_train_aug_scaled, y_train_aug, sample_weight=sample_weights)
        except TypeError:
            clf_aug.fit(X_train_aug_scaled, y_train_aug)

        y_pred_aug = clf_aug.predict(X_test_aug_scaled)
        aug_f1 = f1_score(y_test, y_pred_aug, average="macro")
        augmented_f1s.append(aug_f1)

        # Per-class augmented
        aug_per_class = f1_score(y_test, y_pred_aug, average=None, labels=range(len(unique_labels)))
        for i, label in enumerate(unique_labels):
            per_class_augmented[label].append(aug_per_class[i])

        delta = aug_f1 - base_f1
        deltas.append(delta)

        if (fold_idx + 1) % 5 == 0:
            print(f"      Fold {fold_idx + 1}/{total_folds}: base={base_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}")

    # Statistics
    baseline_mean = np.mean(baseline_f1s)
    baseline_std = np.std(baseline_f1s, ddof=1)
    augmented_mean = np.mean(augmented_f1s)
    augmented_std = np.std(augmented_f1s, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)
    delta_pct = (delta_mean / baseline_mean) * 100 if baseline_mean > 0 else 0

    # 95% CI
    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))

    # Statistical significance
    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    # Win rate
    win_rate = sum(1 for d in deltas if d > 0) / n

    # Per-class deltas
    per_class_delta = {}
    for label in unique_labels:
        base_mean = np.mean(per_class_baselines[label])
        aug_mean = np.mean(per_class_augmented[label])
        per_class_delta[label] = aug_mean - base_mean

    return {
        "classifier": clf_name,
        "n_folds": total_folds,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "augmented_mean": augmented_mean,
        "augmented_std": augmented_std,
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
    }


def main():
    print("=" * 70)
    print("Experiment 14: Multi-Classifier Evaluation for Rare Classes")
    print("=" * 70)
    print(f"\nTarget classes: ESFJ (42), ESFP (48), ESTJ (39)")
    print(f"Testing if powerful classifiers improve these classes")
    print(f"\nClassifiers: {list(CLASSIFIERS.keys())}")

    # Load data
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_original = np.array([label_to_idx[l] for l in labels])

    print(f"\nLoaded {len(texts)} samples, {len(unique_labels)} classes")

    # Generate synthetic data with RARE_massive_oversample
    config_name = "RARE_massive_oversample"
    params = get_config_params(config_name)

    print(f"\n{'#' * 70}")
    print(f"# Generating synthetic data: {config_name}")
    print(f"{'#' * 70}")

    generator = SyntheticGenerator(cache, params)
    synth_emb, synth_labels_raw, synth_texts = generator.generate_all(
        embeddings, labels, texts
    )

    print(f"  Generated {len(synth_texts)} synthetic samples")

    # Encode synthetic labels
    y_synthetic = np.array([label_to_idx[l] for l in synth_labels_raw])

    # Show distribution
    from collections import Counter
    synth_dist = Counter(synth_labels_raw)
    print(f"\n  Synthetic distribution:")
    for cls in ["ESFJ", "ESFP", "ESTJ"]:
        orig_count = sum(1 for l in labels if l == cls)
        synth_count = synth_dist.get(cls, 0)
        print(f"    {cls}: {orig_count} orig + {synth_count} synth = {orig_count + synth_count} total")

    # Run evaluation with each classifier
    results = {}

    for clf_name in CLASSIFIERS.keys():
        print(f"\n{'=' * 70}")
        print(f"  Evaluating: {clf_name}")
        print("=" * 70)

        try:
            clf_results = run_kfold_with_classifier(
                clf_name=clf_name,
                X_original=embeddings,
                y_original=y_original,
                X_synthetic=synth_emb,
                y_synthetic=y_synthetic,
                unique_labels=unique_labels,
                synthetic_weight=1.0,
                n_splits=5,
                n_repeats=3,
                seed=42
            )

            results[clf_name] = clf_results

            # Print summary
            sig = "*" if clf_results["significant"] else ""
            print(f"\n  Results for {clf_name}:")
            print(f"    Baseline:  {clf_results['baseline_mean']:.4f} +/- {clf_results['baseline_std']:.4f}")
            print(f"    Augmented: {clf_results['augmented_mean']:.4f} +/- {clf_results['augmented_std']:.4f}")
            print(f"    Delta:     {clf_results['delta_mean']:+.4f} ({clf_results['delta_pct']:+.2f}%) {sig}")
            print(f"    p-value:   {clf_results['p_value']:.6f}")
            print(f"    Win rate:  {clf_results['win_rate']*100:.1f}%")

            # Rare class deltas
            print(f"\n    Rare class deltas:")
            for cls in ["ESFJ", "ESFP", "ESTJ"]:
                delta = clf_results["per_class_delta"].get(cls, 0)
                print(f"      {cls}: {delta:+.4f}")

        except Exception as e:
            print(f"  Error with {clf_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_dir = RESULTS_DIR / "multiclassifier"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "exp14_multiclassifier_rare.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 14 SUMMARY - MULTI-CLASSIFIER RARE CLASSES")
    print("=" * 70)

    print(f"\n{'Classifier':<20} {'Delta':<12} {'p-value':<10} {'ESFJ':<10} {'ESFP':<10} {'ESTJ':<10}")
    print("-" * 82)

    # Sort by delta
    sorted_results = sorted(results.items(), key=lambda x: x[1]['delta_pct'], reverse=True)

    for clf_name, data in sorted_results:
        sig = "*" if data["significant"] else ""
        esfj = data["per_class_delta"].get("ESFJ", 0)
        esfp = data["per_class_delta"].get("ESFP", 0)
        estj = data["per_class_delta"].get("ESTJ", 0)

        print(f"{clf_name:<20} {data['delta_pct']:+.2f}%{sig:<5} {data['p_value']:.4f}    {esfj:+.4f}    {esfp:+.4f}    {estj:+.4f}")

    print("-" * 82)

    # Find best classifier for each rare class
    print(f"\nBest classifier per rare class:")
    for cls in ["ESFJ", "ESFP", "ESTJ"]:
        best_clf = max(results.items(), key=lambda x: x[1]["per_class_delta"].get(cls, -999))
        best_delta = best_clf[1]["per_class_delta"].get(cls, 0)
        print(f"  {cls}: {best_clf[0]} (delta={best_delta:+.4f})")

    print("=" * 70)


if __name__ == "__main__":
    main()
