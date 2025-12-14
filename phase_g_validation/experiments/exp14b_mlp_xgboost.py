#!/usr/bin/env python3
"""
Experiment 14b: MLP and XGBoost for Rare Classes

Focus on neural networks and XGBoost which work better with embeddings.
Skip slow tree-based methods (RandomForest, GradientBoosting).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from config_definitions import RARE_CLASS_EXPERIMENTS, get_config_params
from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from base_config import RESULTS_DIR

# Try XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

# Try LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Focus on neural networks and gradient boosting (not tree-based)
CLASSIFIERS = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {"max_iter": 2000, "solver": "lbfgs", "n_jobs": -1},
        "needs_scaling": True,
    },
    "MLP_256_128": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (256, 128), "max_iter": 300,
                   "early_stopping": True, "random_state": 42, "verbose": False},
        "needs_scaling": True,
    },
    "MLP_512_256_128": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300,
                   "early_stopping": True, "random_state": 42, "verbose": False},
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
            "n_jobs": -1,
            "random_state": 42,
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
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
        "needs_scaling": False,
    }


def run_kfold(clf_name, X_orig, y_orig, X_synth, y_synth, unique_labels,
              synthetic_weight=1.0, n_splits=5, n_repeats=3, seed=42):
    """Run K-fold with specified classifier."""

    cfg = CLASSIFIERS[clf_name]
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    baseline_f1s, augmented_f1s, deltas = [], [], []
    per_class_base = {l: [] for l in unique_labels}
    per_class_aug = {l: [] for l in unique_labels}

    total_folds = n_splits * n_repeats
    print(f"\n    Running K-Fold with {clf_name}...")

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_orig, y_orig)):
        X_train, y_train = X_orig[train_idx], y_orig[train_idx]
        X_test, y_test = X_orig[test_idx], y_orig[test_idx]

        # Baseline
        clf = cfg["class"](**cfg["params"])
        if cfg["needs_scaling"]:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        base_f1 = f1_score(y_test, y_pred, average="macro")
        baseline_f1s.append(base_f1)

        base_pc = f1_score(y_test, y_pred, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_base[l].append(base_pc[i])

        # Augmented
        clf2 = cfg["class"](**cfg["params"])
        X_train_aug = np.vstack([X_train, X_synth])
        y_train_aug = np.concatenate([y_train, y_synth])
        weights = np.concatenate([np.ones(len(y_train)), np.full(len(y_synth), synthetic_weight)])

        if cfg["needs_scaling"]:
            scaler2 = StandardScaler()
            X_tr2 = scaler2.fit_transform(X_train_aug)
            X_te2 = scaler2.transform(X_test)
        else:
            X_tr2, X_te2 = X_train_aug, X_te

        try:
            clf2.fit(X_tr2, y_train_aug, sample_weight=weights)
        except TypeError:
            clf2.fit(X_tr2, y_train_aug)

        y_pred2 = clf2.predict(X_te2)
        aug_f1 = f1_score(y_test, y_pred2, average="macro")
        augmented_f1s.append(aug_f1)

        aug_pc = f1_score(y_test, y_pred2, average=None, labels=range(len(unique_labels)))
        for i, l in enumerate(unique_labels):
            per_class_aug[l].append(aug_pc[i])

        delta = aug_f1 - base_f1
        deltas.append(delta)

        if (fold_idx + 1) % 5 == 0:
            print(f"      Fold {fold_idx + 1}/{total_folds}: base={base_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}")

    # Stats
    base_mean, base_std = np.mean(baseline_f1s), np.std(baseline_f1s, ddof=1)
    aug_mean, aug_std = np.mean(augmented_f1s), np.std(augmented_f1s, ddof=1)
    delta_mean, delta_std = np.mean(deltas), np.std(deltas, ddof=1)
    delta_pct = (delta_mean / base_mean) * 100 if base_mean > 0 else 0

    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))
    t_stat, p_value = stats.ttest_1samp(deltas, 0)
    win_rate = sum(1 for d in deltas if d > 0) / n

    per_class_delta = {l: np.mean(per_class_aug[l]) - np.mean(per_class_base[l]) for l in unique_labels}

    return {
        "classifier": clf_name,
        "n_folds": total_folds,
        "baseline_mean": base_mean, "baseline_std": base_std,
        "augmented_mean": aug_mean, "augmented_std": aug_std,
        "delta_mean": delta_mean, "delta_std": delta_std, "delta_pct": delta_pct,
        "ci_95_lower": ci_95[0], "ci_95_upper": ci_95[1],
        "t_statistic": t_stat, "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": win_rate,
        "per_class_delta": per_class_delta,
    }


def main():
    print("=" * 70)
    print("Experiment 14b: MLP and XGBoost for Rare Classes")
    print("=" * 70)
    print(f"\nClassifiers: {list(CLASSIFIERS.keys())}")

    # Load data
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_orig = np.array([label_to_idx[l] for l in labels])

    print(f"\nLoaded {len(texts)} samples, {len(unique_labels)} classes")

    # Generate synthetics
    print(f"\n{'#' * 70}")
    print("# Generating RARE_massive_oversample synthetics")
    print("#" * 70)

    params = get_config_params("RARE_massive_oversample")
    generator = SyntheticGenerator(cache, params)
    synth_emb, synth_labels_raw, synth_texts = generator.generate_all(embeddings, labels, texts)

    y_synth = np.array([label_to_idx[l] for l in synth_labels_raw])
    print(f"  Generated {len(synth_texts)} synthetic samples")

    # Distribution
    from collections import Counter
    dist = Counter(synth_labels_raw)
    for cls in ["ESFJ", "ESFP", "ESTJ"]:
        orig = sum(1 for l in labels if l == cls)
        syn = dist.get(cls, 0)
        print(f"    {cls}: {orig} orig + {syn} synth = {orig + syn} total")

    # Run evaluations
    results = {}
    for clf_name in CLASSIFIERS:
        print(f"\n{'=' * 70}")
        print(f"  Evaluating: {clf_name}")
        print("=" * 70)

        try:
            res = run_kfold(clf_name, embeddings, y_orig, synth_emb, y_synth,
                           unique_labels, synthetic_weight=1.0)
            results[clf_name] = res

            sig = "*" if res["significant"] else ""
            print(f"\n  Results:")
            print(f"    Baseline:  {res['baseline_mean']:.4f} +/- {res['baseline_std']:.4f}")
            print(f"    Augmented: {res['augmented_mean']:.4f} +/- {res['augmented_std']:.4f}")
            print(f"    Delta:     {res['delta_mean']:+.4f} ({res['delta_pct']:+.2f}%) {sig}")
            print(f"    p-value:   {res['p_value']:.6f}")

            print(f"\n    Rare class deltas:")
            for cls in ["ESFJ", "ESFP", "ESTJ"]:
                d = res["per_class_delta"].get(cls, 0)
                print(f"      {cls}: {d:+.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save (convert numpy types to native Python)
    out_dir = RESULTS_DIR / "multiclassifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(out_dir / "exp14b_mlp_xgboost.json", 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 14b SUMMARY - MLP & XGBoost")
    print("=" * 70)

    print(f"\n{'Classifier':<20} {'Baseline':<10} {'Delta':<12} {'ESFJ':<10} {'ESFP':<10} {'ESTJ':<10}")
    print("-" * 82)

    for clf, data in sorted(results.items(), key=lambda x: x[1]['delta_pct'], reverse=True):
        sig = "*" if data["significant"] else ""
        esfj = data["per_class_delta"].get("ESFJ", 0)
        esfp = data["per_class_delta"].get("ESFP", 0)
        estj = data["per_class_delta"].get("ESTJ", 0)
        print(f"{clf:<20} {data['baseline_mean']:.4f}    {data['delta_pct']:+.2f}%{sig:<5} {esfj:+.4f}    {esfp:+.4f}    {estj:+.4f}")

    print("-" * 82)

    # Best per class
    print(f"\nBest classifier per rare class:")
    for cls in ["ESFJ", "ESFP", "ESTJ"]:
        best = max(results.items(), key=lambda x: x[1]["per_class_delta"].get(cls, -999))
        d = best[1]["per_class_delta"].get(cls, 0)
        print(f"  {cls}: {best[0]} (delta={d:+.4f})")

    print("=" * 70)


if __name__ == "__main__":
    main()
