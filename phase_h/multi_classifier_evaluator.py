#!/usr/bin/env python3
"""
Multi-Classifier Evaluator for Phase H

Evaluates synthetic data with multiple classification models:
- LogisticRegression (baseline)
- SVM (SVC with RBF kernel)
- RandomForest
- XGBoost
- LightGBM
- MLP (Neural Network)
- Fine-tuned transformers (optional)

Usage:
    python multi_classifier_evaluator.py --synth-csv path/to/synth.csv --classifier all
    python multi_classifier_evaluator.py --synth-csv path/to/synth.csv --classifier xgboost --k 5
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")


# ============================================================================
# Classifier Factory
# ============================================================================

CLASSIFIERS = {
    "logistic": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "params": {"max_iter": 2000, "solver": "lbfgs", "n_jobs": -1},
        "needs_scaling": True,
    },
    "svm": {
        "name": "SVM (RBF)",
        "class": SVC,
        "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "needs_scaling": True,
    },
    "svm_linear": {
        "name": "SVM (Linear)",
        "class": SVC,
        "params": {"kernel": "linear", "C": 1.0},
        "needs_scaling": True,
    },
    "random_forest": {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5, "n_jobs": -1, "random_state": 42},
        "needs_scaling": False,
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
        "needs_scaling": False,
    },
    "mlp": {
        "name": "MLP Neural Network",
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (256, 128, 64), "max_iter": 500, "early_stopping": True, "random_state": 42},
        "needs_scaling": True,
    },
}

# Add XGBoost if available
if HAS_XGBOOST:
    CLASSIFIERS["xgboost"] = {
        "name": "XGBoost",
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
            "use_label_encoder": False,
        },
        "needs_scaling": False,
    }

# Add LightGBM if available
if HAS_LIGHTGBM:
    CLASSIFIERS["lightgbm"] = {
        "name": "LightGBM",
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


def get_classifier(name: str, class_weight: Optional[str] = None) -> Tuple[Any, bool]:
    """
    Get classifier instance by name.

    Returns: (classifier, needs_scaling)
    """
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIERS.keys())}")

    config = CLASSIFIERS[name]
    params = config["params"].copy()

    # Add class_weight if supported
    if class_weight and "class_weight" in config["class"]().get_params():
        params["class_weight"] = class_weight

    clf = config["class"](**params)
    return clf, config["needs_scaling"]


# ============================================================================
# Embedding Functions
# ============================================================================

def load_embeddings_cached(cache_dir: str, prefix: str) -> Optional[np.ndarray]:
    """Load embeddings from cache."""
    cache_path = Path(cache_dir) / f"{prefix}_embeddings.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None


def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for texts."""
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


# ============================================================================
# Evaluation Functions
# ============================================================================

def train_and_evaluate(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    sample_weight: Optional[np.ndarray] = None
) -> Dict:
    """Train classifier and return metrics."""

    # Scale if needed
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train
    if sample_weight is not None:
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)

    # Metrics
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1.tolist(),
    }


def run_kfold_evaluation(
    classifier_name: str,
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: Optional[np.ndarray],
    y_synthetic: Optional[np.ndarray],
    n_splits: int = 5,
    n_repeats: int = 1,
    synthetic_weight: float = 0.5,
    class_weight: Optional[str] = None,
    random_state: int = 42
) -> Dict:
    """
    Run Stratified K-Fold evaluation with specified classifier.
    """

    clf_template, needs_scaling = get_classifier(classifier_name, class_weight)

    if n_repeats > 1:
        kfold = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
        total_folds = n_splits * n_repeats
    else:
        kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        total_folds = n_splits

    baseline_f1s = []
    augmented_f1s = []
    deltas = []
    per_class_deltas = []

    print(f"\n  Running K-Fold with {classifier_name} (k={n_splits}, repeats={n_repeats})")
    print("  " + "-" * 50)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
        X_train_fold = X_original[train_idx]
        y_train_fold = y_original[train_idx]
        X_test_fold = X_original[test_idx]
        y_test_fold = y_original[test_idx]

        # Create fresh classifier for each fold
        clf_baseline, _ = get_classifier(classifier_name, class_weight)
        scaler_baseline = StandardScaler() if needs_scaling else None

        # Baseline
        baseline_metrics = train_and_evaluate(
            clf_baseline, X_train_fold, y_train_fold,
            X_test_fold, y_test_fold, scaler_baseline
        )
        baseline_f1 = baseline_metrics["macro_f1"]
        baseline_f1s.append(baseline_f1)

        # Augmented
        if X_synthetic is not None and len(X_synthetic) > 0:
            clf_augmented, _ = get_classifier(classifier_name, class_weight)
            scaler_augmented = StandardScaler() if needs_scaling else None

            X_train_aug = np.vstack([X_train_fold, X_synthetic])
            y_train_aug = np.concatenate([y_train_fold, y_synthetic])

            sample_weights = np.concatenate([
                np.ones(len(y_train_fold)),
                np.full(len(y_synthetic), synthetic_weight)
            ])

            augmented_metrics = train_and_evaluate(
                clf_augmented, X_train_aug, y_train_aug,
                X_test_fold, y_test_fold, scaler_augmented, sample_weights
            )
            augmented_f1 = augmented_metrics["macro_f1"]

            # Per-class delta
            pc_delta = [a - b for a, b in zip(augmented_metrics["per_class_f1"], baseline_metrics["per_class_f1"])]
            per_class_deltas.append(pc_delta)
        else:
            augmented_f1 = baseline_f1

        augmented_f1s.append(augmented_f1)
        delta = augmented_f1 - baseline_f1
        deltas.append(delta)

        print(f"    Fold {fold_idx + 1:2d}: Base={baseline_f1:.4f}, Aug={augmented_f1:.4f}, Δ={delta:+.4f}")

    # Statistics
    baseline_mean = np.mean(baseline_f1s)
    baseline_std = np.std(baseline_f1s, ddof=1)
    augmented_mean = np.mean(augmented_f1s)
    augmented_std = np.std(augmented_f1s, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)

    # 95% CI
    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n)) if n > 1 else (delta_mean, delta_mean)

    # Statistical significance
    if n > 1:
        t_stat, p_value = stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_value = 0.0, 1.0

    # Win rate
    wins = sum(1 for d in deltas if d > 0)
    win_rate = wins / n

    # Per-class mean delta
    if per_class_deltas:
        mean_per_class_delta = np.mean(per_class_deltas, axis=0).tolist()
    else:
        mean_per_class_delta = []

    return {
        "classifier": classifier_name,
        "classifier_name": CLASSIFIERS[classifier_name]["name"],
        "n_folds": total_folds,
        "baseline": {
            "mean": float(baseline_mean),
            "std": float(baseline_std),
        },
        "augmented": {
            "mean": float(augmented_mean),
            "std": float(augmented_std),
        },
        "delta": {
            "mean": float(delta_mean),
            "std": float(delta_std),
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "win_rate": float(win_rate),
            "wins": int(wins),
        },
        "per_class_delta_mean": mean_per_class_delta,
        "n_synthetic": len(y_synthetic) if y_synthetic is not None else 0,
    }


def print_summary(results: List[Dict], synth_name: str):
    """Print formatted comparison of all classifiers."""
    print("\n" + "=" * 80)
    print(f"  MULTI-CLASSIFIER EVALUATION SUMMARY")
    print(f"  Synthetic Data: {synth_name}")
    print("=" * 80)

    # Sort by delta mean
    results_sorted = sorted(results, key=lambda x: x["delta"]["mean"], reverse=True)

    print(f"\n{'Classifier':<25} {'Baseline':>10} {'Augmented':>10} {'Delta':>12} {'Signif':>8}")
    print("-" * 80)

    for r in results_sorted:
        base = r["baseline"]["mean"]
        aug = r["augmented"]["mean"]
        delta = r["delta"]["mean"]
        sig = "YES" if r["delta"]["significant"] else "no"
        sig_marker = "*" if r["delta"]["significant"] else ""

        print(f"{r['classifier_name']:<25} {base:>10.4f} {aug:>10.4f} {delta:>+10.4f}{sig_marker:>2} {sig:>8}")

    print("-" * 80)

    # Best classifier
    best = results_sorted[0]
    print(f"\n  BEST: {best['classifier_name']}")
    print(f"        Delta: {best['delta']['mean']:+.4f} ({best['delta']['mean']/best['baseline']['mean']*100:+.2f}%)")
    print(f"        95% CI: [{best['delta']['ci_95_lower']:+.4f}, {best['delta']['ci_95_upper']:+.4f}]")
    print(f"        p-value: {best['delta']['p_value']:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multi-Classifier Evaluator")
    parser.add_argument("--synth-csv", type=str, required=True,
                       help="Path to synthetic data CSV")
    parser.add_argument("--classifier", type=str, default="all",
                       help=f"Classifier to use. Options: {list(CLASSIFIERS.keys())} or 'all'")
    parser.add_argument("--data-path", type=str,
                       default="/home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv",
                       help="Path to original dataset")
    parser.add_argument("--cache-dir", type=str,
                       default="/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/embeddings_cache",
                       help="Embedding cache directory")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of folds for K-Fold CV")
    parser.add_argument("--repeated", type=int, default=3,
                       help="Number of repeats for Repeated K-Fold")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--synthetic-weight", type=float, default=0.5,
                       help="Weight for synthetic samples")
    parser.add_argument("--balanced", action="store_true",
                       help="Use class_weight='balanced'")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  PHASE H: Multi-Classifier Evaluation")
    print("=" * 80)
    print(f"  Synthetic: {args.synth_csv}")
    print(f"  Classifier: {args.classifier}")
    print(f"  K-Fold: k={args.k}, repeats={args.repeated}")
    print("=" * 80)

    # Load original data
    print(f"\nLoading original data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    text_col = "posts" if "posts" in df.columns else "text"
    label_col = "type" if "type" in df.columns else "label"
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_original = np.array([label_to_idx[l] for l in labels])

    print(f"  Original samples: {len(texts)}")
    print(f"  Classes: {len(unique_labels)} - {unique_labels}")

    # Load embeddings
    print(f"\nLoading embeddings...")
    X_original = load_embeddings_cached(args.cache_dir, "full")

    if X_original is None:
        print("  Cache miss, computing embeddings...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        X_original = compute_embeddings(texts, model)
        cache_path = Path(args.cache_dir) / "full_embeddings.npy"
        os.makedirs(args.cache_dir, exist_ok=True)
        np.save(cache_path, X_original)
        print(f"  Saved to {cache_path}")
    else:
        print(f"  Loaded from cache: {X_original.shape}")

    # Load synthetic data
    X_synthetic = None
    y_synthetic = None

    synth_path = Path(args.synth_csv)
    if synth_path.exists():
        print(f"\nLoading synthetic data from {synth_path}...")
        synth_df = pd.read_csv(synth_path)

        if len(synth_df) > 0:
            synth_texts = synth_df["text"].tolist()
            synth_labels = synth_df["label"].tolist()
            y_synthetic = np.array([label_to_idx.get(l, -1) for l in synth_labels])

            valid_mask = y_synthetic >= 0
            if not all(valid_mask):
                print(f"  Warning: {sum(~valid_mask)} samples with unknown labels")
                synth_texts = [t for t, v in zip(synth_texts, valid_mask) if v]
                y_synthetic = y_synthetic[valid_mask]

            print(f"  Synthetic samples: {len(synth_texts)}")

            # Compute synthetic embeddings
            print("  Computing synthetic embeddings...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            X_synthetic = compute_embeddings(synth_texts, model)
        else:
            print("  No synthetic samples")
    else:
        print(f"  Error: Synthetic file not found: {synth_path}")
        sys.exit(1)

    # Determine classifiers to run
    if args.classifier == "all":
        classifiers_to_run = list(CLASSIFIERS.keys())
    else:
        classifiers_to_run = [args.classifier]

    # Run evaluations
    class_weight = "balanced" if args.balanced else None
    all_results = []

    for clf_name in classifiers_to_run:
        try:
            results = run_kfold_evaluation(
                classifier_name=clf_name,
                X_original=X_original,
                y_original=y_original,
                X_synthetic=X_synthetic,
                y_synthetic=y_synthetic,
                n_splits=args.k,
                n_repeats=args.repeated,
                synthetic_weight=args.synthetic_weight,
                class_weight=class_weight,
                random_state=args.seed
            )
            all_results.append(results)
        except Exception as e:
            print(f"  Error with {clf_name}: {e}")

    # Print summary
    synth_name = synth_path.stem
    print_summary(all_results, synth_name)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"multiclf_{synth_name}_{timestamp}.json"

    summary = {
        "synth_file": str(synth_path),
        "timestamp": timestamp,
        "params": {
            "k": args.k,
            "repeats": args.repeated,
            "seed": args.seed,
            "synthetic_weight": args.synthetic_weight,
            "balanced": args.balanced,
        },
        "n_original": len(y_original),
        "n_synthetic": len(y_synthetic) if y_synthetic is not None else 0,
        "classes": unique_labels,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
