#!/usr/bin/env python3
"""
Stratified K-Fold Evaluator for Phase G

Evaluates augmented data using Stratified K-Fold CV to reduce variance.
Uses existing synthetic data (no new API calls).

Usage:
    python kfold_evaluator.py --config V3_low_vol --seed 42 --k 5
    python kfold_evaluator.py --config V2_high_vol --seed 42 --k 5 --repeated 5
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer


def load_embeddings_cached(cache_dir: str, prefix: str) -> Optional[np.ndarray]:
    """Load embeddings from cache."""
    cache_path = Path(cache_dir) / f"{prefix}_embeddings.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None


def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Compute embeddings for texts."""
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: Optional[str] = None
) -> Dict:
    """Train classifier and return metrics."""
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
    }


def run_kfold_evaluation(
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: Optional[np.ndarray],
    y_synthetic: Optional[np.ndarray],
    n_splits: int = 5,
    n_repeats: int = 1,
    synthetic_weight: float = 0.5,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    report_per_class: bool = False,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Run Stratified K-Fold evaluation.

    Returns metrics for both baseline and augmented models.
    """

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

    # Per-class tracking
    n_classes = len(np.unique(y_original))
    baseline_per_class = {i: [] for i in range(n_classes)}
    augmented_per_class = {i: [] for i in range(n_classes)}

    print(f"\nRunning {'Repeated ' if n_repeats > 1 else ''}Stratified K-Fold (k={n_splits}, repeats={n_repeats}, total={total_folds} folds)")
    print("=" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
        X_train_fold = X_original[train_idx]
        y_train_fold = y_original[train_idx]
        X_test_fold = X_original[test_idx]
        y_test_fold = y_original[test_idx]

        # Baseline: train on original only
        baseline_metrics = train_and_evaluate(
            X_train_fold, y_train_fold,
            X_test_fold, y_test_fold,
            class_weight=class_weight
        )
        baseline_f1 = baseline_metrics["macro_f1"]
        baseline_f1s.append(baseline_f1)

        # Track per-class F1 for baseline
        if report_per_class:
            for i, f1 in enumerate(baseline_metrics["per_class_f1"]):
                baseline_per_class[i].append(f1)

        # Augmented: train on original + synthetic
        if X_synthetic is not None and len(X_synthetic) > 0:
            # Combine with synthetic data
            X_train_aug = np.vstack([X_train_fold, X_synthetic])
            y_train_aug = np.concatenate([y_train_fold, y_synthetic])

            # Apply sample weights (original=1.0, synthetic=weight)
            sample_weights = np.concatenate([
                np.ones(len(y_train_fold)),
                np.full(len(y_synthetic), synthetic_weight)
            ])

            # Train with sample weights
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight)
            clf.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)
            y_pred = clf.predict(X_test_fold)
            augmented_f1 = f1_score(y_test_fold, y_pred, average="macro")

            # Track per-class F1 for augmented
            if report_per_class:
                aug_per_class_f1 = f1_score(y_test_fold, y_pred, average=None)
                for i, f1 in enumerate(aug_per_class_f1):
                    augmented_per_class[i].append(f1)
        else:
            augmented_f1 = baseline_f1  # No augmentation
            if report_per_class:
                for i in range(n_classes):
                    augmented_per_class[i].append(baseline_per_class[i][-1])

        augmented_f1s.append(augmented_f1)
        delta = augmented_f1 - baseline_f1
        deltas.append(delta)

        print(f"  Fold {fold_idx + 1:2d}: Baseline={baseline_f1:.4f}, Aug={augmented_f1:.4f}, Delta={delta:+.4f} ({delta/baseline_f1*100:+.2f}%)")

    # Compute statistics
    baseline_mean = np.mean(baseline_f1s)
    baseline_std = np.std(baseline_f1s, ddof=1)
    augmented_mean = np.mean(augmented_f1s)
    augmented_std = np.std(augmented_f1s, ddof=1)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)

    # 95% Confidence Interval
    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=delta_std/np.sqrt(n))

    # Statistical significance (one-sample t-test: is delta > 0?)
    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    # Win rate
    wins = sum(1 for d in deltas if d > 0)
    win_rate = wins / n

    results = {
        "n_folds": total_folds,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "baseline": {
            "mean": float(baseline_mean),
            "std": float(baseline_std),
            "values": [float(v) for v in baseline_f1s]
        },
        "augmented": {
            "mean": float(augmented_mean),
            "std": float(augmented_std),
            "values": [float(v) for v in augmented_f1s]
        },
        "delta": {
            "mean": float(delta_mean),
            "std": float(delta_std),
            "values": [float(v) for v in deltas],
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "win_rate": float(win_rate),
            "wins": int(wins)
        },
        "n_synthetic": len(y_synthetic) if y_synthetic is not None else 0
    }

    # Add per-class results
    if report_per_class and class_names is not None:
        per_class_results = {}
        for i, class_name in enumerate(class_names):
            baseline_vals = baseline_per_class[i]
            augmented_vals = augmented_per_class[i]
            per_class_results[class_name] = {
                "baseline_mean": float(np.mean(baseline_vals)),
                "baseline_std": float(np.std(baseline_vals, ddof=1)) if len(baseline_vals) > 1 else 0.0,
                "augmented_mean": float(np.mean(augmented_vals)),
                "augmented_std": float(np.std(augmented_vals, ddof=1)) if len(augmented_vals) > 1 else 0.0,
            }
        results["per_class"] = per_class_results

    return results


def print_summary(results: Dict, config_name: str):
    """Print formatted summary of K-Fold results."""
    print("\n" + "=" * 60)
    print(f"  K-FOLD EVALUATION SUMMARY: {config_name}")
    print("=" * 60)

    baseline = results["baseline"]
    augmented = results["augmented"]
    delta = results["delta"]

    print(f"\n  Folds: {results['n_folds']} ({results['n_splits']} splits x {results['n_repeats']} repeats)")
    print(f"  Synthetic samples used: {results['n_synthetic']}")

    print(f"\n  BASELINE:")
    print(f"    Mean F1: {baseline['mean']:.4f} +/- {baseline['std']:.4f}")

    print(f"\n  AUGMENTED:")
    print(f"    Mean F1: {augmented['mean']:.4f} +/- {augmented['std']:.4f}")

    print(f"\n  IMPROVEMENT (Delta):")
    print(f"    Mean:     {delta['mean']:+.4f} ({delta['mean']/baseline['mean']*100:+.2f}%)")
    print(f"    Std:      {delta['std']:.4f}")
    print(f"    95% CI:   [{delta['ci_95_lower']:+.4f}, {delta['ci_95_upper']:+.4f}]")
    print(f"    Win Rate: {delta['wins']}/{results['n_folds']} ({delta['win_rate']*100:.1f}%)")

    print(f"\n  STATISTICAL SIGNIFICANCE:")
    print(f"    t-statistic: {delta['t_statistic']:.3f}")
    print(f"    p-value:     {delta['p_value']:.4f}")
    if delta['significant']:
        if delta['mean'] > 0:
            print(f"    Result:      SIGNIFICANT IMPROVEMENT (p < 0.05)")
        else:
            print(f"    Result:      SIGNIFICANT DEGRADATION (p < 0.05)")
    else:
        print(f"    Result:      NOT SIGNIFICANT (p >= 0.05)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stratified K-Fold Evaluator")
    parser.add_argument("--config", type=str, required=True,
                       help="Config name (e.g., V3_low_vol, V2_high_vol)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed used for the experiment (to load correct files)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of folds for K-Fold CV")
    parser.add_argument("--repeated", type=int, default=1,
                       help="Number of repeats for Repeated K-Fold (default: 1 = no repeat)")
    parser.add_argument("--results-dir", type=str,
                       default="/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/results",
                       help="Directory with experiment results")
    parser.add_argument("--data-path", type=str,
                       default="/home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv",
                       help="Path to original dataset")
    parser.add_argument("--cache-dir", type=str,
                       default="/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/embeddings_cache",
                       help="Embedding cache directory")
    parser.add_argument("--synthetic-weight", type=float, default=0.5,
                       help="Weight for synthetic samples (default: 0.5)")
    parser.add_argument("--balanced", action="store_true",
                       help="Use class_weight='balanced' in classifier")
    parser.add_argument("--report-per-class", action="store_true",
                       help="Report per-class F1 scores in output")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    # Paths
    results_dir = Path(args.results_dir)
    synth_path = results_dir / f"{args.config}_s{args.seed}_synth.csv"

    print(f"\nStratified K-Fold Evaluator")
    print(f"Config: {args.config}, Seed: {args.seed}, K: {args.k}, Repeats: {args.repeated}")
    print("-" * 60)

    # Load original data
    print(f"\nLoading original data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    # Handle different column names
    text_col = "posts" if "posts" in df.columns else "text"
    label_col = "type" if "type" in df.columns else "label"
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_original = np.array([label_to_idx[l] for l in labels])

    print(f"  Original samples: {len(texts)}")
    print(f"  Classes: {len(unique_labels)}")

    # Load or compute embeddings
    print(f"\nLoading embeddings...")
    X_original = load_embeddings_cached(args.cache_dir, "full")

    if X_original is None:
        print("  Cache miss, computing embeddings...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        X_original = compute_embeddings(texts, model)
        # Save to cache
        cache_path = Path(args.cache_dir) / "full_embeddings.npy"
        np.save(cache_path, X_original)
        print(f"  Saved to {cache_path}")
    else:
        print(f"  Loaded from cache: {X_original.shape}")

    # Load synthetic data
    X_synthetic = None
    y_synthetic = None

    if synth_path.exists():
        print(f"\nLoading synthetic data from {synth_path}...")
        synth_df = pd.read_csv(synth_path)

        if len(synth_df) > 0:
            synth_texts = synth_df["text"].tolist()
            synth_labels = synth_df["label"].tolist()
            y_synthetic = np.array([label_to_idx.get(l, -1) for l in synth_labels])

            # Filter out unknown labels
            valid_mask = y_synthetic >= 0
            if not all(valid_mask):
                print(f"  Warning: {sum(~valid_mask)} synthetic samples with unknown labels")
                synth_texts = [t for t, v in zip(synth_texts, valid_mask) if v]
                y_synthetic = y_synthetic[valid_mask]

            print(f"  Synthetic samples: {len(synth_texts)}")

            # Compute embeddings for synthetic
            print("  Computing synthetic embeddings...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            X_synthetic = compute_embeddings(synth_texts, model)
            print(f"  Shape: {X_synthetic.shape}")
        else:
            print("  No synthetic samples in file")
    else:
        print(f"\nWarning: Synthetic file not found: {synth_path}")
        print("  Running baseline-only evaluation")

    # Run K-Fold evaluation
    class_weight = "balanced" if args.balanced else None

    results = run_kfold_evaluation(
        X_original=X_original,
        y_original=y_original,
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_splits=args.k,
        n_repeats=args.repeated,
        synthetic_weight=args.synthetic_weight,
        class_weight=class_weight,
        random_state=args.seed,
        report_per_class=args.report_per_class,
        class_names=unique_labels if args.report_per_class else None
    )

    # Add metadata
    results["config"] = args.config
    results["seed"] = args.seed
    results["synthetic_weight"] = args.synthetic_weight
    results["class_weight"] = class_weight

    # Print summary
    print_summary(results, args.config)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / f"{args.config}_s{args.seed}_kfold_k{args.k}.json"

    with open(output_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        results_json = json.loads(json.dumps(results, default=convert))
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
