#!/usr/bin/env python3
"""
Multi-Model Evaluation for Replication Results

Evaluates synthetic data ensembles with multiple ML classifiers:
- LogisticRegression (baseline)
- RandomForest
- XGBoost
- GradientBoosting
- SVM (LinearSVC)

Uses hold-out evaluation (80/20 split) for methodological correctness.
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available, will skip")

# Paths - auto-detect
import os
SCRIPT_DIR = Path(__file__).parent
DEMO_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', DEMO_DIR.parent))
DATASET_PATH = PROJECT_ROOT / "mbti_1.csv"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = PROJECT_ROOT / "phase_e" / "embeddings_cache"


def get_classifiers():
    """Return dict of classifiers to evaluate."""
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LinearSVC': LinearSVC(max_iter=2000, random_state=42),
    }

    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    return classifiers


def load_dataset():
    """Load original MBTI dataset."""
    df = pd.read_csv(DATASET_PATH)
    if 'posts' in df.columns:
        df = df.rename(columns={'posts': 'text', 'type': 'label'})
    return df


def get_embeddings_cached(texts, model, cache_name: str = None):
    """Get embeddings with optional disk caching."""
    if cache_name:
        cache_path = CACHE_DIR / f"holdout_eval_{cache_name}.npz"
        if cache_path.exists():
            print(f"  Loading cached embeddings: {cache_path.name}")
            data = np.load(cache_path)
            return data['embeddings']

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Save to cache if name provided
    if cache_name:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"holdout_eval_{cache_name}.npz"
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"  Cached embeddings: {cache_path.name}")

    return embeddings


def evaluate_with_classifier(clf_name, clf, X_train, y_train, X_test, y_test,
                             X_synth, y_synth, synth_weight=0.5):
    """Evaluate baseline and augmented with specific classifier."""

    # Clone classifier for baseline
    from sklearn.base import clone
    clf_base = clone(clf)
    clf_aug = clone(clf)

    # Baseline
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, average='macro')
    f1_base_per_class = f1_score(y_test, y_pred_base, average=None)

    # Augmented
    if len(X_synth) > 0:
        X_train_aug = np.vstack([X_train, X_synth])
        y_train_aug = np.concatenate([y_train, y_synth])

        # Sample weights (not all classifiers support this)
        try:
            weights = np.concatenate([
                np.ones(len(X_train)),
                np.full(len(X_synth), synth_weight)
            ])
            clf_aug.fit(X_train_aug, y_train_aug, sample_weight=weights)
        except TypeError:
            # Classifier doesn't support sample_weight
            clf_aug.fit(X_train_aug, y_train_aug)

        y_pred_aug = clf_aug.predict(X_test)
        f1_aug = f1_score(y_test, y_pred_aug, average='macro')
        f1_aug_per_class = f1_score(y_test, y_pred_aug, average=None)
    else:
        f1_aug = f1_base
        f1_aug_per_class = f1_base_per_class

    delta = f1_aug - f1_base
    delta_pct = (delta / f1_base) * 100 if f1_base > 0 else 0

    return {
        'classifier': clf_name,
        'baseline_f1': float(f1_base),
        'augmented_f1': float(f1_aug),
        'delta_absolute': float(delta),
        'delta_percent': float(delta_pct),
        'baseline_per_class': f1_base_per_class.tolist(),
        'augmented_per_class': f1_aug_per_class.tolist(),
    }


def evaluate_ensemble(synth_path: Path, X_orig, y_orig, le: LabelEncoder,
                      model: SentenceTransformer, seed: int = 42):
    """Evaluate an ensemble with all classifiers."""

    # Load synthetics
    synth_df = pd.read_csv(synth_path)
    print(f"\n  Loading: {synth_path.name} ({len(synth_df)} synthetics)")

    # Compute synthetic embeddings
    X_synth = model.encode(synth_df['text'].tolist(), show_progress_bar=False, batch_size=64)
    y_synth = le.transform(synth_df['label'].values)

    # Fixed split
    y_encoded = le.transform(y_orig)
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_encoded,
        test_size=0.2,
        random_state=seed,
        stratify=y_encoded
    )

    results = {}
    classifiers = get_classifiers()

    for clf_name, clf in classifiers.items():
        print(f"    Evaluating with {clf_name}...", end=" ")
        try:
            result = evaluate_with_classifier(
                clf_name, clf,
                X_train, y_train, X_test, y_test,
                X_synth, y_synth
            )
            results[clf_name] = result
            print(f"Delta: {result['delta_percent']:+.2f}%")
        except Exception as e:
            print(f"Error: {e}")
            results[clf_name] = {'error': str(e)}

    return {
        'ensemble': synth_path.stem,
        'n_synthetic': len(synth_df),
        'results_by_classifier': results
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Evaluation')
    parser.add_argument('--repl-dir', type=str, required=True,
                       help='Replication results directory (e.g., replication_run1/results)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--ensembles', type=str, nargs='+',
                       default=['ENS_Top3_G5', 'ENS_TopG5_Extended', 'ENS_SUPER_G5_F7_v2'],
                       help='Ensembles to evaluate')
    parser.add_argument('--output', type=str, help='Output JSON path')
    args = parser.parse_args()

    repl_dir = Path(args.repl_dir)

    print("=" * 70)
    print("MULTI-MODEL EVALUATION")
    print("=" * 70)
    print(f"\nDirectory: {repl_dir}")
    print(f"Seed: {args.seed}")
    print(f"Ensembles: {args.ensembles}")

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"Dataset: {len(df)} samples, {len(set(labels))} classes")

    # Label encoder
    le = LabelEncoder()
    le.fit(labels)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings
    print("Computing original embeddings...")
    X_orig = get_embeddings_cached(texts, model, cache_name="mbti_original_full")
    y_orig = np.array(labels)

    # Evaluate each ensemble
    all_results = []

    for ens_name in args.ensembles:
        synth_path = repl_dir / f"{ens_name}_synth.csv"
        if synth_path.exists():
            result = evaluate_ensemble(synth_path, X_orig, y_orig, le, model, args.seed)
            all_results.append(result)
        else:
            print(f"\n  WARNING: {synth_path} not found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Create comparison table
    print(f"\n{'Ensemble':<25} {'Classifier':<20} {'Baseline':<10} {'Augmented':<10} {'Delta':<10}")
    print("-" * 75)

    for result in all_results:
        ens_name = result['ensemble'].replace('_synth', '')
        for clf_name, clf_result in result['results_by_classifier'].items():
            if 'error' not in clf_result:
                print(f"{ens_name:<25} {clf_name:<20} "
                      f"{clf_result['baseline_f1']:.4f}     "
                      f"{clf_result['augmented_f1']:.4f}     "
                      f"{clf_result['delta_percent']:+.2f}%")

    # Best classifier per ensemble
    print("\n" + "-" * 75)
    print("Best classifier per ensemble:")
    for result in all_results:
        ens_name = result['ensemble'].replace('_synth', '')
        best_clf = None
        best_delta = -float('inf')
        for clf_name, clf_result in result['results_by_classifier'].items():
            if 'error' not in clf_result and clf_result['delta_percent'] > best_delta:
                best_delta = clf_result['delta_percent']
                best_clf = clf_name
        if best_clf:
            print(f"  {ens_name}: {best_clf} ({best_delta:+.2f}%)")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repl_dir / "multimodel_evaluation.json"

    with open(output_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'ensembles': all_results,
            'classifiers': list(get_classifiers().keys())
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
