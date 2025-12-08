#!/usr/bin/env python3
"""
Correct Hold-out Evaluation - NO DATA LEAKAGE

This evaluator uses the same methodology as Phase A:
1. Fixed 80/20 train/test split with specific seed
2. Synthetics were generated from ONLY the train split
3. Evaluation is done ONLY on the test split (never seen during generation)

This avoids the K-fold leakage issue where the same synthetics
were used across all folds, some of which contained training data
that influenced the synthetic generation.

For robustness, run with multiple seeds (like Phase A's 25-seed validation).
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/home/benja/Desktop/Tesis/SMOTE-LLM")
DATASET_PATH = PROJECT_ROOT / "mbti_1.csv"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = PROJECT_ROOT / "phase_e" / "embeddings_cache"


def load_dataset():
    """Load original MBTI dataset."""
    df = pd.read_csv(DATASET_PATH)
    if 'posts' in df.columns:
        df = df.rename(columns={'posts': 'text', 'type': 'label'})
    return df


def get_cache_path(cache_name: str) -> Path:
    """Get cache file path for embeddings."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"holdout_eval_{cache_name}.npz"


def get_embeddings_cached(texts, model, cache_name: str = None):
    """Get embeddings with optional disk caching."""
    if cache_name:
        cache_path = get_cache_path(cache_name)
        if cache_path.exists():
            print(f"  Loading cached embeddings: {cache_path.name}")
            data = np.load(cache_path)
            return data['embeddings']

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Save to cache if name provided
    if cache_name:
        cache_path = get_cache_path(cache_name)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"  Cached embeddings: {cache_path.name}")

    return embeddings


def get_embeddings(texts, model):
    """Get sentence embeddings (no cache, for backwards compatibility)."""
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def evaluate_holdout(
    X_orig, y_orig,
    X_synth, y_synth,
    synth_weight: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
    classifier: str = 'LogisticRegression'
):
    """
    Evaluate with fixed hold-out split.

    This is methodologically correct because:
    - The synthetics were generated from ONLY the train split (80%)
    - We evaluate on ONLY the test split (20%) which never influenced generation
    - No data from test leaked into training
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)

    # Fixed split - MUST match the split used during synthetic generation
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded
    )

    # Baseline model
    if classifier == 'LogisticRegression':
        clf_base = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
    else:
        clf_base = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, average='macro')
    f1_base_per_class = f1_score(y_test, y_pred_base, average=None)

    # Augmented model
    if len(X_synth) > 0:
        X_train_aug = np.vstack([X_train, X_synth])
        y_synth_encoded = le.transform(y_synth)
        y_train_aug = np.concatenate([y_train, y_synth_encoded])

        # Sample weights
        weights = np.concatenate([
            np.ones(len(X_train)),
            np.full(len(X_synth), synth_weight)
        ])

        if classifier == 'LogisticRegression':
            clf_aug = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
        else:
            clf_aug = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

        clf_aug.fit(X_train_aug, y_train_aug, sample_weight=weights)
        y_pred_aug = clf_aug.predict(X_test)
        f1_aug = f1_score(y_test, y_pred_aug, average='macro')
        f1_aug_per_class = f1_score(y_test, y_pred_aug, average=None)
    else:
        f1_aug = f1_base
        f1_aug_per_class = f1_base_per_class

    # Per-class results
    per_class = {}
    for i, label in enumerate(le.classes_):
        per_class[label] = {
            'baseline': float(f1_base_per_class[i]),
            'augmented': float(f1_aug_per_class[i]),
            'delta_pp': float((f1_aug_per_class[i] - f1_base_per_class[i]) * 100)
        }

    delta = f1_aug - f1_base
    delta_pct = (delta / f1_base) * 100 if f1_base > 0 else 0

    return {
        'baseline_f1': float(f1_base),
        'augmented_f1': float(f1_aug),
        'delta_absolute': float(delta),
        'delta_percent': float(delta_pct),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_synthetic': len(X_synth),
        'per_class': per_class,
        'classifier': classifier,
        'seed': seed
    }


def evaluate_multi_seed(
    X_orig, y_orig,
    X_synth, y_synth,
    seeds: list,
    synth_weight: float = 0.5,
    test_size: float = 0.2,
    classifier: str = 'LogisticRegression'
):
    """
    Evaluate with multiple seeds for robustness (like Phase A 25-seed validation).

    Note: This requires that synthetics were generated independently for each seed,
    OR that we accept the limitation that synthetics may have some overlap with
    test data for seeds other than the generation seed.

    For strict correctness, only use seed=42 (the generation seed).
    """
    results = []
    for seed in seeds:
        result = evaluate_holdout(
            X_orig, y_orig, X_synth, y_synth,
            synth_weight=synth_weight,
            test_size=test_size,
            seed=seed,
            classifier=classifier
        )
        results.append(result)

    # Aggregate
    deltas = [r['delta_percent'] for r in results]

    return {
        'seeds': seeds,
        'results': results,
        'mean_delta': float(np.mean(deltas)),
        'std_delta': float(np.std(deltas)),
        'min_delta': float(np.min(deltas)),
        'max_delta': float(np.max(deltas)),
        'positive_rate': float(sum(1 for d in deltas if d > 0) / len(deltas))
    }


def main():
    parser = argparse.ArgumentParser(description='Correct Hold-out Evaluation (No Data Leakage)')
    parser.add_argument('--synth', type=str, required=True, help='Path to synthetic CSV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--multi-seed', action='store_true', help='Run with multiple seeds')
    parser.add_argument('--classifier', type=str, default='LogisticRegression',
                       choices=['LogisticRegression', 'RandomForest'])
    parser.add_argument('--output', type=str, help='Output JSON path')
    args = parser.parse_args()

    print("=" * 70)
    print("CORRECT HOLD-OUT EVALUATION (No Data Leakage)")
    print("=" * 70)
    print(f"\nMethodology: Fixed 80/20 split, synthetics from train only")
    print(f"Classifier: {args.classifier}")
    print(f"Seed: {args.seed}")

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"Dataset: {len(df)} samples, {len(set(labels))} classes")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Compute embeddings (cached for original dataset - always the same)
    print("Computing original embeddings...")
    X_orig = get_embeddings_cached(texts, model, cache_name="mbti_original_full")
    y_orig = np.array(labels)

    # Load synthetics
    print(f"\nLoading synthetics from: {args.synth}")
    synth_df = pd.read_csv(args.synth)
    print(f"Synthetics: {len(synth_df)} samples")

    # Compute synthetic embeddings (no cache - different each replication)
    print("Computing synthetic embeddings...")
    X_synth = get_embeddings(synth_df['text'].tolist(), model)
    y_synth = synth_df['label'].values

    # Class distribution
    print(f"\nSynthetic class distribution:")
    for label, count in synth_df['label'].value_counts().items():
        print(f"  {label}: {count}")

    if args.multi_seed:
        # Multi-seed evaluation
        seeds = [42, 100, 123, 456, 789]
        print(f"\nRunning multi-seed evaluation with seeds: {seeds}")
        print("WARNING: Only seed=42 is strictly correct (generation seed)")

        result = evaluate_multi_seed(
            X_orig, y_orig, X_synth, y_synth,
            seeds=seeds,
            classifier=args.classifier
        )

        print("\n" + "=" * 70)
        print("MULTI-SEED RESULTS")
        print("=" * 70)
        print(f"\nMean Delta: {result['mean_delta']:+.2f}% ± {result['std_delta']:.2f}%")
        print(f"Range: [{result['min_delta']:+.2f}%, {result['max_delta']:+.2f}%]")
        print(f"Positive Rate: {result['positive_rate']*100:.0f}%")

        print("\nPer-seed results:")
        for r in result['results']:
            print(f"  Seed {r['seed']}: {r['delta_percent']:+.2f}%")

    else:
        # Single seed evaluation (CORRECT methodology)
        print(f"\nRunning evaluation with seed={args.seed}...")

        result = evaluate_holdout(
            X_orig, y_orig, X_synth, y_synth,
            seed=args.seed,
            classifier=args.classifier
        )

        print("\n" + "=" * 70)
        print("RESULTS (Correct Hold-out, No Leakage)")
        print("=" * 70)
        print(f"\nBaseline F1:  {result['baseline_f1']:.4f}")
        print(f"Augmented F1: {result['augmented_f1']:.4f}")
        print(f"Delta:        {result['delta_percent']:+.2f}%")
        print(f"\nTrain size: {result['train_size']}")
        print(f"Test size:  {result['test_size']}")
        print(f"Synthetics: {result['n_synthetic']}")

        # Per-class results
        print("\n" + "-" * 50)
        print("Per-Class Results:")
        print("-" * 50)
        print(f"{'Class':<8} {'Baseline':>10} {'Augmented':>10} {'Delta':>10}")
        print("-" * 50)

        # Sort by delta
        sorted_classes = sorted(
            result['per_class'].items(),
            key=lambda x: x[1]['delta_pp'],
            reverse=True
        )

        for label, metrics in sorted_classes:
            delta_str = f"{metrics['delta_pp']:+.1f}pp"
            print(f"{label:<8} {metrics['baseline']:>10.3f} {metrics['augmented']:>10.3f} {delta_str:>10}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        synth_name = Path(args.synth).stem.replace('_synth', '')
        output_path = Path(args.synth).parent / f"{synth_name}_holdout_correct.json"

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Compare with K-fold if available
    kfold_path = Path(args.synth).parent / f"{Path(args.synth).stem.replace('_synth', '')}_kfold_results.json"
    if kfold_path.exists():
        with open(kfold_path) as f:
            kfold_result = json.load(f)

        if isinstance(kfold_result, dict) and 'delta' in kfold_result:
            kfold_delta = kfold_result['delta']['mean'] * 100
        elif isinstance(kfold_result, list):
            kfold_delta = kfold_result[0].get('delta', {}).get('mean', 0) * 100
        else:
            kfold_delta = None

        if kfold_delta:
            print("\n" + "=" * 70)
            print("COMPARISON: Hold-out vs K-fold (with leakage)")
            print("=" * 70)
            holdout_delta = result['delta_percent'] if not args.multi_seed else result['mean_delta']
            print(f"\nK-fold (with leakage):    {kfold_delta:+.2f}%")
            print(f"Hold-out (no leakage):    {holdout_delta:+.2f}%")
            print(f"Difference (bias):        {kfold_delta - holdout_delta:+.2f}%")


if __name__ == "__main__":
    main()
