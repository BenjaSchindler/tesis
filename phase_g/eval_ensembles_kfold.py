#!/usr/bin/env python3
"""
Evaluate cross-phase ensembles with K-fold.
Only evaluates files in ensembles_v2/ directory.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
ENSEMBLE_DIR = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/results/ensembles_v2")
DATASET_PATH = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv")
OUTPUT_FILE = ENSEMBLE_DIR / "kfold_ensemble_results.json"

# Use same embedding model as kfold_multimodel.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Config
K_FOLDS = 5
N_REPEATS = 3
SEED = 42


def load_dataset():
    """Load original MBTI dataset."""
    df = pd.read_csv(DATASET_PATH)
    # Normalize column names
    if 'posts' in df.columns:
        df = df.rename(columns={'posts': 'text', 'type': 'label'})
    return df


def get_embeddings(texts, model):
    """Get sentence embeddings."""
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def evaluate_kfold(
    X_orig, y_orig,
    X_synth, y_synth,
    synth_weight: float = 0.5,
    k: int = 5,
    repeats: int = 3,
    seed: int = 42
):
    """
    Evaluate with repeated K-fold cross-validation.

    Returns dict with baseline and augmented metrics.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_orig)

    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)

    baseline_scores = []
    augmented_scores = []
    delta_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_orig, y_encoded)):
        X_train, X_test = X_orig[train_idx], X_orig[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Baseline model
        clf_base = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        clf_base.fit(X_train, y_train)
        y_pred_base = clf_base.predict(X_test)
        f1_base = f1_score(y_test, y_pred_base, average='macro')
        baseline_scores.append(f1_base)

        # Augmented model
        if len(X_synth) > 0:
            # Add synthetics to training
            X_train_aug = np.vstack([X_train, X_synth])

            # Encode synthetic labels
            y_synth_encoded = le.transform(y_synth)
            y_train_aug = np.concatenate([y_train, y_synth_encoded])

            # Sample weights (original=1.0, synthetic=synth_weight)
            weights = np.concatenate([
                np.ones(len(X_train)),
                np.full(len(X_synth), synth_weight)
            ])

            clf_aug = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
            clf_aug.fit(X_train_aug, y_train_aug, sample_weight=weights)
            y_pred_aug = clf_aug.predict(X_test)
            f1_aug = f1_score(y_test, y_pred_aug, average='macro')
        else:
            f1_aug = f1_base

        augmented_scores.append(f1_aug)
        delta_scores.append(f1_aug - f1_base)

    # Statistics
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)
    augmented_mean = np.mean(augmented_scores)
    augmented_std = np.std(augmented_scores)
    delta_mean = np.mean(delta_scores)
    delta_std = np.std(delta_scores)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(augmented_scores, baseline_scores)

    # 95% CI for delta
    n = len(delta_scores)
    se = delta_std / np.sqrt(n)
    ci_lower = delta_mean - 1.96 * se
    ci_upper = delta_mean + 1.96 * se

    # Win rate
    wins = sum(1 for d in delta_scores if d > 0)
    win_rate = wins / len(delta_scores)

    return {
        'baseline': {
            'mean': float(baseline_mean),
            'std': float(baseline_std),
            'values': [float(x) for x in baseline_scores],
        },
        'augmented': {
            'mean': float(augmented_mean),
            'std': float(augmented_std),
            'values': [float(x) for x in augmented_scores],
        },
        'delta': {
            'mean': float(delta_mean),
            'std': float(delta_std),
            'values': [float(x) for x in delta_scores],
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'win_rate': float(win_rate),
            'wins': int(wins),
        }
    }


def main():
    print("=" * 70)
    print("Cross-Phase Ensemble K-Fold Evaluation")
    print("=" * 70)
    print(f"\nK={K_FOLDS}, Repeats={N_REPEATS}, Total folds={K_FOLDS * N_REPEATS}")

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing original embeddings...")
    X_orig = get_embeddings(texts, model)
    y_orig = np.array(labels)

    # Find ensemble files
    ensemble_files = list(ENSEMBLE_DIR.glob("ENS_*_synth.csv"))
    print(f"\nFound {len(ensemble_files)} ensembles to evaluate")

    results = []

    for synth_file in sorted(ensemble_files):
        ens_name = synth_file.stem.replace("_s42_synth", "")
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {ens_name}")
        print("=" * 50)

        # Load synthetics
        synth_df = pd.read_csv(synth_file)
        n_synth = len(synth_df)
        print(f"Synthetics: {n_synth}")

        # Get synthetic embeddings
        print("Computing synthetic embeddings...")
        X_synth = get_embeddings(synth_df['text'].tolist(), model)
        y_synth = synth_df['label'].values

        # Class distribution
        class_dist = synth_df['label'].value_counts().to_dict()
        print(f"Class distribution: {class_dist}")

        # Evaluate
        print(f"Running K-fold evaluation...")
        eval_result = evaluate_kfold(
            X_orig, y_orig,
            X_synth, y_synth,
            synth_weight=0.5,
            k=K_FOLDS,
            repeats=N_REPEATS,
            seed=SEED
        )

        # Add metadata
        eval_result['ensemble'] = ens_name
        eval_result['n_synthetic'] = n_synth
        eval_result['class_distribution'] = class_dist

        # Print summary
        delta_pct = eval_result['delta']['mean'] * 100
        p_val = eval_result['delta']['p_value']
        win_rate = eval_result['delta']['win_rate'] * 100
        sig = "✓" if eval_result['delta']['significant'] else ""

        print(f"\nResult: {delta_pct:+.2f}% (p={p_val:.6f}, WR={win_rate:.0f}%) {sig}")

        results.append(eval_result)

    # Sort by delta
    results.sort(key=lambda x: x['delta']['mean'], reverse=True)

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Ensemble':<25} {'Synth':>6} {'Delta':>8} {'p-value':>10} {'WinRate':>8}")
    print("-" * 65)

    for r in results:
        delta_pct = r['delta']['mean'] * 100
        p_val = r['delta']['p_value']
        win_rate = r['delta']['win_rate'] * 100
        sig = "✓" if r['delta']['significant'] else ""
        print(f"{r['ensemble']:<25} {r['n_synthetic']:>6} {delta_pct:>+7.2f}% {p_val:>10.6f} {win_rate:>7.0f}% {sig}")

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Compare to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO ENS_Top3_G5 (+5.98%)")
    print("=" * 70)
    for r in results:
        delta_pct = r['delta']['mean'] * 100
        if delta_pct > 5.98:
            print(f"  🏆 {r['ensemble']}: {delta_pct:+.2f}% BEATS ENS_Top3_G5!")
        elif delta_pct > 2.69:
            print(f"  ✓ {r['ensemble']}: {delta_pct:+.2f}% (beats best individual W9)")
        else:
            print(f"    {r['ensemble']}: {delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
