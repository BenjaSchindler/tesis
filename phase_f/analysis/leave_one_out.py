#!/usr/bin/env python3
"""
Phase F Analysis: Leave-One-Out Analysis

For ENS_Top3_G5 (CMB3 + CF1 + V4 + G5), create:
- ENS without CMB3
- ENS without CF1
- ENS without V4
- ENS without G5

Evaluate each with K-fold to measure marginal contribution.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"
DATA_PATH = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/mbti_1.csv")
CACHE_DIR = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_e/embeddings_cache")

# ENS_Top3_G5 components
COMPONENTS = {
    "CMB3_skip": RESULTS_DIR / "CMB3_skip_s42_synth.csv",
    "CF1_conf_band": RESULTS_DIR / "CF1_conf_band_s42_synth.csv",
    "V4_ultra": RESULTS_DIR / "V4_ultra_s42_synth.csv",
    "G5_K25_medium": RESULTS_DIR / "G5_K25_medium_s42_synth.csv",
}

def load_original_data():
    """Load original dataset."""
    df = pd.read_csv(DATA_PATH)
    # Column names are 'type' and 'posts'
    return df['posts'].tolist(), df['type'].tolist()

def load_synth_data(excluded_component: Optional[str] = None):
    """Load synthetic data, optionally excluding one component."""
    dfs = []
    for name, path in COMPONENTS.items():
        if excluded_component and name == excluded_component:
            continue
        if path.exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        return [], []

    combined = pd.concat(dfs, ignore_index=True)
    return combined['text'].tolist(), combined['label'].tolist()

def get_embeddings(texts: List[str], model: SentenceTransformer, cache_key: str = None) -> np.ndarray:
    """Get embeddings, using cache if available."""
    if cache_key:
        cache_path = CACHE_DIR / f"{cache_key}_embeddings.npy"
        if cache_path.exists():
            print(f"  Loading cached embeddings: {cache_key}")
            return np.load(cache_path)

    print(f"  Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    if cache_key:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings

def run_kfold_evaluation(
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    synthetic_weight: float = 0.5
) -> Dict:
    """Run K-fold evaluation."""

    kfold = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )
    total_folds = n_splits * n_repeats

    baseline_f1s = []
    augmented_f1s = []
    deltas = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_original, y_original)):
        X_train = X_original[train_idx]
        y_train = y_original[train_idx]
        X_test = X_original[test_idx]
        y_test = y_original[test_idx]

        # Baseline
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X_train, y_train)
        baseline_f1 = f1_score(y_test, clf.predict(X_test), average="macro")
        baseline_f1s.append(baseline_f1)

        # Augmented
        X_train_aug = np.vstack([X_train, X_synthetic])
        y_train_aug = np.concatenate([y_train, y_synthetic])
        sample_weights = np.concatenate([
            np.ones(len(y_train)),
            np.full(len(y_synthetic), synthetic_weight)
        ])

        clf.fit(X_train_aug, y_train_aug, sample_weight=sample_weights)
        augmented_f1 = f1_score(y_test, clf.predict(X_test), average="macro")
        augmented_f1s.append(augmented_f1)

        deltas.append(augmented_f1 - baseline_f1)

    delta_mean = np.mean(deltas)
    n = len(deltas)
    ci_95 = stats.t.interval(0.95, n-1, loc=delta_mean, scale=np.std(deltas, ddof=1)/np.sqrt(n))
    t_stat, p_value = stats.ttest_1samp(deltas, 0)

    return {
        "delta_mean": delta_mean * 100,  # Convert to %
        "delta_std": np.std(deltas, ddof=1) * 100,
        "ci_95": (ci_95[0] * 100, ci_95[1] * 100),
        "p_value": p_value,
        "win_rate": sum(1 for d in deltas if d > 0) / n,
        "n_synthetic": len(y_synthetic)
    }

def main():
    print("="*80)
    print("LEAVE-ONE-OUT ANALYSIS: ENS_Top3_G5")
    print("="*80)

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Load original data
    print("Loading original data...")
    orig_texts, orig_labels = load_original_data()
    X_original = get_embeddings(orig_texts, model, cache_key="mbti_original")
    y_original = np.array(orig_labels)

    # Results storage
    results = {}

    # Full ensemble (reference)
    print("\n" + "-"*60)
    print("Evaluating: FULL ENSEMBLE (ENS_Top3_G5)")
    print("-"*60)

    full_texts, full_labels = load_synth_data(excluded_component=None)
    X_full = get_embeddings(full_texts, model, cache_key="ens_top3_g5_synth")
    y_full = np.array(full_labels)

    results["FULL"] = run_kfold_evaluation(X_original, y_original, X_full, y_full)
    print(f"  Delta: {results['FULL']['delta_mean']:+.3f}% (p={results['FULL']['p_value']:.6f})")
    print(f"  Synthetics: {results['FULL']['n_synthetic']}")

    # Leave-one-out for each component
    for component in COMPONENTS.keys():
        print(f"\n" + "-"*60)
        print(f"Evaluating: WITHOUT {component}")
        print("-"*60)

        loo_texts, loo_labels = load_synth_data(excluded_component=component)
        X_loo = model.encode(loo_texts, show_progress_bar=False, convert_to_numpy=True)
        y_loo = np.array(loo_labels)

        results[f"without_{component}"] = run_kfold_evaluation(X_original, y_original, X_loo, y_loo)

        r = results[f"without_{component}"]
        print(f"  Delta: {r['delta_mean']:+.3f}% (p={r['p_value']:.6f})")
        print(f"  Synthetics: {r['n_synthetic']}")

    # Calculate marginal contributions
    print("\n" + "="*80)
    print("MARGINAL CONTRIBUTION ANALYSIS")
    print("="*80)

    full_delta = results["FULL"]["delta_mean"]
    full_synth = results["FULL"]["n_synthetic"]

    print(f"\nFull Ensemble: {full_delta:+.3f}% ({full_synth} synthetics)")
    print("-"*60)
    print(f"{'Component':<20} {'Without':>12} {'Contribution':>14} {'Synth':>8} {'%/synth':>10}")
    print("-"*60)

    contributions = []
    for component in COMPONENTS.keys():
        r = results[f"without_{component}"]
        without_delta = r["delta_mean"]
        contribution = full_delta - without_delta
        synth_count = full_synth - r["n_synthetic"]
        per_synth = contribution / synth_count if synth_count > 0 else 0

        contributions.append({
            "component": component,
            "contribution": contribution,
            "synth_count": synth_count,
            "per_synth": per_synth
        })

        print(f"{component:<20} {without_delta:>+11.3f}% {contribution:>+13.3f}% {synth_count:>8} {per_synth:>+9.4f}%")

    # Rank by contribution
    contributions.sort(key=lambda x: -x["contribution"])

    print("\n" + "="*80)
    print("RANKING BY CONTRIBUTION")
    print("="*80)

    for i, c in enumerate(contributions, 1):
        print(f"  {i}. {c['component']:<20}: {c['contribution']:+.3f}% ({c['synth_count']} synth)")

    # Most valuable
    most_valuable = contributions[0]
    print(f"\n  MOST VALUABLE: {most_valuable['component']}")
    print(f"  Removing it drops delta by {most_valuable['contribution']:+.3f}%")

    # Least valuable
    least_valuable = contributions[-1]
    print(f"\n  LEAST VALUABLE: {least_valuable['component']}")
    print(f"  Removing it only drops delta by {least_valuable['contribution']:+.3f}%")

    # Save results
    output = {
        "full_ensemble": results["FULL"],
        "leave_one_out": {k: v for k, v in results.items() if k != "FULL"},
        "contributions": contributions
    }

    with open(OUTPUT_DIR / "leave_one_out_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {OUTPUT_DIR / 'leave_one_out_results.json'}")

if __name__ == "__main__":
    main()
