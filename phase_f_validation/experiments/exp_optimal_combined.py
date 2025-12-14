#!/usr/bin/env python3
"""
OPTIMAL COMBINED EXPERIMENT
============================
Combines all best configurations found in expanded experiments:

- K_MAX = 18           (exp01: +1.92%)
- K_NEIGHBORS = 200    (exp03: +1.60%)
- TIER_WEIGHTS = (2.0, 0.8, 0.3)  (exp07a: +5.12%)
- TEMPERATURE = 0.9    (exp07b: +2.03%)
- BUDGET = 0.20        (exp07c: +2.35%)

Hypothesis: Combining optimal params should yield > +5% improvement.
"""

import sys
import os
import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_config import *

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# OPTIMAL CONFIGURATION
# ============================================================
OPTIMAL_CONFIG = {
    "K_MAX": 18,                    # Best from exp01
    "K_NEIGHBORS": 200,             # Best from exp03
    "TIER_WEIGHTS": {               # Best from exp07a
        "LOW": 2.0,
        "MID": 0.8,
        "HIGH": 0.3
    },
    "TEMPERATURE": 0.9,             # Best from exp07b
    "BUDGET": 0.20,                 # Best from exp07c
}

# Tier thresholds (from exp06)
TIER_THRESHOLDS = {
    "LOW": (0.0, 0.20),
    "MID": (0.20, 0.45),
    "HIGH": (0.45, 1.0),
}

# ============================================================
# Helper Functions
# ============================================================

def load_data():
    """Load dataset and embeddings from cache."""
    import pandas as pd

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    texts = df['posts'].tolist()
    labels = df['type'].tolist()
    print(f"  Loaded {len(texts)} samples, {len(set(labels))} classes")

    # Load cached embeddings
    if EMBEDDING_CACHE_PATH.exists():
        print(f"  Loading cached embeddings from {EMBEDDING_CACHE_PATH}")
        embeddings = np.load(EMBEDDING_CACHE_PATH)
        print(f"  Loaded {len(embeddings)} embeddings from cache")
    else:
        print("  Computing embeddings...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDING_CACHE_PATH, embeddings)

    return np.array(texts), np.array(labels), embeddings


def get_class_tier(f1_score_value):
    """Determine tier based on F1 score."""
    for tier, (min_val, max_val) in TIER_THRESHOLDS.items():
        if min_val <= f1_score_value < max_val:
            return tier
    return "HIGH"


def generate_synthetic_samples(texts, labels, embeddings, class_name,
                               baseline_f1_per_class, client, embed_model):
    """Generate synthetic samples for a class using optimal config."""

    # Get class indices
    class_mask = labels == class_name
    class_texts = texts[class_mask]
    class_embeddings = embeddings[class_mask]
    n_class = len(class_texts)

    # Calculate budget based on tier
    class_f1 = baseline_f1_per_class.get(class_name, 0.2)
    tier = get_class_tier(class_f1)
    tier_weight = OPTIMAL_CONFIG["TIER_WEIGHTS"][tier]

    # Base budget from OPTIMAL_CONFIG
    base_n_synth = int(n_class * OPTIMAL_CONFIG["BUDGET"])
    weighted_n_synth = int(base_n_synth * tier_weight)

    if weighted_n_synth == 0:
        return [], []

    # Cluster the class
    k_max = min(OPTIMAL_CONFIG["K_MAX"], n_class // 3, 24)
    k_max = max(1, k_max)

    if k_max > 1 and n_class >= k_max * 2:
        kmeans = KMeans(n_clusters=k_max, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)
    else:
        cluster_labels = np.zeros(n_class, dtype=int)

    # Generate from each cluster
    synthetic_texts = []
    synthetic_embeddings = []

    n_clusters = len(set(cluster_labels))
    samples_per_cluster = max(1, weighted_n_synth // n_clusters)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = class_texts[cluster_mask]
        cluster_embeddings = class_embeddings[cluster_mask]

        if len(cluster_texts) < 2:
            continue

        # Select K neighbors for context (use optimal K=200 or available)
        k_neighbors = min(OPTIMAL_CONFIG["K_NEIGHBORS"], len(cluster_texts))

        # Get medoid as anchor
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        medoid_idx = np.argmin(distances)

        # Select diverse examples for prompt
        example_indices = np.random.choice(
            len(cluster_texts),
            size=min(k_neighbors, len(cluster_texts)),
            replace=False
        )
        examples = [cluster_texts[i] for i in example_indices[:10]]  # Limit prompt size

        # Build prompt
        examples_text = "\n---\n".join(examples[:5])
        prompt = f"""Generate {samples_per_cluster} new unique social media posts written by someone with {class_name} personality type.

Examples of {class_name} writing style:
{examples_text}

Generate {samples_per_cluster} NEW posts (different from examples) that capture this personality's communication style.
Format: One post per line, no numbering."""

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=OPTIMAL_CONFIG["TEMPERATURE"],
                max_tokens=min(150 * samples_per_cluster, 4000),
            )

            generated = response.choices[0].message.content.strip().split('\n')
            generated = [g.strip() for g in generated if len(g.strip()) > 20]

            if generated:
                # Embed generated texts
                new_embeddings = embed_model.encode(generated)
                synthetic_texts.extend(generated[:samples_per_cluster])
                synthetic_embeddings.extend(new_embeddings[:samples_per_cluster])

        except Exception as e:
            print(f"    Error generating for {class_name}: {e}")
            continue

    return synthetic_texts, synthetic_embeddings


def run_fold(X, y, embeddings, train_idx, test_idx, client, embed_model, fold_num=1):
    """Run a single fold with optimal configuration."""
    import sys

    print(f"      [Fold {fold_num}] Starting...", flush=True)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]

    # Train baseline classifier
    print(f"      [Fold {fold_num}] Training baseline...", flush=True)
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_baseline.fit(emb_train, y_train)
    y_pred_baseline = clf_baseline.predict(emb_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
    print(f"      [Fold {fold_num}] Baseline F1: {baseline_f1:.4f}", flush=True)

    # Get per-class F1 for tier assignment
    classes = np.unique(y)
    baseline_f1_per_class = {}
    for cls in classes:
        cls_mask = y_test == cls
        if cls_mask.sum() > 0:
            cls_f1 = f1_score(y_test == cls, y_pred_baseline == cls)
            baseline_f1_per_class[cls] = cls_f1

    # Generate synthetic data
    print(f"      [Fold {fold_num}] Generating synthetic for {len(classes)} classes...", flush=True)
    all_synth_texts = []
    all_synth_labels = []
    all_synth_embeddings = []

    for cls_idx, class_name in enumerate(classes):
        if (cls_idx + 1) % 4 == 0:
            print(f"        Class {cls_idx + 1}/{len(classes)}...", flush=True)

        synth_texts, synth_embs = generate_synthetic_samples(
            X_train, y_train, emb_train, class_name,
            baseline_f1_per_class, client, embed_model
        )

        if synth_texts:
            all_synth_texts.extend(synth_texts)
            all_synth_labels.extend([class_name] * len(synth_texts))
            all_synth_embeddings.extend(synth_embs)

    n_synthetic = len(all_synth_texts)

    if n_synthetic == 0:
        return baseline_f1, baseline_f1, 0, 0

    # Create augmented training set with tier-weighted samples
    synth_emb_array = np.array(all_synth_embeddings)
    synth_labels_array = np.array(all_synth_labels)

    # Calculate sample weights based on tier
    sample_weights_train = np.ones(len(y_train))
    sample_weights_synth = np.array([
        OPTIMAL_CONFIG["TIER_WEIGHTS"][get_class_tier(baseline_f1_per_class.get(lbl, 0.2))]
        for lbl in synth_labels_array
    ])

    # Combine datasets
    X_aug = np.vstack([emb_train, synth_emb_array])
    y_aug = np.concatenate([y_train, synth_labels_array])
    weights_aug = np.concatenate([sample_weights_train, sample_weights_synth])

    # Train augmented classifier
    clf_augmented = LogisticRegression(max_iter=1000, random_state=42)
    clf_augmented.fit(X_aug, y_aug, sample_weight=weights_aug)
    y_pred_aug = clf_augmented.predict(emb_test)
    augmented_f1 = f1_score(y_test, y_pred_aug, average='macro')

    delta = augmented_f1 - baseline_f1

    return baseline_f1, augmented_f1, delta, n_synthetic


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("=" * 60)
    print("  OPTIMAL COMBINED CONFIGURATION EXPERIMENT")
    print("=" * 60)
    print()
    print("Configuration:")
    for key, value in OPTIMAL_CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Initialize
    client = OpenAI()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load data
    texts, labels, embeddings = load_data()

    # Setup K-Fold CV
    cv = RepeatedStratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        n_repeats=KFOLD_CONFIG["n_repeats"],
        random_state=KFOLD_CONFIG["random_state"]
    )

    n_folds = KFOLD_CONFIG["n_splits"] * KFOLD_CONFIG["n_repeats"]
    print(f"\nRunning {KFOLD_CONFIG['n_splits']}-fold × {KFOLD_CONFIG['n_repeats']} repeats = {n_folds} folds")
    print("-" * 60)

    # Run folds
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(texts, labels)):
        baseline_f1, aug_f1, delta, n_synth = run_fold(
            texts, labels, embeddings, train_idx, test_idx, client, embed_model,
            fold_num=fold_idx + 1
        )

        results.append({
            "fold": fold_idx + 1,
            "baseline_f1": baseline_f1,
            "augmented_f1": aug_f1,
            "delta": delta,
            "n_synthetic": n_synth,
        })

        print(f"    Fold {fold_idx + 1}/{n_folds}: baseline={baseline_f1:.4f}, "
              f"aug={aug_f1:.4f}, delta={delta:+.4f}, synth={n_synth}")

    # Aggregate results
    baseline_scores = [r["baseline_f1"] for r in results]
    aug_scores = [r["augmented_f1"] for r in results]
    deltas = [r["delta"] for r in results]
    n_synths = [r["n_synthetic"] for r in results]

    mean_baseline = np.mean(baseline_scores)
    mean_aug = np.mean(aug_scores)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    mean_synth = np.mean(n_synths)

    # Statistical test
    t_stat, p_value = ttest_rel(aug_scores, baseline_scores)

    # 95% CI
    se = std_delta / np.sqrt(len(deltas))
    ci_low = mean_delta - 1.96 * se
    ci_high = mean_delta + 1.96 * se

    print()
    print("=" * 60)
    print("  RESULTS: OPTIMAL COMBINED CONFIGURATION")
    print("=" * 60)
    print(f"  Folds: {n_folds}")
    print(f"  Baseline:  {mean_baseline:.4f} +/- {np.std(baseline_scores):.4f}")
    print(f"  Augmented: {mean_aug:.4f} +/- {np.std(aug_scores):.4f}")
    print(f"  Delta:     {mean_delta:+.4f} ({mean_delta*100:+.2f}%)")
    print(f"  95% CI:    [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  p-value:   {p_value:.6f} {'*' if p_value < 0.05 else ''}")
    print(f"  Win rate:  {sum(1 for d in deltas if d > 0) / len(deltas) * 100:.1f}%")
    print(f"  Avg Synth: {mean_synth:.0f}")
    print("=" * 60)

    # Save results
    output = {
        "config": OPTIMAL_CONFIG,
        "n_folds": n_folds,
        "baseline_mean": mean_baseline,
        "baseline_std": np.std(baseline_scores),
        "augmented_mean": mean_aug,
        "augmented_std": np.std(aug_scores),
        "delta_mean": mean_delta,
        "delta_pct": mean_delta * 100,
        "delta_std": std_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": sum(1 for d in deltas if d > 0) / len(deltas),
        "avg_synthetic": mean_synth,
        "fold_results": results,
    }

    # Save to file
    results_dir = RESULTS_DIR / "optimal_combined"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"optimal_combined_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return output


if __name__ == "__main__":
    main()
