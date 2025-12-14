#!/usr/bin/env python3
"""
OPTIMAL COMBINED - FAST VERSION
================================
Genera sintéticos UNA VEZ, luego CV solo del clasificador.
Mucho más rápido (~15 min vs ~2.5 horas).

Configuración óptima:
- K_MAX = 18, K_NEIGHBORS = 200
- TIER_WEIGHTS = (2.0, 0.8, 0.3)
- TEMPERATURE = 0.9, BUDGET = 0.20
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

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
    "K_MAX": 18,
    "K_NEIGHBORS": 200,
    "TIER_WEIGHTS": {"LOW": 2.0, "MID": 0.8, "HIGH": 0.3},
    "TEMPERATURE": 0.9,
    "BUDGET": 0.20,
}

TIER_THRESHOLDS = {
    "LOW": (0.0, 0.20),
    "MID": (0.20, 0.45),
    "HIGH": (0.45, 1.0),
}


def get_class_tier(f1_score_value):
    for tier, (min_val, max_val) in TIER_THRESHOLDS.items():
        if min_val <= f1_score_value < max_val:
            return tier
    return "HIGH"


def generate_all_synthetic(texts, labels, embeddings, client, embed_model):
    """Generate synthetic data for ALL classes ONCE."""
    print("\n" + "=" * 60, flush=True)
    print("  PHASE 1: GENERATING SYNTHETIC DATA (ONE TIME)", flush=True)
    print("=" * 60, flush=True)

    classes = np.unique(labels)

    # First, get baseline F1 per class using a quick split
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings[idx_train], labels[idx_train])
    y_pred = clf.predict(embeddings[idx_test])

    baseline_f1_per_class = {}
    for cls in classes:
        cls_mask = labels[idx_test] == cls
        if cls_mask.sum() > 0:
            cls_f1 = f1_score(labels[idx_test] == cls, y_pred == cls)
            baseline_f1_per_class[cls] = cls_f1

    print(f"  Baseline per-class F1 computed", flush=True)

    # Generate for each class
    all_synth_texts = []
    all_synth_labels = []
    all_synth_embeddings = []

    for cls_idx, class_name in enumerate(classes):
        print(f"  [{cls_idx+1}/{len(classes)}] Generating for {class_name}...", flush=True)

        # Get class data
        class_mask = labels == class_name
        class_texts = texts[class_mask]
        class_embeddings = embeddings[class_mask]
        n_class = len(class_texts)

        # Calculate budget based on tier
        class_f1 = baseline_f1_per_class.get(class_name, 0.2)
        tier = get_class_tier(class_f1)
        tier_weight = OPTIMAL_CONFIG["TIER_WEIGHTS"][tier]

        base_n_synth = int(n_class * OPTIMAL_CONFIG["BUDGET"])
        weighted_n_synth = int(base_n_synth * tier_weight)

        if weighted_n_synth == 0:
            continue

        # Cluster
        k_max = min(OPTIMAL_CONFIG["K_MAX"], n_class // 3, 24)
        k_max = max(1, k_max)

        if k_max > 1 and n_class >= k_max * 2:
            kmeans = KMeans(n_clusters=k_max, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_embeddings)
        else:
            cluster_labels = np.zeros(n_class, dtype=int)

        n_clusters = len(set(cluster_labels))
        samples_per_cluster = max(1, weighted_n_synth // n_clusters)

        class_synth_texts = []
        class_synth_embs = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = class_texts[cluster_mask]

            if len(cluster_texts) < 2:
                continue

            # Select examples for prompt
            examples = list(cluster_texts[:min(10, len(cluster_texts))])
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
                    new_embeddings = embed_model.encode(generated)
                    class_synth_texts.extend(generated[:samples_per_cluster])
                    class_synth_embs.extend(new_embeddings[:samples_per_cluster])

            except Exception as e:
                print(f"    Error: {e}", flush=True)
                continue

        if class_synth_texts:
            all_synth_texts.extend(class_synth_texts)
            all_synth_labels.extend([class_name] * len(class_synth_texts))
            all_synth_embeddings.extend(class_synth_embs)
            print(f"    Generated {len(class_synth_texts)} samples (tier={tier}, weight={tier_weight})", flush=True)

    print(f"\n  TOTAL SYNTHETIC: {len(all_synth_texts)} samples", flush=True)

    return (np.array(all_synth_texts), np.array(all_synth_labels),
            np.array(all_synth_embeddings), baseline_f1_per_class)


def run_cv_with_synthetic(texts, labels, embeddings,
                          synth_texts, synth_labels, synth_embeddings,
                          baseline_f1_per_class):
    """Run CV using pre-generated synthetic data."""
    print("\n" + "=" * 60, flush=True)
    print("  PHASE 2: CROSS-VALIDATION WITH SYNTHETIC DATA", flush=True)
    print("=" * 60, flush=True)

    cv = RepeatedStratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        n_repeats=KFOLD_CONFIG["n_repeats"],
        random_state=KFOLD_CONFIG["random_state"]
    )

    n_folds = KFOLD_CONFIG["n_splits"] * KFOLD_CONFIG["n_repeats"]
    print(f"  Running {n_folds} folds...", flush=True)

    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(texts, labels)):
        # Get fold data
        emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Baseline
        clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
        clf_baseline.fit(emb_train, y_train)
        y_pred_baseline = clf_baseline.predict(emb_test)
        baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')

        # Augmented: combine train + synthetic with tier weights
        X_aug = np.vstack([emb_train, synth_embeddings])
        y_aug = np.concatenate([y_train, synth_labels])

        # Sample weights
        weights_train = np.ones(len(y_train))
        weights_synth = np.array([
            OPTIMAL_CONFIG["TIER_WEIGHTS"][get_class_tier(baseline_f1_per_class.get(lbl, 0.2))]
            for lbl in synth_labels
        ])
        weights_aug = np.concatenate([weights_train, weights_synth])

        clf_aug = LogisticRegression(max_iter=1000, random_state=42)
        clf_aug.fit(X_aug, y_aug, sample_weight=weights_aug)
        y_pred_aug = clf_aug.predict(emb_test)
        aug_f1 = f1_score(y_test, y_pred_aug, average='macro')

        delta = aug_f1 - baseline_f1

        results.append({
            "fold": fold_idx + 1,
            "baseline_f1": baseline_f1,
            "augmented_f1": aug_f1,
            "delta": delta,
        })

        print(f"    Fold {fold_idx+1}/{n_folds}: baseline={baseline_f1:.4f}, "
              f"aug={aug_f1:.4f}, delta={delta:+.4f}", flush=True)

    return results


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("\n" + "=" * 60, flush=True)
    print("  OPTIMAL COMBINED - FAST VERSION", flush=True)
    print("  (Generate once, CV after)", flush=True)
    print("=" * 60, flush=True)

    print("\nConfiguration:", flush=True)
    for key, value in OPTIMAL_CONFIG.items():
        print(f"  {key}: {value}", flush=True)

    # Initialize
    client = OpenAI()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Load data
    import pandas as pd
    print(f"\nLoading data from {DATA_PATH}...", flush=True)
    df = pd.read_csv(DATA_PATH)
    texts = np.array(df['posts'].tolist())
    labels = np.array(df['type'].tolist())
    print(f"  Loaded {len(texts)} samples, {len(set(labels))} classes", flush=True)

    # Load embeddings
    if EMBEDDING_CACHE_PATH.exists():
        embeddings = np.load(EMBEDDING_CACHE_PATH)
        print(f"  Loaded embeddings from cache", flush=True)
    else:
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        np.save(EMBEDDING_CACHE_PATH, embeddings)

    # PHASE 1: Generate synthetic (ONE TIME)
    synth_texts, synth_labels, synth_embeddings, baseline_f1_per_class = \
        generate_all_synthetic(texts, labels, embeddings, client, embed_model)

    # PHASE 2: CV with synthetic
    results = run_cv_with_synthetic(
        texts, labels, embeddings,
        synth_texts, synth_labels, synth_embeddings,
        baseline_f1_per_class
    )

    # Aggregate
    baseline_scores = [r["baseline_f1"] for r in results]
    aug_scores = [r["augmented_f1"] for r in results]
    deltas = [r["delta"] for r in results]

    mean_baseline = np.mean(baseline_scores)
    mean_aug = np.mean(aug_scores)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)

    t_stat, p_value = ttest_rel(aug_scores, baseline_scores)

    se = std_delta / np.sqrt(len(deltas))
    ci_low = mean_delta - 1.96 * se
    ci_high = mean_delta + 1.96 * se

    print("\n" + "=" * 60, flush=True)
    print("  RESULTS: OPTIMAL COMBINED (FAST)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Folds: {len(results)}", flush=True)
    print(f"  Synthetic samples: {len(synth_texts)}", flush=True)
    print(f"  Baseline:  {mean_baseline:.4f} +/- {np.std(baseline_scores):.4f}", flush=True)
    print(f"  Augmented: {mean_aug:.4f} +/- {np.std(aug_scores):.4f}", flush=True)
    print(f"  Delta:     {mean_delta:+.4f} ({mean_delta*100:+.2f}%)", flush=True)
    print(f"  95% CI:    [{ci_low:+.4f}, {ci_high:+.4f}]", flush=True)
    print(f"  p-value:   {p_value:.6f} {'*' if p_value < 0.05 else ''}", flush=True)
    print(f"  Win rate:  {sum(1 for d in deltas if d > 0) / len(deltas) * 100:.1f}%", flush=True)
    print("=" * 60, flush=True)

    # Save
    output = {
        "method": "fast_generate_once",
        "config": OPTIMAL_CONFIG,
        "n_folds": len(results),
        "n_synthetic": len(synth_texts),
        "baseline_mean": mean_baseline,
        "augmented_mean": mean_aug,
        "delta_mean": mean_delta,
        "delta_pct": mean_delta * 100,
        "delta_std": std_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": sum(1 for d in deltas if d > 0) / len(deltas),
        "fold_results": results,
    }

    results_dir = RESULTS_DIR / "optimal_combined"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"optimal_fast_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return output


if __name__ == "__main__":
    main()
