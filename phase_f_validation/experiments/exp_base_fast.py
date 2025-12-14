#!/usr/bin/env python3
"""
BASE CONFIG - FAST VERSION
===========================
Para comparar vs OPTIMAL.
Usa config original CMB3_skip.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_config import *

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# BASE CONFIG (original CMB3_skip)
BASE_CONFIG = {
    "K_MAX": 5,           # Original
    "K_NEIGHBORS": 8,     # Original
    "TEMPERATURE": 0.7,   # Original
    "BUDGET": 0.15,       # Original
    "WEIGHTS": "uniform", # Sin tier weights
}


def generate_all_synthetic(texts, labels, embeddings, client, embed_model):
    """Generate synthetic with BASE config."""
    print("\n" + "=" * 60, flush=True)
    print("  GENERATING WITH BASE CONFIG", flush=True)
    print("=" * 60, flush=True)

    classes = np.unique(labels)

    all_synth_texts = []
    all_synth_labels = []
    all_synth_embeddings = []

    for cls_idx, class_name in enumerate(classes):
        print(f"  [{cls_idx+1}/{len(classes)}] {class_name}...", flush=True)

        class_mask = labels == class_name
        class_texts = texts[class_mask]
        class_embeddings = embeddings[class_mask]
        n_class = len(class_texts)

        # BASE budget (15%, uniform - no tier weighting)
        n_synth = int(n_class * BASE_CONFIG["BUDGET"])
        if n_synth == 0:
            continue

        # BASE clustering (K_max=5)
        k_max = min(BASE_CONFIG["K_MAX"], n_class // 3, 24)
        k_max = max(1, k_max)

        if k_max > 1 and n_class >= k_max * 2:
            kmeans = KMeans(n_clusters=k_max, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_embeddings)
        else:
            cluster_labels = np.zeros(n_class, dtype=int)

        n_clusters = len(set(cluster_labels))
        samples_per_cluster = max(1, n_synth // n_clusters)

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = class_texts[cluster_mask]

            if len(cluster_texts) < 2:
                continue

            # BASE K_NEIGHBORS=8
            examples = list(cluster_texts[:min(BASE_CONFIG["K_NEIGHBORS"], len(cluster_texts))])
            examples_text = "\n---\n".join(examples[:5])

            prompt = f"""Generate {samples_per_cluster} new social media posts by someone with {class_name} personality.

Examples:
{examples_text}

Generate {samples_per_cluster} NEW posts. One per line, no numbers."""

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=BASE_CONFIG["TEMPERATURE"],
                    max_tokens=min(150 * samples_per_cluster, 4000),
                )

                generated = response.choices[0].message.content.strip().split('\n')
                generated = [g.strip() for g in generated if len(g.strip()) > 20]

                if generated:
                    new_embs = embed_model.encode(generated)
                    all_synth_texts.extend(generated[:samples_per_cluster])
                    all_synth_labels.extend([class_name] * len(generated[:samples_per_cluster]))
                    all_synth_embeddings.extend(new_embs[:samples_per_cluster])

            except Exception as e:
                print(f"    Error: {e}", flush=True)

    print(f"\n  TOTAL: {len(all_synth_texts)} samples", flush=True)
    return np.array(all_synth_texts), np.array(all_synth_labels), np.array(all_synth_embeddings)


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("\n" + "=" * 60, flush=True)
    print("  BASE CONFIG TEST (for comparison)", flush=True)
    print("=" * 60, flush=True)

    print("\nBASE Configuration:", flush=True)
    for k, v in BASE_CONFIG.items():
        print(f"  {k}: {v}", flush=True)

    client = OpenAI()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    texts = np.array(df['posts'].tolist())
    labels = np.array(df['type'].tolist())
    print(f"\nLoaded {len(texts)} samples", flush=True)

    embeddings = np.load(EMBEDDING_CACHE_PATH)

    # Generate
    synth_texts, synth_labels, synth_embeddings = \
        generate_all_synthetic(texts, labels, embeddings, client, embed_model)

    # CV
    print("\n" + "=" * 60, flush=True)
    print("  CROSS-VALIDATION", flush=True)
    print("=" * 60, flush=True)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(texts, labels)):
        emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Baseline
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(emb_train, y_train)
        baseline_f1 = f1_score(y_test, clf.predict(emb_test), average='macro')

        # Augmented (uniform weights - BASE config)
        X_aug = np.vstack([emb_train, synth_embeddings])
        y_aug = np.concatenate([y_train, synth_labels])

        clf_aug = LogisticRegression(max_iter=1000, random_state=42)
        clf_aug.fit(X_aug, y_aug)  # No sample weights (uniform)
        aug_f1 = f1_score(y_test, clf_aug.predict(emb_test), average='macro')

        delta = aug_f1 - baseline_f1
        results.append({"baseline": baseline_f1, "aug": aug_f1, "delta": delta})
        print(f"  Fold {fold_idx+1}/15: baseline={baseline_f1:.4f}, aug={aug_f1:.4f}, delta={delta:+.4f}", flush=True)

    # Results
    deltas = [r["delta"] for r in results]
    baseline_scores = [r["baseline"] for r in results]
    aug_scores = [r["aug"] for r in results]

    mean_delta = np.mean(deltas)
    t_stat, p_value = ttest_rel(aug_scores, baseline_scores)

    print("\n" + "=" * 60, flush=True)
    print("  RESULTS: BASE CONFIG", flush=True)
    print("=" * 60, flush=True)
    print(f"  Synthetic: {len(synth_texts)}", flush=True)
    print(f"  Baseline:  {np.mean(baseline_scores):.4f}", flush=True)
    print(f"  Augmented: {np.mean(aug_scores):.4f}", flush=True)
    print(f"  Delta:     {mean_delta:+.4f} ({mean_delta*100:+.2f}%)", flush=True)
    print(f"  p-value:   {p_value:.6f} {'*' if p_value < 0.05 else ''}", flush=True)
    print(f"  Win rate:  {sum(1 for d in deltas if d > 0)/len(deltas)*100:.1f}%", flush=True)
    print("=" * 60, flush=True)

    # Save
    output = {
        "config": BASE_CONFIG,
        "n_synthetic": len(synth_texts),
        "delta_pct": mean_delta * 100,
        "p_value": p_value,
        "results": results,
    }

    results_dir = RESULTS_DIR / "optimal_combined"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"base_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
