#!/usr/bin/env python3
"""
Embedding Model Ablation Study

Tests the best augmentation config (cascade_l1/soft_weighted/minmax/temp=0.5)
across 4 different sentence embedding models to validate that geometric
filtering is embedding-agnostic.

Models tested:
  1. all-mpnet-base-v2 (768d) — current baseline
  2. BAAI/bge-large-en-v1.5 (1024d) — SOTA retrieval model
  3. intfloat/e5-large-v2 (1024d) — SOTA contrastive learning
  4. BAAI/bge-small-en-v1.5 (384d) — efficiency comparison

Configuration: 4 models × 7 datasets × 3 shots × 3 seeds × 2 methods = 504 experiments
Methods: smote (baseline per embedding) and soft_weighted (our method)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import hashlib
from datetime import datetime
from typing import List
from dataclasses import dataclass, asdict
from scipy import stats

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "embedding_ablation"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
SEEDS = [42, 123, 456]

# Best filter config
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

# Best classifier from thesis_final results
CLASSIFIER = "ridge"

EMBEDDING_MODELS = [
    {"name": "mpnet-base", "model_id": "sentence-transformers/all-mpnet-base-v2"},
    {"name": "bge-large", "model_id": "BAAI/bge-large-en-v1.5"},
    {"name": "e5-large", "model_id": "intfloat/e5-large-v2"},
    {"name": "bge-small", "model_id": "BAAI/bge-small-en-v1.5"},
]

DATASETS = [
    "sms_spam_10shot", "sms_spam_25shot", "sms_spam_50shot",
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    "20newsgroups_10shot", "20newsgroups_25shot", "20newsgroups_50shot",
    "ag_news_10shot", "ag_news_25shot", "ag_news_50shot",
    "emotion_10shot", "emotion_25shot", "emotion_50shot",
    "dbpedia14_10shot", "dbpedia14_25shot", "dbpedia14_50shot",
    "20newsgroups_20class_10shot", "20newsgroups_20class_25shot", "20newsgroups_20class_50shot",
]

DATASET_N_CLASSES = {
    "sms_spam": 2, "hate_speech_davidson": 3, "20newsgroups": 4,
    "ag_news": 4, "emotion": 6, "dbpedia14": 14, "20newsgroups_20class": 20,
}

METHODS = ["smote", "soft_weighted"]


# ============================================================================
# INFRASTRUCTURE
# ============================================================================

def load_dataset(dataset_name):
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_dataset_base(dataset_name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if dataset_name.startswith(base + "_"):
            return base
    return dataset_name


def generate_smote_samples(real_embeddings, n_generate, seed=42, k_neighbors=5):
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    rng = np.random.RandomState(seed)
    X = np.vstack([real_embeddings, rng.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)
    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res[np.where(y_res == 0)[0][n_base:]][:n_generate]
    except Exception:
        return np.array([]).reshape(0, real_embeddings.shape[1])


def normalize_scores(raw_scores, method="minmax", temperature=1.0, min_weight=0.0):
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)
    weight_range = 1.0 - min_weight
    if method == "minmax":
        s_min, s_max = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    else:
        normalized = raw_scores
    if temperature != 1.0:
        normalized = np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / temperature)
    return min_weight + weight_range * normalized


def load_llm_texts_from_cache(dataset_name, class_name, n_shot, n_generate):
    """Load previously generated LLM texts from cache (text only, no embeddings)."""
    cache_key = hashlib.md5(f"{dataset_name}_{class_name}_{n_shot}_{n_generate}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        return cached.get("texts", [])
    return []


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class EmbeddingAblationResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    embedding_model: str
    embedding_dim: int
    method: str
    seed: int
    f1_macro: float
    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("=" * 90)
    print("EMBEDDING MODEL ABLATION STUDY")
    print("=" * 90)

    total = len(EMBEDDING_MODELS) * len(DATASETS) * len(SEEDS) * len(METHODS)
    print(f"\nEmbedding models: {[m['name'] for m in EMBEDDING_MODELS]}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"Classifier: {CLASSIFIER}")
    print(f"Total experiments: {total}")

    results: List[EmbeddingAblationResult] = []
    experiment_count = 0

    for model_config in EMBEDDING_MODELS:
        model_name = model_config["name"]
        model_id = model_config["model_id"]

        print(f"\n{'=' * 90}")
        print(f"EMBEDDING MODEL: {model_name} ({model_id})")
        print(f"{'=' * 90}")

        embed_model = SentenceTransformer(model_id)

        for dataset_name in DATASETS:
            dataset_path = DATA_DIR / f"{dataset_name}.json"
            if not dataset_path.exists():
                print(f"  Skipping {dataset_name} (not found)")
                continue

            dataset_base = get_dataset_base(dataset_name)
            n_classes = DATASET_N_CLASSES.get(dataset_base, 0)
            n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

            print(f"\n  {dataset_name} ({n_classes} classes, {n_shot}-shot)")

            train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
            unique_classes = sorted(set(train_labels))
            labels_arr = np.array(train_labels)

            # Embed with current model
            # E5 models require prefix for best performance
            if "e5" in model_id.lower():
                train_input = [f"query: {t}" for t in train_texts]
                test_input = [f"query: {t}" for t in test_texts]
            else:
                train_input = train_texts
                test_input = test_texts

            train_emb = embed_model.encode(train_input, show_progress_bar=False)
            test_emb = embed_model.encode(test_input, show_progress_bar=False)
            dim = train_emb.shape[1]

            print(f"    Embedding dim: {dim}")

            for seed in SEEDS:
                for method in METHODS:
                    experiment_count += 1

                    try:
                        if method == "smote":
                            # SMOTE in this embedding space
                            smote_embs, smote_labels = [], []
                            for cls in unique_classes:
                                cls_emb = train_emb[labels_arr == cls]
                                s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
                                if len(s) > 0:
                                    smote_embs.append(s)
                                    smote_labels.extend([cls] * len(s))

                            if not smote_embs:
                                continue

                            aug_emb = np.vstack([train_emb, np.vstack(smote_embs)])
                            aug_labels = list(train_labels) + smote_labels
                            clf = RidgeClassifier(alpha=1.0)
                            clf.fit(aug_emb, aug_labels)
                            pred = clf.predict(test_emb)
                            f1 = f1_score(test_labels, pred, average="macro")

                            results.append(EmbeddingAblationResult(
                                dataset=dataset_name, dataset_base=dataset_base,
                                n_classes=n_classes, n_shot=n_shot,
                                embedding_model=model_name, embedding_dim=dim,
                                method="smote", seed=seed, f1_macro=float(f1),
                                n_real_samples=len(train_emb),
                                n_synthetic_samples=len(smote_labels),
                                weight_mean=1.0, weight_std=0.0,
                                timestamp=datetime.now().isoformat(),
                            ))

                        elif method == "soft_weighted":
                            # Load cached LLM texts and re-embed with current model
                            cascade = FilterCascade(**FILTER_CONFIG)
                            syn_embs, syn_labels, syn_weights = [], [], []

                            for cls in unique_classes:
                                cls_emb = train_emb[labels_arr == cls]
                                n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
                                cached_texts = load_llm_texts_from_cache(
                                    dataset_name, cls, n_shot, n_gen
                                )

                                if not cached_texts:
                                    continue

                                # Re-embed cached texts with current embedding model
                                if "e5" in model_id.lower():
                                    gen_input = [f"query: {t}" for t in cached_texts]
                                else:
                                    gen_input = cached_texts
                                gen_emb = embed_model.encode(gen_input, show_progress_bar=False)

                                # Score and select
                                anchor = cls_emb.mean(axis=0)
                                scores, _ = cascade.compute_quality_scores(
                                    gen_emb, anchor, train_emb, labels_arr, cls
                                )
                                weights = normalize_scores(scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)
                                target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
                                top_idx = np.argsort(scores)[-target_n:]

                                syn_embs.append(gen_emb[top_idx])
                                syn_labels.extend([cls] * len(gen_emb[top_idx]))
                                syn_weights.append(weights[top_idx])

                            if not syn_embs:
                                continue

                            synthetic_emb = np.vstack(syn_embs)
                            synthetic_weights = np.concatenate(syn_weights)
                            aug_emb = np.vstack([train_emb, synthetic_emb])
                            aug_labels = list(train_labels) + syn_labels
                            sample_weights = np.concatenate([
                                np.ones(len(train_emb)), synthetic_weights
                            ])

                            clf = RidgeClassifier(alpha=1.0)
                            clf.fit(aug_emb, aug_labels, sample_weight=sample_weights)
                            pred = clf.predict(test_emb)
                            f1 = f1_score(test_labels, pred, average="macro")

                            results.append(EmbeddingAblationResult(
                                dataset=dataset_name, dataset_base=dataset_base,
                                n_classes=n_classes, n_shot=n_shot,
                                embedding_model=model_name, embedding_dim=dim,
                                method="soft_weighted", seed=seed, f1_macro=float(f1),
                                n_real_samples=len(train_emb),
                                n_synthetic_samples=len(syn_labels),
                                weight_mean=float(synthetic_weights.mean()),
                                weight_std=float(synthetic_weights.std()),
                                timestamp=datetime.now().isoformat(),
                            ))

                        if experiment_count % 20 == 0:
                            print(f"    [{experiment_count}/{total}] {model_name}/{method}: F1={f1:.4f}")

                    except Exception as e:
                        print(f"    ERROR: {model_name}/{dataset_name}/{method}/seed={seed}: {e}")

            # Save checkpoint after each dataset
            save_results(results)

    # Final save and report
    save_results(results, "final_results.json")
    generate_report(results)
    print(f"\n{'=' * 90}")
    print(f"COMPLETE: {len(results)} experiments saved to {RESULTS_DIR}")
    print(f"{'=' * 90}")


# ============================================================================
# RESULTS I/O + REPORT
# ============================================================================

def save_results(results, filename="partial_results.json"):
    output = {"timestamp": datetime.now().isoformat(), "n_experiments": len(results),
              "results": [asdict(r) for r in results]}
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)


def generate_report(results):
    data = [asdict(r) for r in results]
    datasets = sorted(set(r["dataset"] for r in data))

    print("\n" + "=" * 90)
    print("EMBEDDING ABLATION — REPORT")
    print("=" * 90)

    print(f"\n{'Model':<15} {'Dim':>5} {'Mean F1':>9} {'D SMOTE':>10} {'Win%':>7} {'N':>5}")
    print("-" * 55)

    for model_config in EMBEDDING_MODELS:
        model_name = model_config["name"]

        sw_means, smote_means = [], []
        for ds in datasets:
            sw_f1s = [r["f1_macro"] for r in data
                      if r["dataset"] == ds and r["embedding_model"] == model_name
                      and r["method"] == "soft_weighted"]
            sm_f1s = [r["f1_macro"] for r in data
                      if r["dataset"] == ds and r["embedding_model"] == model_name
                      and r["method"] == "smote"]
            if sw_f1s and sm_f1s:
                sw_means.append(np.mean(sw_f1s))
                smote_means.append(np.mean(sm_f1s))

        if not sw_means:
            continue

        sw_arr = np.array(sw_means)
        sm_arr = np.array(smote_means)
        deltas = sw_arr - sm_arr
        delta_mean = np.mean(deltas) * 100
        win_rate = np.mean(deltas > 0) * 100
        mean_f1 = np.mean(sw_arr)

        dim = [r["embedding_dim"] for r in data if r["embedding_model"] == model_name][0]

        print(f"{model_name:<15} {dim:>5} {mean_f1:>9.4f} {delta_mean:>+9.2f}p {win_rate:>6.1f}% {len(sw_means):>5}")

    # By n-shot per model
    print(f"\n{'Model':<15} {'10-shot':>12} {'25-shot':>12} {'50-shot':>12}")
    print("-" * 55)

    for model_config in EMBEDDING_MODELS:
        model_name = model_config["name"]
        row = f"{model_name:<15}"
        for ns in [10, 25, 50]:
            sw_means, sm_means = [], []
            for ds in datasets:
                ds_nshot = int(ds.split("_")[-1].replace("shot", ""))
                if ds_nshot != ns:
                    continue
                sw_f1s = [r["f1_macro"] for r in data
                          if r["dataset"] == ds and r["embedding_model"] == model_name
                          and r["method"] == "soft_weighted"]
                sm_f1s = [r["f1_macro"] for r in data
                          if r["dataset"] == ds and r["embedding_model"] == model_name
                          and r["method"] == "smote"]
                if sw_f1s and sm_f1s:
                    sw_means.append(np.mean(sw_f1s))
                    sm_means.append(np.mean(sm_f1s))
            if sw_means:
                d = (np.mean(sw_means) - np.mean(sm_means)) * 100
                row += f" {d:>+9.2f}pp "
            else:
                row += f" {'--':>11} "
        print(row)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(data),
        "models": [m["name"] for m in EMBEDDING_MODELS],
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
