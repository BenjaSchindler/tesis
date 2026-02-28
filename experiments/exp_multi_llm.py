#!/usr/bin/env python3
"""
Multi-LLM Robustness Experiment

Tests geometric filtering across 5 different LLMs to demonstrate LLM-agnosticism:
  1. Gemini 3 Flash (Google) — original model
  2. GPT-5-mini (OpenAI) — closed
  3. Claude 4.5 Haiku (Anthropic) — closed
  4. Kimi K2.5 (Moonshot AI) — open-weight
  5. GLM-5 (Zhipu AI) — open-weight

Design: 5 LLMs × 21 datasets × 3 classifiers × 3 methods × 1 seed = 945 configs
  (baselines SMOTE/no_aug are LLM-independent → computed once)

Reuses infrastructure from exp_modern_baselines.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from scipy import stats

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "multi_llm"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
SEED = 42

# Soft weighting config (best from exp_soft_weighting.py)
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

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

CLASSIFIER_NAMES = ["logistic_regression", "svc_linear", "ridge"]

METHODS = ["no_augmentation", "smote", "soft_weighted"]

# LLM configurations: (provider_name, model_name, display_name)
LLM_CONFIGS = [
    ("google",    "gemini-3-flash-preview", "Gemini 3 Flash"),
    ("gpt5",      "gpt-5-mini",             "GPT-5-mini"),
    ("anthropic", "claude-haiku-4-5",       "Claude 4.5 Haiku"),
    ("moonshot",  "kimi-k2.5",             "Kimi K2.5"),
    ("zhipu",     "glm-5",                 "GLM-5"),
]


# ============================================================================
# SHARED INFRASTRUCTURE
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


def create_prompt(class_name, examples, n_generate, n_shot):
    selected = examples[:n_shot]
    examples_text = "\n\n".join([f"Example {i+1}: {ex[:500]}" for i, ex in enumerate(selected)])
    return f"""You are an academic researcher generating synthetic training data for a text classification study.
This is for a peer-reviewed thesis on data augmentation methods in NLP.
The generated examples will be used solely for training machine learning classifiers
in a controlled research setting. All content is for academic evaluation purposes only.

Class: {class_name}

Here are {len(selected)} real examples from this class:
{examples_text}

Generate {n_generate} NEW examples that belong to the "{class_name}" class.
Each example should be similar in style, length, and content to the examples above.
Generate one example per line, without numbering:"""


def get_cache_key(model_name, dataset, class_name, n_shot, n_generate):
    """Cache key includes model name for multi-LLM isolation."""
    return hashlib.md5(
        f"{model_name}_{dataset}_{class_name}_{n_shot}_{n_generate}".encode()
    ).hexdigest()[:16]


def get_cache_dir(model_name):
    """Per-model cache subdirectory."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    cache_dir = CACHE_DIR / safe_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def generate_llm_samples_cached(provider, model_name, dataset, class_name,
                                 class_texts, n_generate, n_shot, embed_model):
    """Generate and cache LLM samples, isolated per model."""
    cache_dir = get_cache_dir(model_name)
    cache_key = get_cache_key(model_name, dataset, class_name, n_shot, n_generate)
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("texts"):
            embeddings = embed_model.encode(cached["texts"], show_progress_bar=False)
            return embeddings, cached["texts"]

    prompt = create_prompt(class_name, class_texts, n_generate, n_shot)

    for attempt in range(5):
        try:
            messages = [{"role": "user", "content": prompt}]
            response, _ = provider.generate(messages, temperature=0.8, max_tokens=4000)
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            generated = []
            for line in lines:
                clean = line.lstrip("0123456789.-):* ")
                if len(clean) > 10:
                    generated.append(clean)

            if not generated:
                return np.array([]).reshape(0, 768), []

            with open(cache_file, "w") as f:
                json.dump({
                    "model": model_name, "dataset": dataset,
                    "class_name": class_name, "n_shot": n_shot,
                    "n_generate": n_generate, "texts": generated,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            embeddings = embed_model.encode(generated, show_progress_bar=False)
            return embeddings, generated

        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** attempt * 5
                print(f"        Rate limit, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"        LLM error: {e}")
            return np.array([]).reshape(0, 768), []

    return np.array([]).reshape(0, 768), []


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


def create_classifier(name, seed):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "svc_linear":
        return SVC(kernel="linear", max_iter=5000, random_state=seed)
    elif name == "ridge":
        return RidgeClassifier(alpha=1.0)
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class MultiLLMResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    classifier: str
    augmentation_method: str
    llm_provider: str
    llm_model: str
    llm_display: str
    seed: int
    f1_macro: float
    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# PRECOMPUTE AUGMENTATION DATA
# ============================================================================

def precompute_baselines(dataset_name, train_embeddings, train_labels, seed):
    """Compute LLM-independent baselines (SMOTE)."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    dim = train_embeddings.shape[1]

    smote_embs, smote_labels = [], []
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
        if len(s) > 0:
            smote_embs.append(s)
            smote_labels.extend([cls] * len(s))

    return {
        "embeddings": np.vstack(smote_embs) if smote_embs else np.zeros((0, dim)),
        "labels": smote_labels,
    }


def precompute_llm_data(provider, model_name, dataset_name, train_texts,
                         train_labels, train_embeddings, n_shot, embed_model):
    """Generate and score LLM samples for a specific provider."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    cascade = FilterCascade(**FILTER_CONFIG)

    llm_data = {}
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        cls_texts = [train_texts[i] for i, l in enumerate(train_labels) if l == cls]
        n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)

        gen_emb, gen_texts = generate_llm_samples_cached(
            provider, model_name, dataset_name, cls, cls_texts, n_gen, n_shot, embed_model
        )

        if len(gen_emb) == 0:
            llm_data[cls] = None
            continue

        anchor = cls_emb.mean(axis=0) if len(cls_emb) > 0 else train_embeddings.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_embeddings, labels_arr, cls)
        weights = normalize_scores(scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(scores)[-target_n:]
        llm_data[cls] = {"top_emb": gen_emb[top_idx], "top_weights": weights[top_idx]}

    return llm_data


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, aug_method, seed,
    train_embeddings, train_labels, test_embeddings, test_labels,
    smote_data, llm_data,
    llm_provider, llm_model, llm_display,
):
    unique_classes = sorted(set(train_labels))

    if aug_method == "no_augmentation":
        clf = create_classifier(classifier_name, seed)
        clf.fit(train_embeddings, train_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return MultiLLMResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            llm_provider=llm_provider, llm_model=llm_model, llm_display=llm_display,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0, weight_std=0, timestamp=datetime.now().isoformat(),
        )

    if aug_method == "smote":
        if len(smote_data["labels"]) == 0:
            return run_single_config(
                dataset_name, dataset_base, n_classes, n_shot,
                classifier_name, "no_augmentation", seed,
                train_embeddings, train_labels, test_embeddings, test_labels,
                smote_data, llm_data, llm_provider, llm_model, llm_display,
            )
        aug_emb = np.vstack([train_embeddings, smote_data["embeddings"]])
        aug_labels = list(train_labels) + smote_data["labels"]
        clf = create_classifier(classifier_name, seed)
        clf.fit(aug_emb, aug_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return MultiLLMResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            llm_provider=llm_provider, llm_model=llm_model, llm_display=llm_display,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings),
            n_synthetic_samples=len(smote_data["labels"]),
            weight_mean=1.0, weight_std=0.0, timestamp=datetime.now().isoformat(),
        )

    # soft_weighted
    syn_embs, syn_labels, syn_weights = [], [], []
    for cls in unique_classes:
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        syn_embs.append(ld["top_emb"])
        syn_labels.extend([cls] * len(ld["top_emb"]))
        syn_weights.append(ld["top_weights"])

    if not syn_embs:
        return run_single_config(
            dataset_name, dataset_base, n_classes, n_shot,
            classifier_name, "no_augmentation", seed,
            train_embeddings, train_labels, test_embeddings, test_labels,
            smote_data, llm_data, llm_provider, llm_model, llm_display,
        )

    synthetic_emb = np.vstack(syn_embs)
    synthetic_weights = np.concatenate(syn_weights)
    aug_emb = np.vstack([train_embeddings, synthetic_emb])
    aug_labels = list(train_labels) + syn_labels
    sample_weights = np.concatenate([np.ones(len(train_embeddings)), synthetic_weights])

    clf = create_classifier(classifier_name, seed)
    clf.fit(aug_emb, aug_labels, sample_weight=sample_weights)
    pred = clf.predict(test_embeddings)
    f1 = f1_score(test_labels, pred, average="macro")
    return MultiLLMResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, augmentation_method=aug_method,
        llm_provider=llm_provider, llm_model=llm_model, llm_display=llm_display,
        seed=seed, f1_macro=float(f1),
        n_real_samples=len(train_embeddings),
        n_synthetic_samples=len(synthetic_emb),
        weight_mean=float(synthetic_weights.mean()),
        weight_std=float(synthetic_weights.std()),
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# RESULTS I/O
# ============================================================================

def save_results(results, filename="partial_results.json"):
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(results),
        "results": [asdict(r) if isinstance(r, MultiLLMResult) else r for r in results],
    }
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)


def load_existing_results():
    partial_path = RESULTS_DIR / "partial_results.json"
    if not partial_path.exists():
        return [], set()
    with open(partial_path) as f:
        data = json.load(f)
    existing = data.get("results", [])
    completed = set()
    for r in existing:
        key = (r["dataset"], r["classifier"], r["augmentation_method"],
               r["llm_model"], r["seed"])
        completed.add(key)
    return existing, completed


# ============================================================================
# MIGRATE EXISTING GEMINI CACHE
# ============================================================================

def migrate_gemini_cache():
    """Copy existing flat-directory Gemini cache to per-model subdirectory."""
    gemini_dir = get_cache_dir("gemini-3-flash-preview")
    flat_files = list(CACHE_DIR.glob("*.json"))

    if not flat_files:
        return

    migrated = 0
    for f in flat_files:
        dest = gemini_dir / f.name
        if not dest.exists():
            # Read and verify it's a Gemini generation (not metadata)
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if "texts" in data and "dataset" in data:
                    import shutil
                    shutil.copy2(f, dest)
                    migrated += 1
            except (json.JSONDecodeError, KeyError):
                continue

    if migrated > 0:
        print(f"  Migrated {migrated} existing Gemini cache files to {gemini_dir.name}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("MULTI-LLM ROBUSTNESS EXPERIMENT")
    print("=" * 90)

    # Check which LLMs are available
    available_llms = []
    for provider_name, model_name, display_name in LLM_CONFIGS:
        try:
            provider = create_provider(provider_name, model_name)
            available_llms.append((provider_name, model_name, display_name))
            print(f"  [OK] {display_name} ({model_name})")
        except ValueError as e:
            print(f"  [SKIP] {display_name}: {e}")

    if not available_llms:
        print("\nERROR: No LLM providers available. Check API keys in .env")
        return

    n_llm_methods = len(available_llms)  # soft_weighted per LLM
    n_baselines = 2  # no_aug + smote (LLM-independent)
    total = len(DATASETS) * len(CLASSIFIER_NAMES) * (n_baselines + n_llm_methods)
    print(f"\nAvailable LLMs: {n_llm_methods}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Classifiers: {CLASSIFIER_NAMES}")
    print(f"Methods per LLM: soft_weighted")
    print(f"Baselines: no_augmentation, smote")
    print(f"Total experiments: {total}")

    # Resume support
    existing_results, completed_keys = load_existing_results()
    if existing_results:
        print(f"\nRESUME MODE: Found {len(existing_results)} existing results")

    # Migrate Gemini cache
    migrate_gemini_cache()

    print("\nLoading embedding model...")
    import torch
    torch.backends.cuda.preferred_blas_library("cublaslt")
    embed_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    results = list(existing_results)
    new_count = 0

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        dataset_base = get_dataset_base(dataset_name)
        n_classes = DATASET_N_CLASSES.get(dataset_base, 0)
        n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

        print(f"\n{'#' * 90}")
        print(f"# {dataset_name} ({n_classes} classes, {n_shot}-shot)")
        print(f"{'#' * 90}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

        print("  Embedding...")
        train_embeddings = embed_model.encode(train_texts, show_progress_bar=False)
        test_embeddings = embed_model.encode(test_texts, show_progress_bar=False)

        # Compute baselines (LLM-independent)
        print("  Computing SMOTE baseline...")
        smote_data = precompute_baselines(dataset_name, train_embeddings, train_labels, SEED)

        # Run baselines first
        for clf_name in CLASSIFIER_NAMES:
            for method in ["no_augmentation", "smote"]:
                # Use first LLM as placeholder for baselines (they don't use LLM)
                prov_name, model_name, disp = available_llms[0]
                config_key = (dataset_name, clf_name, method, model_name, SEED)
                if config_key in completed_keys:
                    continue

                result = run_single_config(
                    dataset_name, dataset_base, n_classes, n_shot,
                    clf_name, method, SEED,
                    train_embeddings, train_labels, test_embeddings, test_labels,
                    smote_data, {},
                    prov_name, model_name, disp,
                )
                results.append(asdict(result))
                new_count += 1

        # Run soft_weighted for each LLM
        for prov_name, model_name, disp in available_llms:
            config_key = (dataset_name, CLASSIFIER_NAMES[0], "soft_weighted", model_name, SEED)
            if config_key in completed_keys:
                print(f"  [{disp}] Already complete, skipping")
                continue

            print(f"  [{disp}] Generating + scoring...")
            try:
                provider = create_provider(prov_name, model_name)
                llm_data = precompute_llm_data(
                    provider, model_name, dataset_name,
                    train_texts, train_labels, train_embeddings, n_shot, embed_model
                )
            except Exception as e:
                print(f"    ERROR generating with {disp}: {e}")
                continue

            for clf_name in CLASSIFIER_NAMES:
                config_key = (dataset_name, clf_name, "soft_weighted", model_name, SEED)
                if config_key in completed_keys:
                    continue

                try:
                    result = run_single_config(
                        dataset_name, dataset_base, n_classes, n_shot,
                        clf_name, "soft_weighted", SEED,
                        train_embeddings, train_labels, test_embeddings, test_labels,
                        smote_data, llm_data,
                        prov_name, model_name, disp,
                    )
                    results.append(asdict(result))
                    new_count += 1
                    print(f"    {clf_name}: F1={result.f1_macro:.4f}")
                except Exception as e:
                    print(f"    ERROR: {clf_name}/{disp}: {e}")

        # Save after each dataset
        save_results(results)

    # Final save
    save_results(results, "final_results.json")

    # Quick summary
    print(f"\n{'=' * 90}")
    print("QUICK SUMMARY")
    print(f"{'=' * 90}")

    data = [r for r in results if isinstance(r, dict)]

    for prov_name, model_name, disp in available_llms:
        sw_f1s = [r["f1_macro"] for r in data
                  if r["augmentation_method"] == "soft_weighted" and r["llm_model"] == model_name]
        sm_f1s = [r["f1_macro"] for r in data
                  if r["augmentation_method"] == "smote"]

        if sw_f1s and sm_f1s:
            # Match by dataset+classifier
            sw_by_key = {}
            for r in data:
                if r["augmentation_method"] == "soft_weighted" and r["llm_model"] == model_name:
                    key = (r["dataset"], r["classifier"])
                    sw_by_key[key] = r["f1_macro"]

            sm_by_key = {}
            for r in data:
                if r["augmentation_method"] == "smote":
                    key = (r["dataset"], r["classifier"])
                    sm_by_key[key] = r["f1_macro"]

            common_keys = set(sw_by_key.keys()) & set(sm_by_key.keys())
            if common_keys:
                deltas = [sw_by_key[k] - sm_by_key[k] for k in common_keys]
                mean_delta = np.mean(deltas) * 100
                win_rate = np.mean([d > 0 for d in deltas]) * 100
                print(f"  {disp:20s}: D={mean_delta:+.2f}pp, Win={win_rate:.1f}% (n={len(common_keys)})")

    print(f"\nTotal: {len(results)} results ({new_count} new)")
    print(f"Saved to {RESULTS_DIR}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
