#!/usr/bin/env python3
"""
Modern Baselines Experiment — Extended Method Comparison

Adds 3 modern augmentation baselines to the thesis comparison:
  1. Embedding Mixup (MixText-inspired interpolation in embedding space)
  2. T5 Paraphrase (local T5 model for paraphrasing)
  3. Contextual BERT Augmentation (BERT MLM word substitution)

Configuration: 10 methods × 3 classifiers × 7 datasets × 3 shots × 3 seeds = 1,890 experiments

Reuses infrastructure from exp_thesis_final.py:
  - Same datasets, same evaluation, same statistical analysis
  - Same LLM cache for binary_filter/soft_weighted
  - Same SMOTE/EDA/BT implementations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import random
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
from dataclasses import dataclass, asdict
from scipy import stats

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade
from core.modern_augmenters import (
    EmbeddingMixupAugmenter,
    T5ParaphraseAugmenter,
    ContextualBERTAugmenter,
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "modern_baselines"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
BT_CACHE_DIR = PROJECT_ROOT / "cache" / "backtranslation"
T5_CACHE_DIR = PROJECT_ROOT / "cache" / "t5_paraphrase"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
T5_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
SEEDS = [42, 123, 456]

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

AUGMENTATION_METHODS = [
    "no_augmentation", "smote", "random_oversample", "eda",
    "back_translation", "binary_filter", "soft_weighted",
    "embedding_mixup", "t5_paraphrase", "contextual_bert",
]


# ============================================================================
# EDA AUGMENTER (from exp_thesis_final.py)
# ============================================================================

class EDAugmenter:
    def __init__(self, alpha_rs=0.1, alpha_ri=0.1, p_rd=0.1, random_state=42):
        self.alpha_rs = alpha_rs
        self.alpha_ri = alpha_ri
        self.p_rd = p_rd
        self.rng = random.Random(random_state)

    def _random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) < 2:
                break
            i, j = self.rng.sample(range(len(new_words)), 2)
            new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words

    def _random_deletion(self, words, p):
        if len(words) <= 1:
            return words
        result = [w for w in words if self.rng.random() > p]
        return result or [self.rng.choice(words)]

    def _random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            if not new_words:
                break
            word = self.rng.choice(new_words)
            pos = self.rng.randint(0, len(new_words))
            new_words.insert(pos, word)
        return new_words

    def augment(self, text, n_aug=4):
        words = text.split()
        if len(words) < 3:
            return [text] * n_aug
        augmented = []
        for _ in range(n_aug):
            w = words.copy()
            n_swap = max(1, int(self.alpha_rs * len(w)))
            w = self._random_swap(w, n_swap)
            w = self._random_deletion(w, self.p_rd)
            n_ins = max(1, int(self.alpha_ri * len(words)))
            w = self._random_insertion(w, n_ins)
            augmented.append(" ".join(w))
        return augmented

    def generate_for_class(self, texts, n_generate, embed_model):
        augmented_texts = []
        n_per = max(1, n_generate // len(texts) + 1)
        for text in texts:
            augmented_texts.extend(self.augment(text, n_aug=n_per))
        self.rng.shuffle(augmented_texts)
        augmented_texts = augmented_texts[:n_generate]
        embeddings = embed_model.encode(augmented_texts, show_progress_bar=False)
        return embeddings, augmented_texts


# ============================================================================
# BACK-TRANSLATION AUGMENTER (from exp_thesis_final.py)
# ============================================================================

class BackTranslationAugmenter:
    def __init__(self, provider, cache_dir):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _translate_roundtrip(self, text):
        import time
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / f"bt_{text_hash}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f).get("backtranslated", text)
        for attempt in range(5):
            try:
                msg1 = [{"role": "user", "content": f"Translate the following text to Spanish. Output ONLY the translation, nothing else.\n\n{text}"}]
                spanish, _ = self.provider.generate(msg1, temperature=0.3, max_tokens=500)
                msg2 = [{"role": "user", "content": f"Translate the following Spanish text to English. Output ONLY the translation, nothing else.\n\n{spanish.strip()}"}]
                english, _ = self.provider.generate(msg2, temperature=0.3, max_tokens=500)
                result = english.strip()
                with open(cache_path, "w") as f:
                    json.dump({"original": text, "spanish": spanish.strip(),
                               "backtranslated": result, "timestamp": datetime.now().isoformat()}, f, indent=2)
                return result
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait = 2 ** attempt * 5
                    print(f"        BT rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"        BT error: {e}")
                return text
        return text

    def generate_for_class(self, texts, n_generate, embed_model):
        bt_texts = []
        source = texts * (n_generate // len(texts) + 1)
        for text in source[:n_generate]:
            bt_texts.append(self._translate_roundtrip(text))
        embeddings = embed_model.encode(bt_texts, show_progress_bar=False)
        return embeddings, bt_texts


# ============================================================================
# SHARED INFRASTRUCTURE
# ============================================================================

def load_dataset(dataset_name):
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_cache_key(dataset, class_name, n_shot, n_generate):
    return hashlib.md5(f"{dataset}_{class_name}_{n_shot}_{n_generate}".encode()).hexdigest()[:16]


def create_prompt(class_name, examples, n_generate, n_shot):
    selected = examples[:n_shot]
    examples_text = "\n\n".join([f"Example {i+1}: {ex[:500]}" for i, ex in enumerate(selected)])
    return f"""You are an expert at generating realistic text examples for classification.

Class: {class_name}

Here are {len(selected)} real examples from this class:
{examples_text}

Generate {n_generate} NEW examples that belong to the "{class_name}" class.
Each example should be similar in style, length, and content to the examples above.
Generate one example per line, without numbering:"""


def generate_llm_samples_cached(provider, dataset, class_name, class_texts,
                                 n_generate, n_shot, model):
    cache_key = get_cache_key(dataset, class_name, n_shot, n_generate)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("texts"):
            embeddings = model.encode(cached["texts"], show_progress_bar=False)
            return embeddings, cached["texts"]
    prompt = create_prompt(class_name, class_texts, n_generate, n_shot)
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
            json.dump({"dataset": dataset, "class_name": class_name,
                        "n_shot": n_shot, "n_generate": n_generate,
                        "texts": generated, "timestamp": datetime.now().isoformat()}, f, indent=2)
        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated
    except Exception as e:
        print(f"        LLM error: {e}")
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


def generate_random_oversample(real_embeddings, n_generate, seed=42, noise_level=0.01):
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(real_embeddings), n_generate, replace=True)
    samples = real_embeddings[indices].copy()
    if noise_level > 0:
        samples += rng.normal(0, noise_level, samples.shape)
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples /= (norms + 1e-8)
    return samples


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


def get_dataset_base(dataset_name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if dataset_name.startswith(base + "_"):
            return base
    return dataset_name


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class BaselineResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    classifier: str
    augmentation_method: str
    seed: int
    f1_macro: float
    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# AUGMENTATION PRE-COMPUTATION
# ============================================================================

def precompute_augmentations(
    dataset_name, train_texts, train_labels, train_embeddings,
    n_shot, model, provider, bt_augmenter, t5_augmenter, bert_augmenter, seed,
):
    # Update BERT augmenter seed without reloading the model
    if bert_augmenter is not None:
        bert_augmenter.rng = np.random.RandomState(seed)
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    dim = train_embeddings.shape[1]

    # --- SMOTE ---
    print("    SMOTE...")
    smote_embs, smote_labels = [], []
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
        if len(s) > 0:
            smote_embs.append(s)
            smote_labels.extend([cls] * len(s))
    smote_data = {
        "embeddings": np.vstack(smote_embs) if smote_embs else np.zeros((0, dim)),
        "labels": smote_labels,
    }

    # --- Random Oversample ---
    print("    Random oversample...")
    rand_embs, rand_labels = [], []
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        r = generate_random_oversample(cls_emb, N_SYNTHETIC_PER_CLASS, seed=seed)
        rand_embs.append(r)
        rand_labels.extend([cls] * len(r))
    random_data = {
        "embeddings": np.vstack(rand_embs),
        "labels": rand_labels,
    }

    # --- EDA ---
    print("    EDA...")
    eda = EDAugmenter(random_state=seed)
    eda_embs, eda_labels = [], []
    for cls in unique_classes:
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        e, _ = eda.generate_for_class(cls_texts, N_SYNTHETIC_PER_CLASS, model)
        if len(e) > 0:
            eda_embs.append(e)
            eda_labels.extend([cls] * len(e))
    eda_data = {
        "embeddings": np.vstack(eda_embs) if eda_embs else np.zeros((0, dim)),
        "labels": eda_labels,
    }

    # --- Back-Translation (cached, seed-independent) ---
    bt_data = {"embeddings": None, "labels": []}

    # --- LLM + cascade scores ---
    print("    LLM generation + cascade scoring...")
    cascade = FilterCascade(**FILTER_CONFIG)
    llm_data = {}
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
        gen_emb, gen_texts = generate_llm_samples_cached(
            provider, dataset_name, cls, cls_texts, n_gen, n_shot, model
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

    # --- NEW: Embedding Mixup (seed-dependent) ---
    print("    Embedding mixup...")
    mixup = EmbeddingMixupAugmenter(random_state=seed)
    mixup_embs, mixup_labels = [], []
    for cls in unique_classes:
        cls_emb = train_embeddings[labels_arr == cls]
        synth = mixup.generate_for_class(cls_emb, N_SYNTHETIC_PER_CLASS)
        if len(synth) > 0:
            mixup_embs.append(synth)
            mixup_labels.extend([cls] * len(synth))
    mixup_data = {
        "embeddings": np.vstack(mixup_embs) if mixup_embs else np.zeros((0, dim)),
        "labels": mixup_labels,
    }

    # --- NEW: T5 Paraphrase (cached, seed-independent) ---
    print("    T5 paraphrase...")
    t5_embs, t5_labels = [], []
    for cls in unique_classes:
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        e, _ = t5_augmenter.generate_for_class(cls_texts, N_SYNTHETIC_PER_CLASS, model)
        if len(e) > 0:
            t5_embs.append(e)
            t5_labels.extend([cls] * len(e))
    t5_data = {
        "embeddings": np.vstack(t5_embs) if t5_embs else np.zeros((0, dim)),
        "labels": t5_labels,
    }

    # --- NEW: Contextual BERT (seed-dependent via random masking) ---
    # bert_augmenter is passed in to avoid reloading model each seed
    print("    BERT contextual augmentation...")
    bert_embs, bert_labels = [], []
    for cls in unique_classes:
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        e, _ = bert_augmenter.generate_for_class(cls_texts, N_SYNTHETIC_PER_CLASS, model)
        if len(e) > 0:
            bert_embs.append(e)
            bert_labels.extend([cls] * len(e))
    bert_data = {
        "embeddings": np.vstack(bert_embs) if bert_embs else np.zeros((0, dim)),
        "labels": bert_labels,
    }

    print("    Augmentation precompute complete.")
    return {
        "smote": smote_data,
        "random_oversample": random_data,
        "eda": eda_data,
        "back_translation": bt_data,
        "llm": llm_data,
        "embedding_mixup": mixup_data,
        "t5_paraphrase": t5_data,
        "contextual_bert": bert_data,
    }


def precompute_back_translation(train_texts, train_labels, model, bt_augmenter):
    unique_classes = sorted(set(train_labels))
    bt_embs, bt_labels = [], []
    for cls in unique_classes:
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        e, _ = bt_augmenter.generate_for_class(cls_texts, N_SYNTHETIC_PER_CLASS, model)
        if len(e) > 0:
            bt_embs.append(e)
            bt_labels.extend([cls] * len(e))
    return {
        "embeddings": np.vstack(bt_embs) if bt_embs else np.zeros((0, 768)),
        "labels": bt_labels,
    }


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, aug_method, seed,
    train_embeddings, train_labels, test_embeddings, test_labels,
    aug_data,
):
    unique_classes = sorted(set(train_labels))

    if aug_method == "no_augmentation":
        clf = create_classifier(classifier_name, seed)
        clf.fit(train_embeddings, train_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return BaselineResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0, weight_std=0, timestamp=datetime.now().isoformat(),
        )

    # Uniform-weight methods (SMOTE, random, EDA, BT, mixup, T5, BERT)
    if aug_method in ("smote", "random_oversample", "eda", "back_translation",
                      "embedding_mixup", "t5_paraphrase", "contextual_bert"):
        method_data = aug_data[aug_method]
        if len(method_data["labels"]) == 0:
            return run_single_config(
                dataset_name, dataset_base, n_classes, n_shot,
                classifier_name, "no_augmentation", seed,
                train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
            )
        aug_emb = np.vstack([train_embeddings, method_data["embeddings"]])
        aug_labels = list(train_labels) + method_data["labels"]
        clf = create_classifier(classifier_name, seed)
        clf.fit(aug_emb, aug_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return BaselineResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings),
            n_synthetic_samples=len(method_data["labels"]),
            weight_mean=1.0, weight_std=0.0, timestamp=datetime.now().isoformat(),
        )

    # binary_filter or soft_weighted
    llm_data = aug_data["llm"]
    syn_embs, syn_labels, syn_weights = [], [], []
    for cls in unique_classes:
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        syn_embs.append(ld["top_emb"])
        syn_labels.extend([cls] * len(ld["top_emb"]))
        if aug_method == "binary_filter":
            syn_weights.append(np.ones(len(ld["top_emb"])))
        else:
            syn_weights.append(ld["top_weights"])

    if not syn_embs:
        return run_single_config(
            dataset_name, dataset_base, n_classes, n_shot,
            classifier_name, "no_augmentation", seed,
            train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
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
    return BaselineResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, augmentation_method=aug_method,
        seed=seed, f1_macro=float(f1),
        n_real_samples=len(train_embeddings),
        n_synthetic_samples=len(synthetic_emb),
        weight_mean=float(synthetic_weights.mean()),
        weight_std=float(synthetic_weights.std()),
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_paired_statistics(method_f1s, baseline_f1s, n_comparisons=9):
    method_f1s = np.array(method_f1s)
    baseline_f1s = np.array(baseline_f1s)
    n = len(method_f1s)
    deltas = method_f1s - baseline_f1s
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1) if n > 1 else 0.0
    se = delta_std / np.sqrt(n) if n > 1 else 0.0

    if n > 1 and se > 0:
        ci_t = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
    else:
        ci_t = (delta_mean, delta_mean)

    rng = np.random.RandomState(42)
    boot = [np.mean(deltas[rng.choice(n, n, replace=True)]) for _ in range(1000)]
    ci_boot = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    if n > 1 and delta_std > 0:
        t_stat, p_val = stats.ttest_rel(method_f1s, baseline_f1s)
    else:
        t_stat, p_val = 0.0, 1.0

    cohen_d = delta_mean / (delta_std + 1e-10) if delta_std > 0 else 0.0
    bonf_p = min(p_val * n_comparisons, 1.0)

    return {
        "delta_mean_pp": float(delta_mean * 100),
        "ci_95_bootstrap": (float(ci_boot[0] * 100), float(ci_boot[1] * 100)),
        "p_value": float(p_val),
        "bonferroni_p": float(bonf_p),
        "cohen_d": float(cohen_d),
        "significant_bonferroni": bool(bonf_p < 0.05),
        "win_rate": float(np.mean(deltas > 0)),
    }


# ============================================================================
# REPORT
# ============================================================================

def generate_report(results):
    data = [asdict(r) for r in results]
    n_comparisons = len(AUGMENTATION_METHODS) - 1

    print("\n" + "=" * 90)
    print("MODERN BASELINES — COMPREHENSIVE STATISTICAL REPORT")
    print("=" * 90)

    print(f"\n{'Method':<20} {'Mean F1':>8} {'D SMOTE':>10} {'95% CI':>22} {'Bonf.p':>8} {'d':>6} {'Win%':>6}")
    print("-" * 85)

    for method in AUGMENTATION_METHODS:
        if method == "smote":
            print(f"{'smote':<20} {'--':>8} {'ref':>10} {'--':>22} {'--':>8} {'--':>6} {'--':>6}")
            continue

        method_means, smote_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            for clf in CLASSIFIER_NAMES:
                m_f1s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == method]
                s_f1s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["classifier"] == clf
                         and r["augmentation_method"] == "smote"]
                if m_f1s and s_f1s:
                    method_means.append(np.mean(m_f1s))
                    smote_means.append(np.mean(s_f1s))

        if not method_means:
            continue

        st = compute_paired_statistics(method_means, smote_means, n_comparisons)
        mean_f1 = np.mean(method_means)
        sig = "*" if st["significant_bonferroni"] else ""
        print(f"{method:<20} {mean_f1:>8.4f} {st['delta_mean_pp']:>+9.2f}p "
              f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
              f"{st['bonferroni_p']:>7.4f}{sig} "
              f"{st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%")

    summary = {"timestamp": datetime.now().isoformat(), "n_experiments": len(data),
               "n_seeds": len(SEEDS), "n_methods": len(AUGMENTATION_METHODS),
               "n_classifiers": len(CLASSIFIER_NAMES)}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# RESULTS I/O
# ============================================================================

def save_results(results, filename="partial_results.json"):
    output = {"timestamp": datetime.now().isoformat(), "n_experiments": len(results),
              "results": [asdict(r) if isinstance(r, BaselineResult) else r for r in results]}
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)


def load_existing_results():
    """Load previous partial results for resume support."""
    partial_path = RESULTS_DIR / "partial_results.json"
    if not partial_path.exists():
        return [], set()
    with open(partial_path) as f:
        data = json.load(f)
    existing = data.get("results", [])
    completed = set()
    for r in existing:
        key = (r["dataset"], r["classifier"], r["augmentation_method"], r["seed"])
        completed.add(key)
    return existing, completed


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("MODERN BASELINES EXPERIMENT")
    print("=" * 90)

    total = len(DATASETS) * len(CLASSIFIER_NAMES) * len(AUGMENTATION_METHODS) * len(SEEDS)
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"Classifiers: {CLASSIFIER_NAMES}")
    print(f"Methods: {AUGMENTATION_METHODS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {total}")

    # Resume support: load existing results and skip completed configs
    existing_results, completed_keys = load_existing_results()
    if existing_results:
        print(f"\nRESUME MODE: Found {len(existing_results)} existing results, skipping those.")
        remaining = total - len(completed_keys)
        print(f"Remaining experiments: {remaining}")
    else:
        remaining = total

    # Check which (dataset, seed) combos are fully done (all methods × classifiers)
    full_combo_size = len(CLASSIFIER_NAMES) * len(AUGMENTATION_METHODS)
    completed_ds_seeds = set()
    combo_counts = Counter()
    for key in completed_keys:
        ds, clf, method, seed = key
        combo_counts[(ds, seed)] += 1
    for combo, count in combo_counts.items():
        if count >= full_combo_size:
            completed_ds_seeds.add(combo)

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Initializing providers and augmenters...")
    provider = create_provider("google", "gemini-2.0-flash")
    bt_augmenter = BackTranslationAugmenter(provider, BT_CACHE_DIR)
    t5_augmenter = T5ParaphraseAugmenter(cache_dir=T5_CACHE_DIR)
    bert_augmenter = ContextualBERTAugmenter()  # loaded once, seed updated per call

    results: List = list(existing_results)  # start with previous results
    new_count = 0
    skipped_count = 0
    experiment_count = len(existing_results)

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        # Skip entire dataset if all (dataset, seed) combos are done
        all_seeds_done = all((dataset_name, s) in completed_ds_seeds for s in SEEDS)
        if all_seeds_done:
            print(f"\n  Skipping {dataset_name} (all seeds complete)")
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
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        print("  Pre-computing back-translation...")
        bt_data = precompute_back_translation(train_texts, train_labels, model, bt_augmenter)

        for seed in SEEDS:
            # Skip entire seed if this (dataset, seed) combo is done
            if (dataset_name, seed) in completed_ds_seeds:
                print(f"\n  --- Seed {seed} --- (skipped, already complete)")
                continue

            print(f"\n  --- Seed {seed} ---")
            aug_data = precompute_augmentations(
                dataset_name, train_texts, train_labels, train_embeddings,
                n_shot, model, provider, bt_augmenter, t5_augmenter, bert_augmenter, seed,
            )
            aug_data["back_translation"] = bt_data

            print(f"    Running {len(CLASSIFIER_NAMES)}×{len(AUGMENTATION_METHODS)} classifier configs...")
            import time as _time
            _seed_start = _time.time()
            for clf_name in CLASSIFIER_NAMES:
                for aug_method in AUGMENTATION_METHODS:
                    config_key = (dataset_name, clf_name, aug_method, seed)
                    if config_key in completed_keys:
                        skipped_count += 1
                        continue

                    experiment_count += 1
                    new_count += 1
                    try:
                        _t0 = _time.time()
                        result = run_single_config(
                            dataset_name, dataset_base, n_classes, n_shot,
                            clf_name, aug_method, seed,
                            train_embeddings, train_labels, test_embeddings, test_labels,
                            aug_data,
                        )
                        results.append(asdict(result))
                        _elapsed = _time.time() - _t0
                        print(f"      {clf_name}/{aug_method}: F1={result.f1_macro:.4f} ({_elapsed:.1f}s)")

                    except Exception as e:
                        print(f"    ERROR: {dataset_name}/{clf_name}/{aug_method}/seed={seed}: {e}")

            _seed_elapsed = _time.time() - _seed_start
            print(f"    Seed {seed} classifiers done in {_seed_elapsed:.1f}s")
            # Save after each seed
            save_results(results)

    # Final save and report
    save_results(results, "final_results.json")

    # Convert dicts back to BaselineResult for report generation
    report_results = []
    for r in results:
        if isinstance(r, dict):
            report_results.append(BaselineResult(**{k: r[k] for k in BaselineResult.__dataclass_fields__}))
        else:
            report_results.append(r)
    generate_report(report_results)

    print(f"\n{'=' * 90}")
    print(f"COMPLETE: {len(results)} total experiments ({new_count} new, {skipped_count} skipped)")
    print(f"Saved to {RESULTS_DIR}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
