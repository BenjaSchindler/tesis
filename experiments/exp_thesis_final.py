#!/usr/bin/env python3
"""
Thesis Final Experiment — Statistical Validation + Extended Baselines + Ablation

Comprehensive experiment combining:
  1. Multi-seed statistical testing (5 seeds with CIs, p-values, Cohen's d)
  2. Extended baselines (EDA, back-translation, random oversampling)
  3. Clean ablation narrative (4 tables)

Configuration: 7 datasets × 3 shots × 5 classifiers × 7 methods × 5 seeds = 3,675 experiments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import random
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
from scipy import stats

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "thesis_final"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
BT_CACHE_DIR = PROJECT_ROOT / "cache" / "backtranslation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
SEEDS = [42, 123, 456, 789, 1011]

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

CLASSIFIER_NAMES = ["logistic_regression", "svc_linear", "random_forest", "ridge", "mlp"]

AUGMENTATION_METHODS = [
    "no_augmentation", "smote", "random_oversample", "eda",
    "back_translation", "binary_filter", "soft_weighted",
]


# ============================================================================
# MLP CLASSIFIER
# ============================================================================

class _MLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPClassifier:
    def __init__(self, hidden_dim=128, epochs=300, lr=1e-3, batch_size=16,
                 random_state=42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.label_map_ = None
        self.inv_label_map_ = None

    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        unique_labels = sorted(set(y))
        self.label_map_ = {l: i for i, l in enumerate(unique_labels)}
        self.inv_label_map_ = {i: l for l, i in self.label_map_.items()}
        n_classes = len(unique_labels)
        y_enc = np.array([self.label_map_[l] for l in y])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)
        w_t = torch.tensor(sample_weight, dtype=torch.float32) if sample_weight is not None else torch.ones(len(y_enc), dtype=torch.float32)

        self.model_ = _MLP(X.shape[1], n_classes, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        effective_bs = min(self.batch_size, max(4, len(X) // 2))
        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True, drop_last=False)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = (loss_fn(logits, yb) * wb).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_t).argmax(dim=1).cpu().numpy()
        return np.array([self.inv_label_map_[p] for p in preds])


# ============================================================================
# CLASSIFIER FACTORY
# ============================================================================

def create_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "svc_linear":
        return SVC(kernel="linear", random_state=seed)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    elif name == "ridge":
        return RidgeClassifier(alpha=1.0)
    elif name == "mlp":
        return MLPClassifier(hidden_dim=128, epochs=300, lr=1e-3, batch_size=16, random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================================
# EDA AUGMENTER (Wei & Zou 2019 — pure Python, no external packages)
# ============================================================================

class EDAugmenter:
    """Easy Data Augmentation: random swap, deletion, insertion on words."""

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
# BACK-TRANSLATION AUGMENTER (via Gemini)
# ============================================================================

class BackTranslationAugmenter:
    """Back-translation EN→ES→EN using the existing LLM provider."""

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
                               "backtranslated": result,
                               "timestamp": datetime.now().isoformat()}, f, indent=2)
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

def load_dataset(dataset_name: str):
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


def get_dataset_base(dataset_name):
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if dataset_name.startswith(base + "_"):
            return base
    return dataset_name


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class ThesisFinalResult:
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
# PRE-COMPUTATION
# ============================================================================

def precompute_augmentations(
    dataset_name, train_texts, train_labels, train_embeddings,
    n_shot, model, provider, bt_augmenter, seed
):
    """Pre-compute all augmentation data for one dataset + seed."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)
    dim = train_embeddings.shape[1]

    # --- SMOTE (seed-dependent) ---
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

    # --- Random Oversample (seed-dependent) ---
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

    # --- EDA (seed-dependent) ---
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
    # Only compute on first call; caller passes pre-computed bt_data if available
    bt_data = {"embeddings": None, "labels": []}

    # --- LLM + cascade scores (cached, seed-independent) ---
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
        llm_data[cls] = {
            "top_emb": gen_emb[top_idx],
            "top_weights": weights[top_idx],
        }

    return {
        "smote": smote_data,
        "random_oversample": random_data,
        "eda": eda_data,
        "back_translation": bt_data,
        "llm": llm_data,
    }


def precompute_back_translation(train_texts, train_labels, model, bt_augmenter):
    """Pre-compute back-translation (once per dataset, shared across seeds)."""
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
        return ThesisFinalResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            seed=seed, f1_macro=float(f1),
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0, weight_std=0, timestamp=datetime.now().isoformat(),
        )

    # Methods using pre-computed uniform-weight data
    if aug_method in ("smote", "random_oversample", "eda", "back_translation"):
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
        return ThesisFinalResult(
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
    return ThesisFinalResult(
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

def compute_paired_statistics(method_f1s, baseline_f1s, n_comparisons=6):
    """Compute paired statistics: CI, t-test, Cohen's d, Bonferroni."""
    method_f1s = np.array(method_f1s)
    baseline_f1s = np.array(baseline_f1s)
    n = len(method_f1s)
    deltas = method_f1s - baseline_f1s
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1) if n > 1 else 0.0
    se = delta_std / np.sqrt(n) if n > 1 else 0.0

    # 95% CI (t-distribution)
    if n > 1 and se > 0:
        ci_t = stats.t.interval(0.95, n - 1, loc=delta_mean, scale=se)
    else:
        ci_t = (delta_mean, delta_mean)

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot = [np.mean(deltas[rng.choice(n, n, replace=True)]) for _ in range(1000)]
    ci_boot = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    # Paired t-test
    if n > 1 and delta_std > 0:
        t_stat, p_val = stats.ttest_rel(method_f1s, baseline_f1s)
    else:
        t_stat, p_val = 0.0, 1.0

    cohen_d = delta_mean / (delta_std + 1e-10) if delta_std > 0 else 0.0
    bonf_p = min(p_val * n_comparisons, 1.0)

    return {
        "delta_mean_pp": float(delta_mean * 100),
        "delta_std_pp": float(delta_std * 100),
        "ci_95_t": (float(ci_t[0] * 100), float(ci_t[1] * 100)),
        "ci_95_bootstrap": (float(ci_boot[0] * 100), float(ci_boot[1] * 100)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "bonferroni_p": float(bonf_p),
        "cohen_d": float(cohen_d),
        "significant_005": bool(p_val < 0.05),
        "significant_bonferroni": bool(bonf_p < 0.05),
        "win_rate": float(np.mean(deltas > 0)),
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results):
    data = [asdict(r) for r in results]
    n_comparisons = len(AUGMENTATION_METHODS) - 1  # all vs smote

    print("\n" + "=" * 90)
    print("THESIS FINAL — COMPREHENSIVE STATISTICAL REPORT")
    print("=" * 90)

    # ---- TABLE 1: Overall Method Comparison ----
    print("\n" + "-" * 90)
    print("TABLE 1: OVERALL METHOD COMPARISON (vs SMOTE, across all datasets/classifiers/shots)")
    print("-" * 90)
    print(f"\n{'Method':<18} {'Mean F1':>8} {'Δ SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'Bonf.':>7} {'d':>6} {'Win%':>6}")
    print("-" * 90)

    for method in AUGMENTATION_METHODS:
        if method == "smote":
            print(f"{'smote':<18} {'—':>8} {'ref':>10} {'—':>22} {'—':>9} {'—':>7} {'—':>6} {'—':>6}")
            continue

        # Collect paired F1s: for each (dataset, classifier, shot), average across seeds,
        # then compare method vs smote
        method_means, smote_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            for clf in CLASSIFIER_NAMES:
                m_f1s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["classifier"] == clf and r["augmentation_method"] == method]
                s_f1s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["classifier"] == clf and r["augmentation_method"] == "smote"]
                if m_f1s and s_f1s:
                    method_means.append(np.mean(m_f1s))
                    smote_means.append(np.mean(s_f1s))

        if not method_means:
            continue

        st = compute_paired_statistics(method_means, smote_means, n_comparisons)
        mean_f1 = np.mean(method_means)
        sig = "*" if st["significant_bonferroni"] else ""
        print(f"{method:<18} {mean_f1:>8.4f} {st['delta_mean_pp']:>+9.2f}p "
              f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
              f"{st['p_value']:>9.4f} {st['bonferroni_p']:>6.4f}{sig} "
              f"{st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%")

    # ---- TABLE 2: By Number of Classes ----
    print("\n" + "-" * 90)
    print("TABLE 2: SOFT WEIGHTED vs SMOTE — BY NUMBER OF CLASSES")
    print("-" * 90)
    print(f"\n{'Classes':>8} {'Δ SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'d':>6} {'Win%':>6} {'N':>5}")
    print("-" * 65)

    for n_cls in sorted(set(r["n_classes"] for r in data)):
        m_means, s_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            for clf in CLASSIFIER_NAMES:
                sub_m = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["n_classes"] == n_cls
                         and r["classifier"] == clf and r["augmentation_method"] == "soft_weighted"]
                sub_s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["n_classes"] == n_cls
                         and r["classifier"] == clf and r["augmentation_method"] == "smote"]
                if sub_m and sub_s:
                    m_means.append(np.mean(sub_m))
                    s_means.append(np.mean(sub_s))
        if m_means:
            st = compute_paired_statistics(m_means, s_means)
            print(f"{n_cls:>8} {st['delta_mean_pp']:>+9.2f}p "
                  f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
                  f"{st['p_value']:>9.4f} {st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}% {len(m_means):>5}")

    # ---- TABLE 3: By N-Shot ----
    print("\n" + "-" * 90)
    print("TABLE 3: SOFT WEIGHTED vs SMOTE — BY N-SHOT")
    print("-" * 90)
    print(f"\n{'N-shot':>8} {'Δ SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'d':>6} {'Win%':>6}")
    print("-" * 65)

    for ns in sorted(set(r["n_shot"] for r in data)):
        m_means, s_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            for clf in CLASSIFIER_NAMES:
                sub_m = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["n_shot"] == ns
                         and r["classifier"] == clf and r["augmentation_method"] == "soft_weighted"]
                sub_s = [r["f1_macro"] for r in data
                         if r["dataset"] == ds and r["n_shot"] == ns
                         and r["classifier"] == clf and r["augmentation_method"] == "smote"]
                if sub_m and sub_s:
                    m_means.append(np.mean(sub_m))
                    s_means.append(np.mean(sub_s))
        if m_means:
            st = compute_paired_statistics(m_means, s_means)
            print(f"{ns:>8} {st['delta_mean_pp']:>+9.2f}p "
                  f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
                  f"{st['p_value']:>9.4f} {st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%")

    # ---- TABLE 4: By Classifier ----
    print("\n" + "-" * 90)
    print("TABLE 4: SOFT WEIGHTED vs SMOTE — BY CLASSIFIER")
    print("-" * 90)
    print(f"\n{'Classifier':<22} {'Δ SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'d':>6} {'Win%':>6}")
    print("-" * 78)

    for clf_name in CLASSIFIER_NAMES:
        m_means, s_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            sub_m = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf_name
                     and r["augmentation_method"] == "soft_weighted"]
            sub_s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf_name
                     and r["augmentation_method"] == "smote"]
            if sub_m and sub_s:
                m_means.append(np.mean(sub_m))
                s_means.append(np.mean(sub_s))
        if m_means:
            st = compute_paired_statistics(m_means, s_means)
            sig = " *" if st["significant_005"] else ""
            print(f"{clf_name:<22} {st['delta_mean_pp']:>+9.2f}p "
                  f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
                  f"{st['p_value']:>9.4f} {st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%{sig}")

    # ---- TABLE 5-8: ABLATION TABLES ----
    print("\n" + "=" * 90)
    print("ABLATION STUDIES")
    print("=" * 90)

    # Ablation 1: Does filtering help? (binary_filter vs no_augmentation)
    print("\n" + "-" * 90)
    print("ABLATION 1: DOES FILTERING HELP? (binary_filter vs no_augmentation)")
    print("-" * 90)
    m_means, s_means = [], []
    for ds in sorted(set(r["dataset"] for r in data)):
        for clf in CLASSIFIER_NAMES:
            sub_m = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf
                     and r["augmentation_method"] == "binary_filter"]
            sub_s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf
                     and r["augmentation_method"] == "no_augmentation"]
            if sub_m and sub_s:
                m_means.append(np.mean(sub_m))
                s_means.append(np.mean(sub_s))
    if m_means:
        st = compute_paired_statistics(m_means, s_means, 1)
        print(f"  binary_filter vs no_aug: {st['delta_mean_pp']:+.2f}pp, "
              f"CI [{st['ci_95_bootstrap'][0]:+.2f}, {st['ci_95_bootstrap'][1]:+.2f}], "
              f"p={st['p_value']:.4f}, d={st['cohen_d']:.2f}, win={st['win_rate']*100:.1f}%")

    # Ablation 2: Soft > binary? (soft_weighted vs binary_filter)
    print("\n" + "-" * 90)
    print("ABLATION 2: SOFT > BINARY? (soft_weighted vs binary_filter)")
    print("-" * 90)
    m_means, s_means = [], []
    for ds in sorted(set(r["dataset"] for r in data)):
        for clf in CLASSIFIER_NAMES:
            sub_m = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf
                     and r["augmentation_method"] == "soft_weighted"]
            sub_s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf
                     and r["augmentation_method"] == "binary_filter"]
            if sub_m and sub_s:
                m_means.append(np.mean(sub_m))
                s_means.append(np.mean(sub_s))
    if m_means:
        st = compute_paired_statistics(m_means, s_means, 1)
        print(f"  soft_weighted vs binary_filter: {st['delta_mean_pp']:+.2f}pp, "
              f"CI [{st['ci_95_bootstrap'][0]:+.2f}, {st['ci_95_bootstrap'][1]:+.2f}], "
              f"p={st['p_value']:.4f}, d={st['cohen_d']:.2f}, win={st['win_rate']*100:.1f}%")

    # Ablation 3: Classifier generalization (per-classifier, soft vs smote)
    print("\n" + "-" * 90)
    print("ABLATION 3: CLASSIFIER GENERALIZATION (soft_weighted vs SMOTE per classifier)")
    print("-" * 90)
    for clf_name in CLASSIFIER_NAMES:
        m_means, s_means = [], []
        for ds in sorted(set(r["dataset"] for r in data)):
            sub_m = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf_name
                     and r["augmentation_method"] == "soft_weighted"]
            sub_s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf_name
                     and r["augmentation_method"] == "smote"]
            if sub_m and sub_s:
                m_means.append(np.mean(sub_m))
                s_means.append(np.mean(sub_s))
        if m_means:
            st = compute_paired_statistics(m_means, s_means, len(CLASSIFIER_NAMES))
            sig = "YES" if st["significant_bonferroni"] else "no"
            print(f"  {clf_name:<22}: {st['delta_mean_pp']:+.2f}pp, p={st['p_value']:.4f}, "
                  f"Bonf={st['bonferroni_p']:.4f} [{sig}], d={st['cohen_d']:.2f}, "
                  f"win={st['win_rate']*100:.1f}%")

    # Ablation 4: Neural networks (MLP only)
    print("\n" + "-" * 90)
    print("ABLATION 4: NEURAL NETWORKS (MLP soft_weighted vs MLP SMOTE)")
    print("-" * 90)
    m_means, s_means = [], []
    for ds in sorted(set(r["dataset"] for r in data)):
        sub_m = [r["f1_macro"] for r in data
                 if r["dataset"] == ds and r["classifier"] == "mlp"
                 and r["augmentation_method"] == "soft_weighted"]
        sub_s = [r["f1_macro"] for r in data
                 if r["dataset"] == ds and r["classifier"] == "mlp"
                 and r["augmentation_method"] == "smote"]
        if sub_m and sub_s:
            m_means.append(np.mean(sub_m))
            s_means.append(np.mean(sub_s))
    if m_means:
        st = compute_paired_statistics(m_means, s_means, 1)
        print(f"  MLP soft_weighted vs MLP SMOTE: {st['delta_mean_pp']:+.2f}pp, "
              f"CI [{st['ci_95_bootstrap'][0]:+.2f}, {st['ci_95_bootstrap'][1]:+.2f}], "
              f"p={st['p_value']:.4f}, d={st['cohen_d']:.2f}, win={st['win_rate']*100:.1f}%")

    # ---- SAVE SUMMARY ----
    summary = {"timestamp": datetime.now().isoformat(), "n_experiments": len(data),
               "n_seeds": len(SEEDS), "n_methods": len(AUGMENTATION_METHODS)}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# RESULTS I/O
# ============================================================================

def save_results(results, filename="partial_results.json"):
    output = {"timestamp": datetime.now().isoformat(), "n_experiments": len(results),
              "results": [asdict(r) for r in results]}
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("THESIS FINAL EXPERIMENT")
    print("=" * 90)

    total = len(DATASETS) * len(CLASSIFIER_NAMES) * len(AUGMENTATION_METHODS) * len(SEEDS)
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"Classifiers: {CLASSIFIER_NAMES}")
    print(f"Methods: {AUGMENTATION_METHODS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {total}")

    print("\nLoading model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-2.0-flash")
    bt_augmenter = BackTranslationAugmenter(provider, BT_CACHE_DIR)

    results: List[ThesisFinalResult] = []
    experiment_count = 0

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
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        # Pre-compute back-translation (once, shared across seeds)
        print("  Pre-computing back-translation...")
        bt_data = precompute_back_translation(train_texts, train_labels, model, bt_augmenter)

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            aug_data = precompute_augmentations(
                dataset_name, train_texts, train_labels, train_embeddings,
                n_shot, model, provider, bt_augmenter, seed,
            )
            aug_data["back_translation"] = bt_data

            for clf_name in CLASSIFIER_NAMES:
                for aug_method in AUGMENTATION_METHODS:
                    experiment_count += 1
                    try:
                        result = run_single_config(
                            dataset_name, dataset_base, n_classes, n_shot,
                            clf_name, aug_method, seed,
                            train_embeddings, train_labels,
                            test_embeddings, test_labels, aug_data,
                        )
                        results.append(result)

                        if aug_method in ("soft_weighted", "binary_filter"):
                            print(f"    [{experiment_count}/{total}] {clf_name}/{aug_method}/s{seed}: "
                                  f"F1={result.f1_macro:.4f}")
                    except Exception as e:
                        print(f"    [{experiment_count}/{total}] ERROR: {clf_name}/{aug_method}/s{seed}: {e}")

                    if experiment_count % 200 == 0:
                        save_results(results)

    save_results(results, "final_results.json")
    if results:
        generate_report(results)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Total: {len(results)} experiments across {len(SEEDS)} seeds")
    print("=" * 90)


if __name__ == "__main__":
    main()
