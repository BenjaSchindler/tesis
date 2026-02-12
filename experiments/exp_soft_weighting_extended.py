#!/usr/bin/env python3
"""
Extended Soft Weighting Experiment — Robustness Analysis

Tests the proven best soft weighting config (cascade_l1/top_n/minmax/temp=0.5)
across three new dimensions:
  1. More datasets (7 total, 2-20 classes)
  2. More n-shot values (10, 25, 50)
  3. Multiple classifiers (LogisticRegression, SVC, RandomForest, Ridge)

Compares 4 augmentation methods:
  - no_augmentation: real data only
  - smote: 50 SMOTE samples/class, uniform weight
  - binary_filter: cascade_l1 top_n, uniform weight=1.0
  - soft_weighted: cascade_l1 top_n, quality-weighted

Total: 7 datasets × 3 shots × 4 classifiers × 4 methods = 336 experiments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib

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


# ============================================================================
# CNN CLASSIFIER (sklearn-compatible wrapper)
# ============================================================================

class _MLP(nn.Module):
    """2-layer MLP for dense embedding classification."""

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
    """sklearn-compatible MLP that supports sample_weight via weighted loss.

    Designed for low-resource classification on pre-computed dense embeddings.
    """

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

        # Encode labels
        unique_labels = sorted(set(y))
        self.label_map_ = {l: i for i, l in enumerate(unique_labels)}
        self.inv_label_map_ = {i: l for l, i in self.label_map_.items()}
        n_classes = len(unique_labels)

        y_enc = np.array([self.label_map_[l] for l in y])
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)

        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32)
        else:
            w_t = torch.ones(len(y_enc), dtype=torch.float32)

        self.model_ = _MLP(X.shape[1], n_classes, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        effective_bs = min(self.batch_size, max(4, len(X) // 2))
        dataset = TensorDataset(X_t, y_t, w_t)
        loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True,
                            drop_last=False)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                per_sample_loss = loss_fn(logits, yb)
                loss = (per_sample_loss * wb).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict(self, X):
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return np.array([self.inv_label_map_[p] for p in preds])

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "soft_weighting_extended"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0

# Fixed best soft weighting config from Phase 2
FILTER_CONFIG = {"filter_level": 1, "k_neighbors": 10}
NORMALIZATION = "minmax"
TEMPERATURE = 0.5
MIN_WEIGHT = 0.0

# All datasets (existing + new)
DATASETS = [
    # 2 classes
    "sms_spam_10shot", "sms_spam_25shot", "sms_spam_50shot",
    # 3 classes
    "hate_speech_davidson_10shot", "hate_speech_davidson_25shot", "hate_speech_davidson_50shot",
    # 4 classes (existing)
    "20newsgroups_10shot", "20newsgroups_25shot", "20newsgroups_50shot",
    # 4 classes (real AG News)
    "ag_news_10shot", "ag_news_25shot", "ag_news_50shot",
    # 6 classes
    "emotion_10shot", "emotion_25shot", "emotion_50shot",
    # 14 classes
    "dbpedia14_10shot", "dbpedia14_25shot", "dbpedia14_50shot",
    # 20 classes
    "20newsgroups_20class_10shot", "20newsgroups_20class_25shot", "20newsgroups_20class_50shot",
]

# Classifiers (all support sample_weight)
CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "svc_linear": lambda: SVC(kernel="linear", random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "ridge": lambda: RidgeClassifier(alpha=1.0),
    "mlp": lambda: MLPClassifier(hidden_dim=128, epochs=300, lr=1e-3, batch_size=16, random_state=42),
}

AUGMENTATION_METHODS = ["no_augmentation", "smote", "binary_filter", "soft_weighted"]

# Dataset name to number of classes mapping
DATASET_N_CLASSES = {
    "sms_spam": 2,
    "hate_speech_davidson": 3,
    "20newsgroups": 4,
    "ag_news": 4,
    "emotion": 6,
    "dbpedia14": 14,
    "20newsgroups_20class": 20,
}


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class ExtendedResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    classifier: str
    augmentation_method: str

    f1_macro: float
    baseline_f1: float
    smote_f1: float
    delta_vs_baseline_pp: float
    delta_vs_smote_pp: float

    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float

    timestamp: str


# ============================================================================
# SHARED INFRASTRUCTURE
# ============================================================================

def load_dataset(dataset_name: str):
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return data["train_texts"], data["train_labels"], data["test_texts"], data["test_labels"]


def get_cache_key(dataset, class_name, n_shot, n_generate):
    key_str = f"{dataset}_{class_name}_{n_shot}_{n_generate}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def create_prompt(class_name, examples, n_generate, n_shot):
    selected_examples = examples[:n_shot]
    examples_text = "\n\n".join([
        f"Example {i+1}: {ex[:500]}" for i, ex in enumerate(selected_examples)
    ])
    return f"""You are an expert at generating realistic text examples for classification.

Class: {class_name}

Here are {len(selected_examples)} real examples from this class:
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
            json.dump({
                "dataset": dataset, "class_name": class_name,
                "n_shot": n_shot, "n_generate": n_generate,
                "texts": generated, "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated

    except Exception as e:
        print(f"        Error generating: {e}")
        return np.array([]).reshape(0, 768), []


def generate_smote_samples(real_embeddings, n_generate, k_neighbors=5):
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])
    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])

    n_base = len(real_embeddings)
    n_dummy = max(n_base + n_generate, n_base * 2)
    X = np.vstack([real_embeddings, np.random.randn(n_dummy, real_embeddings.shape[1])])
    y = np.array([0] * n_base + [1] * n_dummy)

    try:
        smote = SMOTE(k_neighbors=k, sampling_strategy={0: n_base + n_generate, 1: n_dummy},
                      random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        class0_indices = np.where(y_res == 0)[0]
        return X_res[class0_indices[n_base:]][:n_generate]
    except Exception as e:
        print(f"        SMOTE error: {e}")
        return np.array([]).reshape(0, real_embeddings.shape[1])


def normalize_scores(raw_scores, method="minmax", temperature=1.0, min_weight=0.0):
    from scipy.special import expit as sigmoid_fn
    n = len(raw_scores)
    if n == 0:
        return np.array([])
    if np.std(raw_scores) < 1e-10:
        return np.ones(n)

    weight_range = 1.0 - min_weight

    if method == "minmax":
        s_min, s_max = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - s_min) / (s_max - s_min + 1e-10)
    elif method == "sigmoid":
        median = np.median(raw_scores)
        iqr = np.percentile(raw_scores, 75) - np.percentile(raw_scores, 25)
        if iqr < 1e-10:
            iqr = np.std(raw_scores) + 1e-10
        normalized = sigmoid_fn((raw_scores - median) / (iqr + 1e-10))
    elif method == "rank":
        order = raw_scores.argsort().argsort()
        normalized = order / (n - 1 + 1e-10)
    else:
        raise ValueError(f"Unknown normalization: {method}")

    if temperature != 1.0:
        normalized = np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / temperature)

    return min_weight + weight_range * normalized


# ============================================================================
# PRE-COMPUTATION (once per dataset)
# ============================================================================

def precompute_dataset_augmentations(
    dataset_name, train_texts, train_labels, train_embeddings,
    n_shot, model, provider
):
    """Pre-compute LLM candidates, SMOTE, and cascade scores for a dataset."""
    unique_classes = sorted(set(train_labels))
    train_labels_arr = np.array(train_labels)

    # SMOTE data
    smote_embs, smote_labels = [], []
    for cls in unique_classes:
        cls_mask = train_labels_arr == cls
        cls_emb = train_embeddings[cls_mask]
        s = generate_smote_samples(cls_emb, N_SYNTHETIC_PER_CLASS)
        if len(s) > 0:
            smote_embs.append(s)
            smote_labels.extend([cls] * len(s))

    smote_data = {
        "embeddings": np.vstack(smote_embs) if smote_embs else np.array([]).reshape(0, train_embeddings.shape[1]),
        "labels": smote_labels,
    }

    # LLM candidates + cascade scores
    llm_data = {}
    cascade = FilterCascade(**FILTER_CONFIG)

    for cls in unique_classes:
        cls_mask = train_labels_arr == cls
        cls_emb = train_embeddings[cls_mask]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        n_generate = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
        gen_emb, gen_texts = generate_llm_samples_cached(
            provider, dataset_name, cls, cls_texts, n_generate, n_shot, model
        )

        if len(gen_emb) == 0:
            llm_data[cls] = None
            continue

        # Cascade scores
        anchor = cls_emb.mean(axis=0) if len(cls_emb) > 0 else train_embeddings.mean(axis=0)
        composite_scores, _ = cascade.compute_quality_scores(
            gen_emb, anchor, train_embeddings, train_labels_arr, cls
        )

        # Normalize and select top_n
        weights = normalize_scores(composite_scores, NORMALIZATION, TEMPERATURE, MIN_WEIGHT)
        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        top_idx = np.argsort(composite_scores)[-target_n:]

        llm_data[cls] = {
            "all_emb": gen_emb,
            "all_scores": composite_scores,
            "all_weights": weights,
            "top_idx": top_idx,
            "top_emb": gen_emb[top_idx],
            "top_weights": weights[top_idx],
        }

    return smote_data, llm_data


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, classifier_factory,
    aug_method,
    train_embeddings, train_labels,
    test_embeddings, test_labels,
    smote_data, llm_data,
    baseline_f1, smote_f1,
):
    """Run one augmentation method with one classifier."""
    unique_classes = sorted(set(train_labels))

    if aug_method == "no_augmentation":
        clf = classifier_factory()
        clf.fit(train_embeddings, train_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return ExtendedResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            f1_macro=float(f1), baseline_f1=baseline_f1, smote_f1=smote_f1,
            delta_vs_baseline_pp=float((f1 - baseline_f1) * 100),
            delta_vs_smote_pp=float((f1 - smote_f1) * 100),
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0, weight_std=0,
            timestamp=datetime.now().isoformat(),
        )

    if aug_method == "smote":
        if len(smote_data["labels"]) == 0:
            # No SMOTE data — fallback to no augmentation
            return run_single_config(
                dataset_name, dataset_base, n_classes, n_shot,
                classifier_name, classifier_factory, "no_augmentation",
                train_embeddings, train_labels, test_embeddings, test_labels,
                smote_data, llm_data, baseline_f1, smote_f1,
            )
        aug_emb = np.vstack([train_embeddings, smote_data["embeddings"]])
        aug_labels = list(train_labels) + smote_data["labels"]
        clf = classifier_factory()
        clf.fit(aug_emb, aug_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return ExtendedResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, augmentation_method=aug_method,
            f1_macro=float(f1), baseline_f1=baseline_f1, smote_f1=smote_f1,
            delta_vs_baseline_pp=float((f1 - baseline_f1) * 100),
            delta_vs_smote_pp=float((f1 - smote_f1) * 100),
            n_real_samples=len(train_embeddings),
            n_synthetic_samples=len(smote_data["labels"]),
            weight_mean=1.0, weight_std=0.0,
            timestamp=datetime.now().isoformat(),
        )

    # binary_filter or soft_weighted
    syn_embs, syn_labels, syn_weights = [], [], []
    for cls in unique_classes:
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        syn_embs.append(ld["top_emb"])
        syn_labels.extend([cls] * len(ld["top_emb"]))
        if aug_method == "binary_filter":
            syn_weights.append(np.ones(len(ld["top_emb"])))
        else:  # soft_weighted
            syn_weights.append(ld["top_weights"])

    if not syn_embs:
        return run_single_config(
            dataset_name, dataset_base, n_classes, n_shot,
            classifier_name, classifier_factory, "no_augmentation",
            train_embeddings, train_labels, test_embeddings, test_labels,
            smote_data, llm_data, baseline_f1, smote_f1,
        )

    synthetic_emb = np.vstack(syn_embs)
    synthetic_weights = np.concatenate(syn_weights)

    aug_emb = np.vstack([train_embeddings, synthetic_emb])
    aug_labels = list(train_labels) + syn_labels
    sample_weights = np.concatenate([np.ones(len(train_embeddings)), synthetic_weights])

    clf = classifier_factory()
    clf.fit(aug_emb, aug_labels, sample_weight=sample_weights)
    pred = clf.predict(test_embeddings)
    f1 = f1_score(test_labels, pred, average="macro")

    return ExtendedResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, augmentation_method=aug_method,
        f1_macro=float(f1), baseline_f1=baseline_f1, smote_f1=smote_f1,
        delta_vs_baseline_pp=float((f1 - baseline_f1) * 100),
        delta_vs_smote_pp=float((f1 - smote_f1) * 100),
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
    output_path = RESULTS_DIR / filename
    data = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(results),
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def get_dataset_base(dataset_name):
    """Extract base dataset name from full name."""
    for base in sorted(DATASET_N_CLASSES.keys(), key=len, reverse=True):
        if dataset_name.startswith(base + "_"):
            return base
    return dataset_name


def generate_report(results):
    print("\n" + "=" * 80)
    print("EXTENDED SOFT WEIGHTING EXPERIMENT — SUMMARY")
    print("=" * 80)

    data = [asdict(r) for r in results]

    # Table 1: Performance vs Number of Classes
    print("\n" + "-" * 80)
    print("TABLE 1: PERFORMANCE vs NUMBER OF CLASSES (soft_weighted vs SMOTE)")
    print("-" * 80)
    print(f"\n{'Classes':>8} {'Method':<16} {'Mean F1':>10} {'vs SMOTE':>12} {'Win Rate':>10} {'N':>5}")
    print("-" * 65)

    for n_cls in sorted(set(r["n_classes"] for r in data)):
        for method in ["soft_weighted", "binary_filter", "smote"]:
            subset = [r for r in data if r["n_classes"] == n_cls and r["augmentation_method"] == method]
            if not subset:
                continue
            mean_f1 = np.mean([r["f1_macro"] for r in subset])
            deltas = [r["delta_vs_smote_pp"] for r in subset]
            mean_d = np.mean(deltas)
            win = 100 * sum(1 for x in deltas if x > 0) / len(deltas) if method != "smote" else "-"
            win_str = f"{win:>9.1f}%" if isinstance(win, float) else f"{'—':>10}"
            print(f"{n_cls:>8} {method:<16} {mean_f1:>10.4f} {mean_d:>+11.2f}pp {win_str} {len(subset):>5}")
        print()

    # Table 2: Performance vs N-shot
    print("-" * 80)
    print("TABLE 2: PERFORMANCE vs N-SHOT (soft_weighted only)")
    print("-" * 80)
    print(f"\n{'N-shot':>8} {'Mean vs SMOTE':>15} {'Std':>8} {'Win Rate':>10} {'N':>5}")
    print("-" * 50)

    for ns in sorted(set(r["n_shot"] for r in data)):
        subset = [r for r in data if r["n_shot"] == ns and r["augmentation_method"] == "soft_weighted"]
        if not subset:
            continue
        deltas = [r["delta_vs_smote_pp"] for r in subset]
        print(f"{ns:>8} {np.mean(deltas):>+15.2f}pp {np.std(deltas):>8.2f} "
              f"{100 * sum(1 for x in deltas if x > 0) / len(deltas):>9.1f}% {len(subset):>5}")

    # Table 3: Performance vs Classifier
    print("\n" + "-" * 80)
    print("TABLE 3: PERFORMANCE vs CLASSIFIER")
    print("-" * 80)
    print(f"\n{'Classifier':<22} {'Method':<16} {'Mean vs SMOTE':>14} {'Win Rate':>10} {'Soft-Binary':>12}")
    print("-" * 78)

    for clf_name in CLASSIFIERS:
        soft_deltas = [r["delta_vs_smote_pp"] for r in data
                       if r["classifier"] == clf_name and r["augmentation_method"] == "soft_weighted"]
        binary_deltas = [r["delta_vs_smote_pp"] for r in data
                         if r["classifier"] == clf_name and r["augmentation_method"] == "binary_filter"]
        if soft_deltas:
            soft_mean = np.mean(soft_deltas)
            soft_win = 100 * sum(1 for x in soft_deltas if x > 0) / len(soft_deltas)
            binary_mean = np.mean(binary_deltas) if binary_deltas else 0
            benefit = soft_mean - binary_mean
            print(f"{clf_name:<22} {'soft_weighted':<16} {soft_mean:>+13.2f}pp {soft_win:>9.1f}% {benefit:>+11.2f}pp")
            if binary_deltas:
                binary_win = 100 * sum(1 for x in binary_deltas if x > 0) / len(binary_deltas)
                print(f"{'':<22} {'binary_filter':<16} {binary_mean:>+13.2f}pp {binary_win:>9.1f}%")

    # Table 4: Interaction — Classifier × N_classes (soft_weighted delta vs smote)
    print("\n" + "-" * 80)
    print("TABLE 4: SOFT WEIGHTED vs SMOTE — CLASSIFIER × N_CLASSES")
    print("-" * 80)

    n_classes_list = sorted(set(r["n_classes"] for r in data))
    header = f"{'Classifier':<22}" + "".join(f"{'C=' + str(c):>10}" for c in n_classes_list)
    print(f"\n{header}")
    print("-" * (22 + 10 * len(n_classes_list)))

    for clf_name in CLASSIFIERS:
        row = f"{clf_name:<22}"
        for n_cls in n_classes_list:
            subset = [r for r in data
                      if r["classifier"] == clf_name and r["n_classes"] == n_cls
                      and r["augmentation_method"] == "soft_weighted"]
            if subset:
                mean_d = np.mean([r["delta_vs_smote_pp"] for r in subset])
                row += f"{mean_d:>+9.1f}p"
            else:
                row += f"{'—':>10}"
        print(row)

    # Table 5: Best per dataset
    print("\n" + "-" * 80)
    print("TABLE 5: BEST CONFIGURATION PER DATASET")
    print("-" * 80)

    for ds in sorted(set(r["dataset"] for r in data)):
        ds_results = [r for r in data if r["dataset"] == ds and r["augmentation_method"] == "soft_weighted"]
        if not ds_results:
            continue
        best = max(ds_results, key=lambda x: x["f1_macro"])
        print(f"\n  {ds} ({best['n_classes']} classes):")
        print(f"    Best clf: {best['classifier']}, F1={best['f1_macro']:.4f}, "
              f"vs SMOTE: {best['delta_vs_smote_pp']:+.2f}pp")

    # Table 6: Overall summary
    print("\n" + "-" * 80)
    print("TABLE 6: OVERALL SUMMARY")
    print("-" * 80)
    print(f"\n{'Method':<16} {'Mean vs SMOTE':>14} {'Median':>10} {'Std':>8} {'Win Rate':>10} {'N':>6}")
    print("-" * 68)

    for method in AUGMENTATION_METHODS:
        subset = [r for r in data if r["augmentation_method"] == method]
        if not subset:
            continue
        deltas = [r["delta_vs_smote_pp"] for r in subset]
        win = 100 * sum(1 for x in deltas if x > 0) / len(deltas) if method != "smote" else 0
        print(f"{method:<16} {np.mean(deltas):>+13.2f}pp {np.median(deltas):>+9.2f} "
              f"{np.std(deltas):>8.2f} {win:>9.1f}% {len(subset):>6}")

    # Save summary JSON
    summary = {
        "overall": {
            method: {
                "mean_vs_smote": float(np.mean([r["delta_vs_smote_pp"] for r in data if r["augmentation_method"] == method])),
                "win_rate": float(100 * sum(1 for r in data if r["augmentation_method"] == method and r["delta_vs_smote_pp"] > 0) /
                                  max(1, sum(1 for r in data if r["augmentation_method"] == method))),
                "n": sum(1 for r in data if r["augmentation_method"] == method),
            }
            for method in AUGMENTATION_METHODS
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EXTENDED SOFT WEIGHTING EXPERIMENT")
    print("=" * 80)

    total = len(DATASETS) * len(CLASSIFIERS) * len(AUGMENTATION_METHODS)
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"Methods: {AUGMENTATION_METHODS}")
    print(f"Total experiments: {total}")

    print("\nLoading model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-2.0-flash")

    results: List[ExtendedResult] = []
    experiment_count = 0

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        dataset_base = get_dataset_base(dataset_name)
        n_classes = DATASET_N_CLASSES.get(dataset_base, 0)
        n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

        print(f"\n{'#' * 80}")
        print(f"# {dataset_name} ({n_classes} classes, {n_shot}-shot)")
        print(f"{'#' * 80}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

        print("  Embedding...")
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        # Compute baselines with LogisticRegression (reference)
        clf_bl = LogisticRegression(max_iter=1000, random_state=42)
        clf_bl.fit(train_embeddings, train_labels)
        baseline_f1 = float(f1_score(test_labels, clf_bl.predict(test_embeddings), average="macro"))

        # Pre-compute augmentation data (shared across classifiers)
        print("  Pre-computing augmentations...")
        smote_data, llm_data = precompute_dataset_augmentations(
            dataset_name, train_texts, train_labels, train_embeddings,
            n_shot, model, provider,
        )

        # Compute SMOTE F1 with LogisticRegression (reference for delta)
        if len(smote_data["labels"]) > 0:
            aug_emb = np.vstack([train_embeddings, smote_data["embeddings"]])
            aug_labels = list(train_labels) + smote_data["labels"]
            clf_sm = LogisticRegression(max_iter=1000, random_state=42)
            clf_sm.fit(aug_emb, aug_labels)
            smote_f1 = float(f1_score(test_labels, clf_sm.predict(test_embeddings), average="macro"))
        else:
            smote_f1 = baseline_f1

        print(f"  Baseline F1: {baseline_f1:.4f}, SMOTE F1: {smote_f1:.4f}")

        n_llm_classes = sum(1 for v in llm_data.values() if v is not None)
        print(f"  LLM data for {n_llm_classes}/{n_classes} classes")

        # Run all classifier × method combinations
        for clf_name, clf_factory in CLASSIFIERS.items():
            for aug_method in AUGMENTATION_METHODS:
                experiment_count += 1

                try:
                    result = run_single_config(
                        dataset_name, dataset_base, n_classes, n_shot,
                        clf_name, clf_factory, aug_method,
                        train_embeddings, train_labels,
                        test_embeddings, test_labels,
                        smote_data, llm_data,
                        baseline_f1, smote_f1,
                    )
                    results.append(result)

                    if aug_method in ("soft_weighted", "binary_filter"):
                        status = " BEATS SMOTE" if result.delta_vs_smote_pp > 0 else ""
                        print(f"    [{experiment_count}/{total}] {clf_name}/{aug_method}: "
                              f"F1={result.f1_macro:.4f}, vs SMOTE: {result.delta_vs_smote_pp:+.2f}pp{status}")

                except Exception as e:
                    print(f"    [{experiment_count}/{total}] {clf_name}/{aug_method}: ERROR: {e}")
                    import traceback
                    traceback.print_exc()

                if experiment_count % 50 == 0:
                    save_results(results)

    # Final save and report
    save_results(results, "final_results.json")
    if results:
        generate_report(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
