#!/usr/bin/env python3
"""
Autoresearch Experiment — Filter vs No-Filter Comparison

Adds the missing `llm_unfiltered` baseline to cleanly answer:
"Does geometric filtering help, or is the LLM generation itself doing all the work?"

Configuration: 10 datasets x 3 shots x 3 classifiers x 7 methods x 10 seeds = 6,300 experiments

All results go to autoresearch/results/ — no existing files are modified.
"""

import sys
from pathlib import Path

# Import from existing src/core/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from scipy import stats

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade
from core.geometric_filter import LOFFilter, CombinedGeometricFilter


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0
SEEDS = [42, 123, 456, 789, 1011, 2024, 3333, 5555, 7777, 9999]

# Soft weighting config
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
    "trec6_10shot", "trec6_25shot", "trec6_50shot",
    # banking77/clinc150 excluded — too many classes makes LOF/combined filter too slow
    # "banking77_10shot", "banking77_25shot", "banking77_50shot",
    # "clinc150_10shot", "clinc150_25shot", "clinc150_50shot",
]

DATASET_N_CLASSES = {
    "sms_spam": 2, "hate_speech_davidson": 3, "20newsgroups": 4,
    "ag_news": 4, "emotion": 6, "dbpedia14": 14, "20newsgroups_20class": 20,
    "trec6": 6, "banking77": 77, "clinc150": 150,
}

CLASSIFIER_NAMES = ["logistic_regression", "svc_linear", "ridge"]

AUGMENTATION_METHODS = [
    "no_augmentation", "smote", "llm_unfiltered",
    "cascade_l1", "cascade_l1_soft", "lof_relaxed", "combined",
]


# ============================================================================
# CLASSIFIER FACTORY
# ============================================================================

def create_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "svc_linear":
        return SVC(kernel="linear", random_state=seed)
    elif name == "ridge":
        return RidgeClassifier(alpha=1.0)
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================================
# SHARED INFRASTRUCTURE (from exp_thesis_final.py)
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
# RESUME SUPPORT
# ============================================================================

def get_result_key(dataset, classifier, method, seed):
    return f"{dataset}|{classifier}|{method}|{seed}"


def load_completed():
    """Load completed experiment keys for resume."""
    partial_file = RESULTS_DIR / "partial_results.json"
    if partial_file.exists():
        with open(partial_file) as f:
            data = json.load(f)
        results = [r for r in data["results"]]
        keys = {get_result_key(r["dataset"], r["classifier"], r["method"], r["seed"])
                for r in results}
        return results, keys
    return [], set()


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class AutoresearchResult:
    dataset: str
    dataset_base: str
    n_classes: int
    n_shot: int
    classifier: str
    method: str
    seed: int
    f1_macro: float
    n_real_samples: int
    n_synthetic_samples: int
    n_candidates: int
    acceptance_rate: float
    weight_mean: float
    weight_std: float
    timestamp: str


# ============================================================================
# PRECOMPUTATION
# ============================================================================

def precompute_llm_data(dataset_name, train_texts, train_labels, train_embeddings,
                        n_shot, model, provider):
    """Pre-compute LLM embeddings + filter results ONCE per dataset (seed-independent)."""
    unique_classes = sorted(set(train_labels))
    labels_arr = np.array(train_labels)

    cascade = FilterCascade(**FILTER_CONFIG)
    lof_filter = LOFFilter(n_neighbors=5, threshold=-0.5)
    combined_filter = CombinedGeometricFilter(lof_threshold=0.0, sim_threshold=0.5)

    llm_data = {}
    n_total = len(unique_classes)
    for ci, cls in enumerate(unique_classes):
        if ci % max(1, n_total // 10) == 0:
            print(f"    LLM precompute: class {ci+1}/{n_total}")
        cls_emb = train_embeddings[labels_arr == cls]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]
        n_gen = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
        gen_emb, gen_texts = generate_llm_samples_cached(
            provider, dataset_name, cls, cls_texts, n_gen, n_shot, model
        )
        if len(gen_emb) == 0:
            llm_data[cls] = None
            continue

        target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
        n_candidates = len(gen_emb)

        # Cascade L1: top-N by distance score (deterministic)
        anchor = cls_emb.mean(axis=0) if len(cls_emb) > 0 else train_embeddings.mean(axis=0)
        scores, _ = cascade.compute_quality_scores(gen_emb, anchor, train_embeddings, labels_arr, cls)
        cascade_top_idx = np.argsort(scores)[-target_n:]
        cascade_weights = normalize_scores(scores[cascade_top_idx], NORMALIZATION, TEMPERATURE, MIN_WEIGHT)

        # LOF relaxed (deterministic)
        _, lof_mask, lof_scores = lof_filter.filter(gen_emb, train_embeddings, labels_arr, cls)
        lof_passed_indices = np.where(lof_mask)[0]
        if len(lof_passed_indices) > target_n:
            lof_top = lof_passed_indices[np.argsort(lof_scores[lof_passed_indices])[-target_n:]]
        elif len(lof_passed_indices) > 0:
            lof_top = lof_passed_indices
        else:
            lof_top = np.argsort(lof_scores)[-target_n:]

        # Combined (deterministic)
        _, combined_mask, _ = combined_filter.filter(gen_emb, train_embeddings, labels_arr, cls)
        combined_idx = np.where(combined_mask)[0]

        llm_data[cls] = {
            "all_emb": gen_emb,
            "n_candidates": n_candidates,
            "target_n": target_n,
            "cascade_idx": cascade_top_idx,
            "cascade_weights": cascade_weights,
            "lof_idx": lof_top,
            "lof_n_passed": int(lof_mask.sum()),
            "combined_idx": combined_idx,
            "combined_n_passed": int(combined_mask.sum()),
        }

    return llm_data


def precompute_augmentations(train_labels, train_embeddings, llm_data, seed):
    """Pre-compute seed-dependent parts only (SMOTE + random unfiltered selection)."""
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

    # --- Unfiltered random selection (seed-dependent) ---
    for cls in unique_classes:
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        rng = np.random.RandomState(seed)
        ld[f"unfiltered_idx_{seed}"] = rng.choice(ld["n_candidates"], ld["target_n"], replace=False)

    return {"smote": smote_data, "llm": llm_data}


# ============================================================================
# SINGLE EXPERIMENT
# ============================================================================

def run_single_config(
    dataset_name, dataset_base, n_classes, n_shot,
    classifier_name, method, seed,
    train_embeddings, train_labels, test_embeddings, test_labels,
    aug_data,
):
    unique_classes = sorted(set(train_labels))
    n_real = len(train_embeddings)

    # --- no_augmentation ---
    if method == "no_augmentation":
        clf = create_classifier(classifier_name, seed)
        clf.fit(train_embeddings, train_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return AutoresearchResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, method=method, seed=seed,
            f1_macro=float(f1), n_real_samples=n_real, n_synthetic_samples=0,
            n_candidates=0, acceptance_rate=0.0,
            weight_mean=0, weight_std=0, timestamp=datetime.now().isoformat(),
        )

    # --- smote ---
    if method == "smote":
        smote_data = aug_data["smote"]
        if len(smote_data["labels"]) == 0:
            return run_single_config(
                dataset_name, dataset_base, n_classes, n_shot,
                classifier_name, "no_augmentation", seed,
                train_embeddings, train_labels, test_embeddings, test_labels, aug_data,
            )
        aug_emb = np.vstack([train_embeddings, smote_data["embeddings"]])
        aug_labels = list(train_labels) + smote_data["labels"]
        clf = create_classifier(classifier_name, seed)
        clf.fit(aug_emb, aug_labels)
        pred = clf.predict(test_embeddings)
        f1 = f1_score(test_labels, pred, average="macro")
        return AutoresearchResult(
            dataset=dataset_name, dataset_base=dataset_base,
            n_classes=n_classes, n_shot=n_shot,
            classifier=classifier_name, method=method, seed=seed,
            f1_macro=float(f1), n_real_samples=n_real,
            n_synthetic_samples=len(smote_data["labels"]),
            n_candidates=0, acceptance_rate=1.0,
            weight_mean=1.0, weight_std=0.0, timestamp=datetime.now().isoformat(),
        )

    # --- LLM-based methods ---
    llm_data = aug_data["llm"]
    syn_embs, syn_labels, syn_weights = [], [], []
    total_candidates, total_accepted = 0, 0

    for cls in unique_classes:
        if llm_data.get(cls) is None:
            continue
        ld = llm_data[cls]
        all_emb = ld["all_emb"]
        n_cand = ld["n_candidates"]
        total_candidates += n_cand

        if method == "llm_unfiltered":
            idx = ld.get(f"unfiltered_idx_{seed}", ld.get("unfiltered_idx"))
            weights = np.ones(len(idx))
        elif method == "cascade_l1":
            idx = ld["cascade_idx"]
            weights = np.ones(len(idx))
        elif method == "cascade_l1_soft":
            idx = ld["cascade_idx"]
            weights = ld["cascade_weights"]
        elif method == "lof_relaxed":
            idx = ld["lof_idx"]
            weights = np.ones(len(idx))
        elif method == "combined":
            idx = ld["combined_idx"]
            weights = np.ones(len(idx))
        else:
            raise ValueError(f"Unknown method: {method}")

        if len(idx) > 0:
            syn_embs.append(all_emb[idx])
            syn_labels.extend([cls] * len(idx))
            syn_weights.append(weights)
            total_accepted += len(idx)

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
    sample_weights = np.concatenate([np.ones(n_real), synthetic_weights])

    clf = create_classifier(classifier_name, seed)
    clf.fit(aug_emb, aug_labels, sample_weight=sample_weights)
    pred = clf.predict(test_embeddings)
    f1 = f1_score(test_labels, pred, average="macro")

    acceptance_rate = total_accepted / total_candidates if total_candidates > 0 else 0.0

    return AutoresearchResult(
        dataset=dataset_name, dataset_base=dataset_base,
        n_classes=n_classes, n_shot=n_shot,
        classifier=classifier_name, method=method, seed=seed,
        f1_macro=float(f1), n_real_samples=n_real,
        n_synthetic_samples=len(synthetic_emb),
        n_candidates=total_candidates,
        acceptance_rate=float(acceptance_rate),
        weight_mean=float(synthetic_weights.mean()),
        weight_std=float(synthetic_weights.std()),
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_paired_statistics(method_f1s, baseline_f1s, n_comparisons=6):
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


def _collect_paired(data, method, baseline, classifiers=CLASSIFIER_NAMES):
    """Collect paired F1 means for method vs baseline across (dataset, classifier)."""
    m_means, b_means = [], []
    for ds in sorted(set(r["dataset"] for r in data)):
        for clf in classifiers:
            m_f1s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf and r["method"] == method]
            b_f1s = [r["f1_macro"] for r in data
                     if r["dataset"] == ds and r["classifier"] == clf and r["method"] == baseline]
            if m_f1s and b_f1s:
                m_means.append(np.mean(m_f1s))
                b_means.append(np.mean(b_f1s))
    return m_means, b_means


def generate_report(results):
    data = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results]

    print("\n" + "=" * 95)
    print("AUTORESEARCH — FILTER vs NO-FILTER COMPARISON")
    print("=" * 95)

    # ---- TABLE 1: All methods vs SMOTE ----
    n_comp = len(AUGMENTATION_METHODS) - 1
    print("\n" + "-" * 95)
    print("TABLE 1: ALL METHODS vs SMOTE")
    print("-" * 95)
    print(f"\n{'Method':<18} {'Mean F1':>8} {'D SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'Bonf.':>7} {'d':>6} {'Win%':>6}")
    print("-" * 95)

    for method in AUGMENTATION_METHODS:
        if method == "smote":
            print(f"{'smote':<18} {'---':>8} {'ref':>10} {'---':>22} {'---':>9} {'---':>7} {'---':>6} {'---':>6}")
            continue
        m_means, s_means = _collect_paired(data, method, "smote")
        if not m_means:
            continue
        st = compute_paired_statistics(m_means, s_means, n_comp)
        mean_f1 = np.mean(m_means)
        sig = "*" if st["significant_bonferroni"] else ""
        print(f"{method:<18} {mean_f1:>8.4f} {st['delta_mean_pp']:>+9.2f}p "
              f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
              f"{st['p_value']:>9.4f} {st['bonferroni_p']:>6.4f}{sig} "
              f"{st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%")

    # ---- TABLE 2: Filtered methods vs llm_unfiltered (KEY TABLE) ----
    filter_methods = ["cascade_l1", "cascade_l1_soft", "lof_relaxed", "combined"]
    n_comp2 = len(filter_methods)
    print("\n" + "-" * 95)
    print("TABLE 2: FILTERED METHODS vs LLM_UNFILTERED (does filtering help?)")
    print("-" * 95)
    print(f"\n{'Method':<18} {'Mean F1':>8} {'D Unfilt':>10} {'95% CI':>22} {'p-value':>9} {'Bonf.':>7} {'d':>6} {'Win%':>6}")
    print("-" * 95)

    for method in ["llm_unfiltered"] + filter_methods:
        if method == "llm_unfiltered":
            uf_means, _ = _collect_paired(data, "llm_unfiltered", "smote")
            mean_f1 = np.mean(uf_means) if uf_means else 0
            print(f"{'llm_unfiltered':<18} {mean_f1:>8.4f} {'ref':>10} {'---':>22} {'---':>9} {'---':>7} {'---':>6} {'---':>6}")
            continue
        m_means, u_means = _collect_paired(data, method, "llm_unfiltered")
        if not m_means:
            continue
        st = compute_paired_statistics(m_means, u_means, n_comp2)
        mean_f1 = np.mean(m_means)
        sig = "*" if st["significant_bonferroni"] else ""
        print(f"{method:<18} {mean_f1:>8.4f} {st['delta_mean_pp']:>+9.2f}p "
              f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
              f"{st['p_value']:>9.4f} {st['bonferroni_p']:>6.4f}{sig} "
              f"{st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%")

    # ---- TABLE 3: By N-shot (cascade_l1_soft vs both baselines) ----
    print("\n" + "-" * 95)
    print("TABLE 3: CASCADE_L1_SOFT — BY N-SHOT")
    print("-" * 95)
    print(f"\n{'N-shot':>8} {'vs SMOTE':>10} {'Win%':>6} {'vs Unfilt':>10} {'Win%':>6}")
    print("-" * 50)

    for ns in sorted(set(r["n_shot"] for r in data)):
        sub = [r for r in data if r["n_shot"] == ns]
        m_s, s_s = _collect_paired(sub, "cascade_l1_soft", "smote")
        m_u, u_u = _collect_paired(sub, "cascade_l1_soft", "llm_unfiltered")
        if m_s and m_u:
            st_s = compute_paired_statistics(m_s, s_s)
            st_u = compute_paired_statistics(m_u, u_u)
            print(f"{ns:>8} {st_s['delta_mean_pp']:>+9.2f}p {st_s['win_rate']*100:>5.1f}% "
                  f"{st_u['delta_mean_pp']:>+9.2f}p {st_u['win_rate']*100:>5.1f}%")

    # ---- TABLE 4: By N-classes ----
    print("\n" + "-" * 95)
    print("TABLE 4: CASCADE_L1_SOFT — BY NUMBER OF CLASSES")
    print("-" * 95)
    print(f"\n{'Classes':>8} {'vs SMOTE':>10} {'Win%':>6} {'vs Unfilt':>10} {'Win%':>6}")
    print("-" * 50)

    for n_cls in sorted(set(r["n_classes"] for r in data)):
        sub = [r for r in data if r["n_classes"] == n_cls]
        m_s, s_s = _collect_paired(sub, "cascade_l1_soft", "smote")
        m_u, u_u = _collect_paired(sub, "cascade_l1_soft", "llm_unfiltered")
        if m_s and m_u:
            st_s = compute_paired_statistics(m_s, s_s)
            st_u = compute_paired_statistics(m_u, u_u)
            print(f"{n_cls:>8} {st_s['delta_mean_pp']:>+9.2f}p {st_s['win_rate']*100:>5.1f}% "
                  f"{st_u['delta_mean_pp']:>+9.2f}p {st_u['win_rate']*100:>5.1f}%")

    # ---- TABLE 5: By classifier ----
    print("\n" + "-" * 95)
    print("TABLE 5: CASCADE_L1_SOFT vs SMOTE — BY CLASSIFIER")
    print("-" * 95)
    print(f"\n{'Classifier':<22} {'D SMOTE':>10} {'95% CI':>22} {'p-value':>9} {'d':>6} {'Win%':>6}")
    print("-" * 78)

    for clf_name in CLASSIFIER_NAMES:
        m_means, s_means = _collect_paired(data, "cascade_l1_soft", "smote", [clf_name])
        if m_means:
            st = compute_paired_statistics(m_means, s_means)
            sig = " *" if st["significant_005"] else ""
            print(f"{clf_name:<22} {st['delta_mean_pp']:>+9.2f}p "
                  f"[{st['ci_95_bootstrap'][0]:>+6.2f},{st['ci_95_bootstrap'][1]:>+6.2f}] "
                  f"{st['p_value']:>9.4f} {st['cohen_d']:>6.2f} {st['win_rate']*100:>5.1f}%{sig}")

    # ---- TABLE 6: Acceptance rates ----
    print("\n" + "-" * 95)
    print("TABLE 6: FILTER ACCEPTANCE RATES (mean samples accepted / 150 candidates)")
    print("-" * 95)

    llm_methods = ["llm_unfiltered", "cascade_l1", "lof_relaxed", "combined"]
    for method in llm_methods:
        rates = [r["acceptance_rate"] for r in data if r["method"] == method and r["n_candidates"] > 0]
        n_syn = [r["n_synthetic_samples"] for r in data if r["method"] == method]
        if rates:
            print(f"  {method:<18}: mean accept={np.mean(rates)*100:.1f}%, "
                  f"mean samples={np.mean(n_syn):.0f}, "
                  f"min samples={np.min(n_syn):.0f}, max={np.max(n_syn):.0f}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(data),
        "n_seeds": len(SEEDS),
        "n_methods": len(AUGMENTATION_METHODS),
        "n_datasets": len(DATASETS),
        "n_classifiers": len(CLASSIFIER_NAMES),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# RESULTS I/O
# ============================================================================

def save_results(results, filename="partial_results.json"):
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(results),
        "results": [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results],
    }
    with open(RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 95)
    print("AUTORESEARCH EXPERIMENT — Filter vs No-Filter")
    print("=" * 95)

    total = len(DATASETS) * len(CLASSIFIER_NAMES) * len(AUGMENTATION_METHODS) * len(SEEDS)
    print(f"\nDatasets: {len(DATASETS)}")
    print(f"Classifiers: {CLASSIFIER_NAMES}")
    print(f"Methods: {AUGMENTATION_METHODS}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Total experiments: {total}")

    # Resume support
    existing_results, completed_keys = load_completed()
    if existing_results:
        print(f"\nResuming: {len(existing_results)} completed, {total - len(completed_keys)} remaining")

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    print("Initializing LLM provider (cache-only)...")
    provider = create_provider("google", "gemini-2.0-flash")

    results = existing_results.copy() if existing_results else []
    # Convert dicts back to dataclass for new results tracking
    experiment_count = len(completed_keys)
    new_count = 0

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        dataset_base = get_dataset_base(dataset_name)
        n_classes = DATASET_N_CLASSES.get(dataset_base, 0)
        n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

        # Check if all configs for this dataset are done
        ds_total = len(CLASSIFIER_NAMES) * len(AUGMENTATION_METHODS) * len(SEEDS)
        ds_done = sum(1 for k in completed_keys if k.startswith(f"{dataset_name}|"))
        if ds_done >= ds_total:
            print(f"\n  Skipping {dataset_name} (all {ds_total} configs done)")
            continue

        print(f"\n{'#' * 95}")
        print(f"# {dataset_name} ({n_classes} classes, {n_shot}-shot) — {ds_done}/{ds_total} done")
        print(f"{'#' * 95}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")

        print("  Embedding...")
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        print("  Pre-computing LLM embeddings + filters (once)...")
        llm_data = precompute_llm_data(
            dataset_name, train_texts, train_labels, train_embeddings,
            n_shot, model, provider,
        )

        for seed in SEEDS:
            # Check if all configs for this seed are done
            seed_keys = [get_result_key(dataset_name, c, m, seed)
                         for c in CLASSIFIER_NAMES for m in AUGMENTATION_METHODS]
            if all(k in completed_keys for k in seed_keys):
                continue

            print(f"\n  --- Seed {seed} ---")
            aug_data = precompute_augmentations(
                train_labels, train_embeddings, llm_data, seed,
            )

            for clf_name in CLASSIFIER_NAMES:
                for method in AUGMENTATION_METHODS:
                    key = get_result_key(dataset_name, clf_name, method, seed)
                    if key in completed_keys:
                        continue

                    experiment_count += 1
                    new_count += 1
                    try:
                        result = run_single_config(
                            dataset_name, dataset_base, n_classes, n_shot,
                            clf_name, method, seed,
                            train_embeddings, train_labels,
                            test_embeddings, test_labels, aug_data,
                        )
                        results.append(result)
                        completed_keys.add(key)

                        if method in ("cascade_l1_soft", "llm_unfiltered"):
                            print(f"    [{experiment_count}/{total}] {clf_name}/{method}/s{seed}: "
                                  f"F1={result.f1_macro:.4f}")
                    except Exception as e:
                        print(f"    [{experiment_count}/{total}] ERROR: {clf_name}/{method}/s{seed}: {e}")

                    if new_count % 50 == 0:
                        save_results(results)
                        print(f"    [checkpoint: {len(results)} results saved]")

    save_results(results, "final_results.json")
    if results:
        generate_report(results)

    print("\n" + "=" * 95)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Total: {len(results)} experiments across {len(SEEDS)} seeds")
    print("=" * 95)


if __name__ == "__main__":
    main()
