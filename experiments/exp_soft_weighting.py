#!/usr/bin/env python3
"""
Soft Weighting Experiment

Instead of binary keep/reject, uses geometric filter scores as continuous
sample weights in LogisticRegression.fit(sample_weight=...).

This directly addresses the 'over-filtering hurts' finding: no samples are
fully rejected, but lower-quality samples get less influence on the classifier.

Two strategies:
  - keep_all: Keep ALL 3x oversampled candidates, weight by score
  - top_n:    Select top N by score, but pass scores as weights (not uniform)

Two-phase execution:
  - Phase 1 (screening): Fixed normalization, sweep filters x strategies x datasets
  - Phase 2 (tuning):    Top combos from Phase 1, sweep norm/temp/min_weight
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter
import hashlib
from scipy.special import expit as sigmoid

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter
from core.filter_cascade import FilterCascade

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "soft_weighting"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SYNTHETIC_PER_CLASS = 50
OVERSAMPLE_FACTOR = 3.0

BENCHMARK_DATASETS = [
    "20newsgroups_10shot",
    "20newsgroups_25shot",
    "sms_spam_10shot",
    "sms_spam_25shot",
    "hate_speech_davidson_10shot",
    "hate_speech_davidson_25shot",
]

# Filters to test
FILTER_CONFIGS = [
    {"name": "none",         "type": "none",    "params": {}},
    {"name": "lof_k10",      "type": "lof",     "params": {"n_neighbors": 10, "threshold": 0.0}},
    {"name": "lof_k20",      "type": "lof",     "params": {"n_neighbors": 20, "threshold": 0.0}},
    {"name": "cascade_l1",   "type": "cascade",  "params": {"filter_level": 1, "k_neighbors": 10}},
    {"name": "cascade_l2",   "type": "cascade",  "params": {"filter_level": 2, "k_neighbors": 10}},
    {"name": "cascade_full", "type": "cascade",  "params": {"filter_level": 4, "k_neighbors": 10}},
]

STRATEGIES = ["keep_all", "top_n"]
NORMALIZATIONS = ["minmax", "sigmoid", "rank"]
TEMPERATURES = [0.5, 1.0, 2.0]
MIN_WEIGHTS = [0.0, 0.1, 0.3]

# Phase 1 defaults
PHASE1_NORM = "minmax"
PHASE1_TEMP = 1.0
PHASE1_MIN_WEIGHT = 0.1


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SoftWeightResult:
    dataset: str
    filter_name: str
    filter_type: str
    filter_params: Dict[str, Any]
    strategy: str
    normalization: str
    temperature: float
    min_weight: float

    baseline_f1: float
    smote_f1: float
    soft_weighted_f1: float

    delta_vs_baseline_pp: float
    delta_vs_smote_pp: float

    n_real_samples: int
    n_synthetic_samples: int
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    weight_median: float

    timestamp: str
    phase: int = 1


# ============================================================================
# SHARED INFRASTRUCTURE (from exp_filter_comparison.py)
# ============================================================================

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return data['train_texts'], data['train_labels'], data['test_texts'], data['test_labels']


def get_cache_key(dataset: str, class_name: str, n_shot: int, n_generate: int) -> str:
    key_str = f"{dataset}_{class_name}_{n_shot}_{n_generate}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def create_prompt(class_name: str, examples: List[str], n_generate: int, n_shot: int) -> str:
    selected_examples = examples[:n_shot]
    examples_text = "\n\n".join([
        f"Example {i+1}: {ex[:500]}"
        for i, ex in enumerate(selected_examples)
    ])
    return f"""You are an expert at generating realistic text examples for classification.

Class: {class_name}

Here are {len(selected_examples)} real examples from this class:
{examples_text}

Generate {n_generate} NEW examples that belong to the "{class_name}" class.
Each example should be similar in style, length, and content to the examples above.
Generate one example per line, without numbering:"""


def generate_llm_samples_cached(
    provider,
    dataset: str,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    n_shot: int,
    model: SentenceTransformer
) -> Tuple[np.ndarray, List[str]]:
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

        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        generated = []
        for line in lines:
            clean = line.lstrip('0123456789.-):* ')
            if len(clean) > 10:
                generated.append(clean)

        if not generated:
            return np.array([]).reshape(0, 768), []

        with open(cache_file, 'w') as f:
            json.dump({
                "dataset": dataset,
                "class_name": class_name,
                "n_shot": n_shot,
                "n_generate": n_generate,
                "texts": generated,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated

    except Exception as e:
        print(f"        Error generating: {e}")
        return np.array([]).reshape(0, 768), []


def generate_smote_samples(
    real_embeddings: np.ndarray,
    n_generate: int,
    k_neighbors: int = 5
) -> np.ndarray:
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
        smote = SMOTE(
            k_neighbors=k,
            sampling_strategy={0: n_base + n_generate, 1: n_dummy},
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X, y)
        class0_indices = np.where(y_res == 0)[0]
        new_indices = class0_indices[n_base:]
        return X_res[new_indices][:n_generate]

    except Exception as e:
        print(f"        SMOTE error: {e}")
        return np.array([]).reshape(0, real_embeddings.shape[1])


def run_smote_baseline(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str]
) -> Tuple[float, float]:
    """Returns (baseline_f1, smote_f1)."""
    unique_classes = list(set(train_labels))
    train_labels_arr = np.array(train_labels)

    all_synthetic_emb = []
    all_synthetic_labels = []

    for cls in unique_classes:
        cls_mask = train_labels_arr == cls
        cls_embeddings = train_embeddings[cls_mask]
        smote_emb = generate_smote_samples(cls_embeddings, N_SYNTHETIC_PER_CLASS)
        if len(smote_emb) > 0:
            all_synthetic_emb.append(smote_emb)
            all_synthetic_labels.extend([cls] * len(smote_emb))

    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])

    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = list(train_labels) + all_synthetic_labels

    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_baseline.fit(train_embeddings, train_labels)
    baseline_f1 = f1_score(test_labels, clf_baseline.predict(test_embeddings), average='macro')

    clf_augmented = LogisticRegression(max_iter=1000, random_state=42)
    clf_augmented.fit(aug_embeddings, aug_labels)
    smote_f1 = f1_score(test_labels, clf_augmented.predict(test_embeddings), average='macro')

    return baseline_f1, smote_f1


# ============================================================================
# CORE: SCORE EXTRACTION
# ============================================================================

def compute_scores_for_candidates(
    filter_type: str,
    filter_params: Dict,
    candidate_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_class: str,
) -> np.ndarray:
    """
    Compute raw continuous quality scores for ALL candidates (no rejection).

    Returns:
        scores: (N,) array — higher = better quality
    """
    n = len(candidate_embeddings)

    if filter_type == "none":
        return np.ones(n)

    if filter_type == "lof":
        lof = LOFFilter(**filter_params)
        # .filter() returns (filtered_emb, mask, lof_scores)
        # lof_scores has values for ALL candidates (computed before threshold)
        _, _, lof_scores = lof.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )
        return lof_scores

    if filter_type == "cascade":
        cascade = FilterCascade(**filter_params)
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]
        anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)
        composite_scores, _ = cascade.compute_quality_scores(
            candidate_embeddings, anchor, real_embeddings, real_labels, target_class
        )
        return composite_scores

    raise ValueError(f"Unknown filter type for scoring: {filter_type}")


# ============================================================================
# CORE: SCORE NORMALIZATION
# ============================================================================

def normalize_scores(
    raw_scores: np.ndarray,
    method: str = "minmax",
    temperature: float = 1.0,
    min_weight: float = 0.1
) -> np.ndarray:
    """
    Normalize raw filter scores to [min_weight, 1.0] for use as sample weights.

    Args:
        raw_scores: per-sample scores (any range)
        method: "minmax" | "sigmoid" | "rank"
        temperature: controls spread (lower = more peaked, higher = more uniform)
        min_weight: floor for weights (0.0 = allow near-zero influence)

    Returns:
        weights in [min_weight, 1.0]
    """
    n = len(raw_scores)
    if n == 0:
        return np.array([])

    # Edge case: all scores identical
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
        normalized = sigmoid((raw_scores - median) / (iqr + 1e-10))

    elif method == "rank":
        # Rank-based: purely ordinal
        order = raw_scores.argsort().argsort()  # rank indices
        normalized = order / (n - 1 + 1e-10)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Apply temperature: w^(1/temperature)
    # temperature < 1 → more peaked (extremes amplified)
    # temperature > 1 → more uniform (extremes compressed)
    if temperature != 1.0:
        normalized = np.power(np.clip(normalized, 1e-10, 1.0), 1.0 / temperature)

    # Scale to [min_weight, 1.0]
    weights = min_weight + weight_range * normalized

    return weights


# ============================================================================
# CORE: SOFT-WEIGHTED TRAINING
# ============================================================================

def run_soft_weight_experiment(
    dataset_name: str,
    train_texts: List[str],
    train_labels: List[str],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: List[str],
    filter_config: Dict,
    strategy: str,
    normalization: str,
    temperature: float,
    min_weight: float,
    baseline_f1: float,
    smote_f1: float,
    model: SentenceTransformer,
    provider,
    phase: int = 1,
) -> SoftWeightResult:
    """Run a single soft-weighting experiment configuration."""

    unique_classes = list(set(train_labels))
    train_labels_arr = np.array(train_labels)
    n_shot = int(dataset_name.split("_")[-1].replace("shot", ""))

    all_synthetic_emb = []
    all_synthetic_labels = []
    all_synthetic_weights = []

    for cls in unique_classes:
        cls_mask = train_labels_arr == cls
        cls_embeddings = train_embeddings[cls_mask]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        # Generate 3x oversampled candidates
        n_generate = int(N_SYNTHETIC_PER_CLASS * OVERSAMPLE_FACTOR)
        gen_emb, gen_texts = generate_llm_samples_cached(
            provider, dataset_name, cls, cls_texts, n_generate, n_shot, model
        )

        if len(gen_emb) == 0:
            continue

        # Compute raw scores for ALL candidates
        raw_scores = compute_scores_for_candidates(
            filter_config["type"], filter_config["params"],
            gen_emb, train_embeddings, train_labels_arr, cls
        )

        # Normalize per class
        weights = normalize_scores(raw_scores, normalization, temperature, min_weight)

        # Apply strategy
        if strategy == "keep_all":
            selected_emb = gen_emb
            selected_weights = weights
        elif strategy == "top_n":
            target_n = min(N_SYNTHETIC_PER_CLASS, len(gen_emb))
            top_idx = np.argsort(raw_scores)[-target_n:]
            selected_emb = gen_emb[top_idx]
            selected_weights = weights[top_idx]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_synthetic_emb.append(selected_emb)
        all_synthetic_labels.extend([cls] * len(selected_emb))
        all_synthetic_weights.append(selected_weights)

    # Assemble training data
    if not all_synthetic_emb:
        # Fallback: no synthetic data generated
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_embeddings, train_labels)
        soft_f1 = f1_score(test_labels, clf.predict(test_embeddings), average='macro')
        return SoftWeightResult(
            dataset=dataset_name, filter_name=filter_config["name"],
            filter_type=filter_config["type"], filter_params=filter_config["params"],
            strategy=strategy, normalization=normalization,
            temperature=temperature, min_weight=min_weight,
            baseline_f1=baseline_f1, smote_f1=smote_f1, soft_weighted_f1=soft_f1,
            delta_vs_baseline_pp=(soft_f1 - baseline_f1) * 100,
            delta_vs_smote_pp=(soft_f1 - smote_f1) * 100,
            n_real_samples=len(train_embeddings), n_synthetic_samples=0,
            weight_mean=0, weight_std=0, weight_min=0, weight_max=0, weight_median=0,
            timestamp=datetime.now().isoformat(), phase=phase,
        )

    synthetic_embeddings = np.vstack(all_synthetic_emb)
    synthetic_weights = np.concatenate(all_synthetic_weights)

    # Build full weight vector: real=1.0, synthetic=quality score
    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = list(train_labels) + all_synthetic_labels
    sample_weights = np.concatenate([
        np.ones(len(train_embeddings)),
        synthetic_weights
    ])

    # Train with sample weights
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_embeddings, aug_labels, sample_weight=sample_weights)
    pred = clf.predict(test_embeddings)
    soft_f1 = f1_score(test_labels, pred, average='macro')

    return SoftWeightResult(
        dataset=dataset_name,
        filter_name=filter_config["name"],
        filter_type=filter_config["type"],
        filter_params=filter_config["params"],
        strategy=strategy,
        normalization=normalization,
        temperature=temperature,
        min_weight=min_weight,
        baseline_f1=float(baseline_f1),
        smote_f1=float(smote_f1),
        soft_weighted_f1=float(soft_f1),
        delta_vs_baseline_pp=float((soft_f1 - baseline_f1) * 100),
        delta_vs_smote_pp=float((soft_f1 - smote_f1) * 100),
        n_real_samples=len(train_embeddings),
        n_synthetic_samples=len(synthetic_embeddings),
        weight_mean=float(synthetic_weights.mean()),
        weight_std=float(synthetic_weights.std()),
        weight_min=float(synthetic_weights.min()),
        weight_max=float(synthetic_weights.max()),
        weight_median=float(np.median(synthetic_weights)),
        timestamp=datetime.now().isoformat(),
        phase=phase,
    )


# ============================================================================
# RESULTS I/O
# ============================================================================

def save_results(results: List[SoftWeightResult], filename: str = "partial_results.json"):
    output_path = RESULTS_DIR / filename
    data = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(results),
        "results": [asdict(r) for r in results]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(results: List[SoftWeightResult]):
    print("\n" + "=" * 80)
    print("SOFT WEIGHTING EXPERIMENT SUMMARY")
    print("=" * 80)

    data = [asdict(r) for r in results]

    # 1. Strategy comparison: keep_all vs top_n
    print("\n" + "-" * 80)
    print("STRATEGY COMPARISON")
    print("-" * 80)
    print(f"\n{'Strategy':<12} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12} {'N':>6}")
    print("-" * 58)
    for strategy in ["keep_all", "top_n"]:
        deltas = [r['delta_vs_smote_pp'] for r in data if r['strategy'] == strategy]
        if deltas:
            mean = np.mean(deltas)
            std = np.std(deltas)
            win = 100 * sum(1 for x in deltas if x > 0) / len(deltas)
            print(f"{strategy:<12} {mean:>+15.2f}pp {std:>10.2f} {win:>11.1f}% {len(deltas):>6}")

    # 2. Filter ranking under soft weighting
    print("\n" + "-" * 80)
    print("FILTER RANKING (avg delta vs SMOTE)")
    print("-" * 80)

    filter_names = sorted(set(r['filter_name'] for r in data))
    filter_stats = []
    for fn in filter_names:
        deltas = [r['delta_vs_smote_pp'] for r in data if r['filter_name'] == fn]
        if deltas:
            filter_stats.append({
                'filter': fn,
                'mean': np.mean(deltas),
                'std': np.std(deltas),
                'win_rate': 100 * sum(1 for x in deltas if x > 0) / len(deltas),
                'n': len(deltas),
            })
    filter_stats.sort(key=lambda x: x['mean'], reverse=True)

    print(f"\n{'Filter':<16} {'Mean':>10} {'Std':>10} {'Win Rate':>12} {'N':>6}")
    print("-" * 58)
    for fs in filter_stats:
        print(f"{fs['filter']:<16} {fs['mean']:>+10.2f}pp {fs['std']:>10.2f} {fs['win_rate']:>11.1f}% {fs['n']:>6}")

    # 3. Soft weighting vs SMOTE per dataset
    print("\n" + "-" * 80)
    print("BEST SOFT WEIGHTING PER DATASET")
    print("-" * 80)

    datasets = sorted(set(r['dataset'] for r in data))
    for ds in datasets:
        ds_results = [r for r in data if r['dataset'] == ds]
        if not ds_results:
            continue
        best = max(ds_results, key=lambda x: x['delta_vs_smote_pp'])
        print(f"\n  {ds}:")
        print(f"    Best: {best['filter_name']} / {best['strategy']}")
        print(f"    F1: {best['soft_weighted_f1']:.4f}, vs SMOTE: {best['delta_vs_smote_pp']:+.2f}pp")
        print(f"    Weights: mean={best['weight_mean']:.3f}, std={best['weight_std']:.3f}")

    # 4. Weight distribution analysis
    print("\n" + "-" * 80)
    print("WEIGHT DISTRIBUTION SUMMARY")
    print("-" * 80)
    print(f"\n{'Filter':<16} {'Strategy':<10} {'W.Mean':>8} {'W.Std':>8} {'W.Min':>8} {'W.Max':>8}")
    print("-" * 62)
    for fn in filter_names:
        for strategy in ["keep_all", "top_n"]:
            subset = [r for r in data if r['filter_name'] == fn and r['strategy'] == strategy]
            if subset:
                w_mean = np.mean([r['weight_mean'] for r in subset])
                w_std = np.mean([r['weight_std'] for r in subset])
                w_min = np.mean([r['weight_min'] for r in subset])
                w_max = np.mean([r['weight_max'] for r in subset])
                print(f"{fn:<16} {strategy:<10} {w_mean:>8.3f} {w_std:>8.3f} {w_min:>8.3f} {w_max:>8.3f}")

    # 5. If Phase 2 results present: normalization/temperature sensitivity
    phase2_data = [r for r in data if r.get('phase', 1) == 2]
    if phase2_data:
        print("\n" + "-" * 80)
        print("NORMALIZATION METHOD COMPARISON (Phase 2)")
        print("-" * 80)
        print(f"\n{'Method':<10} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12}")
        print("-" * 50)
        for norm in NORMALIZATIONS:
            deltas = [r['delta_vs_smote_pp'] for r in phase2_data if r['normalization'] == norm]
            if deltas:
                print(f"{norm:<10} {np.mean(deltas):>+15.2f}pp {np.std(deltas):>10.2f} "
                      f"{100 * sum(1 for x in deltas if x > 0) / len(deltas):>11.1f}%")

        print("\n" + "-" * 80)
        print("TEMPERATURE SENSITIVITY (Phase 2)")
        print("-" * 80)
        print(f"\n{'Temp':<8} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12}")
        print("-" * 48)
        for temp in TEMPERATURES:
            deltas = [r['delta_vs_smote_pp'] for r in phase2_data if r['temperature'] == temp]
            if deltas:
                print(f"{temp:<8} {np.mean(deltas):>+15.2f}pp {np.std(deltas):>10.2f} "
                      f"{100 * sum(1 for x in deltas if x > 0) / len(deltas):>11.1f}%")

        print("\n" + "-" * 80)
        print("MIN WEIGHT SENSITIVITY (Phase 2)")
        print("-" * 80)
        print(f"\n{'MinW':<8} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12}")
        print("-" * 48)
        for mw in MIN_WEIGHTS:
            deltas = [r['delta_vs_smote_pp'] for r in phase2_data if r['min_weight'] == mw]
            if deltas:
                print(f"{mw:<8} {np.mean(deltas):>+15.2f}pp {np.std(deltas):>10.2f} "
                      f"{100 * sum(1 for x in deltas if x > 0) / len(deltas):>11.1f}%")

    # Save summary JSON
    summary = {
        "filter_ranking": filter_stats,
        "strategy_comparison": {
            s: {
                "mean": float(np.mean([r['delta_vs_smote_pp'] for r in data if r['strategy'] == s])),
                "win_rate": float(100 * sum(1 for r in data if r['strategy'] == s and r['delta_vs_smote_pp'] > 0) /
                                  max(1, sum(1 for r in data if r['strategy'] == s)))
            }
            for s in ["keep_all", "top_n"]
        },
        "best_per_dataset": {
            ds: asdict(max(
                [r for r in results if r.dataset == ds],
                key=lambda x: x.delta_vs_smote_pp
            ))
            for ds in datasets
            if any(r.dataset == ds for r in results)
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SOFT WEIGHTING EXPERIMENT")
    print("=" * 80)
    print(f"\nFilters: {[f['name'] for f in FILTER_CONFIGS]}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Datasets: {len(BENCHMARK_DATASETS)}")

    # Phase 1: screening
    phase1_total = len(FILTER_CONFIGS) * len(STRATEGIES) * len(BENCHMARK_DATASETS)
    print(f"\n--- Phase 1: Screening ({phase1_total} configs) ---")
    print(f"  norm={PHASE1_NORM}, temp={PHASE1_TEMP}, min_weight={PHASE1_MIN_WEIGHT}")

    print("\nLoading model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-2.0-flash")

    results: List[SoftWeightResult] = []
    experiment_count = 0

    # Pre-compute baselines per dataset
    dataset_cache = {}

    for dataset_name in BENCHMARK_DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")

        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        print(f"  Classes: {sorted(set(train_labels))}")
        print(f"  Distribution: {dict(Counter(train_labels))}")

        print("  Embedding texts...")
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        print("  Running baselines...")
        baseline_f1, smote_f1 = run_smote_baseline(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
        print(f"    Baseline F1: {baseline_f1:.4f}")
        print(f"    SMOTE F1:    {smote_f1:.4f}")

        dataset_cache[dataset_name] = {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "train_embeddings": train_embeddings,
            "test_embeddings": test_embeddings,
            "test_labels": test_labels,
            "baseline_f1": baseline_f1,
            "smote_f1": smote_f1,
        }

        # Phase 1 runs
        for filter_config in FILTER_CONFIGS:
            for strategy in STRATEGIES:
                experiment_count += 1
                config_str = f"{filter_config['name']}_{strategy}"
                print(f"\n  [{experiment_count}/{phase1_total}] {config_str}")

                try:
                    result = run_soft_weight_experiment(
                        dataset_name,
                        train_texts, train_labels, train_embeddings,
                        test_embeddings, test_labels,
                        filter_config, strategy,
                        PHASE1_NORM, PHASE1_TEMP, PHASE1_MIN_WEIGHT,
                        baseline_f1, smote_f1,
                        model, provider, phase=1,
                    )
                    results.append(result)

                    status = "BEATS SMOTE" if result.delta_vs_smote_pp > 0 else ""
                    print(f"    F1: {result.soft_weighted_f1:.4f}, vs SMOTE: {result.delta_vs_smote_pp:+.2f}pp {status}")
                    print(f"    Weights: mean={result.weight_mean:.3f}, std={result.weight_std:.3f}, "
                          f"range=[{result.weight_min:.3f}, {result.weight_max:.3f}]")

                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()

                if experiment_count % 10 == 0:
                    save_results(results)

    # Phase 1 report
    save_results(results, "phase1_results.json")
    if results:
        print("\n\n--- Phase 1 Results ---")
        generate_summary_report(results)

    # Phase 2: Identify top 3 filter+strategy combos and tune
    if len(results) >= 6:
        print("\n" + "=" * 80)
        print("PHASE 2: TUNING TOP PERFORMERS")
        print("=" * 80)

        # Rank filter+strategy combos by mean delta_vs_smote
        combo_stats = {}
        for r in results:
            key = (r.filter_name, r.strategy)
            if key not in combo_stats:
                combo_stats[key] = []
            combo_stats[key].append(r.delta_vs_smote_pp)

        ranked = sorted(combo_stats.items(), key=lambda x: np.mean(x[1]), reverse=True)
        # Skip "none" filter (uniform scores = no weighting effect)
        top_combos = [(k, v) for k, v in ranked if k[0] != "none"][:3]

        print(f"\nTop combos to tune:")
        for (fn, strat), deltas in top_combos:
            print(f"  {fn} / {strat}: mean={np.mean(deltas):+.2f}pp")

        phase2_configs = []
        for (fn, strat), _ in top_combos:
            fc = next(f for f in FILTER_CONFIGS if f["name"] == fn)
            for norm in NORMALIZATIONS:
                for temp in TEMPERATURES:
                    for mw in MIN_WEIGHTS:
                        # Skip Phase 1 default (already tested)
                        if norm == PHASE1_NORM and temp == PHASE1_TEMP and mw == PHASE1_MIN_WEIGHT:
                            continue
                        phase2_configs.append((fc, strat, norm, temp, mw))

        phase2_total = len(phase2_configs) * len(dataset_cache)
        print(f"\nPhase 2: {phase2_total} experiments")

        p2_count = 0
        for dataset_name, dc in dataset_cache.items():
            print(f"\n  Dataset: {dataset_name}")

            for fc, strat, norm, temp, mw in phase2_configs:
                p2_count += 1
                config_str = f"{fc['name']}_{strat}_{norm}_t{temp}_mw{mw}"

                if p2_count % 20 == 0 or p2_count <= 3:
                    print(f"    [{p2_count}/{phase2_total}] {config_str}")

                try:
                    result = run_soft_weight_experiment(
                        dataset_name,
                        dc["train_texts"], dc["train_labels"], dc["train_embeddings"],
                        dc["test_embeddings"], dc["test_labels"],
                        fc, strat, norm, temp, mw,
                        dc["baseline_f1"], dc["smote_f1"],
                        model, provider, phase=2,
                    )
                    results.append(result)

                except Exception as e:
                    print(f"    Error: {e}")

                if p2_count % 50 == 0:
                    save_results(results)

    # Final save and report
    save_results(results, "final_results.json")
    if results:
        print("\n\n--- Final Combined Results ---")
        generate_summary_report(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
