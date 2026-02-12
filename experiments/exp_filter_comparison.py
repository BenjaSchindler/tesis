#!/usr/bin/env python3
"""
Comprehensive Filter Comparison Experiment

Tests ALL combinations of filtering methods and generation configurations
on low-resource benchmark datasets to find the best approach.

Filter Types:
- none: No filtering (random selection)
- lof: LOF-based outlier detection
- combined: LOF + cosine similarity
- cascade: Multi-level filter cascade (levels 0-4)
- embedding_guided: Coverage + quality based selection

Generation Configs:
- LLM percentages: 5%, 25%, 50%, 100%
- N-shot: 10, 25, 50 examples in prompt

Always compares against SMOTE baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter, CombinedGeometricFilter
from core.filter_cascade import FilterCascade
from core.embedding_guided_sampler import EmbeddingGuidedSampler

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "filter_comparison"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm_generations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION MATRICES
# ============================================================================

FILTER_CONFIGS = [
    # No filter (baseline)
    {"filter_type": "none", "params": {}},

    # LOFFilter configurations
    {"filter_type": "lof", "params": {"n_neighbors": 5, "threshold": -0.5}},
    {"filter_type": "lof", "params": {"n_neighbors": 5, "threshold": 0.0}},
    {"filter_type": "lof", "params": {"n_neighbors": 10, "threshold": -0.5}},
    {"filter_type": "lof", "params": {"n_neighbors": 10, "threshold": 0.0}},
    {"filter_type": "lof", "params": {"n_neighbors": 20, "threshold": -0.3}},
    {"filter_type": "lof", "params": {"n_neighbors": 20, "threshold": 0.0}},

    # CombinedGeometricFilter configurations
    {"filter_type": "combined", "params": {"lof_threshold": 0.0, "sim_threshold": 0.3}},
    {"filter_type": "combined", "params": {"lof_threshold": 0.0, "sim_threshold": 0.5}},
    {"filter_type": "combined", "params": {"lof_threshold": 0.0, "sim_threshold": 0.7}},
    {"filter_type": "combined", "params": {"lof_threshold": -0.3, "sim_threshold": 0.5}},

    # FilterCascade configurations
    {"filter_type": "cascade", "params": {"filter_level": 0, "k_neighbors": 10}},
    {"filter_type": "cascade", "params": {"filter_level": 1, "k_neighbors": 10}},
    {"filter_type": "cascade", "params": {"filter_level": 2, "k_neighbors": 10}},
    {"filter_type": "cascade", "params": {"filter_level": 3, "k_neighbors": 10}},
    {"filter_type": "cascade", "params": {"filter_level": 4, "k_neighbors": 10}},

    # EmbeddingGuidedSampler configurations
    {"filter_type": "embedding_guided", "params": {"coverage_weight": 0.4, "quality_weight": 0.6, "min_distance_threshold": 0.1}},
    {"filter_type": "embedding_guided", "params": {"coverage_weight": 0.6, "quality_weight": 0.4, "min_distance_threshold": 0.1}},
    {"filter_type": "embedding_guided", "params": {"coverage_weight": 0.8, "quality_weight": 0.2, "min_distance_threshold": 0.1}},
    {"filter_type": "embedding_guided", "params": {"coverage_weight": 0.6, "quality_weight": 0.4, "min_distance_threshold": 0.05}},
]

LLM_PERCENTAGES = [5, 25, 50, 100]
N_SHOT_VALUES = [10, 25, 50]
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


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExperimentResult:
    """Single experiment result."""
    dataset: str
    filter_type: str
    filter_params: Dict[str, Any]
    llm_pct: int
    n_shot: int
    baseline_f1: float
    augmented_f1: float
    smote_only_f1: float
    delta_pp: float
    delta_vs_smote: float
    generation_stats: Dict[str, Dict]
    filter_stats: Dict[str, Any]
    timestamp: str


# ============================================================================
# FILTER FACTORY
# ============================================================================

class FilterFactory:
    """Factory for creating filter instances."""

    @staticmethod
    def create(filter_type: str, params: dict):
        """Create a filter instance."""
        if filter_type == "none":
            return None
        elif filter_type == "lof":
            return LOFFilter(**params)
        elif filter_type == "combined":
            return CombinedGeometricFilter(**params)
        elif filter_type == "cascade":
            return FilterCascade(**params)
        elif filter_type == "embedding_guided":
            return EmbeddingGuidedSampler(**params)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


# ============================================================================
# UNIFIED FILTER APPLICATION
# ============================================================================

def apply_filter(
    filter_obj,
    filter_type: str,
    candidate_embeddings: np.ndarray,
    candidate_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_class: str,
    target_n: int
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Apply any filter type and return filtered embeddings, texts, and stats.

    Unified interface for all filter types.
    """
    if len(candidate_embeddings) == 0:
        return np.array([]).reshape(0, 768), [], {"n_candidates": 0, "n_selected": 0}

    # No filter - random selection
    if filter_obj is None or filter_type == "none":
        n_select = min(target_n, len(candidate_embeddings))
        if n_select == len(candidate_embeddings):
            indices = np.arange(n_select)
        else:
            indices = np.random.choice(len(candidate_embeddings), n_select, replace=False)
        return (
            candidate_embeddings[indices],
            [candidate_texts[i] for i in indices],
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": n_select,
                "method": "random",
                "pct_accepted": 100.0
            }
        )

    # LOF Filter
    if filter_type == "lof":
        filtered_emb, mask, scores = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )

        # Select top target_n from filtered by score
        n_select = min(target_n, len(filtered_emb))
        if n_select == 0:
            return (
                np.array([]).reshape(0, candidate_embeddings.shape[1]),
                [],
                {
                    "n_candidates": len(candidate_embeddings),
                    "n_passed_filter": int(mask.sum()),
                    "n_selected": 0,
                    "pct_accepted": 100 * mask.sum() / max(1, len(mask)),
                    "mean_lof_score": float(scores.mean()) if len(scores) > 0 else 0.0,
                }
            )

        # Get indices that passed filter, then select top by LOF score
        passed_indices = np.where(mask)[0]
        passed_scores = scores[mask]
        top_local_idx = np.argsort(passed_scores)[-n_select:]
        top_global_idx = passed_indices[top_local_idx]

        return (
            candidate_embeddings[top_global_idx],
            [candidate_texts[i] for i in top_global_idx],
            {
                "n_candidates": len(candidate_embeddings),
                "n_passed_filter": int(mask.sum()),
                "n_selected": n_select,
                "pct_accepted": 100 * mask.sum() / len(mask),
                "mean_lof_score": float(scores.mean()),
            }
        )

    # Combined Geometric Filter
    if filter_type == "combined":
        filtered_emb, mask, stats = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_class
        )

        n_select = min(target_n, len(filtered_emb))
        if n_select == 0:
            stats["n_selected"] = 0
            return np.array([]).reshape(0, candidate_embeddings.shape[1]), [], stats

        # Random selection from filtered
        if n_select < len(filtered_emb):
            local_idx = np.random.choice(len(filtered_emb), n_select, replace=False)
        else:
            local_idx = np.arange(len(filtered_emb))

        global_idx = np.where(mask)[0][local_idx]
        stats["n_selected"] = n_select

        return (
            filtered_emb[local_idx],
            [candidate_texts[i] for i in global_idx],
            stats
        )

    # Filter Cascade
    if filter_type == "cascade":
        filtered_emb, avg_quality, details = filter_obj.filter_samples(
            candidates=candidate_embeddings,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=target_class,
            target_count=target_n
        )

        # Get indices by computing scores
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]
        anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)

        scores, _ = filter_obj.compute_quality_scores(
            candidate_embeddings, anchor, real_embeddings, real_labels, target_class
        )
        top_idx = np.argsort(scores)[-len(filtered_emb):]

        return (
            filtered_emb,
            [candidate_texts[i] for i in top_idx],
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(filtered_emb),
                "avg_quality": avg_quality,
                "filter_level": filter_obj.filter_level,
                "pct_accepted": 100 * len(filtered_emb) / max(1, len(candidate_embeddings)),
                **details
            }
        )

    # Embedding Guided Sampler
    if filter_type == "embedding_guided":
        class_mask = real_labels == target_class
        class_embs = real_embeddings[class_mask]

        selected_texts, selected_embs, scores = filter_obj.select_samples(
            candidate_embeddings,
            candidate_texts,
            class_embs,
            target_n,
            class_label=target_class
        )

        avg_coverage = np.mean([s.coverage_gain for s in scores]) if scores else 0
        avg_quality = np.mean([s.quality_score for s in scores]) if scores else 0

        return (
            selected_embs if len(selected_embs) > 0 else np.array([]).reshape(0, candidate_embeddings.shape[1]),
            selected_texts,
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(selected_texts),
                "avg_coverage_gain": avg_coverage,
                "avg_quality_score": avg_quality,
                "pct_accepted": 100 * len(selected_texts) / max(1, len(candidate_embeddings)),
            }
        )

    raise ValueError(f"Unknown filter type: {filter_type}")


# ============================================================================
# LLM GENERATION WITH CACHING
# ============================================================================

def get_cache_key(dataset: str, class_name: str, n_shot: int, n_generate: int) -> str:
    """Generate cache key for LLM generations."""
    key_str = f"{dataset}_{class_name}_{n_shot}_{n_generate}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def create_prompt(class_name: str, examples: List[str], n_generate: int, n_shot: int) -> str:
    """Create generation prompt with n_shot examples."""
    # Use up to n_shot examples
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
    """Generate LLM samples with caching."""

    cache_key = get_cache_key(dataset, class_name, n_shot, n_generate)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("texts"):
            embeddings = model.encode(cached["texts"], show_progress_bar=False)
            return embeddings, cached["texts"]

    # Generate new
    prompt = create_prompt(class_name, class_texts, n_generate, n_shot)

    try:
        messages = [{"role": "user", "content": prompt}]
        response, _ = provider.generate(messages, temperature=0.8, max_tokens=4000)

        # Parse response
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        generated = []
        for line in lines:
            clean = line.lstrip('0123456789.-):* ')
            if len(clean) > 10:
                generated.append(clean)

        if not generated:
            return np.array([]).reshape(0, 768), []

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                "dataset": dataset,
                "class_name": class_name,
                "n_shot": n_shot,
                "n_generate": n_generate,
                "texts": generated,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        # Embed
        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated

    except Exception as e:
        print(f"        Error generating: {e}")
        return np.array([]).reshape(0, 768), []


# ============================================================================
# SMOTE GENERATION
# ============================================================================

def generate_smote_samples(
    real_embeddings: np.ndarray,
    n_generate: int,
    k_neighbors: int = 5
) -> np.ndarray:
    """Generate SMOTE samples for a single class."""
    if len(real_embeddings) < 2:
        return np.array([]).reshape(0, real_embeddings.shape[1])

    k = min(k_neighbors, len(real_embeddings) - 1)
    if k < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])

    # Create dummy binary problem
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


# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def load_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a benchmark dataset."""
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return (
        data['train_texts'],
        data['train_labels'],
        data['test_texts'],
        data['test_labels']
    )


def run_smote_baseline(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str]
) -> Tuple[float, float, float]:
    """Run SMOTE-only baseline and return (baseline_f1, smote_f1, delta_pp)."""

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

    # Combine
    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])

    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = list(train_labels) + all_synthetic_labels

    # Train and evaluate
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_augmented = LogisticRegression(max_iter=1000, random_state=42)

    clf_baseline.fit(train_embeddings, train_labels)
    baseline_pred = clf_baseline.predict(test_embeddings)
    baseline_f1 = f1_score(test_labels, baseline_pred, average='macro')

    clf_augmented.fit(aug_embeddings, aug_labels)
    aug_pred = clf_augmented.predict(test_embeddings)
    smote_f1 = f1_score(test_labels, aug_pred, average='macro')

    delta_pp = (smote_f1 - baseline_f1) * 100

    return baseline_f1, smote_f1, delta_pp


def run_single_experiment(
    dataset_name: str,
    train_texts: List[str],
    train_labels: List[str],
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: List[str],
    filter_config: Dict,
    llm_pct: int,
    n_shot: int,
    baseline_f1: float,
    smote_only_f1: float,
    model: SentenceTransformer,
    provider
) -> ExperimentResult:
    """Run a single experiment configuration."""

    unique_classes = list(set(train_labels))
    train_labels_arr = np.array(train_labels)

    # Calculate LLM vs SMOTE split
    n_llm_per_class = int(N_SYNTHETIC_PER_CLASS * llm_pct / 100)
    n_smote_per_class = N_SYNTHETIC_PER_CLASS - n_llm_per_class

    # Create filter
    filter_obj = FilterFactory.create(filter_config["filter_type"], filter_config["params"])

    all_synthetic_emb = []
    all_synthetic_labels = []
    generation_stats = {}
    filter_stats_all = {}

    for cls in unique_classes:
        cls_mask = train_labels_arr == cls
        cls_embeddings = train_embeddings[cls_mask]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        llm_emb = np.array([]).reshape(0, train_embeddings.shape[1])

        # Generate LLM samples
        if n_llm_per_class > 0 and llm_pct > 0:
            n_generate = int(n_llm_per_class * OVERSAMPLE_FACTOR)

            gen_emb, gen_texts = generate_llm_samples_cached(
                provider, dataset_name, cls, cls_texts, n_generate, n_shot, model
            )

            if len(gen_emb) > 0:
                # Apply filter
                filtered_emb, filtered_texts, filter_stats = apply_filter(
                    filter_obj, filter_config["filter_type"],
                    gen_emb, gen_texts,
                    train_embeddings, train_labels_arr,
                    cls, n_llm_per_class
                )

                llm_emb = filtered_emb
                filter_stats_all[cls] = filter_stats

        # Generate SMOTE samples
        smote_emb = np.array([]).reshape(0, train_embeddings.shape[1])
        if n_smote_per_class > 0:
            # Use LLM samples as additional anchors
            if len(llm_emb) > 0:
                base_emb = np.vstack([cls_embeddings, llm_emb])
            else:
                base_emb = cls_embeddings

            smote_emb = generate_smote_samples(base_emb, n_smote_per_class)

        # Combine
        parts = []
        if len(llm_emb) > 0:
            parts.append(llm_emb)
        if len(smote_emb) > 0:
            parts.append(smote_emb)

        if parts:
            combined = np.vstack(parts)
            all_synthetic_emb.append(combined)
            all_synthetic_labels.extend([cls] * len(combined))

        generation_stats[cls] = {
            'llm_samples': len(llm_emb),
            'smote_samples': len(smote_emb),
            'total': len(llm_emb) + len(smote_emb)
        }

    # Combine synthetic data
    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])

    # Train and evaluate
    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = list(train_labels) + all_synthetic_labels

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_embeddings, aug_labels)
    pred = clf.predict(test_embeddings)
    augmented_f1 = f1_score(test_labels, pred, average='macro')

    delta_pp = (augmented_f1 - baseline_f1) * 100
    delta_vs_smote = (augmented_f1 - smote_only_f1) * 100

    return ExperimentResult(
        dataset=dataset_name,
        filter_type=filter_config["filter_type"],
        filter_params=filter_config["params"],
        llm_pct=llm_pct,
        n_shot=n_shot,
        baseline_f1=float(baseline_f1),
        augmented_f1=float(augmented_f1),
        smote_only_f1=float(smote_only_f1),
        delta_pp=float(delta_pp),
        delta_vs_smote=float(delta_vs_smote),
        generation_stats=generation_stats,
        filter_stats=filter_stats_all,
        timestamp=datetime.now().isoformat()
    )


def save_results(results: List[ExperimentResult], partial: bool = True):
    """Save results incrementally."""
    filename = "partial_results.json" if partial else "final_results.json"
    output_path = RESULTS_DIR / filename

    data = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(results),
        "results": [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def generate_summary_report(results: List[ExperimentResult]):
    """Generate comprehensive summary tables."""

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Convert to simple dict list for easier processing
    data = [asdict(r) for r in results]

    # 1. Best filter per dataset
    print("\n" + "-"*80)
    print("BEST FILTER PER DATASET")
    print("-"*80)

    datasets = set(r['dataset'] for r in data)

    for dataset in sorted(datasets):
        dataset_results = [r for r in data if r['dataset'] == dataset]

        if not dataset_results:
            continue

        best = max(dataset_results, key=lambda x: x['delta_vs_smote'])

        params_str = str(best['filter_params'])[:30]
        print(f"\n{dataset}:")
        print(f"  Best: {best['filter_type']} ({params_str})")
        print(f"  LLM%: {best['llm_pct']}%, N-shot: {best['n_shot']}")
        print(f"  F1: {best['augmented_f1']:.4f}, vs SMOTE: {best['delta_vs_smote']:+.2f}pp")

    # 2. Filter type ranking
    print("\n" + "-"*80)
    print("FILTER TYPE RANKING (avg delta vs SMOTE)")
    print("-"*80)

    filter_types = set(r['filter_type'] for r in data)

    filter_stats = []
    for ft in filter_types:
        ft_results = [r['delta_vs_smote'] for r in data if r['filter_type'] == ft]
        if ft_results:
            filter_stats.append({
                'filter_type': ft,
                'mean': np.mean(ft_results),
                'std': np.std(ft_results),
                'n': len(ft_results),
                'win_rate': 100 * sum(1 for x in ft_results if x > 0) / len(ft_results)
            })

    filter_stats.sort(key=lambda x: x['mean'], reverse=True)

    print(f"\n{'Filter Type':<20} {'Mean':>10} {'Std':>10} {'Win Rate':>12} {'N':>8}")
    print("-" * 60)
    for fs in filter_stats:
        print(f"{fs['filter_type']:<20} {fs['mean']:>+10.2f}pp {fs['std']:>10.2f} {fs['win_rate']:>11.1f}% {fs['n']:>8}")

    # 3. LLM percentage impact
    print("\n" + "-"*80)
    print("LLM PERCENTAGE IMPACT")
    print("-"*80)

    llm_pcts = sorted(set(r['llm_pct'] for r in data))

    print(f"\n{'LLM%':<10} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12}")
    print("-" * 50)
    for pct in llm_pcts:
        pct_results = [r['delta_vs_smote'] for r in data if r['llm_pct'] == pct]
        if pct_results:
            mean = np.mean(pct_results)
            std = np.std(pct_results)
            win_rate = 100 * sum(1 for x in pct_results if x > 0) / len(pct_results)
            print(f"{pct:<10} {mean:>+15.2f}pp {std:>10.2f} {win_rate:>11.1f}%")

    # 4. N-shot impact
    print("\n" + "-"*80)
    print("N-SHOT IMPACT")
    print("-"*80)

    n_shots = sorted(set(r['n_shot'] for r in data))

    print(f"\n{'N-shot':<10} {'Mean vs SMOTE':>15} {'Std':>10} {'Win Rate':>12}")
    print("-" * 50)
    for ns in n_shots:
        ns_results = [r['delta_vs_smote'] for r in data if r['n_shot'] == ns]
        if ns_results:
            mean = np.mean(ns_results)
            std = np.std(ns_results)
            win_rate = 100 * sum(1 for x in ns_results if x > 0) / len(ns_results)
            print(f"{ns:<10} {mean:>+15.2f}pp {std:>10.2f} {win_rate:>11.1f}%")

    # 5. Top 10 configurations
    print("\n" + "-"*80)
    print("TOP 10 CONFIGURATIONS (by delta vs SMOTE)")
    print("-"*80)

    sorted_results = sorted(data, key=lambda x: x['delta_vs_smote'], reverse=True)[:10]

    for i, r in enumerate(sorted_results, 1):
        print(f"\n{i}. {r['dataset']}")
        print(f"   Filter: {r['filter_type']} {r['filter_params']}")
        print(f"   LLM: {r['llm_pct']}%, N-shot: {r['n_shot']}")
        print(f"   F1: {r['augmented_f1']:.4f}, vs SMOTE: {r['delta_vs_smote']:+.2f}pp")

    # Save summary to JSON
    summary_data = {
        "filter_type_ranking": filter_stats,
        "llm_pct_stats": [
            {
                "llm_pct": pct,
                "mean": float(np.mean([r['delta_vs_smote'] for r in data if r['llm_pct'] == pct])),
                "std": float(np.std([r['delta_vs_smote'] for r in data if r['llm_pct'] == pct]))
            }
            for pct in llm_pcts
        ],
        "n_shot_stats": [
            {
                "n_shot": ns,
                "mean": float(np.mean([r['delta_vs_smote'] for r in data if r['n_shot'] == ns])),
                "std": float(np.std([r['delta_vs_smote'] for r in data if r['n_shot'] == ns]))
            }
            for ns in n_shots
        ],
        "top_10": sorted_results
    }

    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"\n\nSummary saved to: {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("="*80)
    print("COMPREHENSIVE FILTER COMPARISON EXPERIMENT")
    print("="*80)
    print(f"\nFilter configs: {len(FILTER_CONFIGS)}")
    print(f"LLM percentages: {LLM_PERCENTAGES}")
    print(f"N-shot values: {N_SHOT_VALUES}")
    print(f"Datasets: {len(BENCHMARK_DATASETS)}")

    total_experiments = len(FILTER_CONFIGS) * len(LLM_PERCENTAGES) * len(N_SHOT_VALUES) * len(BENCHMARK_DATASETS)
    print(f"\nTotal experiments: {total_experiments}")

    # Initialize
    print("\nLoading model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-2.0-flash")

    results = []
    experiment_count = 0

    for dataset_name in BENCHMARK_DATASETS:
        # Check if dataset exists
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")

        # Load data
        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_name)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        print(f"  Classes: {set(train_labels)}")
        print(f"  Distribution: {dict(Counter(train_labels))}")

        # Embed
        print("  Embedding texts...")
        train_embeddings = model.encode(train_texts, show_progress_bar=False)
        test_embeddings = model.encode(test_texts, show_progress_bar=False)

        # Run SMOTE baseline
        print("  Running SMOTE baseline...")
        baseline_f1, smote_only_f1, smote_delta = run_smote_baseline(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
        print(f"    Baseline F1: {baseline_f1:.4f}")
        print(f"    SMOTE F1: {smote_only_f1:.4f}")
        print(f"    SMOTE delta: {smote_delta:+.2f}pp")

        # Run all configurations
        for filter_config in FILTER_CONFIGS:
            for llm_pct in LLM_PERCENTAGES:
                for n_shot in N_SHOT_VALUES:
                    experiment_count += 1

                    config_name = f"{filter_config['filter_type']}_{llm_pct}%_{n_shot}shot"
                    print(f"\n  [{experiment_count}/{total_experiments}] {config_name}")

                    try:
                        result = run_single_experiment(
                            dataset_name,
                            train_texts, train_labels, train_embeddings,
                            test_embeddings, test_labels,
                            filter_config, llm_pct, n_shot,
                            baseline_f1, smote_only_f1,
                            model, provider
                        )

                        results.append(result)

                        status = "BEATS SMOTE" if result.delta_vs_smote > 0 else ""
                        print(f"    F1: {result.augmented_f1:.4f}, vs SMOTE: {result.delta_vs_smote:+.2f}pp {status}")

                    except Exception as e:
                        print(f"    Error: {e}")
                        import traceback
                        traceback.print_exc()

                    # Save checkpoint every 10 experiments
                    if experiment_count % 10 == 0:
                        save_results(results, partial=True)

    # Final save
    save_results(results, partial=False)

    # Generate summary
    if results:
        generate_summary_report(results)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
