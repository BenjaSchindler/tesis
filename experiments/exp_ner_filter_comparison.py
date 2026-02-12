#!/usr/bin/env python3
"""
NER Geometric Filter Comparison Experiment

Tests whether geometric filtering of LLM-generated synthetic NER data
improves Named Entity Recognition performance in low-resource settings.

Research question: Does the "simple beats complex" finding from text
classification (cascade L1 > all others) hold for NER?

Configuration: 3 datasets × 3 n-shot × 7 filters = 63 configs

Baselines:
- No augmentation (real data only)
- Unfiltered LLM augmentation (none filter)

Metrics:
- Entity-level F1 (strict match, macro average)
- Per-entity-type F1
- LLM calls needed, acceptance rates, efficiency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter

from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter, CombinedGeometricFilter
from core.filter_cascade import FilterCascade
from core.embedding_guided_sampler import EmbeddingGuidedSampler
from core.ner_generator import generate_ner_batch
from core.ner_filter_adapter import assign_dominant_entity_types, apply_ner_filter
from core.ner_evaluator import (
    evaluate_ner_augmentation,
    compute_ner_baseline,
    evaluate_with_cv,
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks" / "ner"
RESULTS_DIR = PROJECT_ROOT / "results" / "ner_filter_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SAMPLES_PER_TYPE = 50
MAX_LLM_CALLS_PER_TYPE = 100
BATCH_SIZE = 15
N_EXAMPLES = 10  # Few-shot examples in prompt
NER_EPOCHS = 30
N_EVAL_SEEDS = 3  # Multiple seeds for stable NER results
EARLY_STOP_ACCEPTANCE = 0.02

# Filters to compare (same as classification experiment)
FILTERS = [
    {"name": "none", "type": "none", "params": {}},
    {"name": "lof_relaxed", "type": "lof", "params": {"n_neighbors": 10, "threshold": -0.5}},
    {"name": "lof_strict", "type": "lof", "params": {"n_neighbors": 10, "threshold": 0.0}},
    {"name": "cascade_l1", "type": "cascade", "params": {"filter_level": 1, "k_neighbors": 10}},
    {"name": "cascade_l2", "type": "cascade", "params": {"filter_level": 2, "k_neighbors": 10}},
    {"name": "cascade_full", "type": "cascade", "params": {"filter_level": 4, "k_neighbors": 10}},
    {"name": "combined", "type": "combined", "params": {"lof_threshold": 0.0, "sim_threshold": 0.5}},
]

# Datasets to test
DATASETS = [
    "multinerd_10shot",
    "multinerd_25shot",
    "multinerd_50shot",
    "wikiann_10shot",
    "wikiann_25shot",
    "wikiann_50shot",
    "fewnerd_10shot",
    "fewnerd_25shot",
    "fewnerd_50shot",
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NERGenerationResult:
    """Result of generating NER samples for one entity type."""
    valid_sentences: List[Dict]
    valid_embeddings: np.ndarray
    valid_texts: List[str]
    llm_calls: int
    total_generated: int
    acceptance_rate: float
    status: str

@dataclass
class NERFilterResult:
    """Result for one filter on one dataset."""
    filter_name: str
    filter_type: str
    filter_params: Dict
    mean_f1: float
    std_f1: float
    f1_delta_vs_baseline: float
    total_llm_calls: int
    avg_acceptance_rate: float
    efficiency_score: float
    per_type_stats: Dict[str, Dict]
    per_type_f1: Dict[str, float]
    all_reached_target: bool

@dataclass
class NERDatasetResult:
    """Results for one dataset."""
    dataset: str
    entity_types: List[str]
    n_train_sentences: int
    baseline_mean_f1: float
    baseline_std_f1: float
    baseline_per_type_f1: Dict[str, float]
    filter_results: List[NERFilterResult]
    timestamp: str


# ============================================================================
# FILTER FACTORY
# ============================================================================

def create_filter(filter_config: Dict):
    """Create a filter instance from config."""
    filter_type = filter_config["type"]
    params = filter_config["params"]

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
# NER DATASET LOADING
# ============================================================================

def load_ner_dataset(dataset_name: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Load an NER benchmark dataset.

    Returns:
        (train_sentences, test_sentences, entity_types)
    """
    path = DATA_DIR / f"{dataset_name}.json"
    with open(path) as f:
        data = json.load(f)
    return (
        data["train_sentences"],
        data["test_sentences"],
        data["entity_types"]
    )


def get_dataset_base_name(dataset_name: str) -> str:
    """Extract base dataset name (without shot suffix)."""
    for base in ["multinerd", "conll2003", "wikiann", "fewnerd"]:
        if dataset_name.startswith(base):
            return base
    return dataset_name


# ============================================================================
# CORE: GENERATE UNTIL N VALID FOR NER
# ============================================================================

def generate_ner_until_n_valid(
    provider,
    filter_obj,
    filter_type: str,
    target_n: int,
    entity_type: str,
    real_sentences: List[Dict],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    embed_model,
    dataset_name: str
) -> NERGenerationResult:
    """Generate NER sentences iteratively until n valid samples pass the filter.

    Uses pool-based approach: generate batches, filter entire pool.
    """
    base_dataset = get_dataset_base_name(dataset_name)

    # Special case: no filter — still need multiple calls since NER parsing
    # has ~30% success rate (not all LLM lines parse as valid annotations)
    if filter_type == "none" or filter_obj is None:
        all_parsed = []
        all_embs = []
        all_texts = []
        llm_calls = 0

        while len(all_parsed) < target_n and llm_calls < MAX_LLM_CALLS_PER_TYPE:
            parsed, embs, texts = generate_ner_batch(
                provider, entity_type, real_sentences,
                BATCH_SIZE, embed_model, base_dataset, N_EXAMPLES
            )
            llm_calls += 1
            if len(parsed) > 0:
                all_parsed.extend(parsed)
                all_embs.append(embs)
                all_texts.extend(texts)

        if all_embs:
            combined_embs = np.vstack(all_embs)
        else:
            combined_embs = np.array([]).reshape(0, 768)

        # Truncate to target_n (random selection)
        if len(all_parsed) > target_n:
            indices = np.random.choice(len(all_parsed), target_n, replace=False)
            all_parsed = [all_parsed[i] for i in indices]
            combined_embs = combined_embs[indices]
            all_texts = [all_texts[i] for i in indices]

        return NERGenerationResult(
            valid_sentences=all_parsed,
            valid_embeddings=combined_embs,
            valid_texts=all_texts,
            llm_calls=llm_calls,
            total_generated=len(all_parsed),
            acceptance_rate=1.0,
            status="SUCCESS" if len(all_parsed) >= target_n else "INSUFFICIENT"
        )

    # Filtered generation: iterative pool approach
    pool_sentences = []
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while True:
        # Generate batch
        parsed, batch_embs, batch_texts = generate_ner_batch(
            provider, entity_type, real_sentences,
            BATCH_SIZE, embed_model, base_dataset, N_EXAMPLES
        )
        llm_calls += 1

        if len(parsed) > 0:
            pool_sentences.extend(parsed)
            pool_embeddings.append(batch_embs)
            pool_texts.extend(batch_texts)

        if not pool_embeddings:
            if llm_calls >= MAX_LLM_CALLS_PER_TYPE:
                return NERGenerationResult(
                    valid_sentences=[],
                    valid_embeddings=np.array([]).reshape(0, 768),
                    valid_texts=[],
                    llm_calls=llm_calls,
                    total_generated=0,
                    acceptance_rate=0.0,
                    status="MAX_CALLS_REACHED"
                )
            continue

        # Combine pool
        pool_arr = np.vstack(pool_embeddings)

        # Apply filter
        filtered_sents, filtered_embs, filtered_texts, stats = apply_ner_filter(
            filter_obj, filter_type,
            pool_sentences, pool_arr, pool_texts,
            real_embeddings, real_labels,
            entity_type, target_n
        )

        n_valid = len(filtered_sents)
        acceptance_rate = n_valid / len(pool_arr) if len(pool_arr) > 0 else 0

        # Check success
        if n_valid >= target_n:
            return NERGenerationResult(
                valid_sentences=filtered_sents[:target_n],
                valid_embeddings=filtered_embs[:target_n],
                valid_texts=filtered_texts[:target_n],
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="SUCCESS"
            )

        # Check max calls
        if llm_calls >= MAX_LLM_CALLS_PER_TYPE:
            return NERGenerationResult(
                valid_sentences=filtered_sents,
                valid_embeddings=filtered_embs,
                valid_texts=filtered_texts,
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="MAX_CALLS_REACHED"
            )

        # Early stop on very low acceptance
        if len(pool_arr) > 50 and acceptance_rate < EARLY_STOP_ACCEPTANCE:
            return NERGenerationResult(
                valid_sentences=filtered_sents,
                valid_embeddings=filtered_embs,
                valid_texts=filtered_texts,
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="LOW_ACCEPTANCE"
            )

        if llm_calls % 5 == 0:
            print(f"          ... {llm_calls} calls, {n_valid}/{target_n} valid ({acceptance_rate*100:.1f}%)")


# ============================================================================
# EXPERIMENT PER FILTER
# ============================================================================

def run_filter_experiment(
    filter_config: Dict,
    train_sentences: List[Dict],
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_sentences: List[Dict],
    entity_types: List[str],
    baseline_metrics: Dict,
    provider,
    embed_model,
    dataset_name: str
) -> NERFilterResult:
    """Run NER experiment for one filter configuration."""
    filter_name = filter_config["name"]
    filter_type = filter_config["type"]
    filter_params = filter_config["params"]

    print(f"    Filter: {filter_name}")

    filter_obj = create_filter(filter_config)

    all_synthetic_sents = []
    total_llm_calls = 0
    per_type_stats = {}
    all_reached_target = True

    for etype in entity_types:
        # Get sentences containing this entity type
        type_sentences = [
            s for s in train_sentences
            if any(t.startswith(f"B-{etype}") for t in s["ner_tags"])
        ]

        print(f"      Entity type: {etype} ({len(type_sentences)} real sentences)")

        result = generate_ner_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            filter_type=filter_type,
            target_n=TARGET_SAMPLES_PER_TYPE,
            entity_type=etype,
            real_sentences=type_sentences,
            real_embeddings=train_embeddings,
            real_labels=train_labels,
            embed_model=embed_model,
            dataset_name=dataset_name
        )

        print(f"        -> {len(result.valid_sentences)}/{TARGET_SAMPLES_PER_TYPE} valid, "
              f"{result.llm_calls} calls, {result.acceptance_rate*100:.1f}% accept, "
              f"status: {result.status}")

        all_synthetic_sents.extend(result.valid_sentences)
        total_llm_calls += result.llm_calls

        per_type_stats[etype] = {
            "n_valid": len(result.valid_sentences),
            "llm_calls": result.llm_calls,
            "total_generated": result.total_generated,
            "acceptance_rate": result.acceptance_rate,
            "status": result.status
        }

        if result.status != "SUCCESS":
            all_reached_target = False

    # Evaluate with multiple seeds
    print(f"      Evaluating ({N_EVAL_SEEDS} seeds)...")
    if all_synthetic_sents:
        cv_results = evaluate_with_cv(
            train_sentences, all_synthetic_sents, test_sentences,
            n_folds=N_EVAL_SEEDS, n_epochs=NER_EPOCHS
        )
        mean_f1 = cv_results["mean_f1"]
        std_f1 = cv_results["std_f1"]

        # Per-type F1 from last run
        per_type_f1 = {}
        if cv_results["runs"]:
            last_run = cv_results["runs"][-1]
            for etype, metrics in last_run.get("per_type", {}).items():
                per_type_f1[etype] = metrics.get("f1", 0.0)
    else:
        mean_f1 = baseline_metrics["mean_f1"]
        std_f1 = baseline_metrics["std_f1"]
        per_type_f1 = {}

    f1_delta = (mean_f1 - baseline_metrics["mean_f1"]) * 100
    avg_acceptance = np.mean([s["acceptance_rate"] for s in per_type_stats.values()])
    efficiency = (f1_delta / max(1, total_llm_calls)) * 1000

    print(f"      => F1: {mean_f1:.4f} +/- {std_f1:.4f} ({f1_delta:+.2f}pp vs baseline), "
          f"Calls: {total_llm_calls}, Efficiency: {efficiency:.2f}")

    return NERFilterResult(
        filter_name=filter_name,
        filter_type=filter_type,
        filter_params=filter_params,
        mean_f1=float(mean_f1),
        std_f1=float(std_f1),
        f1_delta_vs_baseline=float(f1_delta),
        total_llm_calls=total_llm_calls,
        avg_acceptance_rate=float(avg_acceptance),
        efficiency_score=float(efficiency),
        per_type_stats=per_type_stats,
        per_type_f1=per_type_f1,
        all_reached_target=all_reached_target
    )


# ============================================================================
# DATASET-LEVEL EXPERIMENT
# ============================================================================

def run_dataset_experiment(
    dataset_name: str,
    provider,
    embed_model
) -> NERDatasetResult:
    """Run all filter experiments on one NER dataset."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    train_sentences, test_sentences, entity_types = load_ner_dataset(dataset_name)
    print(f"  Train: {len(train_sentences)} sentences, Test: {len(test_sentences)} sentences")
    print(f"  Entity types: {entity_types}")

    # Count entities per type
    for etype in entity_types:
        count = sum(
            1 for s in train_sentences
            if any(t.startswith(f"B-{etype}") for t in s["ner_tags"])
        )
        print(f"    {etype}: {count} sentences")

    # Embed all training sentences
    print("  Embedding sentences...")
    train_texts = [" ".join(s["tokens"]) for s in train_sentences]
    train_embeddings = embed_model.encode(train_texts, show_progress_bar=False)

    # Assign dominant entity type labels
    train_labels = assign_dominant_entity_types(train_sentences)
    print(f"  Label distribution: {dict(Counter(train_labels))}")

    # Compute baseline (no augmentation)
    print("  Computing baseline...")
    baseline_cv = evaluate_with_cv(
        train_sentences, [], test_sentences,
        n_folds=N_EVAL_SEEDS, n_epochs=NER_EPOCHS
    )
    baseline_mean_f1 = baseline_cv["mean_f1"]
    baseline_std_f1 = baseline_cv["std_f1"]

    baseline_per_type_f1 = {}
    if baseline_cv["runs"]:
        last = baseline_cv["runs"][-1]
        for etype, metrics in last.get("per_type", {}).items():
            baseline_per_type_f1[etype] = metrics.get("f1", 0.0)

    print(f"  Baseline F1: {baseline_mean_f1:.4f} +/- {baseline_std_f1:.4f}")

    baseline_metrics = {"mean_f1": baseline_mean_f1, "std_f1": baseline_std_f1}

    # Run each filter
    filter_results = []
    for filter_config in FILTERS:
        try:
            result = run_filter_experiment(
                filter_config,
                train_sentences, train_embeddings, train_labels,
                test_sentences, entity_types, baseline_metrics,
                provider, embed_model, dataset_name
            )
            filter_results.append(result)
        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback; traceback.print_exc()

    return NERDatasetResult(
        dataset=dataset_name,
        entity_types=entity_types,
        n_train_sentences=len(train_sentences),
        baseline_mean_f1=float(baseline_mean_f1),
        baseline_std_f1=float(baseline_std_f1),
        baseline_per_type_f1=baseline_per_type_f1,
        filter_results=filter_results,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(all_results: List[NERDatasetResult]):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 80)
    print("NER GEOMETRIC FILTER EXPERIMENT - SUMMARY")
    print("=" * 80)
    print(f"Target samples per entity type: {TARGET_SAMPLES_PER_TYPE}")
    print(f"Datasets tested: {len(all_results)}")
    print(f"Evaluation seeds: {N_EVAL_SEEDS}")

    # Aggregate by filter
    filter_agg = {}
    for fc in FILTERS:
        filter_agg[fc["name"]] = {
            "f1_deltas": [], "llm_calls": [], "acceptance_rates": [],
            "efficiencies": [], "mean_f1s": [], "success_count": 0, "total_count": 0
        }

    for dr in all_results:
        for fr in dr.filter_results:
            agg = filter_agg[fr.filter_name]
            agg["f1_deltas"].append(fr.f1_delta_vs_baseline)
            agg["llm_calls"].append(fr.total_llm_calls)
            agg["acceptance_rates"].append(fr.avg_acceptance_rate)
            agg["efficiencies"].append(fr.efficiency_score)
            agg["mean_f1s"].append(fr.mean_f1)
            agg["total_count"] += 1
            if fr.all_reached_target:
                agg["success_count"] += 1

    # Print comparison table
    print("\n" + "-" * 80)
    print("FILTER RANKING (averaged across all NER datasets)")
    print("-" * 80)
    print(f"\n{'Filter':<15} {'F1 Delta':>10} {'Mean F1':>10} {'LLM Calls':>12} {'Accept%':>10} {'Efficiency':>12}")
    print("-" * 70)

    sorted_filters = sorted(
        filter_agg.items(),
        key=lambda x: np.mean(x[1]["f1_deltas"]) if x[1]["f1_deltas"] else -999,
        reverse=True
    )

    for fname, agg in sorted_filters:
        if not agg["f1_deltas"]:
            continue
        print(f"{fname:<15} {np.mean(agg['f1_deltas']):>+10.2f}pp "
              f"{np.mean(agg['mean_f1s']):>10.4f} "
              f"{np.mean(agg['llm_calls']):>12.0f} "
              f"{np.mean(agg['acceptance_rates'])*100:>9.1f}% "
              f"{np.mean(agg['efficiencies']):>12.2f}")

    # Per-dataset results
    print("\n" + "-" * 80)
    print("PER-DATASET RESULTS")
    print("-" * 80)

    for dr in all_results:
        print(f"\n{dr.dataset}:")
        print(f"  Baseline F1: {dr.baseline_mean_f1:.4f} +/- {dr.baseline_std_f1:.4f}")
        print(f"  {'Filter':<15} {'F1':>10} {'Delta':>10} {'Calls':>8}")

        sorted_results = sorted(dr.filter_results, key=lambda x: x.f1_delta_vs_baseline, reverse=True)
        for fr in sorted_results:
            marker = "*" if not fr.all_reached_target else ""
            print(f"  {fr.filter_name:<15} {fr.mean_f1:>10.4f} {fr.f1_delta_vs_baseline:>+10.2f}pp {fr.total_llm_calls:>8}{marker}")

    # Save results
    output_data = {
        "config": {
            "target_samples_per_type": TARGET_SAMPLES_PER_TYPE,
            "max_llm_calls_per_type": MAX_LLM_CALLS_PER_TYPE,
            "batch_size": BATCH_SIZE,
            "n_examples": N_EXAMPLES,
            "ner_epochs": NER_EPOCHS,
            "n_eval_seeds": N_EVAL_SEEDS,
        },
        "results": [asdict(dr) for dr in all_results],
        "summary": {
            fname: {
                "mean_f1_delta": float(np.mean(agg["f1_deltas"])) if agg["f1_deltas"] else 0,
                "mean_f1": float(np.mean(agg["mean_f1s"])) if agg["mean_f1s"] else 0,
                "mean_llm_calls": float(np.mean(agg["llm_calls"])) if agg["llm_calls"] else 0,
                "mean_acceptance_rate": float(np.mean(agg["acceptance_rates"])) if agg["acceptance_rates"] else 0,
                "mean_efficiency": float(np.mean(agg["efficiencies"])) if agg["efficiencies"] else 0,
            }
            for fname, agg in filter_agg.items()
        }
    }

    output_path = RESULTS_DIR / "experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("NER GEOMETRIC FILTER COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Target samples per entity type: {TARGET_SAMPLES_PER_TYPE}")
    print(f"  Max LLM calls per type: {MAX_LLM_CALLS_PER_TYPE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  NER epochs: {NER_EPOCHS}")
    print(f"  Eval seeds: {N_EVAL_SEEDS}")
    print(f"  Filters: {len(FILTERS)}")
    print(f"  Datasets: {len(DATASETS)}")

    # Initialize
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-3-flash-preview")

    # Run experiments
    all_results = []

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"\n  Skipping {dataset_name} (not found)")
            continue

        try:
            result = run_dataset_experiment(dataset_name, provider, embed_model)
            all_results.append(result)

            # Save intermediate results
            intermediate_path = RESULTS_DIR / "intermediate_results.json"
            intermediate_data = {
                "completed": [r.dataset for r in all_results],
                "results": [asdict(r) for r in all_results]
            }
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate_data, f, indent=2, default=str)

        except Exception as e:
            print(f"\n  Error on {dataset_name}: {e}")
            import traceback; traceback.print_exc()

    # Generate report
    if all_results:
        generate_summary_report(all_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
