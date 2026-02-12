#!/usr/bin/env python3
"""
LLM Provider Comparison Experiment

Compares different LLM providers for synthetic data generation quality:
- Google Gemini 3 Flash
- OpenAI GPT-5 mini

Tests on low-resource scenarios where LLM augmentation shows most benefit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# Import prompt function
from exp_fixed_output_count import create_prompt, DATASET_PROMPTS, get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# LLM Providers to compare
LLM_PROVIDERS = [
    {"name": "gemini_flash", "provider": "google", "model": "gemini-3-flash-preview"},
    {"name": "gpt5_mini", "provider": "gpt5", "model": "gpt-5-mini"},
]

# Datasets (10-shot for low resource)
DATASETS = ["sms_spam_10shot", "20newsgroups_10shot", "hate_speech_davidson_10shot"]

# Generation config
SYNTHETIC_PER_CLASS = 50
N_SHOT = 25
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25

# Filter config (best from previous experiments)
LLM_FILTER = {"filter_level": 1, "k_neighbors": 10}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LLMComparisonResult:
    """Result for one LLM provider on one dataset."""
    dataset: str
    llm_name: str
    llm_provider: str
    llm_model: str
    baseline_f1: float
    llm_f1: float
    delta: float
    llm_calls: int
    acceptance_rate: float
    n_synthetic: int
    avg_generation_time: float


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_llm_batch(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    model: SentenceTransformer,
    dataset_name: str
) -> Tuple[np.ndarray, List[str], float]:
    """Generate a batch of LLM samples. Returns embeddings, texts, and time."""
    import time

    prompt = create_prompt(class_name, class_texts, n_generate, dataset_name)
    messages = [{"role": "user", "content": prompt}]

    try:
        start_time = time.time()
        response, usage = provider.generate(messages, temperature=1.0, max_tokens=4000)
        gen_time = time.time() - start_time
    except Exception as e:
        print(f"    LLM error: {e}")
        return np.array([]).reshape(0, 768), [], 0.0

    # Parse response
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 10:
            generated.append(clean)

    if not generated:
        return np.array([]).reshape(0, 768), [], gen_time

    embeddings = model.encode(generated, show_progress_bar=False)
    return embeddings, generated, gen_time


def generate_until_n_valid(
    provider,
    filter_obj: FilterCascade,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    model: SentenceTransformer,
    dataset_name: str
) -> Dict:
    """Generate until we have target_n valid samples."""
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0
    total_time = 0.0

    while llm_calls < MAX_LLM_CALLS_PER_CLASS:
        batch_emb, batch_texts, gen_time = generate_llm_batch(
            provider, class_name, class_texts, BATCH_SIZE, model, dataset_name
        )
        llm_calls += 1
        total_time += gen_time

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        if not pool_embeddings:
            continue

        pool_arr = np.vstack(pool_embeddings)

        # Apply filter
        class_mask = real_labels == class_name
        if not class_mask.any():
            if len(pool_arr) >= target_n:
                return {
                    "embeddings": pool_arr[:target_n],
                    "texts": pool_texts[:target_n],
                    "llm_calls": llm_calls,
                    "acceptance_rate": 1.0,
                    "avg_time": total_time / llm_calls,
                    "status": "SUCCESS"
                }
            continue

        filtered_emb, _, _ = filter_obj.filter_samples(
            candidates=pool_arr,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=class_name,
            target_count=target_n
        )

        n_valid = len(filtered_emb)
        acceptance_rate = n_valid / len(pool_arr) if len(pool_arr) > 0 else 0

        if n_valid >= target_n:
            class_embs = real_embeddings[class_mask]
            anchor = class_embs.mean(axis=0)
            scores, _ = filter_obj.compute_quality_scores(
                pool_arr, anchor, real_embeddings, real_labels, class_name
            )
            top_idx = np.argsort(scores)[-target_n:]

            return {
                "embeddings": pool_arr[top_idx],
                "texts": [pool_texts[i] for i in top_idx],
                "llm_calls": llm_calls,
                "acceptance_rate": acceptance_rate,
                "avg_time": total_time / llm_calls,
                "status": "SUCCESS"
            }

    # Max calls reached
    if pool_embeddings:
        pool_arr = np.vstack(pool_embeddings)
        return {
            "embeddings": pool_arr[:target_n] if len(pool_arr) >= target_n else pool_arr,
            "texts": pool_texts[:target_n] if len(pool_texts) >= target_n else pool_texts,
            "llm_calls": llm_calls,
            "acceptance_rate": len(pool_arr) / (llm_calls * BATCH_SIZE) if llm_calls > 0 else 0,
            "avg_time": total_time / llm_calls if llm_calls > 0 else 0,
            "status": "MAX_CALLS"
        }

    return {
        "embeddings": np.array([]).reshape(0, 768),
        "texts": [],
        "llm_calls": llm_calls,
        "acceptance_rate": 0,
        "avg_time": 0,
        "status": "NO_DATA"
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


def evaluate_with_llm(
    train_emb, train_labels, test_emb, test_labels,
    synth_emb, synth_labels
) -> float:
    """Evaluate with LLM augmentation."""
    if len(synth_emb) == 0:
        return evaluate_baseline(train_emb, train_labels, test_emb, test_labels)

    aug_emb = np.vstack([train_emb, synth_emb])
    aug_labels = list(train_labels) + list(synth_labels)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_llm_experiment(
    dataset_name: str,
    llm_config: Dict,
    embed_model: SentenceTransformer
) -> LLMComparisonResult:
    """Run experiment for one LLM on one dataset."""
    print(f"\n  LLM: {llm_config['name']}")
    print(f"  {'-' * 40}")

    # Initialize provider
    try:
        provider = create_provider(llm_config["provider"], llm_config["model"])
    except Exception as e:
        print(f"    ERROR initializing provider: {e}")
        raise

    filter_obj = FilterCascade(**LLM_FILTER)

    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels"]
    test_texts = data["test_texts"]
    test_labels = data["test_labels"]
    classes = list(set(train_labels))

    # Embed
    train_emb = embed_model.encode(train_texts, show_progress_bar=False)
    test_emb = embed_model.encode(test_texts, show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    # Baseline
    baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, test_labels)
    print(f"    Baseline F1: {baseline_f1:.4f}")

    # Generate with LLM
    all_synth_emb = []
    all_synth_labels = []
    total_calls = 0
    acceptance_rates = []
    avg_times = []

    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        n_shot_actual = min(N_SHOT, len(cls_texts))

        result = generate_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            target_n=SYNTHETIC_PER_CLASS,
            class_name=cls,
            class_texts=cls_texts[:n_shot_actual],
            real_embeddings=train_emb,
            real_labels=train_labels_arr,
            model=embed_model,
            dataset_name=dataset_name
        )

        total_calls += result["llm_calls"]
        acceptance_rates.append(result["acceptance_rate"])
        avg_times.append(result["avg_time"])

        if len(result["embeddings"]) > 0:
            all_synth_emb.append(result["embeddings"])
            all_synth_labels.extend([cls] * len(result["embeddings"]))

    # Evaluate
    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
    else:
        synth_emb = np.array([]).reshape(0, 768)

    llm_f1 = evaluate_with_llm(
        train_emb, train_labels_arr, test_emb, test_labels,
        synth_emb, all_synth_labels
    )
    delta = (llm_f1 - baseline_f1) * 100
    avg_acceptance = np.mean(acceptance_rates) if acceptance_rates else 0
    avg_time = np.mean(avg_times) if avg_times else 0

    print(f"    LLM F1: {llm_f1:.4f} ({delta:+.2f}pp)")
    print(f"    Calls: {total_calls}, Acceptance: {avg_acceptance*100:.1f}%, Avg time: {avg_time:.2f}s")

    return LLMComparisonResult(
        dataset=dataset_name,
        llm_name=llm_config["name"],
        llm_provider=llm_config["provider"],
        llm_model=llm_config["model"],
        baseline_f1=baseline_f1,
        llm_f1=llm_f1,
        delta=delta,
        llm_calls=total_calls,
        acceptance_rate=avg_acceptance,
        n_synthetic=len(synth_emb),
        avg_generation_time=avg_time
    )


def main():
    print("=" * 70)
    print("LLM PROVIDER COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  LLM Providers: {[p['name'] for p in LLM_PROVIDERS]}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Synthetic per class: {SYNTHETIC_PER_CLASS}")

    # Initialize embedding model
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        for llm_config in LLM_PROVIDERS:
            try:
                result = run_llm_experiment(dataset_name, llm_config, embed_model)
                all_results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[LLMComparisonResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: LLM COMPARISON")
    print("=" * 70)

    # Group by dataset
    for dataset in DATASETS:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{dataset}:")
        print(f"{'LLM':<20} {'F1':<10} {'Delta':<12} {'Calls':<8} {'Accept%':<10} {'Time':<8}")
        print("-" * 70)

        baseline = dataset_results[0].baseline_f1
        print(f"{'BASELINE':<20} {baseline:<10.4f} {'-':<12} {'-':<8} {'-':<10} {'-':<8}")

        for r in sorted(dataset_results, key=lambda x: x.delta, reverse=True):
            print(f"{r.llm_name:<20} {r.llm_f1:<10.4f} {r.delta:+10.2f}pp "
                  f"{r.llm_calls:<8} {r.acceptance_rate*100:<9.1f}% {r.avg_generation_time:<7.2f}s")

    # Overall comparison
    print("\n" + "-" * 70)
    print("OVERALL (Average across datasets):")
    print("-" * 70)

    for llm_config in LLM_PROVIDERS:
        llm_results = [r for r in all_results if r.llm_name == llm_config["name"]]
        if llm_results:
            avg_delta = np.mean([r.delta for r in llm_results])
            avg_calls = np.mean([r.llm_calls for r in llm_results])
            avg_accept = np.mean([r.acceptance_rate for r in llm_results])
            avg_time = np.mean([r.avg_generation_time for r in llm_results])
            print(f"{llm_config['name']:<20} Avg Delta: {avg_delta:+.2f}pp, "
                  f"Calls: {avg_calls:.1f}, Accept: {avg_accept*100:.1f}%, Time: {avg_time:.2f}s")


def save_results(all_results: List[LLMComparisonResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"llm_comparison_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "llm_providers": LLM_PROVIDERS,
            "datasets": DATASETS,
            "synthetic_per_class": SYNTHETIC_PER_CLASS,
            "llm_filter": LLM_FILTER,
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Latest summary
    summary_file = RESULTS_DIR / "latest_summary.json"
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
