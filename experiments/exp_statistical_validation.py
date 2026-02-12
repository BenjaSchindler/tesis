#!/usr/bin/env python3
"""
Statistical Validation Experiment

Runs multiple trials with different random seeds to obtain:
- Mean and standard deviation of results
- 95% confidence intervals
- p-values for significance testing
- Win rates

This validates whether previous findings are statistically significant.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict, field
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# Import prompt function
from exp_fixed_output_count import create_prompt, DATASET_PROMPTS, get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "statistical_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Statistical parameters
N_RUNS = 5  # Number of independent runs
SEEDS = [42, 123, 456, 789, 1011]
N_FOLDS = 5  # For cross-validation within each run

# Datasets to validate
DATASETS = ["sms_spam_10shot", "20newsgroups_10shot", "hate_speech_davidson_10shot"]

# Methods to compare
METHODS = ["baseline", "smote", "llm"]

# Generation config
SYNTHETIC_PER_CLASS = 50
N_SHOT = 25
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25

# Filter config
LLM_FILTER = {"filter_level": 1, "k_neighbors": 10}

# LLM Provider
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StatisticalResult:
    """Statistical result for one method on one dataset."""
    dataset: str
    method: str
    f1_mean: float
    f1_std: float
    f1_values: List[float]
    ci_95_lower: float
    ci_95_upper: float
    delta_mean: float  # vs baseline
    delta_std: float
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    win_rate: float  # % of runs where method > baseline


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_llm_samples(
    provider,
    filter_obj: FilterCascade,
    train_texts: List[str],
    train_labels: List[str],
    train_emb: np.ndarray,
    embed_model: SentenceTransformer,
    dataset_name: str,
    seed: int
) -> Tuple[np.ndarray, List[str]]:
    """Generate LLM samples with filtering."""
    np.random.seed(seed)

    classes = list(set(train_labels))
    train_labels_arr = np.array(train_labels)

    all_synth_emb = []
    all_synth_labels = []

    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        n_shot_actual = min(N_SHOT, len(cls_texts))

        # Shuffle and select examples based on seed
        np.random.shuffle(cls_texts)
        selected_texts = cls_texts[:n_shot_actual]

        # Generate
        pool_embeddings = []
        pool_texts = []
        llm_calls = 0

        while llm_calls < MAX_LLM_CALLS_PER_CLASS:
            prompt = create_prompt(cls, selected_texts, BATCH_SIZE, dataset_name)
            messages = [{"role": "user", "content": prompt}]

            try:
                response, _ = provider.generate(messages, temperature=1.0, max_tokens=4000)
            except Exception as e:
                llm_calls += 1
                continue

            # Parse
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            generated = []
            for line in lines:
                clean = line.lstrip('0123456789.-):* ')
                if len(clean) > 10:
                    generated.append(clean)

            llm_calls += 1

            if generated:
                batch_emb = embed_model.encode(generated, show_progress_bar=False)
                pool_embeddings.append(batch_emb)
                pool_texts.extend(generated)

            if pool_embeddings:
                pool_arr = np.vstack(pool_embeddings)
                if len(pool_arr) >= SYNTHETIC_PER_CLASS * 1.5:
                    break

        if not pool_embeddings:
            continue

        pool_arr = np.vstack(pool_embeddings)

        # Filter
        class_mask = train_labels_arr == cls
        if class_mask.any():
            filtered_emb, _, _ = filter_obj.filter_samples(
                candidates=pool_arr,
                real_embeddings=train_emb,
                real_labels=train_labels_arr,
                target_class=cls,
                target_count=SYNTHETIC_PER_CLASS
            )
        else:
            filtered_emb = pool_arr[:SYNTHETIC_PER_CLASS]

        if len(filtered_emb) > 0:
            all_synth_emb.append(filtered_emb[:SYNTHETIC_PER_CLASS])
            all_synth_labels.extend([cls] * min(len(filtered_emb), SYNTHETIC_PER_CLASS))

    if all_synth_emb:
        return np.vstack(all_synth_emb), all_synth_labels
    return np.array([]).reshape(0, 768), []


def generate_smote_samples(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, List[str]]:
    """Generate SMOTE samples."""
    classes = list(set(train_labels))
    all_synth_emb = []
    all_synth_labels = []

    for cls in classes:
        class_mask = train_labels == cls
        n_class = class_mask.sum()

        k_neighbors = min(5, n_class - 1)
        if k_neighbors < 1:
            continue

        binary_labels = class_mask.astype(int)
        target_count = n_class + SYNTHETIC_PER_CLASS

        try:
            smote = SMOTE(
                k_neighbors=k_neighbors,
                sampling_strategy={1: target_count},
                random_state=seed
            )
            X_resampled, y_resampled = smote.fit_resample(train_emb, binary_labels)

            new_samples = X_resampled[len(train_emb):]
            if len(new_samples) > 0:
                all_synth_emb.append(new_samples[:SYNTHETIC_PER_CLASS])
                all_synth_labels.extend([cls] * min(len(new_samples), SYNTHETIC_PER_CLASS))

        except Exception:
            continue

    if all_synth_emb:
        return np.vstack(all_synth_emb), all_synth_labels
    return np.array([]).reshape(0, 768), []


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_method(
    method: str,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: List[str],
    synth_emb: np.ndarray = None,
    synth_labels: List[str] = None,
    seed: int = 42
) -> float:
    """Evaluate a method and return F1 score."""
    if method == "baseline" or synth_emb is None or len(synth_emb) == 0:
        X_train = train_emb
        y_train = train_labels
    else:
        X_train = np.vstack([train_emb, synth_emb])
        y_train = list(train_labels) + list(synth_labels)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    predictions = clf.predict(test_emb)
    return f1_score(test_labels, predictions, average='macro')


def run_single_trial(
    dataset_name: str,
    method: str,
    embed_model: SentenceTransformer,
    provider,
    filter_obj: FilterCascade,
    seed: int
) -> float:
    """Run a single trial for one method."""
    # Load dataset
    with open(DATA_DIR / f"{dataset_name}.json") as f:
        data = json.load(f)

    train_texts = data["train_texts"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]

    # Embed
    train_emb = embed_model.encode(train_texts, show_progress_bar=False)
    test_emb = embed_model.encode(data["test_texts"], show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    # Generate synthetic data based on method
    synth_emb = None
    synth_labels = None

    if method == "smote":
        synth_emb, synth_labels = generate_smote_samples(train_emb, train_labels_arr, seed)
    elif method == "llm":
        synth_emb, synth_labels = generate_llm_samples(
            provider, filter_obj, train_texts, train_labels,
            train_emb, embed_model, dataset_name, seed
        )

    # Evaluate
    f1 = evaluate_method(
        method, train_emb, train_labels_arr, test_emb, test_labels,
        synth_emb, synth_labels, seed
    )

    return f1


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(
    f1_values: List[float],
    baseline_values: List[float]
) -> Dict:
    """Compute statistical metrics."""
    n = len(f1_values)

    f1_mean = np.mean(f1_values)
    f1_std = np.std(f1_values, ddof=1)

    # Confidence interval
    se = f1_std / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=f1_mean, scale=se)

    # Delta vs baseline
    deltas = np.array(f1_values) - np.array(baseline_values)
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(f1_values, baseline_values)

    # Win rate
    win_rate = np.mean(np.array(f1_values) > np.array(baseline_values))

    return {
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "ci_95_lower": ci[0],
        "ci_95_upper": ci[1],
        "delta_mean": delta_mean * 100,  # percentage points
        "delta_std": delta_std * 100,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "win_rate": win_rate
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_statistical_validation(
    dataset_name: str,
    embed_model: SentenceTransformer,
    provider,
    filter_obj: FilterCascade
) -> List[StatisticalResult]:
    """Run statistical validation for one dataset."""
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    results = {}
    for method in METHODS:
        results[method] = []

    # Run trials
    for i, seed in enumerate(SEEDS):
        print(f"\n  Run {i+1}/{N_RUNS} (seed={seed})")

        for method in METHODS:
            print(f"    {method}...", end=" ", flush=True)
            f1 = run_single_trial(
                dataset_name, method, embed_model, provider, filter_obj, seed
            )
            results[method].append(f1)
            print(f"F1={f1:.4f}")

    # Compute statistics
    stat_results = []
    baseline_values = results["baseline"]

    for method in METHODS:
        f1_values = results[method]

        if method == "baseline":
            # For baseline, compare against itself
            stats_dict = {
                "f1_mean": np.mean(f1_values),
                "f1_std": np.std(f1_values, ddof=1),
                "ci_95_lower": np.mean(f1_values) - 1.96 * np.std(f1_values, ddof=1) / np.sqrt(len(f1_values)),
                "ci_95_upper": np.mean(f1_values) + 1.96 * np.std(f1_values, ddof=1) / np.sqrt(len(f1_values)),
                "delta_mean": 0.0,
                "delta_std": 0.0,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "win_rate": 0.0
            }
        else:
            stats_dict = compute_statistics(f1_values, baseline_values)

        stat_results.append(StatisticalResult(
            dataset=dataset_name,
            method=method,
            f1_mean=stats_dict["f1_mean"],
            f1_std=stats_dict["f1_std"],
            f1_values=f1_values,
            ci_95_lower=stats_dict["ci_95_lower"],
            ci_95_upper=stats_dict["ci_95_upper"],
            delta_mean=stats_dict["delta_mean"],
            delta_std=stats_dict["delta_std"],
            t_statistic=stats_dict["t_statistic"],
            p_value=stats_dict["p_value"],
            significant=stats_dict["significant"],
            win_rate=stats_dict["win_rate"]
        ))

    return stat_results


def main():
    print("=" * 70)
    print("STATISTICAL VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  N_RUNS: {N_RUNS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Methods: {METHODS}")
    print(f"  Datasets: {DATASETS}")

    # Initialize
    print("\nLoading models...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    filter_obj = FilterCascade(**LLM_FILTER)

    all_results = []

    for dataset_name in DATASETS:
        try:
            results = run_statistical_validation(
                dataset_name, embed_model, provider, filter_obj
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[StatisticalResult]):
    """Print summary of statistical results."""
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    for dataset in DATASETS:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        if not dataset_results:
            continue

        print(f"\n{dataset}:")
        print(f"{'Method':<12} {'F1 Mean':<12} {'F1 Std':<10} {'CI 95%':<20} "
              f"{'Delta':<12} {'p-value':<10} {'Sig?':<6} {'Win%':<8}")
        print("-" * 95)

        for r in dataset_results:
            sig = "YES" if r.significant else "no"
            ci_str = f"[{r.ci_95_lower:.4f}, {r.ci_95_upper:.4f}]"
            delta_str = f"{r.delta_mean:+.2f}pp" if r.method != "baseline" else "-"
            p_str = f"{r.p_value:.4f}" if r.method != "baseline" else "-"
            win_str = f"{r.win_rate*100:.0f}%" if r.method != "baseline" else "-"

            print(f"{r.method:<12} {r.f1_mean:<12.4f} {r.f1_std:<10.4f} {ci_str:<20} "
                  f"{delta_str:<12} {p_str:<10} {sig:<6} {win_str:<8}")

    # Overall summary
    print("\n" + "-" * 80)
    print("OVERALL SIGNIFICANCE:")
    print("-" * 80)

    for method in METHODS:
        if method == "baseline":
            continue

        method_results = [r for r in all_results if r.method == method]
        n_significant = sum(1 for r in method_results if r.significant)
        avg_delta = np.mean([r.delta_mean for r in method_results])
        avg_win = np.mean([r.win_rate for r in method_results])

        print(f"{method}: {n_significant}/{len(method_results)} datasets significant, "
              f"Avg delta: {avg_delta:+.2f}pp, Avg win rate: {avg_win*100:.0f}%")


def save_results(all_results: List[StatisticalResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"statistical_validation_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "n_runs": N_RUNS,
            "seeds": SEEDS,
            "methods": METHODS,
            "datasets": DATASETS,
            "synthetic_per_class": SYNTHETIC_PER_CLASS,
            "llm_filter": LLM_FILTER,
            "llm_model": LLM_MODEL,
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
