#!/usr/bin/env python3
"""
Dataset Scaling Experiment: LLM+Filter vs SMOTE

Compares LLM augmentation (with cascade_l1) vs SMOTE across different dataset sizes.
Goal: Find the crossover point where SMOTE surpasses LLM augmentation.

Key metrics:
- F1 score at each dataset scale
- Crossover point (N samples/class where SMOTE >= LLM)
- Max advantage (largest LLM-SMOTE difference)
- Convergence behavior at full dataset

Supports two modes:
- fixed: Always generate SYNTHETIC_FIXED samples per class
- proportional: Generate samples proportional to real data (SYNTHETIC_RATIO * n_real)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade
from baselines import SMOTEBaseline

# Import prompt and generation functions from fixed_output_count
from exp_fixed_output_count import (
    create_prompt,
    generate_llm_batch,
    GenerationResult,
    DATASET_PROMPTS,
    get_dataset_base_name,
)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "scaling_experiment"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset-specific scales (limited by minimum class size)
DATASET_SCALES = {
    "sms_spam": [10, 25, 50, 100, 250, 500],              # Min class: 598
    "20newsgroups": [10, 25, 50, 100, 250],               # Min class: 480
    "hate_speech_davidson": [10, 25, 50, 100, 250, 500, 1000],  # Min class: 1203
}

# Augmentation config - Synthetic mode
SYNTHETIC_MODE = "fixed"  # "fixed" | "proportional" - set via --mode argument
SYNTHETIC_FIXED = 50      # Used when mode=fixed
SYNTHETIC_RATIO = 0.5     # Used when mode=proportional (synthetic = real * ratio)
SYNTHETIC_MIN = 10        # Minimum synthetic samples in proportional mode

N_SHOT = 25               # Examples in prompt
N_RUNS = 1                # Runs per config (increase for stability)


def get_synthetic_count(n_real_per_class: int, mode: str = None) -> int:
    """
    Calculate synthetic sample count based on mode.

    Args:
        n_real_per_class: Number of real samples per class
        mode: "fixed" or "proportional" (uses global SYNTHETIC_MODE if None)

    Returns:
        Number of synthetic samples to generate per class
    """
    mode = mode or SYNTHETIC_MODE
    if mode == "fixed":
        return SYNTHETIC_FIXED
    else:  # proportional
        return max(SYNTHETIC_MIN, int(n_real_per_class * SYNTHETIC_RATIO))

# LLM Filter (best performing from previous experiments)
LLM_FILTER = {"type": "cascade", "params": {"filter_level": 1, "k_neighbors": 10}}

# Generation limits
MAX_LLM_CALLS_PER_CLASS = 50
BATCH_SIZE = 25
EARLY_STOP_ACCEPTANCE = 0.02

# LLM Provider
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-3-flash-preview"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ScaleResult:
    """Result for one dataset at one scale."""
    dataset: str
    scale: int  # samples per class, or -1 for "full"
    n_train: int
    n_classes: int
    synthetic_per_class: int  # actual synthetic samples generated
    baseline_f1: float
    smote_f1: float
    llm_f1: float
    smote_delta: float  # smote_f1 - baseline_f1
    llm_delta: float    # llm_f1 - baseline_f1
    llm_vs_smote: float # llm_f1 - smote_f1
    llm_calls: int
    llm_acceptance_rate: float


# ============================================================================
# DATASET LOADING AND SUBSETTING
# ============================================================================

def load_full_dataset(dataset_name: str) -> Dict:
    """Load a full dataset from JSON."""
    path = DATA_DIR / f"{dataset_name}_full.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    data["dataset_name"] = dataset_name
    return data


def get_min_class_size(data: Dict) -> int:
    """Get the minimum class size in a dataset."""
    labels = data["train_labels"]
    counts = Counter(labels)
    return min(counts.values())


def create_stratified_subset(
    full_data: Dict,
    n_per_class: int,
    seed: int = 42
) -> Dict:
    """
    Create a stratified subset of the dataset.

    Args:
        full_data: Full dataset with train_texts, train_labels, etc.
        n_per_class: Number of samples per class
        seed: Random seed for reproducibility

    Returns:
        New dataset dict with the subset
    """
    train_texts = full_data["train_texts"]
    train_labels = full_data["train_labels"]
    classes = list(set(train_labels))

    np.random.seed(seed)
    selected_indices = []

    for cls in classes:
        cls_indices = [i for i, l in enumerate(train_labels) if l == cls]

        if len(cls_indices) < n_per_class:
            raise ValueError(
                f"Class '{cls}' has only {len(cls_indices)} samples, "
                f"but {n_per_class} requested"
            )

        selected = np.random.choice(cls_indices, n_per_class, replace=False)
        selected_indices.extend(selected)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    return {
        "train_texts": [train_texts[i] for i in selected_indices],
        "train_labels": [train_labels[i] for i in selected_indices],
        "test_texts": full_data["test_texts"],
        "test_labels": full_data["test_labels"],
        "n_train": len(selected_indices),
        "n_per_class": n_per_class,
        "dataset_name": full_data.get("dataset_name", "unknown"),
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_baseline(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray
) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    pred = clf.predict(test_emb)
    return f1_score(test_labels, pred, average='macro')


def evaluate_with_smote(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    target_per_class: int
) -> float:
    """Evaluate with SMOTE augmentation."""
    smote = SMOTEBaseline(k_neighbors=5, random_state=42)

    try:
        synth_emb, synth_labels = smote.generate(
            train_emb,
            train_labels,
            target_n_per_class=target_per_class
        )
    except Exception as e:
        print(f"    SMOTE failed: {e}")
        return evaluate_baseline(train_emb, train_labels, test_emb, test_labels)

    if len(synth_emb) == 0:
        return evaluate_baseline(train_emb, train_labels, test_emb, test_labels)

    aug_emb = np.vstack([train_emb, synth_emb])
    aug_labels = np.concatenate([train_labels, synth_labels])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    pred = clf.predict(test_emb)
    return f1_score(test_labels, pred, average='macro')


def generate_until_n_valid(
    provider,
    filter_obj,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    model: SentenceTransformer,
    dataset_name: str
) -> GenerationResult:
    """
    Generate samples iteratively until we have target_n valid samples.
    Uses cascade_l1 filter.
    """
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while True:
        # Generate batch
        batch_emb, batch_texts = generate_llm_batch(
            provider, class_name, class_texts, BATCH_SIZE, model, dataset_name
        )
        llm_calls += 1

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        # Skip if no pool yet
        if not pool_embeddings:
            if llm_calls >= MAX_LLM_CALLS_PER_CLASS:
                return GenerationResult(
                    valid_embeddings=np.array([]).reshape(0, 768),
                    valid_texts=[],
                    llm_calls=llm_calls,
                    total_generated=0,
                    acceptance_rate=0.0,
                    status="MAX_CALLS_REACHED"
                )
            continue

        # Stack pool
        pool_arr = np.vstack(pool_embeddings)

        # Apply cascade filter
        class_mask = real_labels == class_name
        if not class_mask.any():
            # No samples of this class - skip filtering
            if len(pool_arr) >= target_n:
                return GenerationResult(
                    valid_embeddings=pool_arr[:target_n],
                    valid_texts=pool_texts[:target_n],
                    llm_calls=llm_calls,
                    total_generated=len(pool_arr),
                    acceptance_rate=1.0,
                    status="SUCCESS"
                )
            continue

        # Use cascade filter
        filtered_emb, _, _ = filter_obj.filter_samples(
            candidates=pool_arr,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=class_name,
            target_count=target_n
        )

        n_valid = len(filtered_emb)
        acceptance_rate = n_valid / len(pool_arr) if len(pool_arr) > 0 else 0

        # Check success
        if n_valid >= target_n:
            # Get corresponding texts
            class_embs = real_embeddings[class_mask]
            anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)
            scores, _ = filter_obj.compute_quality_scores(
                pool_arr, anchor, real_embeddings, real_labels, class_name
            )
            top_idx = np.argsort(scores)[-target_n:]

            return GenerationResult(
                valid_embeddings=pool_arr[top_idx],
                valid_texts=[pool_texts[i] for i in top_idx],
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="SUCCESS"
            )

        # Check limits
        if llm_calls >= MAX_LLM_CALLS_PER_CLASS:
            # Return what we have
            if n_valid > 0:
                class_embs = real_embeddings[class_mask]
                anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)
                scores, _ = filter_obj.compute_quality_scores(
                    pool_arr, anchor, real_embeddings, real_labels, class_name
                )
                top_idx = np.argsort(scores)[-n_valid:]
                return GenerationResult(
                    valid_embeddings=pool_arr[top_idx],
                    valid_texts=[pool_texts[i] for i in top_idx],
                    llm_calls=llm_calls,
                    total_generated=len(pool_arr),
                    acceptance_rate=acceptance_rate,
                    status="MAX_CALLS_REACHED"
                )
            return GenerationResult(
                valid_embeddings=np.array([]).reshape(0, 768),
                valid_texts=[],
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="MAX_CALLS_REACHED"
            )

        # Early stop on low acceptance
        if len(pool_arr) > 100 and acceptance_rate < EARLY_STOP_ACCEPTANCE:
            return GenerationResult(
                valid_embeddings=filtered_emb if n_valid > 0 else np.array([]).reshape(0, 768),
                valid_texts=pool_texts[:n_valid] if n_valid > 0 else [],
                llm_calls=llm_calls,
                total_generated=len(pool_arr),
                acceptance_rate=acceptance_rate,
                status="LOW_ACCEPTANCE"
            )


def evaluate_with_llm(
    data: Dict,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    target_per_class: int,
    provider,
    model: SentenceTransformer
) -> Tuple[float, int, float]:
    """
    Evaluate with LLM augmentation + cascade filter.

    Returns:
        (f1_score, total_llm_calls, avg_acceptance_rate)
    """
    train_labels_arr = np.array(data["train_labels"])
    test_labels = data["test_labels"]
    dataset_name = data.get("dataset_name", "unknown")

    # Create filter
    filter_obj = FilterCascade(**LLM_FILTER["params"])

    all_synth_emb = []
    all_synth_labels = []
    total_llm_calls = 0
    acceptance_rates = []

    classes = list(set(data["train_labels"]))

    for cls in classes:
        cls_texts = [t for t, l in zip(data["train_texts"], data["train_labels"]) if l == cls]

        # Adjust n_shot based on available samples
        n_shot_actual = min(N_SHOT, len(cls_texts))

        result = generate_until_n_valid(
            provider=provider,
            filter_obj=filter_obj,
            target_n=target_per_class,
            class_name=cls,
            class_texts=cls_texts[:n_shot_actual],
            real_embeddings=train_emb,
            real_labels=train_labels_arr,
            model=model,
            dataset_name=dataset_name
        )

        total_llm_calls += result.llm_calls
        acceptance_rates.append(result.acceptance_rate)

        if len(result.valid_embeddings) > 0:
            all_synth_emb.append(result.valid_embeddings)
            all_synth_labels.extend([cls] * len(result.valid_embeddings))

    # Combine and train
    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
        aug_emb = np.vstack([train_emb, synth_emb])
        aug_labels = list(data["train_labels"]) + all_synth_labels
    else:
        aug_emb = train_emb
        aug_labels = data["train_labels"]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(aug_emb, aug_labels)
    pred = clf.predict(test_emb)
    f1 = f1_score(test_labels, pred, average='macro')

    avg_acceptance = np.mean(acceptance_rates) if acceptance_rates else 0.0

    return f1, total_llm_calls, avg_acceptance


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_scaling_experiment_for_dataset(
    dataset_name: str,
    model: SentenceTransformer,
    provider
) -> List[ScaleResult]:
    """Run scaling experiment for one dataset."""

    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print("="*70)

    # Load full dataset
    full_data = load_full_dataset(dataset_name)
    min_class_size = get_min_class_size(full_data)
    n_classes = len(set(full_data["train_labels"]))

    print(f"  Total train: {len(full_data['train_texts'])}")
    print(f"  Classes: {n_classes}")
    print(f"  Min class size: {min_class_size}")

    # Get valid scales for this dataset
    scales = DATASET_SCALES.get(dataset_name, [10, 25, 50, 100])
    valid_scales = [s for s in scales if s <= min_class_size]

    print(f"  Scales to test: {valid_scales}")

    results = []

    for scale in valid_scales:
        # Calculate synthetic count based on mode
        synthetic_per_class = get_synthetic_count(scale)

        print(f"\n  Scale: {scale} samples/class")
        print(f"  Synthetic: {synthetic_per_class} per class (mode={SYNTHETIC_MODE})")
        print(f"  {'-'*50}")

        # Create subset
        subset = create_stratified_subset(full_data, n_per_class=scale)

        # Embed
        print("    Embedding...")
        train_emb = model.encode(subset["train_texts"], show_progress_bar=False)
        test_emb = model.encode(subset["test_texts"], show_progress_bar=False)
        train_labels_arr = np.array(subset["train_labels"])

        # Baseline
        baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, subset["test_labels"])
        print(f"    Baseline F1: {baseline_f1:.4f}")

        # SMOTE
        smote_f1 = evaluate_with_smote(
            train_emb, train_labels_arr, test_emb, subset["test_labels"],
            synthetic_per_class
        )
        smote_delta = (smote_f1 - baseline_f1) * 100
        print(f"    SMOTE F1:    {smote_f1:.4f} ({smote_delta:+.2f}pp)")

        # LLM + Filter
        llm_f1, llm_calls, llm_acceptance = evaluate_with_llm(
            subset, train_emb, test_emb, synthetic_per_class, provider, model
        )
        llm_delta = (llm_f1 - baseline_f1) * 100
        llm_vs_smote = (llm_f1 - smote_f1) * 100
        print(f"    LLM F1:      {llm_f1:.4f} ({llm_delta:+.2f}pp)")
        print(f"    LLM vs SMOTE: {llm_vs_smote:+.2f}pp")
        print(f"    LLM calls: {llm_calls}, acceptance: {llm_acceptance*100:.1f}%")

        results.append(ScaleResult(
            dataset=dataset_name,
            scale=scale,
            n_train=subset["n_train"],
            n_classes=n_classes,
            synthetic_per_class=synthetic_per_class,
            baseline_f1=baseline_f1,
            smote_f1=smote_f1,
            llm_f1=llm_f1,
            smote_delta=smote_delta,
            llm_delta=llm_delta,
            llm_vs_smote=llm_vs_smote,
            llm_calls=llm_calls,
            llm_acceptance_rate=llm_acceptance
        ))

    return results


def print_summary(all_results: List[ScaleResult]):
    """Print summary of all results."""
    print("\n" + "="*80)
    print("SUMMARY: LLM vs SMOTE ACROSS DATASET SIZES")
    print("="*80)

    # Group by dataset
    datasets = list(set(r.dataset for r in all_results))

    for dataset in datasets:
        dataset_results = [r for r in all_results if r.dataset == dataset]
        dataset_results.sort(key=lambda x: x.scale)

        print(f"\n{dataset}:")
        print(f"{'Scale':<8} {'N_train':<10} {'Baseline':<10} {'SMOTE':<10} {'LLM':<10} {'LLM-SMOTE':<12}")
        print("-" * 70)

        crossover_found = False
        for r in dataset_results:
            marker = ""
            if r.llm_vs_smote < 0 and not crossover_found:
                marker = " <-- CROSSOVER"
                crossover_found = True

            print(f"{r.scale:<8} {r.n_train:<10} {r.baseline_f1:<10.4f} "
                  f"{r.smote_f1:<10.4f} {r.llm_f1:<10.4f} {r.llm_vs_smote:+.2f}pp{marker}")

        # Find crossover point
        if crossover_found:
            crossover_scale = next(r.scale for r in dataset_results if r.llm_vs_smote < 0)
            print(f"\n  CROSSOVER at ~{crossover_scale} samples/class")
        else:
            print(f"\n  LLM wins at all scales tested")

        # Max advantage
        max_adv = max(r.llm_vs_smote for r in dataset_results)
        max_adv_scale = next(r.scale for r in dataset_results if r.llm_vs_smote == max_adv)
        print(f"  MAX LLM ADVANTAGE: {max_adv:+.2f}pp at {max_adv_scale} samples/class")


def save_results(all_results: List[ScaleResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = f"_{SYNTHETIC_MODE}"
    output_file = RESULTS_DIR / f"scaling_results{mode_suffix}_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "synthetic_mode": SYNTHETIC_MODE,
            "synthetic_fixed": SYNTHETIC_FIXED,
            "synthetic_ratio": SYNTHETIC_RATIO,
            "synthetic_min": SYNTHETIC_MIN,
            "n_shot": N_SHOT,
            "llm_filter": LLM_FILTER,
            "llm_model": LLM_MODEL,
        },
        "results": [asdict(r) for r in all_results]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save summary with mode suffix
    summary_file = RESULTS_DIR / f"latest_summary_{SYNTHETIC_MODE}.json"
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)


def main():
    global SYNTHETIC_MODE

    # Parse arguments
    parser = argparse.ArgumentParser(description="Dataset Scaling Experiment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed", "proportional"],
        default="fixed",
        help="Synthetic sample mode: 'fixed' (always 50) or 'proportional' (50%% of real)"
    )
    args = parser.parse_args()

    # Set global mode
    SYNTHETIC_MODE = args.mode

    print("="*80)
    print("DATASET SCALING EXPERIMENT: LLM+Filter vs SMOTE")
    print("="*80)
    print(f"\nConfig:")
    print(f"  Synthetic mode: {SYNTHETIC_MODE}")
    if SYNTHETIC_MODE == "fixed":
        print(f"  Synthetic per class: {SYNTHETIC_FIXED}")
    else:
        print(f"  Synthetic ratio: {SYNTHETIC_RATIO} (min={SYNTHETIC_MIN})")
    print(f"  LLM Filter: {LLM_FILTER}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Datasets: {list(DATASET_SCALES.keys())}")

    # Initialize
    print("\nLoading models...")
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    # Run experiments
    all_results = []

    for dataset_name in DATASET_SCALES.keys():
        try:
            results = run_scaling_experiment_for_dataset(dataset_name, model, provider)
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    main()
