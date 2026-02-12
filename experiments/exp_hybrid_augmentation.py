#!/usr/bin/env python3
"""
Hybrid LLM + SMOTE Augmentation Experiment

Tests combining LLM-generated samples with SMOTE for scenarios where LLM alone
doesn't outperform SMOTE (typically >100 samples/class).

Hypothesis: LLM adds semantic diversity, SMOTE expands geometrically on that base.

Pipeline:
1. Generate LLM samples (K = llm_ratio * target)
2. Filter with cascade_l1
3. Apply SMOTE on (Real + Filtered LLM) to generate remaining samples
4. Evaluate combined augmentation
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
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.filter_cascade import FilterCascade

# Import prompt function from fixed output count experiment
from exp_fixed_output_count import create_prompt, DATASET_PROMPTS, get_dataset_base_name

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "hybrid_augmentation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Hybrid modes: ratio of LLM vs SMOTE
HYBRID_MODES = [
    {"name": "llm_25_smote_75", "llm_ratio": 0.25, "smote_ratio": 0.75},
    {"name": "llm_50_smote_50", "llm_ratio": 0.50, "smote_ratio": 0.50},
    {"name": "llm_75_smote_25", "llm_ratio": 0.75, "smote_ratio": 0.25},
]

# Baselines to compare
BASELINES = ["llm_only", "smote_only"]

# Dataset scales where LLM alone typically doesn't win
SCALES = [100, 250, 500]

# Full datasets to use
DATASETS = ["sms_spam", "20newsgroups", "hate_speech_davidson"]

# Total synthetic samples per class
SYNTHETIC_PER_CLASS = 50

# Generation config
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
class HybridResult:
    """Result for one hybrid configuration."""
    dataset: str
    scale: int
    mode: str
    llm_ratio: float
    smote_ratio: float
    baseline_f1: float
    augmented_f1: float
    delta: float
    llm_samples: int
    smote_samples: int
    total_samples: int
    llm_calls: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_full_dataset(dataset_name: str) -> Dict:
    """Load full dataset from benchmarks."""
    # Try to load the full dataset file
    full_path = DATA_DIR / f"{dataset_name}_full.json"
    if full_path.exists():
        with open(full_path) as f:
            return json.load(f)

    # Fall back to largest available
    for suffix in ["50shot", "25shot", "10shot"]:
        path = DATA_DIR / f"{dataset_name}_{suffix}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(f"No dataset found for {dataset_name}")


def create_stratified_subset(data: Dict, n_per_class: int) -> Dict:
    """Create a stratified subset with n samples per class."""
    train_texts = data["train_texts"]
    train_labels = data["train_labels"]

    # Group by class
    class_indices = {}
    for i, label in enumerate(train_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    # Sample n_per_class from each class
    subset_indices = []
    for label, indices in class_indices.items():
        if len(indices) < n_per_class:
            # Not enough samples, use all
            subset_indices.extend(indices)
        else:
            np.random.seed(42)
            selected = np.random.choice(indices, n_per_class, replace=False)
            subset_indices.extend(selected)

    # Create subset
    subset_texts = [train_texts[i] for i in subset_indices]
    subset_labels = [train_labels[i] for i in subset_indices]

    return {
        "train_texts": subset_texts,
        "train_labels": subset_labels,
        "test_texts": data["test_texts"],
        "test_labels": data["test_labels"],
        "n_train": len(subset_texts)
    }


def generate_llm_batch(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    model: SentenceTransformer,
    dataset_name: str
) -> Tuple[np.ndarray, List[str]]:
    """Generate a batch of LLM samples."""
    prompt = create_prompt(class_name, class_texts, n_generate, dataset_name)
    messages = [{"role": "user", "content": prompt}]

    try:
        response, _ = provider.generate(messages, temperature=1.0, max_tokens=4000)
    except Exception as e:
        print(f"    LLM error: {e}")
        return np.array([]).reshape(0, 768), []

    # Parse response
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    generated = []
    for line in lines:
        clean = line.lstrip('0123456789.-):* ')
        if len(clean) > 10:
            generated.append(clean)

    if not generated:
        return np.array([]).reshape(0, 768), []

    embeddings = model.encode(generated, show_progress_bar=False)
    return embeddings, generated


def generate_llm_filtered(
    provider,
    filter_obj: FilterCascade,
    target_n: int,
    class_name: str,
    class_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    model: SentenceTransformer,
    dataset_name: str
) -> Tuple[np.ndarray, int]:
    """Generate LLM samples and filter to target_n."""
    pool_embeddings = []
    pool_texts = []
    llm_calls = 0

    while llm_calls < MAX_LLM_CALLS_PER_CLASS:
        batch_emb, batch_texts = generate_llm_batch(
            provider, class_name, class_texts, BATCH_SIZE, model, dataset_name
        )
        llm_calls += 1

        if len(batch_emb) > 0:
            pool_embeddings.append(batch_emb)
            pool_texts.extend(batch_texts)

        if not pool_embeddings:
            continue

        pool_arr = np.vstack(pool_embeddings)

        # Check if we have enough for filtering
        if len(pool_arr) >= target_n * 1.5:  # Buffer for filtering
            break

    if not pool_embeddings:
        return np.array([]).reshape(0, 768), llm_calls

    pool_arr = np.vstack(pool_embeddings)

    # Apply filter
    class_mask = real_labels == class_name
    if not class_mask.any():
        return pool_arr[:target_n], llm_calls

    filtered_emb, _, _ = filter_obj.filter_samples(
        candidates=pool_arr,
        real_embeddings=real_embeddings,
        real_labels=real_labels,
        target_class=class_name,
        target_count=target_n
    )

    return filtered_emb[:target_n], llm_calls


def generate_smote_samples(
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_class: str,
    n_generate: int
) -> np.ndarray:
    """Generate SMOTE samples for a specific class."""
    # Get class mask
    class_mask = real_labels == target_class

    if not class_mask.any():
        return np.array([]).reshape(0, real_embeddings.shape[1])

    # SMOTE needs at least k_neighbors + 1 samples
    n_class = class_mask.sum()
    k_neighbors = min(5, n_class - 1)
    if k_neighbors < 1:
        return np.array([]).reshape(0, real_embeddings.shape[1])

    # Create binary labels for SMOTE (target class vs rest)
    binary_labels = class_mask.astype(int)

    # Calculate target count
    n_majority = (~class_mask).sum()
    target_count = n_class + n_generate

    try:
        smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy={1: target_count},
            random_state=42
        )
        X_resampled, y_resampled = smote.fit_resample(real_embeddings, binary_labels)

        # Extract only new samples (those beyond original)
        new_samples = X_resampled[len(real_embeddings):]
        return new_samples[:n_generate]

    except Exception as e:
        print(f"    SMOTE error for {target_class}: {e}")
        return np.array([]).reshape(0, real_embeddings.shape[1])


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_baseline(train_emb, train_labels, test_emb, test_labels) -> float:
    """Evaluate baseline (no augmentation)."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    return f1_score(test_labels, clf.predict(test_emb), average='macro')


def evaluate_augmented(
    train_emb, train_labels, test_emb, test_labels,
    synth_emb, synth_labels
) -> float:
    """Evaluate with augmented data."""
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

def run_hybrid_experiment(
    dataset_name: str,
    scale: int,
    mode: Dict,
    model: SentenceTransformer,
    provider,
    filter_obj: FilterCascade
) -> HybridResult:
    """Run one hybrid configuration."""
    print(f"\n    Mode: {mode['name']}")

    # Load and subset dataset
    full_data = load_full_dataset(dataset_name)
    subset = create_stratified_subset(full_data, n_per_class=scale)

    train_texts = subset["train_texts"]
    train_labels = subset["train_labels"]
    classes = list(set(train_labels))
    n_classes = len(classes)

    # Embed
    train_emb = model.encode(train_texts, show_progress_bar=False)
    test_emb = model.encode(subset["test_texts"], show_progress_bar=False)
    train_labels_arr = np.array(train_labels)

    # Baseline
    baseline_f1 = evaluate_baseline(train_emb, train_labels_arr, test_emb, subset["test_labels"])

    # Calculate split
    llm_per_class = int(SYNTHETIC_PER_CLASS * mode["llm_ratio"])
    smote_per_class = SYNTHETIC_PER_CLASS - llm_per_class

    all_synth_emb = []
    all_synth_labels = []
    total_llm_calls = 0
    total_llm_samples = 0
    total_smote_samples = 0

    for cls in classes:
        cls_texts = [t for t, l in zip(train_texts, train_labels) if l == cls]
        n_shot_actual = min(N_SHOT, len(cls_texts))

        # Step 1: Generate LLM samples (if any)
        llm_emb = np.array([]).reshape(0, 768)
        if llm_per_class > 0:
            llm_emb, llm_calls = generate_llm_filtered(
                provider=provider,
                filter_obj=filter_obj,
                target_n=llm_per_class,
                class_name=cls,
                class_texts=cls_texts[:n_shot_actual],
                real_embeddings=train_emb,
                real_labels=train_labels_arr,
                model=model,
                dataset_name=dataset_name
            )
            total_llm_calls += llm_calls
            total_llm_samples += len(llm_emb)

        # Step 2: Generate SMOTE on (Real + LLM)
        smote_emb = np.array([]).reshape(0, 768)
        if smote_per_class > 0 and len(train_emb) > 0:
            # Combine real + LLM for SMOTE base
            if len(llm_emb) > 0:
                combined_emb = np.vstack([train_emb, llm_emb])
                combined_labels = np.array(list(train_labels_arr) + [cls] * len(llm_emb))
            else:
                combined_emb = train_emb
                combined_labels = train_labels_arr

            smote_emb = generate_smote_samples(
                combined_emb, combined_labels, cls, smote_per_class
            )
            total_smote_samples += len(smote_emb)

        # Combine both
        if len(llm_emb) > 0:
            all_synth_emb.append(llm_emb)
            all_synth_labels.extend([cls] * len(llm_emb))
        if len(smote_emb) > 0:
            all_synth_emb.append(smote_emb)
            all_synth_labels.extend([cls] * len(smote_emb))

    # Evaluate
    if all_synth_emb:
        synth_emb = np.vstack(all_synth_emb)
    else:
        synth_emb = np.array([]).reshape(0, 768)

    augmented_f1 = evaluate_augmented(
        train_emb, train_labels_arr, test_emb, subset["test_labels"],
        synth_emb, all_synth_labels
    )
    delta = (augmented_f1 - baseline_f1) * 100

    print(f"      Baseline: {baseline_f1:.4f}, Augmented: {augmented_f1:.4f} ({delta:+.2f}pp)")
    print(f"      LLM: {total_llm_samples}, SMOTE: {total_smote_samples}, Calls: {total_llm_calls}")

    return HybridResult(
        dataset=dataset_name,
        scale=scale,
        mode=mode["name"],
        llm_ratio=mode["llm_ratio"],
        smote_ratio=mode["smote_ratio"],
        baseline_f1=baseline_f1,
        augmented_f1=augmented_f1,
        delta=delta,
        llm_samples=total_llm_samples,
        smote_samples=total_smote_samples,
        total_samples=len(synth_emb),
        llm_calls=total_llm_calls
    )


def run_baseline_experiment(
    dataset_name: str,
    scale: int,
    baseline_type: str,
    model: SentenceTransformer,
    provider,
    filter_obj: FilterCascade
) -> HybridResult:
    """Run baseline (LLM only or SMOTE only)."""
    print(f"\n    Baseline: {baseline_type}")

    if baseline_type == "llm_only":
        mode = {"name": "llm_only", "llm_ratio": 1.0, "smote_ratio": 0.0}
    else:  # smote_only
        mode = {"name": "smote_only", "llm_ratio": 0.0, "smote_ratio": 1.0}

    return run_hybrid_experiment(dataset_name, scale, mode, model, provider, filter_obj)


def main():
    print("=" * 70)
    print("HYBRID LLM + SMOTE AUGMENTATION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Hybrid modes: {[m['name'] for m in HYBRID_MODES]}")
    print(f"  Baselines: {BASELINES}")
    print(f"  Scales: {SCALES}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Synthetic per class: {SYNTHETIC_PER_CLASS}")

    # Initialize
    print("\nLoading models...")
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    print("Initializing LLM provider...")
    provider = create_provider(LLM_PROVIDER, LLM_MODEL)

    filter_obj = FilterCascade(**LLM_FILTER)

    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        for scale in SCALES:
            print(f"\n  Scale: {scale} samples/class")
            print(f"  {'-' * 50}")

            # Run baselines
            for baseline in BASELINES:
                try:
                    result = run_baseline_experiment(
                        dataset_name, scale, baseline, model, provider, filter_obj
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"    ERROR ({baseline}): {e}")

            # Run hybrid modes
            for mode in HYBRID_MODES:
                try:
                    result = run_hybrid_experiment(
                        dataset_name, scale, mode, model, provider, filter_obj
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"    ERROR ({mode['name']}): {e}")

    # Summary
    if all_results:
        print_summary(all_results)
        save_results(all_results)
    else:
        print("\nNo results collected!")


def print_summary(all_results: List[HybridResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: HYBRID AUGMENTATION")
    print("=" * 70)

    for dataset in DATASETS:
        print(f"\n{dataset}:")
        print(f"{'Scale':<8} {'Mode':<20} {'F1':<10} {'Delta':<12}")
        print("-" * 55)

        for scale in SCALES:
            scale_results = [r for r in all_results
                          if r.dataset == dataset and r.scale == scale]

            for r in sorted(scale_results, key=lambda x: x.delta, reverse=True):
                print(f"{r.scale:<8} {r.mode:<20} {r.augmented_f1:<10.4f} {r.delta:+10.2f}pp")

            print()


def save_results(all_results: List[HybridResult]):
    """Save results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"hybrid_results_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": {
            "hybrid_modes": HYBRID_MODES,
            "scales": SCALES,
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
