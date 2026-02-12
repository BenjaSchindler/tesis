#!/usr/bin/env python3
"""
Experiment: LLM Augmentation on Benchmark Datasets

Tests different LLM percentages (5%, 25%, 50%, 100%) with geometric filtering
on low-resource datasets where LLM should have advantage over SMOTE.

Hypothesis: In low-resource settings (<50 samples/class), LLM + filtering
will outperform SMOTE because:
1. SMOTE has too few points to interpolate well
2. LLM can use domain knowledge
3. Geometric filtering removes bad generations

Datasets:
- hate_speech_davidson (10-shot, 25-shot, 50-shot)
- ag_news_synthetic (for pipeline testing)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE

from core.llm_providers import create_provider
from core.geometric_filter import LOFFilter

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configs
LLM_PERCENTAGES = [5, 25, 50, 100]  # % of synthetic data from LLM
N_SYNTHETIC_PER_CLASS = 50  # Target synthetic samples per class
LOF_THRESHOLD = -0.3
N_FOLDS = 5


def load_dataset(dataset_path: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load a benchmark dataset."""
    with open(dataset_path) as f:
        data = json.load(f)
    return (
        data['train_texts'],
        data['train_labels'],
        data['test_texts'],
        data['test_labels']
    )


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Embed texts using sentence transformer."""
    return model.encode(texts, show_progress_bar=False)


def create_prompt(class_name: str, examples: List[str], n_generate: int) -> str:
    """Create a generation prompt."""
    examples_text = "\n\n".join([f"Example {i+1}: {ex[:300]}" for i, ex in enumerate(examples[:10])])

    return f"""You are an expert at generating realistic text examples for classification.

Class: {class_name}

Here are real examples from this class:
{examples_text}

Generate {n_generate} NEW examples that belong to the "{class_name}" class.
Each example should be 1-3 sentences, similar in style to the examples above.
Generate one example per line:"""


def generate_llm_samples(
    provider,
    class_name: str,
    class_texts: List[str],
    n_generate: int,
    model: SentenceTransformer
) -> Tuple[np.ndarray, List[str]]:
    """Generate LLM samples for a class."""
    prompt = create_prompt(class_name, class_texts, n_generate)

    try:
        messages = [{"role": "user", "content": prompt}]
        response, _ = provider.generate(messages, temperature=0.8, max_tokens=2000)

        # Parse response
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        generated = []
        for line in lines:
            clean = line.lstrip('0123456789.-):* ')
            if len(clean) > 10:
                generated.append(clean)

        if not generated:
            return np.array([]).reshape(0, 768), []

        # Embed
        embeddings = model.encode(generated, show_progress_bar=False)
        return embeddings, generated

    except Exception as e:
        print(f"      Error generating: {e}")
        return np.array([]).reshape(0, 768), []


def apply_lof_filter(
    gen_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    threshold: float = LOF_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply LOF filtering to generated embeddings."""
    if len(gen_embeddings) == 0:
        return gen_embeddings, np.array([])

    from sklearn.neighbors import LocalOutlierFactor

    # Fit LOF on real data
    n_neighbors = min(20, len(real_embeddings) - 1)
    if n_neighbors < 2:
        return gen_embeddings, np.zeros(len(gen_embeddings))

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(real_embeddings)

    # Score generated samples
    scores = lof.decision_function(gen_embeddings)

    # Filter
    mask = scores > threshold
    filtered = gen_embeddings[mask]

    return filtered, scores


def generate_smote_samples(
    real_embeddings: np.ndarray,
    n_generate: int,
    k_neighbors: int = 5
) -> np.ndarray:
    """Generate SMOTE samples."""
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
        print(f"      SMOTE error: {e}")
        return np.array([]).reshape(0, real_embeddings.shape[1])


def run_experiment_config(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    llm_pct: int,
    model: SentenceTransformer,
    provider
) -> Dict:
    """Run experiment with specific LLM percentage."""

    print(f"\n  Config: {llm_pct}% LLM, {100-llm_pct}% SMOTE")

    # Embed all data
    train_embeddings = embed_texts(train_texts, model)
    test_embeddings = embed_texts(test_texts, model)

    unique_classes = list(set(train_labels))
    n_classes = len(unique_classes)

    # Generate synthetic data
    all_synthetic_emb = []
    all_synthetic_labels = []
    generation_stats = {}

    for cls in unique_classes:
        cls_mask = np.array([l == cls for l in train_labels])
        cls_embeddings = train_embeddings[cls_mask]
        cls_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i] == cls]

        n_llm = int(N_SYNTHETIC_PER_CLASS * llm_pct / 100)
        n_smote = N_SYNTHETIC_PER_CLASS - n_llm

        llm_emb = np.array([]).reshape(0, train_embeddings.shape[1])
        smote_emb = np.array([]).reshape(0, train_embeddings.shape[1])

        # Generate LLM samples
        if n_llm > 0:
            n_generate = int(n_llm * 2)  # Oversample for filtering
            gen_emb, gen_texts = generate_llm_samples(
                provider, cls, cls_texts, n_generate, model
            )

            if len(gen_emb) > 0:
                # Apply LOF filter
                filtered_emb, lof_scores = apply_lof_filter(gen_emb, cls_embeddings)

                # Take top n_llm by LOF score
                if len(filtered_emb) > n_llm:
                    # Select best by similarity to centroid
                    centroid = cls_embeddings.mean(axis=0)
                    from sklearn.metrics.pairwise import cosine_similarity
                    sims = cosine_similarity(filtered_emb, [centroid]).flatten()
                    top_idx = np.argsort(sims)[-n_llm:]
                    llm_emb = filtered_emb[top_idx]
                else:
                    llm_emb = filtered_emb[:n_llm]

        # Generate SMOTE samples
        if n_smote > 0:
            # Use LLM samples as additional anchors if available
            if len(llm_emb) > 0:
                base_emb = np.vstack([cls_embeddings, llm_emb])
            else:
                base_emb = cls_embeddings

            smote_emb = generate_smote_samples(base_emb, n_smote)

        # Combine
        if len(llm_emb) > 0 and len(smote_emb) > 0:
            combined = np.vstack([llm_emb, smote_emb])
        elif len(llm_emb) > 0:
            combined = llm_emb
        elif len(smote_emb) > 0:
            combined = smote_emb
        else:
            combined = np.array([]).reshape(0, train_embeddings.shape[1])

        if len(combined) > 0:
            all_synthetic_emb.append(combined)
            all_synthetic_labels.extend([cls] * len(combined))

        generation_stats[cls] = {
            'llm_samples': len(llm_emb),
            'smote_samples': len(smote_emb),
            'total': len(combined)
        }

    # Combine synthetic data
    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])
    synthetic_labels = np.array(all_synthetic_labels)

    print(f"    Generated: {len(synthetic_labels)} synthetic samples")

    # Evaluate with cross-validation on test set
    # Combine train + synthetic for training
    if len(synthetic_embeddings) > 0:
        aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
        aug_labels = train_labels + list(synthetic_labels)
    else:
        aug_embeddings = train_embeddings
        aug_labels = train_labels

    # Train classifier and evaluate
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_augmented = LogisticRegression(max_iter=1000, random_state=42)

    # Baseline (no augmentation)
    clf_baseline.fit(train_embeddings, train_labels)
    baseline_pred = clf_baseline.predict(test_embeddings)
    baseline_f1 = f1_score(test_labels, baseline_pred, average='macro')

    # Augmented
    clf_augmented.fit(aug_embeddings, aug_labels)
    aug_pred = clf_augmented.predict(test_embeddings)
    aug_f1 = f1_score(test_labels, aug_pred, average='macro')

    delta = aug_f1 - baseline_f1

    print(f"    Baseline F1: {baseline_f1:.4f}")
    print(f"    Augmented F1: {aug_f1:.4f}")
    print(f"    Delta: {delta*100:+.2f} pp")

    return {
        'llm_pct': llm_pct,
        'baseline_f1': float(baseline_f1),
        'augmented_f1': float(aug_f1),
        'delta_pp': float(delta * 100),
        'n_synthetic': len(synthetic_labels),
        'generation_stats': generation_stats
    }


def run_smote_baseline(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    model: SentenceTransformer
) -> Dict:
    """Run pure SMOTE baseline."""

    print(f"\n  SMOTE Baseline (100% SMOTE)")

    train_embeddings = embed_texts(train_texts, model)
    test_embeddings = embed_texts(test_texts, model)

    unique_classes = list(set(train_labels))

    all_synthetic_emb = []
    all_synthetic_labels = []

    for cls in unique_classes:
        cls_mask = np.array([l == cls for l in train_labels])
        cls_embeddings = train_embeddings[cls_mask]

        smote_emb = generate_smote_samples(cls_embeddings, N_SYNTHETIC_PER_CLASS)

        if len(smote_emb) > 0:
            all_synthetic_emb.append(smote_emb)
            all_synthetic_labels.extend([cls] * len(smote_emb))

    if all_synthetic_emb:
        synthetic_embeddings = np.vstack(all_synthetic_emb)
    else:
        synthetic_embeddings = np.array([]).reshape(0, train_embeddings.shape[1])
    synthetic_labels = np.array(all_synthetic_labels)

    # Combine and evaluate
    aug_embeddings = np.vstack([train_embeddings, synthetic_embeddings])
    aug_labels = train_labels + list(synthetic_labels)

    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_augmented = LogisticRegression(max_iter=1000, random_state=42)

    clf_baseline.fit(train_embeddings, train_labels)
    baseline_pred = clf_baseline.predict(test_embeddings)
    baseline_f1 = f1_score(test_labels, baseline_pred, average='macro')

    clf_augmented.fit(aug_embeddings, aug_labels)
    aug_pred = clf_augmented.predict(test_embeddings)
    aug_f1 = f1_score(test_labels, aug_pred, average='macro')

    delta = aug_f1 - baseline_f1

    print(f"    Baseline F1: {baseline_f1:.4f}")
    print(f"    SMOTE F1: {aug_f1:.4f}")
    print(f"    Delta: {delta*100:+.2f} pp")

    return {
        'llm_pct': 0,
        'baseline_f1': float(baseline_f1),
        'augmented_f1': float(aug_f1),
        'delta_pp': float(delta * 100),
        'n_synthetic': len(synthetic_labels)
    }


def main():
    """Main entry point."""
    print("="*70)
    print("BENCHMARK EXPERIMENT: LLM % vs SMOTE")
    print("Testing on low-resource datasets")
    print("="*70)

    # Initialize
    print("\nLoading model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Initializing LLM provider...")
    provider = create_provider("google", "gemini-2.0-flash")

    # Find datasets
    datasets = list(DATA_DIR.glob("*_*shot.json"))
    print(f"\nFound {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d.stem}")

    all_results = {}

    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")

        # Load data
        train_texts, train_labels, test_texts, test_labels = load_dataset(dataset_path)
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
        print(f"  Classes: {set(train_labels)}")
        print(f"  Distribution: {dict(Counter(train_labels))}")

        results = {'dataset': dataset_name, 'configs': []}

        # Run SMOTE baseline
        smote_result = run_smote_baseline(
            train_texts, train_labels, test_texts, test_labels, model
        )
        results['smote_baseline'] = smote_result

        # Run each LLM percentage
        for llm_pct in LLM_PERCENTAGES:
            try:
                config_result = run_experiment_config(
                    train_texts, train_labels, test_texts, test_labels,
                    llm_pct, model, provider
                )
                results['configs'].append(config_result)
            except Exception as e:
                print(f"    Error with {llm_pct}% LLM: {e}")
                import traceback
                traceback.print_exc()

        all_results[dataset_name] = results

        # Save checkpoint
        output_path = RESULTS_DIR / f"benchmark_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Dataset':<35} {'SMOTE':>10} {'5% LLM':>10} {'25% LLM':>10} {'50% LLM':>10} {'100% LLM':>10}")
    print("-" * 90)

    for dataset_name, results in all_results.items():
        smote_delta = results['smote_baseline']['delta_pp']

        row = f"{dataset_name:<35} {smote_delta:>+9.2f}pp"

        for config in results['configs']:
            delta = config['delta_pp']
            row += f" {delta:>+9.2f}pp"

        print(row)

    # Find best config per dataset
    print("\n" + "-"*70)
    print("BEST CONFIG PER DATASET:")
    print("-"*70)

    for dataset_name, results in all_results.items():
        all_configs = [results['smote_baseline']] + results['configs']
        best = max(all_configs, key=lambda x: x['delta_pp'])

        if best['llm_pct'] == 0:
            best_name = "SMOTE"
        else:
            best_name = f"{best['llm_pct']}% LLM"

        beats_smote = best['delta_pp'] > results['smote_baseline']['delta_pp']
        status = "BEATS SMOTE" if beats_smote and best['llm_pct'] > 0 else ""

        print(f"  {dataset_name}: {best_name} ({best['delta_pp']:+.2f} pp) {status}")


if __name__ == "__main__":
    main()
