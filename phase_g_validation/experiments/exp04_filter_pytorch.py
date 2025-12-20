#!/usr/bin/env python3
"""
Experiment 04: Filter Cascade validation with PyTorch MLP

Tests:
- none: No filters
- length_only: Length filter (10-200 words)
- length_similarity: + Similarity filter (θ≥0.65)
- three_filters: + K-NN purity (θ≥0.40)
- full_cascade: + Anchor coherence (θ≥0.55)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from pytorch_mlp_classifier import run_pytorch_kfold, print_result_summary, DEVICE
from base_config import RESULTS_DIR

print(f"Using device: {DEVICE}")

# Filter configurations
FILTER_CONFIGS = {
    "none": {
        "length_filter": False,
        "similarity_filter": False,
        "knn_purity_filter": False,
        "anchor_coherence_filter": False,
    },
    "length_only": {
        "length_filter": True,
        "min_words": 10,
        "max_words": 200,
        "similarity_filter": False,
        "knn_purity_filter": False,
        "anchor_coherence_filter": False,
    },
    "length_similarity": {
        "length_filter": True,
        "min_words": 10,
        "max_words": 200,
        "similarity_filter": True,
        "similarity_threshold": 0.65,
        "knn_purity_filter": False,
        "anchor_coherence_filter": False,
    },
    "three_filters": {
        "length_filter": True,
        "min_words": 10,
        "max_words": 200,
        "similarity_filter": True,
        "similarity_threshold": 0.65,
        "knn_purity_filter": True,
        "knn_purity_threshold": 0.40,
        "anchor_coherence_filter": False,
    },
    "full_cascade": {
        "length_filter": True,
        "min_words": 10,
        "max_words": 200,
        "similarity_filter": True,
        "similarity_threshold": 0.65,
        "knn_purity_filter": True,
        "knn_purity_threshold": 0.40,
        "anchor_coherence_filter": True,
        "anchor_coherence_threshold": 0.55,
    },
}

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def apply_filters(texts, embeddings, class_embeddings, filter_config):
    """Apply filter cascade to synthetic texts."""
    if not texts:
        return [], []

    accepted_texts = []
    accepted_emb = []

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        # Length filter
        if filter_config.get("length_filter", False):
            word_count = len(text.split())
            min_w = filter_config.get("min_words", 10)
            max_w = filter_config.get("max_words", 200)
            if word_count < min_w or word_count > max_w:
                continue

        # Similarity filter
        if filter_config.get("similarity_filter", False):
            threshold = filter_config.get("similarity_threshold", 0.65)
            sims = cosine_similarity([emb], class_embeddings)[0]
            max_sim = np.max(sims)
            if max_sim < threshold:
                continue

        # K-NN purity filter (simplified)
        if filter_config.get("knn_purity_filter", False):
            threshold = filter_config.get("knn_purity_threshold", 0.40)
            sims = cosine_similarity([emb], class_embeddings)[0]
            avg_sim = np.mean(np.sort(sims)[-5:])  # Top 5 neighbors
            if avg_sim < threshold:
                continue

        # Anchor coherence filter (simplified)
        if filter_config.get("anchor_coherence_filter", False):
            threshold = filter_config.get("anchor_coherence_threshold", 0.55)
            centroid = np.mean(class_embeddings, axis=0)
            coherence = cosine_similarity([emb], [centroid])[0][0]
            if coherence < threshold:
                continue

        accepted_texts.append(text)
        accepted_emb.append(emb)

    return accepted_texts, np.array(accepted_emb) if accepted_emb else np.array([])


def main():
    print("=" * 70)
    print("Experiment 04: Filter Cascade Validation (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    results = []

    for filter_name, filter_config in FILTER_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Testing filter: {filter_name}")
        print("=" * 70)

        params = BASE_CONFIG.copy()

        # Generate synthetics
        print("Generating synthetics...")
        generator = SyntheticGenerator(cache, params)

        all_synth_emb = []
        all_synth_labels = []
        total_generated = 0
        total_accepted = 0

        for label in unique_labels:
            class_mask = np.array(labels) == label
            class_texts = texts[class_mask]
            class_emb = embeddings[class_mask]

            try:
                synth_texts, _ = generator.generate_for_class(
                    class_texts, class_emb, label
                )
                if synth_texts:
                    total_generated += len(synth_texts)

                    # Embed and filter
                    synth_emb = cache.embed_synthetic(synth_texts)
                    filtered_texts, filtered_emb = apply_filters(
                        synth_texts, synth_emb, class_emb, filter_config
                    )

                    if len(filtered_emb) > 0:
                        all_synth_emb.append(filtered_emb)
                        all_synth_labels.extend([label] * len(filtered_emb))
                        total_accepted += len(filtered_emb)

                    print(f"  {label}: {len(synth_texts)} gen -> {len(filtered_emb)} accepted")
            except Exception as e:
                print(f"  {label}: Error - {e}")

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        acceptance_rate = total_accepted / total_generated if total_generated > 0 else 0
        print(f"Total: {total_generated} generated -> {len(X_synth)} accepted ({acceptance_rate*100:.1f}%)")

        result = run_pytorch_kfold(
            embeddings, labels,
            X_synth, y_synth,
            unique_labels,
            config_name="filter",
            config_value=filter_name,
            dropout=0.2
        )
        result.extra_metrics = {
            "filter_config": filter_name,
            "total_generated": total_generated,
            "acceptance_rate": acceptance_rate
        }
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "filter_cascade",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "filter_configs": FILTER_CONFIGS,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp04_filter.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Filter Cascade Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_filter}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Filter & $\Delta$ (pp) & p-value & Accept\% & Synth \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r.significant else ""
        acc = r.extra_metrics.get("acceptance_rate", 0) * 100
        latex_lines.append(
            f"{r.config_value} & {r.delta_pp:+.2f}{sig} & {r.p_value:.4f} & "
            f"{acc:.0f}\\% & {r.n_synthetic} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_filter.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Filter Cascade Experiment")
    print("=" * 70)
    print(f"{'Filter':<18} {'Δ (pp)':<10} {'Accept%':<10} {'Synth':<8}")
    print("-" * 48)
    for r in results:
        acc = r.extra_metrics.get("acceptance_rate", 0) * 100
        print(f"{r.config_value:<18} {r.delta_pp:+.2f}      {acc:.0f}%       {r.n_synthetic}")

    print(f"\nResults saved to {output_dir / 'exp04_filter.json'}")


if __name__ == "__main__":
    main()
