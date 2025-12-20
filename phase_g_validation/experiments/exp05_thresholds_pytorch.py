#!/usr/bin/env python3
"""
Experiment 05: Adaptive Thresholds validation with PyTorch MLP

Tests:
- strict: 0.90
- medium: 0.70
- permissive: 0.60
- adaptive: based on anchor purity
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

# Threshold configurations
THRESHOLD_CONFIGS = {
    "strict": {"similarity_threshold": 0.90, "adaptive": False},
    "medium": {"similarity_threshold": 0.70, "adaptive": False},
    "permissive": {"similarity_threshold": 0.60, "adaptive": False},
    "adaptive": {"similarity_threshold": None, "adaptive": True},
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


def compute_adaptive_threshold(class_embeddings):
    """Compute adaptive threshold based on class cohesion."""
    if len(class_embeddings) < 5:
        return 0.70  # Default for small classes

    # Compute average pairwise similarity
    sims = cosine_similarity(class_embeddings)
    np.fill_diagonal(sims, 0)
    avg_sim = np.mean(sims)

    # Adaptive threshold: more cohesive classes get stricter thresholds
    if avg_sim > 0.7:
        return 0.85
    elif avg_sim > 0.5:
        return 0.70
    else:
        return 0.55


def filter_by_threshold(synth_texts, synth_emb, class_emb, threshold_config):
    """Filter synthetics by similarity threshold."""
    if not synth_texts:
        return [], np.array([])

    accepted_texts = []
    accepted_emb = []

    if threshold_config["adaptive"]:
        threshold = compute_adaptive_threshold(class_emb)
    else:
        threshold = threshold_config["similarity_threshold"]

    for text, emb in zip(synth_texts, synth_emb):
        sims = cosine_similarity([emb], class_emb)[0]
        max_sim = np.max(sims)
        if max_sim >= threshold:
            accepted_texts.append(text)
            accepted_emb.append(emb)

    return accepted_texts, np.array(accepted_emb) if accepted_emb else np.array([])


def main():
    print("=" * 70)
    print("Experiment 05: Adaptive Thresholds Validation (PyTorch MLP)")
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

    for thresh_name, thresh_config in THRESHOLD_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Testing threshold: {thresh_name}")
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
                    synth_emb = cache.embed_synthetic(synth_texts)

                    # Apply threshold filter
                    filtered_texts, filtered_emb = filter_by_threshold(
                        synth_texts, synth_emb, class_emb, thresh_config
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
            config_name="threshold",
            config_value=thresh_name,
            dropout=0.2
        )
        result.extra_metrics = {
            "threshold_config": thresh_name,
            "total_generated": total_generated,
            "acceptance_rate": acceptance_rate
        }
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "adaptive_thresholds",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "threshold_configs": {k: str(v) for k, v in THRESHOLD_CONFIGS.items()},
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp05_thresholds.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Adaptive Thresholds Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_thresholds}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Threshold & $\Delta$ (pp) & p-value & Accept\% & Synth \\",
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

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_thresholds.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Thresholds Experiment")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Δ (pp)':<10} {'Accept%':<10} {'Synth':<8}")
    print("-" * 42)
    for r in results:
        acc = r.extra_metrics.get("acceptance_rate", 0) * 100
        print(f"{r.config_value:<12} {r.delta_pp:+.2f}      {acc:.0f}%       {r.n_synthetic}")

    print(f"\nResults saved to {output_dir / 'exp05_thresholds.json'}")


if __name__ == "__main__":
    main()
