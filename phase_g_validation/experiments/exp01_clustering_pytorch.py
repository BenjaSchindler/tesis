#!/usr/bin/env python3
"""
Experiment 01: Clustering (K_max) validation with PyTorch MLP

Tests: K_max = 1, 3, 6, 12, 18, 24
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import asdict

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from pytorch_mlp_classifier import run_pytorch_kfold, print_result_summary, DEVICE
from base_config import RESULTS_DIR

print(f"Using device: {DEVICE}")

# K_max values to test
KMAX_VALUES = [1, 3, 6, 12, 18, 24]

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def main():
    print("=" * 70)
    print("Experiment 01: Clustering (K_max) Validation (PyTorch MLP)")
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

    for k_max in KMAX_VALUES:
        print(f"\n{'='*70}")
        print(f"Testing K_max = {k_max}")
        print("=" * 70)

        params = BASE_CONFIG.copy()
        params["max_clusters"] = k_max

        # Generate synthetics
        print("Generating synthetics...")
        generator = SyntheticGenerator(cache, params)

        all_synth_emb = []
        all_synth_labels = []

        for label in unique_labels:
            class_mask = np.array(labels) == label
            class_texts = texts[class_mask]
            class_emb = embeddings[class_mask]

            try:
                synth_texts, _ = generator.generate_for_class(
                    class_texts, class_emb, label
                )
                if synth_texts:
                    synth_emb = cache.embed_synthetic(synth_texts)
                    all_synth_emb.append(synth_emb)
                    all_synth_labels.extend([label] * len(synth_emb))
                    print(f"  {label}: +{len(synth_emb)} synthetic")
            except Exception as e:
                print(f"  {label}: Error - {e}")

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        print(f"Total synthetic: {len(X_synth)}")

        result = run_pytorch_kfold(
            embeddings, labels,
            X_synth, y_synth,
            unique_labels,
            config_name="max_clusters",
            config_value=k_max,
            dropout=0.2
        )
        result.extra_metrics = {"max_clusters": k_max}
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "clustering",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "values_tested": KMAX_VALUES,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp01_clustering.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Clustering ($K_{max}$) Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_clustering}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$K_{max}$ & Baseline & Augmented & $\Delta$ (pp) & p-value & Win\% & Synth \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r.significant else ""
        latex_lines.append(
            f"{r.config_value} & {r.baseline_mean:.4f} & {r.augmented_mean:.4f} & "
            f"{r.delta_pp:+.2f}{sig} & {r.p_value:.4f} & {r.win_rate*100:.0f}\\% & {r.n_synthetic} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_clustering.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Clustering Experiment")
    print("=" * 70)
    print(f"{'K_max':<8} {'Δ (pp)':<10} {'p-value':<10} {'Synth':<8}")
    print("-" * 40)
    for r in results:
        print(f"{r.config_value:<8} {r.delta_pp:+.2f}      {r.p_value:.4f}     {r.n_synthetic}")

    print(f"\nResults saved to {output_dir / 'exp01_clustering.json'}")


if __name__ == "__main__":
    main()
