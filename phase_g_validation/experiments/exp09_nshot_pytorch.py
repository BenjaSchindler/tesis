#!/usr/bin/env python3
"""
Experiment 09: N-shot validation with PyTorch MLP

Tests: n_shot = 0, 10, 30, 60, 100, 150, 200
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

# N-shot values to test
NSHOT_VALUES = [0, 10, 30, 60, 100, 150, 200]

# Base config (from high_quality_few that worked)
BASE_CONFIG = {
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def main():
    print("=" * 70)
    print("Experiment 09: N-shot Validation (PyTorch MLP)")
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

    for n_shot in NSHOT_VALUES:
        print(f"\n{'='*70}")
        print(f"Testing n_shot = {n_shot}")
        print("=" * 70)

        params = BASE_CONFIG.copy()
        params["n_shot"] = n_shot

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
            config_name="n_shot",
            config_value=n_shot,
            dropout=0.2
        )
        result.extra_metrics = {"n_shot": n_shot, "temperature": params["temperature"]}
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "n_shot",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "values_tested": NSHOT_VALUES,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp09_nshot.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{N-shot Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_nshot}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$n_{shot}$ & Baseline & Augmented & $\Delta$ (pp) & p-value & Win\% & Synth \\",
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

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_nshot.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: N-shot Experiment")
    print("=" * 70)
    print(f"{'n_shot':<8} {'Δ (pp)':<10} {'p-value':<10} {'Sig':<6}")
    print("-" * 38)
    for r in results:
        sig = "YES" if r.significant else "no"
        print(f"{r.config_value:<8} {r.delta_pp:+.2f}      {r.p_value:.4f}     {sig}")

    print(f"\nResults saved to {output_dir / 'exp09_nshot.json'}")


if __name__ == "__main__":
    main()
