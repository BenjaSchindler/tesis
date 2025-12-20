#!/usr/bin/env python3
"""
Experiment 07c: Budget validation with PyTorch MLP

Tests: budget = 5%, 10%, 15%, 20%, 25%, 30% of class size
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

# Budget percentages to test
BUDGET_PERCENTAGES = [5, 10, 15, 20, 25, 30]

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def main():
    print("=" * 70)
    print("Experiment 07c: Budget Validation (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    # Calculate class sizes
    class_sizes = {l: np.sum(np.array(labels) == l) for l in unique_labels}
    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")
    print(f"Class size range: {min(class_sizes.values())} - {max(class_sizes.values())}")

    results = []

    for budget_pct in BUDGET_PERCENTAGES:
        print(f"\n{'='*70}")
        print(f"Testing budget = {budget_pct}%")
        print("=" * 70)

        params = BASE_CONFIG.copy()

        # Generate synthetics with budget cap
        print("Generating synthetics...")
        generator = SyntheticGenerator(cache, params)

        all_synth_emb = []
        all_synth_labels = []
        total_budget = 0

        for label in unique_labels:
            class_mask = np.array(labels) == label
            class_texts = texts[class_mask]
            class_emb = embeddings[class_mask]

            # Calculate budget for this class
            class_size = len(class_texts)
            max_synth = int(class_size * budget_pct / 100)
            total_budget += max_synth

            try:
                synth_texts, _ = generator.generate_for_class(
                    class_texts, class_emb, label
                )
                if synth_texts:
                    # Cap to budget
                    synth_texts = synth_texts[:max_synth]
                    if synth_texts:
                        synth_emb = cache.embed_synthetic(synth_texts)
                        all_synth_emb.append(synth_emb)
                        all_synth_labels.extend([label] * len(synth_emb))
                        print(f"  {label}: +{len(synth_emb)} synthetic (budget: {max_synth})")
            except Exception as e:
                print(f"  {label}: Error - {e}")

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        print(f"Total synthetic: {len(X_synth)} (max budget: {total_budget})")

        result = run_pytorch_kfold(
            embeddings, labels,
            X_synth, y_synth,
            unique_labels,
            config_name="budget",
            config_value=f"{budget_pct}%",
            dropout=0.2
        )
        result.extra_metrics = {"budget_pct": budget_pct, "total_budget": total_budget}
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "budget",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "values_tested": BUDGET_PERCENTAGES,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp07c_budget.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Budget Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_budget}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Budget & Baseline & Augmented & $\Delta$ (pp) & p-value & Win\% & Synth \\",
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

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_budget.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Budget Experiment")
    print("=" * 70)
    print(f"{'Budget':<10} {'Δ (pp)':<10} {'p-value':<10} {'Synth':<8}")
    print("-" * 40)
    for r in results:
        print(f"{r.config_value:<10} {r.delta_pp:+.2f}      {r.p_value:.4f}     {r.n_synthetic}")

    print(f"\nResults saved to {output_dir / 'exp07c_budget.json'}")


if __name__ == "__main__":
    main()
