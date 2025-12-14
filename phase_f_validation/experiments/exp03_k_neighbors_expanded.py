#!/usr/bin/env python3
"""
Experiment 03 Expanded: K Neighbors Validation - Extended Range

Tests K values: [5, 8, 10, 12, 15, 18, 20, 25, 30, 50, 75, 100, 125, 150, 200]
Focus:
- Granular 10-25 range (8, 12, 18, 20, 30)
- Explore beyond 100 (125, 150, 200)

Metrics: Macro F1, Delta%, Acceptance Rate, Context Quality

Output: tab:k_neighbors_expanded for Metodologia.tex
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from typing import List, Tuple

from base_config import (
    RESULTS_DIR, LATEX_DIR, BASE_PARAMS, KFOLD_CONFIG, EXPERIMENT_PARAMS,
    LLM_MODEL, TEMPERATURE, MAX_TOKENS
)
from validation_runner import load_data, EmbeddingCache, KFoldEvaluator, print_summary


# Use expanded K values
K_VALUES = EXPERIMENT_PARAMS["k_neighbors"]["K_VALUES_EXPANDED"]
EXPERIMENT_NAME = "k_neighbors_expanded"


@dataclass
class KNeighborsResult:
    k_value: int
    macro_f1: float
    acceptance_rate: float
    context_quality: str
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool
    win_rate: float


def assess_context_quality(k: int) -> str:
    """Assess context quality based on K value (extended range)."""
    if k <= 5:
        return "Insuficiente"
    elif k <= 10:
        return "Limitado"
    elif k <= 20:
        return "Optimo"
    elif k <= 50:
        return "Redundante"
    elif k <= 100:
        return "Ruidoso"
    elif k <= 150:
        return "Muy Ruidoso"
    else:
        return "Excesivo"


def generate_synthetic_with_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    k_neighbors: int,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate synthetic samples using K neighbors as context for LLM prompt."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_texts = []
    synthetic_labels_list = []
    acceptance_counts = []
    total_counts = []

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Cluster the class
        n_clusters = min(BASE_PARAMS["max_clusters"], len(class_embeddings) // 20)
        if n_clusters < 1:
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        for c_id in range(n_clusters):
            c_mask = cluster_labels_arr == c_id
            c_points = class_embeddings[c_mask]
            c_texts = class_texts[c_mask]

            if len(c_points) < 3:
                continue

            centroid = kmeans.cluster_centers_[c_id]

            # Get K nearest neighbors for context (key experiment variable)
            dists = np.linalg.norm(c_points - centroid, axis=1)
            k_actual = min(k_neighbors, len(c_points))
            nearest_indices = np.argsort(dists)[:k_actual]

            # Use K neighbors as examples in prompt (max 5 for LLM context)
            example_texts = [c_texts[i] for i in nearest_indices[:min(5, k_actual)]]

            # Context quality based on K neighbors coverage
            context_quality = min(1.0, k_actual / k_neighbors) if k_neighbors > 0 else 0.0

            # Create prompt with K context examples
            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])
            n_samples = max(1, BASE_PARAMS["samples_per_prompt"] // 2)

            prompt = f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are {len(example_texts)} examples of posts from this personality type:
{examples_text}

Generate {n_samples} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS * n_samples,
                )

                generated_text = response.choices[0].message.content.strip()
                samples = [s.strip() for s in generated_text.split('\n')
                          if s.strip() and len(s.strip()) > 10]

                accepted = 0
                for sample in samples[:n_samples]:
                    synthetic_texts.append(sample)
                    synthetic_labels_list.append(target_class)
                    accepted += 1

                acceptance_counts.append(accepted)
                total_counts.append(n_samples)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                acceptance_counts.append(0)
                total_counts.append(n_samples)
                continue

    acceptance_rate = sum(acceptance_counts) / sum(total_counts) if total_counts else 0.0

    if not synthetic_texts:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), 0.0

    # Embed synthetic texts
    print(f"    Embedding {len(synthetic_texts)} synthetic texts...", flush=True)
    synthetic_embeddings = cache.embed_synthetic(synthetic_texts)

    return synthetic_embeddings, np.array(synthetic_labels_list), acceptance_rate


def run_k_neighbors_experiment() -> list:
    """Run expanded K neighbors validation experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 03 EXPANDED: K NEIGHBORS VALIDATION")
    print("="*60)
    print(f"  K values: {K_VALUES}")
    print(f"  Total configurations: {len(K_VALUES)}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for i, k in enumerate(K_VALUES):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(K_VALUES)}] Testing K = {k}")
        print(f"{'─'*60}")

        X_synth, y_synth, acceptance_rate = generate_synthetic_with_k(embeddings, labels, texts, k, cache)
        context_quality = assess_context_quality(k)

        print(f"  Acceptance Rate: {acceptance_rate*100:.1f}%")
        print(f"  Context: {context_quality}")
        print(f"  Generated {len(X_synth)} synthetic samples")

        kfold_result = evaluator.evaluate(
            X_original=embeddings,
            y_original=labels,
            X_synthetic=X_synth if len(X_synth) > 0 else None,
            y_synthetic=y_synth if len(y_synth) > 0 else None,
            config_name=f"K{k}"
        )

        result = KNeighborsResult(
            k_value=int(k),
            macro_f1=float(kfold_result.augmented_mean),
            acceptance_rate=float(acceptance_rate),
            context_quality=context_quality,
            delta_pct=float(kfold_result.delta_pct),
            n_synthetic=int(kfold_result.n_synthetic),
            p_value=float(kfold_result.p_value),
            significant=bool(kfold_result.significant),
            win_rate=float(kfold_result.win_rate)
        )
        results.append(result)

        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"K{k}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print_summary(kfold_result)

    return results


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table for expanded K neighbors."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto expandido del numero de vecinos K en rendimiento}
\label{tab:k_neighbors_expanded}
\begin{tabular}{lcccc}
\hline
K vecinos & Macro F1 & $\Delta$ F1 & Acceptance & Contexto \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sig_marker = "*" if r.significant else ""
        if i == best_idx:
            latex += f"\\textbf{{{r.k_value}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.delta_pct:+.2f}\\%{sig_marker}}} & \\textbf{{{r.acceptance_rate*100:.0f}\\%}} & \\textbf{{{r.context_quality}}} \\\\\n"
        else:
            latex += f"{r.k_value} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig_marker} & {r.acceptance_rate*100:.0f}\\% & {r.context_quality} \\\\\n"

    latex += r"""
\hline
\multicolumn{5}{l}{\footnotesize *Estadisticamente significativo (p $<$ 0.05)} \\
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_k_neighbors_experiment()

    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_k_neighbors_expanded.tex", 'w') as f:
        f.write(latex_table)

    # Save all results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print("\n" + "="*70)
    print("  SUMMARY: Expanded K Neighbors Results")
    print("="*70)
    print(f"{'K':>5} | {'F1':>8} | {'Delta':>8} | {'Acc':>6} | {'Context':>12} | {'Sig':>4}")
    print("-"*70)
    for r in results:
        sig_str = "*" if r.significant else ""
        print(f"{r.k_value:>5} | {r.macro_f1:.4f} | {r.delta_pct:+.2f}% | {r.acceptance_rate*100:>5.0f}% | {r.context_quality:>12} | {sig_str:>4}")

    # Find best
    best_idx = np.argmax([r.delta_pct for r in results])
    best = results[best_idx]
    print("-"*70)
    print(f"  BEST: K={best.k_value} with Delta F1={best.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
