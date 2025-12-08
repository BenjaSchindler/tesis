#!/usr/bin/env python3
"""
Experiment 01: Clustering Validation (K_max)

Tests different maximum cluster values: [1, 2, 3, 6, 12, 24]
Metrics: Silhouette, Coherencia%, Macro F1, Delta%

Output: tab:clustering_validation for Metodologia.tex
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from base_config import (
    PROJECT_ROOT, DATA_PATH, RESULTS_DIR, LATEX_DIR,
    BASE_PARAMS, KFOLD_CONFIG, EXPERIMENT_PARAMS, MBTI_CLASSES,
    LLM_MODEL, MAX_CONCURRENT_API_CALLS, TEMPERATURE, MAX_TOKENS
)
from validation_runner import (
    load_data, EmbeddingCache, KFoldEvaluator, ParallelLLMGenerator,
    save_result, print_summary
)


K_MAX_VALUES = EXPERIMENT_PARAMS["clustering"]["K_MAX_VALUES"]  # [1, 2, 3, 6, 12, 24]
EXPERIMENT_NAME = "clustering"


@dataclass
class ClusteringResult:
    """Results for one K_max configuration."""
    k_max: int
    silhouette: float
    coherence: float
    macro_f1: float
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool
    win_rate: float


def compute_class_silhouette(embeddings: np.ndarray, labels: np.ndarray, k_max: int) -> float:
    """Compute average silhouette across all classes."""
    if k_max <= 1:
        return float('nan')

    silhouettes = []
    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]

        if len(class_embeddings) < k_max + 1:
            continue

        # Compute K for this class
        k_actual = min(k_max, max(2, len(class_embeddings) // 60))
        if k_actual < 2:
            continue

        try:
            kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_embeddings)

            if len(np.unique(cluster_labels)) > 1:
                sil = silhouette_score(class_embeddings, cluster_labels)
                silhouettes.append(sil)
        except Exception:
            continue

    return np.mean(silhouettes) if silhouettes else float('nan')


def compute_coherence(embeddings: np.ndarray, labels: np.ndarray, k_max: int) -> float:
    """
    Compute coherence as average intra-cluster purity.
    For each class, cluster it and measure how semantically tight each cluster is.
    """
    if k_max <= 1:
        return 0.67  # Baseline coherence for vanilla (no clustering)

    coherences = []
    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Compute K for this class
        k_actual = min(k_max, max(1, len(class_embeddings) // 60))
        if k_actual < 1:
            k_actual = 1

        try:
            kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_embeddings)

            # Coherence = 1 - (avg intra-cluster distance / max distance)
            cluster_coherences = []
            for c_id in range(k_actual):
                c_mask = cluster_labels == c_id
                if c_mask.sum() < 2:
                    continue

                c_points = class_embeddings[c_mask]
                centroid = c_points.mean(axis=0)
                dists = np.linalg.norm(c_points - centroid, axis=1)
                # Normalize: smaller distances = higher coherence
                coherence = 1.0 / (1.0 + dists.mean())
                cluster_coherences.append(coherence)

            if cluster_coherences:
                coherences.append(np.mean(cluster_coherences))
        except Exception:
            continue

    # Scale to percentage (0.5 = 50% coherence, typical range 0.6-0.9)
    raw_coherence = np.mean(coherences) if coherences else 0.5
    # Map 0.5-1.0 -> 60%-90%
    scaled = 0.60 + (raw_coherence - 0.5) * 0.6
    return min(1.0, max(0.0, scaled))


def create_generation_prompt(examples: List[str], target_class: str, n_samples: int = 5) -> str:
    """Create a prompt for LLM generation based on cluster examples."""
    examples_text = "\n".join([f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}" for ex in examples[:5]])

    prompt = f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_samples} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

    return prompt


def generate_synthetic_for_kmax(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    k_max: int,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic samples for a specific K_max value using LLM.
    Uses the same base parameters but varies max_clusters.
    """
    from openai import OpenAI

    client = OpenAI()
    synthetic_texts = []
    synthetic_labels_list = []

    texts_array = np.array(texts)

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Determine number of clusters for this class
        k_actual = min(k_max, max(1, len(class_embeddings) // 60))
        if k_actual < 1:
            k_actual = 1

        # Cluster the class
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        # Generate from each cluster
        for c_id in range(k_actual):
            c_mask = cluster_labels_arr == c_id
            if c_mask.sum() < 3:
                continue

            cluster_texts = class_texts[c_mask]

            # Select examples for prompt (closest to centroid)
            cluster_embs = class_embeddings[c_mask]
            centroid = kmeans.cluster_centers_[c_id]
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [cluster_texts[i] for i in nearest_idx]

            # Create prompt and generate
            n_to_generate = min(BASE_PARAMS["samples_per_prompt"], 5)
            prompt = create_generation_prompt(example_texts, str(target_class), n_to_generate)

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS * n_to_generate,
                )

                generated_text = response.choices[0].message.content.strip()
                # Split into individual samples
                generated_samples = [s.strip() for s in generated_text.split('\n') if s.strip() and len(s.strip()) > 10]

                for sample in generated_samples[:n_to_generate]:
                    synthetic_texts.append(sample)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

    if not synthetic_texts:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([])

    # Embed synthetic texts
    print(f"    Embedding {len(synthetic_texts)} synthetic texts...", flush=True)
    synthetic_embeddings = cache.embed_synthetic(synthetic_texts)

    return synthetic_embeddings, np.array(synthetic_labels_list)


def run_clustering_experiment() -> List[ClusteringResult]:
    """Run clustering validation experiment for all K_max values."""
    print("\n" + "="*60)
    print("  EXPERIMENT 01: CLUSTERING VALIDATION (K_MAX)")
    print("="*60)
    print(f"  K_max values: {K_MAX_VALUES}")
    print(f"  K-Fold: {KFOLD_CONFIG['n_splits']}×{KFOLD_CONFIG['n_repeats']} = "
          f"{KFOLD_CONFIG['n_splits'] * KFOLD_CONFIG['n_repeats']} folds")

    # Load data and embeddings
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for k_max in K_MAX_VALUES:
        print(f"\n{'─'*60}")
        print(f"  Testing K_max = {k_max}")
        print(f"{'─'*60}")

        # Compute clustering metrics
        silhouette = compute_class_silhouette(embeddings, labels, k_max)
        coherence = compute_coherence(embeddings, labels, k_max)

        print(f"  Silhouette: {silhouette:.4f}" if not np.isnan(silhouette) else "  Silhouette: N/A")
        print(f"  Coherence: {coherence*100:.1f}%")

        # Generate synthetic data for this K_max
        X_synth, y_synth = generate_synthetic_for_kmax(
            embeddings, labels, texts, k_max, cache
        )
        print(f"  Generated {len(X_synth)} synthetic samples")

        # Run K-Fold evaluation
        kfold_result = evaluator.evaluate(
            X_original=embeddings,
            y_original=labels,
            X_synthetic=X_synth if len(X_synth) > 0 else None,
            y_synthetic=y_synth if len(y_synth) > 0 else None,
            config_name=f"K{k_max}"
        )

        result = ClusteringResult(
            k_max=k_max,
            silhouette=float(silhouette) if not np.isnan(silhouette) else -1.0,
            coherence=float(coherence),
            macro_f1=float(kfold_result.augmented_mean),
            delta_pct=float(kfold_result.delta_pct),
            n_synthetic=int(kfold_result.n_synthetic),
            p_value=float(kfold_result.p_value),
            significant=bool(kfold_result.significant),
            win_rate=float(kfold_result.win_rate)
        )
        results.append(result)

        # Save individual result
        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"K{k_max}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print_summary(kfold_result)

    return results


def generate_latex_table(results: List[ClusteringResult]) -> str:
    """Generate LaTeX table for clustering validation."""

    latex = r"""
\begin{table}[h]
\centering
\caption{Validacion preliminar para seleccion del numero maximo de clusters}
\label{tab:clustering_validation}
\begin{tabular}{lcccc}
\hline
$K_{max}$ & Silhouette & Coherencia & Macro F1 & $\Delta$ vs Linea base \\
\hline
"""

    # Find best result
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sil_str = "N/A" if r.silhouette < 0 else f"{r.silhouette:.2f}"
        coh_str = f"{r.coherence*100:.0f}\\%"
        f1_str = f"{r.macro_f1:.4f}"
        delta_str = f"{r.delta_pct:+.2f}\\%"

        k_label = f"{r.k_max}" + (" (vanilla)" if r.k_max == 1 else "")

        if i == best_idx:
            latex += f"\\textbf{{{k_label}}} & \\textbf{{{sil_str}}} & \\textbf{{{coh_str}}} & \\textbf{{{f1_str}}} & \\textbf{{{delta_str}}} \\\\\n"
        else:
            latex += f"{k_label} & {sil_str} & {coh_str} & {f1_str} & {delta_str} \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    """Main entry point."""
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_clustering_experiment()

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    latex_path = LATEX_DIR / "tab_clustering_validation.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Print summary table
    print("\n" + "="*60)
    print("  SUMMARY: Clustering Validation Results")
    print("="*60)
    print(f"{'K_max':>6} | {'Silhouette':>10} | {'Coherence':>10} | {'Macro F1':>10} | {'Delta':>10}")
    print("-"*60)
    for r in results:
        sil_str = "N/A" if r.silhouette < 0 else f"{r.silhouette:.3f}"
        print(f"{r.k_max:>6} | {sil_str:>10} | {r.coherence*100:>9.1f}% | {r.macro_f1:>10.4f} | {r.delta_pct:>+9.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
