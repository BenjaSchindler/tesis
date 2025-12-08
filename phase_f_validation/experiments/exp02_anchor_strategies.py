#!/usr/bin/env python3
"""
Experiment 02: Anchor Strategy Validation

Tests different anchor selection strategies:
- random, nearest_neighbor, medoid, quality_gated, diverse, ensemble

Metrics: Macro F1, Quality (0-1), Diversity (0-1), Delta%

Output: tab:anchor_strategies for Metodologia.tex
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from base_config import (
    PROJECT_ROOT, DATA_PATH, RESULTS_DIR, LATEX_DIR,
    BASE_PARAMS, KFOLD_CONFIG, EXPERIMENT_PARAMS, MBTI_CLASSES
)
from validation_runner import (
    load_data, EmbeddingCache, KFoldEvaluator, print_summary, LLMSyntheticGenerator
)
from base_config import LLM_MODEL, TEMPERATURE, MAX_TOKENS


STRATEGIES = EXPERIMENT_PARAMS["anchor_strategies"]["STRATEGIES"]
EXPERIMENT_NAME = "anchor_strategies"


@dataclass
class AnchorResult:
    """Results for one anchor strategy."""
    strategy: str
    macro_f1: float
    quality: float
    diversity: float
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool
    win_rate: float


class AnchorSelector:
    """Implements different anchor selection strategies."""

    def __init__(self, k_clusters: int = 5):
        self.k_clusters = k_clusters

    def select_random(
        self,
        class_embeddings: np.ndarray,
        n_anchors: int
    ) -> Tuple[List[int], Dict]:
        """Random anchor selection."""
        indices = np.random.choice(len(class_embeddings), min(n_anchors, len(class_embeddings)), replace=False)
        return list(indices), {"strategy": "random"}

    def select_nearest_neighbor(
        self,
        class_embeddings: np.ndarray,
        n_anchors: int
    ) -> Tuple[List[int], Dict]:
        """
        Select anchors as the nearest neighbor to each cluster centroid.
        """
        if len(class_embeddings) < self.k_clusters:
            return list(range(min(n_anchors, len(class_embeddings)))), {"strategy": "nearest_neighbor"}

        kmeans = KMeans(n_clusters=min(self.k_clusters, len(class_embeddings)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        anchors = []
        for c_id in range(kmeans.n_clusters):
            c_mask = cluster_labels == c_id
            if not c_mask.any():
                continue

            c_points = class_embeddings[c_mask]
            c_indices = np.where(c_mask)[0]

            # Find nearest to centroid
            centroid = kmeans.cluster_centers_[c_id]
            dists = np.linalg.norm(c_points - centroid, axis=1)
            nearest_idx = c_indices[np.argmin(dists)]
            anchors.append(nearest_idx)

        return anchors[:n_anchors], {"strategy": "nearest_neighbor"}

    def select_medoid(
        self,
        class_embeddings: np.ndarray,
        n_anchors: int
    ) -> Tuple[List[int], Dict]:
        """
        Select medoid (most central point) for each cluster.
        More robust than centroid for non-convex clusters.
        """
        if len(class_embeddings) < self.k_clusters:
            return list(range(min(n_anchors, len(class_embeddings)))), {"strategy": "medoid"}

        kmeans = KMeans(n_clusters=min(self.k_clusters, len(class_embeddings)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        anchors = []
        for c_id in range(kmeans.n_clusters):
            c_mask = cluster_labels == c_id
            if not c_mask.any():
                continue

            c_points = class_embeddings[c_mask]
            c_indices = np.where(c_mask)[0]

            if len(c_points) == 1:
                anchors.append(c_indices[0])
                continue

            # Find medoid: point with minimum sum of distances to all others
            pairwise = pairwise_distances(c_points, metric='cosine')
            medoid_idx_local = np.argmin(pairwise.sum(axis=1))
            anchors.append(c_indices[medoid_idx_local])

        return anchors[:n_anchors], {"strategy": "medoid"}

    def select_quality_gated(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        n_anchors: int,
        min_purity: float = 0.60
    ) -> Tuple[List[int], Dict]:
        """
        Select anchors only from high-purity clusters.
        """
        if len(class_embeddings) < self.k_clusters:
            return list(range(min(n_anchors, len(class_embeddings)))), {"strategy": "quality_gated", "avg_purity": 0.0}

        kmeans = KMeans(n_clusters=min(self.k_clusters, len(class_embeddings)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_embeddings)

        class_indices = np.where(all_labels == target_class)[0]
        anchors = []
        purities = []

        for c_id in range(kmeans.n_clusters):
            c_mask = cluster_labels == c_id
            if not c_mask.any():
                continue

            c_points = class_embeddings[c_mask]
            c_local_indices = np.where(c_mask)[0]

            # Calculate purity: check neighbors in full dataset
            centroid = kmeans.cluster_centers_[c_id]
            dists = np.linalg.norm(all_embeddings - centroid, axis=1)
            k_neighbors = min(15, len(all_embeddings))
            nearest_indices = np.argsort(dists)[:k_neighbors]
            nearest_labels = all_labels[nearest_indices]
            purity = (nearest_labels == target_class).mean()
            purities.append(purity)

            if purity >= min_purity:
                # Select medoid of high-purity cluster
                if len(c_points) > 1:
                    pairwise = pairwise_distances(c_points, metric='cosine')
                    medoid_idx_local = np.argmin(pairwise.sum(axis=1))
                    anchors.append(c_local_indices[medoid_idx_local])
                else:
                    anchors.append(c_local_indices[0])

        return anchors[:n_anchors], {"strategy": "quality_gated", "avg_purity": np.mean(purities) if purities else 0.0}

    def select_diverse(
        self,
        class_embeddings: np.ndarray,
        n_anchors: int
    ) -> Tuple[List[int], Dict]:
        """
        Select diverse anchors using max-min distance (farthest-first traversal).
        Maximizes coverage of the embedding space.
        """
        if len(class_embeddings) <= n_anchors:
            return list(range(len(class_embeddings))), {"strategy": "diverse"}

        anchors = []
        remaining = set(range(len(class_embeddings)))

        # Start with random point
        first = np.random.choice(list(remaining))
        anchors.append(first)
        remaining.remove(first)

        # Iteratively select farthest point
        for _ in range(n_anchors - 1):
            if not remaining:
                break

            anchor_embs = class_embeddings[anchors]
            remaining_embs = class_embeddings[list(remaining)]
            remaining_list = list(remaining)

            # Distance to nearest anchor
            dists = cdist(remaining_embs, anchor_embs, metric='cosine')
            min_dists = dists.min(axis=1)

            # Select point with max min-distance
            farthest_local = np.argmax(min_dists)
            farthest_idx = remaining_list[farthest_local]

            anchors.append(farthest_idx)
            remaining.remove(farthest_idx)

        return anchors, {"strategy": "diverse"}

    def select_ensemble(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        n_anchors: int
    ) -> Tuple[List[int], Dict]:
        """
        Ensemble selection combining multiple strategies.
        Weights: diversity=0.3, quality=0.4, stability=0.3
        """
        # Get candidates from each strategy
        medoid_anchors, _ = self.select_medoid(class_embeddings, n_anchors * 2)
        quality_anchors, quality_metrics = self.select_quality_gated(
            class_embeddings, all_embeddings, all_labels, target_class, n_anchors * 2
        )
        diverse_anchors, _ = self.select_diverse(class_embeddings, n_anchors * 2)

        # Combine all candidates
        all_candidates = list(set(medoid_anchors + quality_anchors + diverse_anchors))

        if len(all_candidates) <= n_anchors:
            return all_candidates, {"strategy": "ensemble", "n_candidates": len(all_candidates)}

        # Score each candidate
        scores = []
        for idx in all_candidates:
            anchor_emb = class_embeddings[idx:idx+1]

            # Diversity score: avg distance to other candidates
            other_embs = class_embeddings[[i for i in all_candidates if i != idx]]
            if len(other_embs) > 0:
                dists = cdist(anchor_emb, other_embs, metric='cosine')[0]
                diversity_score = dists.mean()
            else:
                diversity_score = 1.0

            # Quality score: purity of neighborhood
            full_dists = np.linalg.norm(all_embeddings - anchor_emb, axis=1)
            k = min(15, len(all_embeddings))
            nearest = np.argsort(full_dists)[:k]
            purity = (all_labels[nearest] == target_class).mean()
            quality_score = purity

            # Stability score: inverse variance of distances to class
            class_dists = cdist(anchor_emb, class_embeddings, metric='cosine')[0]
            stability_score = 1.0 / (1.0 + class_dists.std())

            # Weighted ensemble score
            ensemble_score = 0.3 * diversity_score + 0.4 * quality_score + 0.3 * stability_score
            scores.append((idx, ensemble_score))

        # Select top n_anchors
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in scores[:n_anchors]]

        return selected, {"strategy": "ensemble", "n_candidates": len(all_candidates)}


def compute_anchor_metrics(
    anchors: List[int],
    class_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str
) -> Tuple[float, float]:
    """Compute quality and diversity metrics for selected anchors."""

    if not anchors:
        return 0.0, 0.0

    # Quality: average purity of anchor neighborhoods
    qualities = []
    for idx in anchors:
        anchor_emb = class_embeddings[idx:idx+1]
        dists = np.linalg.norm(all_embeddings - anchor_emb, axis=1)
        k = min(15, len(all_embeddings))
        nearest = np.argsort(dists)[:k]
        purity = (all_labels[nearest] == target_class).mean()
        qualities.append(purity)
    quality = np.mean(qualities)

    # Diversity: average pairwise distance
    if len(anchors) > 1:
        anchor_embs = class_embeddings[anchors]
        pairwise = pairwise_distances(anchor_embs, metric='cosine')
        # Upper triangle (excluding diagonal)
        upper = pairwise[np.triu_indices(len(anchors), k=1)]
        diversity = upper.mean()
    else:
        diversity = 0.0

    return quality, diversity


def generate_synthetic_with_strategy(
    strategy: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache,
    selector: AnchorSelector
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate synthetic samples using a specific anchor strategy with LLM."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_texts = []
    synthetic_labels_list = []
    all_qualities = []
    all_diversities = []

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        n_anchors = min(BASE_PARAMS["max_clusters"], len(class_embeddings) // 10)
        if n_anchors < 1:
            n_anchors = 1

        # Select anchors using the specified strategy
        if strategy == "random":
            anchors, _ = selector.select_random(class_embeddings, n_anchors)
        elif strategy == "nearest_neighbor":
            anchors, _ = selector.select_nearest_neighbor(class_embeddings, n_anchors)
        elif strategy == "medoid":
            anchors, _ = selector.select_medoid(class_embeddings, n_anchors)
        elif strategy == "quality_gated":
            anchors, _ = selector.select_quality_gated(
                class_embeddings, embeddings, labels, target_class, n_anchors
            )
        elif strategy == "diverse":
            anchors, _ = selector.select_diverse(class_embeddings, n_anchors)
        elif strategy == "ensemble":
            anchors, _ = selector.select_ensemble(
                class_embeddings, embeddings, labels, target_class, n_anchors
            )
        else:
            anchors = list(range(min(n_anchors, len(class_embeddings))))

        # Compute metrics
        quality, diversity = compute_anchor_metrics(
            anchors, class_embeddings, embeddings, labels, target_class
        )
        all_qualities.append(quality)
        all_diversities.append(diversity)

        # Generate synthetic using LLM for each anchor
        for anchor_idx in anchors:
            # Get example texts around anchor
            anchor_emb = class_embeddings[anchor_idx]
            dists = np.linalg.norm(class_embeddings - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [class_texts[i] for i in nearest_idx]

            # Create prompt
            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])
            n_samples = max(1, BASE_PARAMS["samples_per_prompt"] // 2)

            prompt = f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
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

                for sample in samples[:n_samples]:
                    synthetic_texts.append(sample)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class}: {e}", flush=True)
                continue

    if not synthetic_texts:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), 0.0, 0.0

    # Embed synthetic texts
    print(f"    Embedding {len(synthetic_texts)} synthetic texts...", flush=True)
    synthetic_embeddings = cache.embed_synthetic(synthetic_texts)

    avg_quality = np.mean(all_qualities) if all_qualities else 0.0
    avg_diversity = np.mean(all_diversities) if all_diversities else 0.0

    return synthetic_embeddings, np.array(synthetic_labels_list), avg_quality, avg_diversity


def run_anchor_experiment() -> List[AnchorResult]:
    """Run anchor strategy validation experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 02: ANCHOR STRATEGY VALIDATION")
    print("="*60)
    print(f"  Strategies: {STRATEGIES}")
    print(f"  K-Fold: {KFOLD_CONFIG['n_splits']}×{KFOLD_CONFIG['n_repeats']} = "
          f"{KFOLD_CONFIG['n_splits'] * KFOLD_CONFIG['n_repeats']} folds")

    # Load data and embeddings
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    selector = AnchorSelector(k_clusters=BASE_PARAMS["max_clusters"])
    results = []

    for strategy in STRATEGIES:
        print(f"\n{'─'*60}")
        print(f"  Testing strategy: {strategy}")
        print(f"{'─'*60}")

        # Generate synthetic data with this strategy using LLM
        X_synth, y_synth, quality, diversity = generate_synthetic_with_strategy(
            strategy, embeddings, labels, texts, cache, selector
        )
        print(f"  Quality: {quality:.2f}")
        print(f"  Diversity: {diversity:.2f}")
        print(f"  Generated {len(X_synth)} synthetic samples")

        # Run K-Fold evaluation
        kfold_result = evaluator.evaluate(
            X_original=embeddings,
            y_original=labels,
            X_synthetic=X_synth if len(X_synth) > 0 else None,
            y_synthetic=y_synth if len(y_synth) > 0 else None,
            config_name=strategy
        )

        result = AnchorResult(
            strategy=strategy,
            macro_f1=float(kfold_result.augmented_mean),
            quality=float(quality),
            diversity=float(diversity),
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
        with open(output_dir / f"{strategy}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print_summary(kfold_result)

    return results


def generate_latex_table(results: List[AnchorResult]) -> str:
    """Generate LaTeX table for anchor strategies."""

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparacion de estrategias de seleccion de anclas}
\label{tab:anchor_strategies}
\begin{tabular}{lcccc}
\hline
Estrategia & Macro F1 & Quality & Diversity & $\Delta$ \\
\hline
"""

    # Find best result
    best_idx = np.argmax([r.delta_pct for r in results])

    # Map strategy names to Spanish
    name_map = {
        "random": "Random",
        "nearest_neighbor": "Nearest Neighbor",
        "medoid": "Medoid",
        "quality_gated": "Quality-gated",
        "diverse": "Diverse",
        "ensemble": "Ensemble"
    }

    for i, r in enumerate(results):
        name = name_map.get(r.strategy, r.strategy)
        f1_str = f"{r.macro_f1:.4f}"
        quality_str = f"{r.quality:.2f}"
        diversity_str = f"{r.diversity:.2f}"
        delta_str = f"{r.delta_pct:+.1f}\\%"

        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{f1_str}}} & \\textbf{{{quality_str}}} & \\textbf{{{diversity_str}}} & \\textbf{{{delta_str}}} \\\\\n"
        else:
            latex += f"{name} & {f1_str} & {quality_str} & {diversity_str} & {delta_str} \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    """Main entry point."""
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_anchor_experiment()

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    latex_path = LATEX_DIR / "tab_anchor_strategies.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Print summary table
    print("\n" + "="*60)
    print("  SUMMARY: Anchor Strategy Results")
    print("="*60)
    print(f"{'Strategy':>18} | {'Macro F1':>10} | {'Quality':>8} | {'Diversity':>10} | {'Delta':>10}")
    print("-"*65)
    for r in results:
        print(f"{r.strategy:>18} | {r.macro_f1:>10.4f} | {r.quality:>8.2f} | {r.diversity:>10.2f} | {r.delta_pct:>+9.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
