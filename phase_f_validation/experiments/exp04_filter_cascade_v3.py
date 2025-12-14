#!/usr/bin/env python3
"""
Experiment 04 v3: Filter Cascade with Adaptive Relaxation

Problem in v2: three_filters and full_cascade got 0 samples because
the composite quality score was too low to pass MIN_QUALITY_FLOOR.

Solution: Use same adaptive relaxation as exp05_v2:
1. Compute quality scores using filter cascade
2. Rank candidates by score
3. Select top-N regardless of absolute threshold
4. Compare quality vs delta to see if more filters = better quality

Target: ~100 synthetic samples per config (fair comparison)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from base_config import RESULTS_DIR, LATEX_DIR, BASE_PARAMS, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from validation_runner import load_data, EmbeddingCache, KFoldEvaluator, print_summary
from typing import List, Tuple


EXPERIMENT_NAME = "filter_cascade_v3"
TARGET_SYNTH_PER_CLASS = 7  # ~7 per class * 16 classes = ~112 total


# Filter configs - each adds more criteria to ranking
FILTER_CONFIGS = [
    {"name": "length_only", "filters": ["length"], "description": "Solo distancia al anchor"},
    {"name": "length_similarity", "filters": ["length", "similarity"], "description": "+ similitud coseno"},
    {"name": "three_filters", "filters": ["length", "similarity", "knn"], "description": "+ K-NN purity"},
    {"name": "full_cascade", "filters": ["length", "similarity", "knn", "confidence"], "description": "+ confianza"},
]


@dataclass
class FilterResult:
    config: str
    n_filters: int
    avg_quality: float
    macro_f1: float
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool


def compute_quality_scores(
    candidates: np.ndarray,
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str,
    filters: List[str]
) -> np.ndarray:
    """Compute composite quality score for each candidate."""
    if len(candidates) == 0:
        return np.array([])

    n_candidates = len(candidates)

    # Initialize component scores
    scores_dict = {}

    # Ensure anchor is 2D
    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)

    # Filter 1: Length/distance from anchor (closer = better)
    if "length" in filters:
        dists_to_anchor = np.linalg.norm(candidates - anchor_emb, axis=1)
        max_dist = np.max(dists_to_anchor) + 1e-6
        scores_dict["length"] = 1 - (dists_to_anchor / max_dist)

    # Filter 2: Cosine similarity to anchor
    if "similarity" in filters:
        similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()
        scores_dict["similarity"] = np.clip(similarities, 0, 1)

    # Filter 3: K-NN purity
    if "knn" in filters:
        class_mask = all_labels == target_class
        class_embs = all_embeddings[class_mask]

        knn_scores = np.zeros(n_candidates)
        if len(class_embs) > 0:
            for i, cand in enumerate(candidates):
                dists = np.linalg.norm(class_embs - cand, axis=1)
                k = min(10, len(class_embs))
                nearest_dists = np.sort(dists)[:k]
                # Inverse of average distance (closer = better)
                knn_scores[i] = 1.0 / (1.0 + nearest_dists.mean())
        scores_dict["knn"] = knn_scores

    # Filter 4: Confidence (distance to class centroid)
    if "confidence" in filters:
        class_mask = all_labels == target_class
        class_embs = all_embeddings[class_mask]

        if len(class_embs) > 0:
            centroid = class_embs.mean(axis=0)
            dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
            max_dist = np.max(dists_to_centroid) + 1e-6
            scores_dict["confidence"] = 1 - (dists_to_centroid / max_dist)
        else:
            scores_dict["confidence"] = np.ones(n_candidates) * 0.5

    # Combine scores: geometric mean (more robust than product)
    if not scores_dict:
        return np.ones(n_candidates)

    combined = np.ones(n_candidates)
    for score_array in scores_dict.values():
        combined *= score_array

    # Take nth root where n = number of filters
    n_filters = len(scores_dict)
    combined = np.power(combined, 1.0 / n_filters)

    return combined


def select_top_by_ranking(
    candidates: np.ndarray,
    scores: np.ndarray,
    target: int
) -> Tuple[np.ndarray, float]:
    """Select top candidates by ranking (no absolute threshold)."""
    if len(candidates) == 0:
        return np.array([]).reshape(0, 768), 0.0

    # Simply take top-N by score
    n_select = min(target, len(candidates))
    top_idx = np.argsort(scores)[-n_select:]

    selected = candidates[top_idx]
    avg_quality = scores[top_idx].mean()

    return selected, avg_quality


def generate_synthetic_with_adaptive_filters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    filter_config: dict,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate synthetic samples with adaptive filter ranking."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_qualities = []

    filters = filter_config["filters"]

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        n_clusters = min(3, len(class_embeddings) // 30)
        if n_clusters < 1:
            n_clusters = 1

        target_per_cluster = max(3, TARGET_SYNTH_PER_CLASS // n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        for c_id in range(n_clusters):
            c_mask = cluster_labels_arr == c_id
            c_points = class_embeddings[c_mask]
            c_texts = class_texts[c_mask]

            if len(c_points) < 3:
                continue

            anchor_emb = kmeans.cluster_centers_[c_id]

            # Get example texts
            dists = np.linalg.norm(c_points - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [c_texts[i] for i in nearest_idx]

            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])

            # Generate more candidates for better selection pool
            n_candidates = 20

            prompt = f"""Generate {n_candidates} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_candidates} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS * n_candidates,
                )

                generated_text = response.choices[0].message.content.strip()
                samples = [s.strip() for s in generated_text.split('\n')
                          if s.strip() and len(s.strip()) > 10]

                if not samples:
                    continue

                # Embed candidates
                candidate_embeddings = cache.embed_synthetic(samples)

                # Compute quality scores using filter cascade
                scores = compute_quality_scores(
                    candidate_embeddings, anchor_emb,
                    embeddings, labels, target_class, filters
                )

                # Select top by ranking (adaptive - no hard threshold)
                selected, avg_quality = select_top_by_ranking(
                    candidate_embeddings, scores, target_per_cluster
                )

                all_qualities.append(avg_quality)

                for emb in selected:
                    synthetic_embeddings.append(emb)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

    avg_quality = np.mean(all_qualities) if all_qualities else 0.0

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), avg_quality

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), avg_quality


def run_filter_experiment() -> list:
    """Run filter cascade v3 with adaptive ranking."""
    print("\n" + "="*60)
    print("  EXPERIMENT 04 v3: FILTER CASCADE (ADAPTIVE RANKING)")
    print("="*60)
    print(f"  Target: ~{TARGET_SYNTH_PER_CLASS} samples/class (~{TARGET_SYNTH_PER_CLASS*16} total)")
    print("  Method: Rank by quality, select top-N (no hard threshold)")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for config in FILTER_CONFIGS:
        print(f"\n{'─'*60}")
        print(f"  Testing: {config['name']}")
        print(f"  {config['description']}")
        print(f"  Filters: {config['filters']}")
        print(f"{'─'*60}")

        X_synth, y_synth, avg_quality = generate_synthetic_with_adaptive_filters(
            embeddings, labels, texts, config, cache
        )

        print(f"  Avg quality score: {avg_quality:.3f}")
        print(f"  Total synthetic: {len(X_synth)}")

        kfold_result = evaluator.evaluate(
            X_original=embeddings,
            y_original=labels,
            X_synthetic=X_synth if len(X_synth) > 0 else None,
            y_synthetic=y_synth if len(y_synth) > 0 else None,
            config_name=config['name']
        )

        result = FilterResult(
            config=config['name'],
            n_filters=len(config['filters']),
            avg_quality=float(avg_quality),
            macro_f1=float(kfold_result.augmented_mean),
            delta_pct=float(kfold_result.delta_pct),
            n_synthetic=int(kfold_result.n_synthetic),
            p_value=float(kfold_result.p_value),
            significant=bool(kfold_result.significant)
        )
        results.append(result)

        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{config['name']}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print_summary(kfold_result)

    return results


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table for filter cascade v3."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Cascada de filtros con seleccion adaptativa por ranking}
\label{tab:filter_cascade_v3}
\begin{tabular}{lcccc}
\hline
Configuracion & N Filtros & N Synth & Avg Quality & $\Delta$ \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sig_mark = "*" if r.significant else ""
        if i == best_idx:
            latex += f"\\textbf{{{r.config}}} & \\textbf{{{r.n_filters}}} & \\textbf{{{r.n_synthetic}}} & \\textbf{{{r.avg_quality:.3f}}} & \\textbf{{{r.delta_pct:+.2f}\\%{sig_mark}}} \\\\\n"
        else:
            latex += f"{r.config} & {r.n_filters} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.delta_pct:+.2f}\\%{sig_mark} \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_filter_experiment()

    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_filter_cascade_v3.tex", 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("  SUMMARY: Filter Cascade v3 (Adaptive Ranking)")
    print("="*60)
    print(f"  {'Config':<20} | Filters | N Synth | Quality | Delta")
    print("  " + "-"*58)
    for r in results:
        sig = "*" if r.significant else " "
        print(f"  {r.config:<20} |    {r.n_filters}    |   {r.n_synthetic:3}   |  {r.avg_quality:.3f}  | {r.delta_pct:+.2f}%{sig}")

    # Analysis: does more filters = higher quality?
    print("\n  Analysis: Quality vs Number of Filters")
    print("  " + "-"*40)
    for r in results:
        print(f"    {r.n_filters} filters -> quality={r.avg_quality:.3f}, delta={r.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
