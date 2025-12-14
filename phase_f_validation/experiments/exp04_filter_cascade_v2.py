#!/usr/bin/env python3
"""
Experiment 04 v2: Filter Cascade with Quota-Based Selection

Instead of hard thresholds that reject most samples, this version:
1. Applies filters to compute quality scores
2. Ranks candidates by score
3. Selects top-N (quota) per class to ensure fair comparison

Target: ~100 synthetic samples per config (controlled comparison)
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


EXPERIMENT_NAME = "filter_cascade_v2"
TARGET_SYNTH_PER_CLASS = 8  # ~8 per class * 16 classes = ~128 total
MIN_QUALITY_FLOOR = 0.1    # Absolute minimum quality


# Filter configs - each adds more criteria to the score
FILTER_CONFIGS = [
    {"name": "length_only", "filters": ["length"]},
    {"name": "length_similarity", "filters": ["length", "similarity"]},
    {"name": "three_filters", "filters": ["length", "similarity", "knn"]},
    {"name": "full_cascade", "filters": ["length", "similarity", "knn", "confidence"]},
]


@dataclass
class FilterResult:
    config: str
    acceptance_rate: float
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

    scores = np.ones(len(candidates))

    # Ensure anchor is 2D
    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)

    # Filter 1: Length/distance from anchor (closer = better)
    if "length" in filters:
        dists_to_anchor = np.linalg.norm(candidates - anchor_emb, axis=1)
        max_dist = np.max(dists_to_anchor) + 1e-6
        length_score = 1 - (dists_to_anchor / max_dist)  # Normalize to [0,1]
        scores *= length_score

    # Filter 2: Cosine similarity to anchor
    if "similarity" in filters:
        similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()
        similarities = np.clip(similarities, 0, 1)
        scores *= similarities

    # Filter 3: K-NN purity (how many neighbors are same class)
    if "knn" in filters:
        class_mask = all_labels == target_class
        class_embs = all_embeddings[class_mask]

        knn_scores = []
        for cand in candidates:
            if len(class_embs) > 0:
                dists = np.linalg.norm(class_embs - cand, axis=1)
                k = min(10, len(class_embs))
                nearest_dists = np.sort(dists)[:k]
                # Score based on how close to class members
                avg_dist = nearest_dists.mean()
                knn_score = np.exp(-avg_dist)  # Exponential decay
            else:
                knn_score = 0.5
            knn_scores.append(knn_score)
        scores *= np.array(knn_scores)

    # Filter 4: Simulated classifier confidence
    if "confidence" in filters:
        # Use distance to class centroid as proxy for confidence
        class_mask = all_labels == target_class
        class_embs = all_embeddings[class_mask]
        if len(class_embs) > 0:
            centroid = class_embs.mean(axis=0)
            dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
            max_dist = np.max(dists_to_centroid) + 1e-6
            conf_score = 1 - (dists_to_centroid / max_dist)
            scores *= conf_score

    return scores


def select_top_by_quota(
    candidates: np.ndarray,
    scores: np.ndarray,
    target: int = TARGET_SYNTH_PER_CLASS,
    min_quality: float = MIN_QUALITY_FLOOR
) -> Tuple[np.ndarray, float, float]:
    """Select top candidates by quota with quality floor."""
    if len(candidates) == 0:
        return np.array([]).reshape(0, candidates.shape[1] if len(candidates.shape) > 1 else 768), 0.0, 0.0

    # Apply minimum quality floor
    valid_mask = scores >= min_quality
    valid_candidates = candidates[valid_mask]
    valid_scores = scores[valid_mask]

    acceptance_rate = valid_mask.sum() / len(candidates)

    if len(valid_candidates) == 0:
        return np.array([]).reshape(0, candidates.shape[1]), acceptance_rate, 0.0

    # Select top-N by score
    if len(valid_candidates) > target:
        top_idx = np.argsort(valid_scores)[-target:]
        selected = valid_candidates[top_idx]
        selected_scores = valid_scores[top_idx]
    else:
        selected = valid_candidates
        selected_scores = valid_scores

    avg_quality = selected_scores.mean() if len(selected_scores) > 0 else 0.0

    return selected, acceptance_rate, avg_quality


def generate_synthetic_with_quota(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    filter_config: dict,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate synthetic samples with quota-based selection."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_acceptance = []
    all_quality = []

    filters = filter_config["filters"]

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Use fewer clusters for this smaller experiment
        n_clusters = min(3, len(class_embeddings) // 30)
        if n_clusters < 1:
            n_clusters = 1

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        class_candidates = []
        class_scores = []

        for c_id in range(n_clusters):
            c_mask = cluster_labels_arr == c_id
            c_points = class_embeddings[c_mask]
            c_texts = class_texts[c_mask]

            if len(c_points) < 3:
                continue

            anchor_emb = kmeans.cluster_centers_[c_id]

            # Get example texts near anchor
            dists = np.linalg.norm(c_points - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [c_texts[i] for i in nearest_idx]

            # Generate more candidates to have selection pool
            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])
            n_candidates = 15  # Generate more to select from

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

                # Compute quality scores
                scores = compute_quality_scores(
                    candidate_embeddings, anchor_emb,
                    embeddings, labels, target_class, filters
                )

                class_candidates.extend(candidate_embeddings)
                class_scores.extend(scores)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

        # Select top candidates for this class
        if class_candidates:
            class_candidates = np.array(class_candidates)
            class_scores = np.array(class_scores)

            selected, acc_rate, avg_qual = select_top_by_quota(
                class_candidates, class_scores, TARGET_SYNTH_PER_CLASS
            )

            all_acceptance.append(acc_rate)
            all_quality.append(avg_qual)

            for emb in selected:
                synthetic_embeddings.append(emb)
                synthetic_labels_list.append(target_class)

    avg_acceptance = np.mean(all_acceptance) if all_acceptance else 0.0
    avg_quality = np.mean(all_quality) if all_quality else 0.0

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), avg_acceptance, avg_quality

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), avg_acceptance, avg_quality


def run_filter_experiment() -> list:
    """Run filter cascade v2 experiment with quota-based selection."""
    print("\n" + "="*60)
    print("  EXPERIMENT 04 v2: FILTER CASCADE (QUOTA-BASED)")
    print("="*60)
    print(f"  Target: ~{TARGET_SYNTH_PER_CLASS} samples/class")
    print(f"  Min quality floor: {MIN_QUALITY_FLOOR}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for config in FILTER_CONFIGS:
        print(f"\n{'─'*60}")
        print(f"  Testing: {config['name']}")
        print(f"  Filters: {config['filters']}")
        print(f"{'─'*60}")

        X_synth, y_synth, acceptance, quality = generate_synthetic_with_quota(
            embeddings, labels, texts, config, cache
        )

        print(f"  Samples passing floor: {acceptance*100:.1f}%")
        print(f"  Avg quality (selected): {quality:.3f}")
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
            acceptance_rate=float(acceptance),
            avg_quality=float(quality),
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
    """Generate LaTeX table for filter cascade v2."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impacto de filtros con seleccion por cuota (controlado por cantidad)}
\label{tab:filter_cascade_v2}
\begin{tabular}{lcccc}
\hline
Configuracion & N Synth & Avg Quality & Macro F1 & $\Delta$ \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        if i == best_idx:
            latex += f"\\textbf{{{r.config}}} & \\textbf{{{r.n_synthetic}}} & \\textbf{{{r.avg_quality:.3f}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.delta_pct:+.2f}\\%}} \\\\\n"
        else:
            latex += f"{r.config} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\% \\\\\n"

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
    with open(LATEX_DIR / "tab_filter_cascade_v2.tex", 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("  SUMMARY: Filter Cascade v2 Results (Quota-Based)")
    print("="*60)
    for r in results:
        print(f"{r.config:>20} | N={r.n_synthetic:3} | Q={r.avg_quality:.3f} | F1={r.macro_f1:.4f} | Delta={r.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
