#!/usr/bin/env python3
"""
Experiment 05 v2: Adaptive Thresholds with Relaxation

Instead of fixed thresholds that reject most samples, this version:
1. Starts with strict threshold
2. Progressively relaxes until quota is met
3. Ensures fair comparison with similar sample counts

Approach: Adaptive Relaxation (Option 3)
Target: ~100 synthetic samples per config
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


EXPERIMENT_NAME = "adaptive_thresholds_v2"
TARGET_SYNTH_TOTAL = 120  # Target total synthetic samples
MIN_THRESHOLD = 0.40      # Never go below this


# Threshold strategies to compare
THRESHOLD_CONFIGS = [
    {
        "name": "strict_relaxing",
        "start_threshold": 0.90,
        "description": "Start strict (0.90), relax as needed"
    },
    {
        "name": "medium_relaxing",
        "start_threshold": 0.70,
        "description": "Start medium (0.70), relax as needed"
    },
    {
        "name": "permissive_relaxing",
        "start_threshold": 0.50,
        "description": "Start permissive (0.50), relax as needed"
    },
    {
        "name": "purity_adaptive",
        "start_threshold": None,  # Computed from anchor purity
        "description": "Threshold based on anchor purity"
    },
]


@dataclass
class ThresholdResult:
    config: str
    macro_f1: float
    avg_threshold_used: float
    avg_quality: float
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool


def get_purity_based_threshold(
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str
) -> float:
    """Compute threshold based on anchor neighborhood purity."""
    dists = np.linalg.norm(all_embeddings - anchor_emb, axis=1)
    k = min(15, len(all_embeddings))
    nearest = np.argsort(dists)[:k]
    purity = (all_labels[nearest] == target_class).mean()

    # Higher purity = can be more permissive
    if purity >= 0.8:
        return 0.50  # Very pure neighborhood
    elif purity >= 0.6:
        return 0.60
    elif purity >= 0.4:
        return 0.70
    else:
        return 0.80  # Contaminated neighborhood, be strict


def apply_threshold_with_relaxation(
    candidates: np.ndarray,
    anchor_emb: np.ndarray,
    start_threshold: float,
    target_per_cluster: int = 5
) -> Tuple[np.ndarray, float, float]:
    """Apply threshold with progressive relaxation until quota met."""
    if len(candidates) == 0:
        return np.array([]).reshape(0, 768), 0.0, 0.0

    # Ensure anchor is 2D
    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)

    # Compute similarities
    similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()

    # Try progressively lower thresholds
    thresholds_to_try = np.arange(start_threshold, MIN_THRESHOLD - 0.05, -0.10)

    for thresh in thresholds_to_try:
        accepted_mask = similarities >= thresh
        n_accepted = accepted_mask.sum()

        if n_accepted >= target_per_cluster:
            # Got enough, take top ones by similarity
            accepted_idx = np.where(accepted_mask)[0]
            if n_accepted > target_per_cluster:
                # Sort by similarity and take best
                sorted_idx = np.argsort(similarities[accepted_idx])[-target_per_cluster:]
                final_idx = accepted_idx[sorted_idx]
            else:
                final_idx = accepted_idx

            selected = candidates[final_idx]
            avg_quality = similarities[final_idx].mean()
            return selected, thresh, avg_quality

    # If still not enough at minimum threshold, take whatever we can
    accepted_mask = similarities >= MIN_THRESHOLD
    if accepted_mask.sum() > 0:
        accepted_idx = np.where(accepted_mask)[0]
        selected = candidates[accepted_idx]
        avg_quality = similarities[accepted_idx].mean()
        return selected, MIN_THRESHOLD, avg_quality

    # Last resort: take top-K by similarity regardless of threshold
    if len(candidates) > 0:
        top_k = min(target_per_cluster, len(candidates))
        top_idx = np.argsort(similarities)[-top_k:]
        selected = candidates[top_idx]
        avg_quality = similarities[top_idx].mean()
        return selected, similarities[top_idx].min(), avg_quality

    return np.array([]).reshape(0, candidates.shape[1]), 0.0, 0.0


def generate_synthetic_with_adaptive_threshold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    config: dict,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate synthetic samples with adaptive threshold relaxation."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_thresholds = []
    all_qualities = []

    n_classes = len(np.unique(labels))
    target_per_class = max(5, TARGET_SYNTH_TOTAL // n_classes)

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        n_clusters = min(3, len(class_embeddings) // 30)
        if n_clusters < 1:
            n_clusters = 1

        target_per_cluster = max(3, target_per_class // n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        for c_id in range(n_clusters):
            c_mask = cluster_labels_arr == c_id
            c_points = class_embeddings[c_mask]
            c_texts = class_texts[c_mask]

            if len(c_points) < 3:
                continue

            anchor_emb = kmeans.cluster_centers_[c_id]

            # Determine starting threshold
            if config["start_threshold"] is None:
                # Purity-adaptive
                start_thresh = get_purity_based_threshold(
                    anchor_emb, embeddings, labels, target_class
                )
            else:
                start_thresh = config["start_threshold"]

            # Get example texts
            dists = np.linalg.norm(c_points - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [c_texts[i] for i in nearest_idx]

            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])

            # Generate more candidates for selection pool
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

                # Apply adaptive threshold
                selected, thresh_used, quality = apply_threshold_with_relaxation(
                    candidate_embeddings, anchor_emb, start_thresh, target_per_cluster
                )

                all_thresholds.append(thresh_used)
                all_qualities.append(quality)

                for emb in selected:
                    synthetic_embeddings.append(emb)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

    avg_threshold = np.mean(all_thresholds) if all_thresholds else 0.0
    avg_quality = np.mean(all_qualities) if all_qualities else 0.0

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), avg_threshold, avg_quality

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), avg_threshold, avg_quality


def run_threshold_experiment() -> list:
    """Run adaptive thresholds v2 experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 05 v2: ADAPTIVE THRESHOLDS (RELAXATION)")
    print("="*60)
    print(f"  Target total: ~{TARGET_SYNTH_TOTAL} samples")
    print(f"  Min threshold: {MIN_THRESHOLD}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for config in THRESHOLD_CONFIGS:
        print(f"\n{'─'*60}")
        print(f"  Testing: {config['name']}")
        print(f"  {config['description']}")
        print(f"{'─'*60}")

        X_synth, y_synth, avg_thresh, avg_quality = generate_synthetic_with_adaptive_threshold(
            embeddings, labels, texts, config, cache
        )

        print(f"  Avg threshold used: {avg_thresh:.2f}")
        print(f"  Avg quality: {avg_quality:.3f}")
        print(f"  Total synthetic: {len(X_synth)}")

        kfold_result = evaluator.evaluate(
            X_original=embeddings,
            y_original=labels,
            X_synthetic=X_synth if len(X_synth) > 0 else None,
            y_synthetic=y_synth if len(y_synth) > 0 else None,
            config_name=config['name']
        )

        result = ThresholdResult(
            config=config['name'],
            macro_f1=float(kfold_result.augmented_mean),
            avg_threshold_used=float(avg_thresh),
            avg_quality=float(avg_quality),
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
    """Generate LaTeX table for adaptive thresholds v2."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Umbrales adaptativos con relajacion progresiva}
\label{tab:adaptive_thresholds_v2}
\begin{tabular}{lcccc}
\hline
Estrategia & Umbral Prom. & N Synth & Macro F1 & $\Delta$ \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        if i == best_idx:
            latex += f"\\textbf{{{r.config}}} & \\textbf{{{r.avg_threshold_used:.2f}}} & \\textbf{{{r.n_synthetic}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.delta_pct:+.2f}\\%}} \\\\\n"
        else:
            latex += f"{r.config} & {r.avg_threshold_used:.2f} & {r.n_synthetic} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\% \\\\\n"

    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_threshold_experiment()

    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_adaptive_thresholds_v2.tex", 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("  SUMMARY: Adaptive Thresholds v2 Results")
    print("="*60)
    for r in results:
        print(f"{r.config:>20} | Thresh={r.avg_threshold_used:.2f} | N={r.n_synthetic:3} | F1={r.macro_f1:.4f} | Delta={r.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
