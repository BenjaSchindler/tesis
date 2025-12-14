#!/usr/bin/env python3
"""
Experiment 07b: Temperature Diversity Analysis

Tests different LLM temperatures: [0.3, 0.5, 0.7, 0.9]
with additional diversity and quality metrics:

1. text_diversity: Average pairwise distance between generated texts (embedding space)
2. vocab_diversity: Unique words / total words ratio
3. filter_rejection_rate: % of samples rejected by each filter
4. semantic_drift: Average distance from generated text to anchor embedding

This experiment provides insights into the trade-off between
diversity (higher temp) and quality/consistency (lower temp).

Output: tab:temperature_diversity for Metodologia.tex
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from typing import List, Tuple, Dict

from base_config import (
    RESULTS_DIR, LATEX_DIR, LLM_MODEL, MAX_TOKENS, EXPERIMENT_PARAMS
)
from validation_runner import load_data, EmbeddingCache
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats


TEMPERATURES = EXPERIMENT_PARAMS["temperature"]["VALUES"]  # [0.3, 0.5, 0.7, 0.9]
EXPERIMENT_NAME = "temperature_diversity"


@dataclass
class TemperatureDiversityResult:
    temperature: float
    n_generated: int  # Before filtering
    n_synthetic: int  # After filtering
    rejection_rate: float  # % rejected by filters
    text_diversity: float  # Avg pairwise embedding distance
    vocab_diversity: float  # Unique words / total words
    semantic_drift: float  # Avg distance from anchor
    avg_quality: float  # Average quality score
    macro_f1: float
    delta_pct: float
    baseline_f1: float
    p_value: float
    significant: bool


def compute_vocab_diversity(texts: List[str]) -> float:
    """Compute vocabulary diversity: unique words / total words."""
    all_words = []
    for text in texts:
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(words)

    if not all_words:
        return 0.0

    unique_words = len(set(all_words))
    total_words = len(all_words)

    return unique_words / total_words


def compute_text_diversity(embeddings: np.ndarray) -> float:
    """Compute average pairwise distance between embeddings."""
    if len(embeddings) < 2:
        return 0.0

    # Compute pairwise distances
    distances = pdist(embeddings, metric='euclidean')
    return np.mean(distances)


def compute_semantic_drift(generated_embeddings: np.ndarray, anchor_embeddings: np.ndarray) -> float:
    """Compute average distance from generated samples to their anchors."""
    if len(generated_embeddings) == 0 or len(anchor_embeddings) == 0:
        return 0.0

    # For each generated sample, compute distance to nearest anchor
    dists = cdist(generated_embeddings, anchor_embeddings, metric='euclidean')
    min_dists = np.min(dists, axis=1)
    return np.mean(min_dists)


def compute_filter_scores(
    candidate_emb: np.ndarray,
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str
) -> Dict[str, float]:
    """Compute individual filter scores for a candidate."""
    scores = {}

    # Distance from anchor (lower is better -> higher score)
    dist_to_anchor = np.linalg.norm(candidate_emb - anchor_emb)
    scores["length"] = 1.0 / (1.0 + dist_to_anchor)

    # Cosine similarity
    cos_sim = 1 - cdist(candidate_emb.reshape(1, -1), anchor_emb.reshape(1, -1), metric='cosine')[0, 0]
    scores["similarity"] = max(0, cos_sim)

    # K-NN purity
    class_mask = all_labels == target_class
    class_embs = all_embeddings[class_mask]
    if len(class_embs) > 0:
        dists = np.linalg.norm(class_embs - candidate_emb, axis=1)
        k = min(10, len(class_embs))
        nearest_dists = np.sort(dists)[:k]
        scores["knn"] = 1.0 / (1.0 + nearest_dists.mean())
    else:
        scores["knn"] = 0.5

    # Confidence (distance to centroid)
    if len(class_embs) > 0:
        centroid = class_embs.mean(axis=0)
        dist_to_centroid = np.linalg.norm(candidate_emb - centroid)
        scores["confidence"] = 1.0 / (1.0 + dist_to_centroid)
    else:
        scores["confidence"] = 0.5

    return scores


def generate_with_diversity_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache,
    temperature: float
) -> Dict:
    """Generate synthetic samples and compute diversity metrics."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    all_generated_texts = []
    all_generated_embeddings = []
    all_anchor_embeddings = []
    all_filter_results = {"length": [], "similarity": [], "knn": [], "confidence": []}

    accepted_embeddings = []
    accepted_labels = []
    all_quality_scores = []

    unique_classes = np.unique(labels)

    for target_class in unique_classes:
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        n_clusters = min(3, max(1, len(class_embeddings) // 40))
        target_per_cluster = 5

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_arr = kmeans.fit_predict(class_embeddings)

        for c_id in range(n_clusters):
            c_mask = cluster_labels_arr == c_id
            c_points = class_embeddings[c_mask]
            c_texts = class_texts[c_mask]

            if len(c_points) < 3:
                continue

            anchor_emb = kmeans.cluster_centers_[c_id]

            # Get examples
            dists = np.linalg.norm(c_points - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [c_texts[i] for i in nearest_idx]

            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])

            n_candidates = 15

            prompt = f"""Generate {n_candidates} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_candidates} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,  # Variable temperature
                    max_tokens=MAX_TOKENS * n_candidates,
                )

                generated_text = response.choices[0].message.content.strip()
                samples = [s.strip() for s in generated_text.split('\n')
                          if s.strip() and len(s.strip()) > 10]

                if not samples:
                    continue

                # Store generated texts for diversity metrics
                all_generated_texts.extend(samples)

                # Embed candidates
                candidate_embeddings = cache.embed_synthetic(samples)
                all_generated_embeddings.extend(candidate_embeddings)
                all_anchor_embeddings.extend([anchor_emb] * len(candidate_embeddings))

                # Compute filter scores for each candidate
                for cand_emb in candidate_embeddings:
                    scores = compute_filter_scores(
                        cand_emb, anchor_emb, embeddings, labels, target_class
                    )
                    for filter_name, score in scores.items():
                        all_filter_results[filter_name].append(score)

                # Combined quality score (geometric mean)
                combined_scores = np.zeros(len(candidate_embeddings))
                for i, cand_emb in enumerate(candidate_embeddings):
                    scores = compute_filter_scores(
                        cand_emb, anchor_emb, embeddings, labels, target_class
                    )
                    combined = np.prod([s for s in scores.values()]) ** (1/len(scores))
                    combined_scores[i] = combined

                # Select top candidates by quality
                n_select = min(target_per_cluster, len(candidate_embeddings))
                top_idx = np.argsort(combined_scores)[-n_select:]

                for idx in top_idx:
                    accepted_embeddings.append(candidate_embeddings[idx])
                    accepted_labels.append(target_class)
                    all_quality_scores.append(combined_scores[idx])

            except Exception as e:
                print(f"    API error for {target_class}: {e}", flush=True)
                continue

    # Compute diversity metrics
    metrics = {
        "n_generated": len(all_generated_texts),
        "n_accepted": len(accepted_embeddings),
        "rejection_rate": 1 - (len(accepted_embeddings) / max(1, len(all_generated_texts))),
    }

    # Text diversity (embedding space)
    if all_generated_embeddings:
        metrics["text_diversity"] = compute_text_diversity(np.array(all_generated_embeddings))
    else:
        metrics["text_diversity"] = 0.0

    # Vocabulary diversity
    metrics["vocab_diversity"] = compute_vocab_diversity(all_generated_texts)

    # Semantic drift
    if all_generated_embeddings and all_anchor_embeddings:
        metrics["semantic_drift"] = compute_semantic_drift(
            np.array(all_generated_embeddings),
            np.array(all_anchor_embeddings)
        )
    else:
        metrics["semantic_drift"] = 0.0

    # Average quality
    metrics["avg_quality"] = np.mean(all_quality_scores) if all_quality_scores else 0.0

    # Per-filter average scores
    for filter_name, scores in all_filter_results.items():
        metrics[f"filter_{filter_name}_avg"] = np.mean(scores) if scores else 0.0

    # Return synthetic data and metrics
    if accepted_embeddings:
        X_synth = np.array(accepted_embeddings)
        y_synth = np.array(accepted_labels)
    else:
        X_synth = np.array([]).reshape(0, embeddings.shape[1])
        y_synth = np.array([])

    return {
        "X_synth": X_synth,
        "y_synth": y_synth,
        "metrics": metrics
    }


def evaluate_classification(
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
) -> Dict:
    """Evaluate classification performance."""
    kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    baseline_scores = []
    augmented_scores = []

    for train_idx, test_idx in kfold.split(X_original, y_original):
        X_train, X_test = X_original[train_idx], X_original[test_idx]
        y_train, y_test = y_original[train_idx], y_original[test_idx]

        # Baseline
        clf_base = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf_base.fit(X_train, y_train)
        y_pred_base = clf_base.predict(X_test)
        baseline_scores.append(f1_score(y_test, y_pred_base, average='macro'))

        # Augmented
        if len(X_synthetic) > 0:
            X_aug = np.vstack([X_train, X_synthetic])
            y_aug = np.concatenate([y_train, y_synthetic])
            weights = np.concatenate([
                np.ones(len(X_train)),
                np.full(len(X_synthetic), 0.5)
            ])

            clf_aug = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf_aug.fit(X_aug, y_aug, sample_weight=weights)
            y_pred_aug = clf_aug.predict(X_test)
            augmented_scores.append(f1_score(y_test, y_pred_aug, average='macro'))
        else:
            augmented_scores.append(baseline_scores[-1])

    baseline_mean = np.mean(baseline_scores)
    augmented_mean = np.mean(augmented_scores)
    delta = augmented_mean - baseline_mean
    delta_pct = (delta / baseline_mean) * 100

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(augmented_scores, baseline_scores)

    return {
        "baseline_mean": baseline_mean,
        "augmented_mean": augmented_mean,
        "delta_pct": delta_pct,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def run_temperature_diversity_experiment() -> List[TemperatureDiversityResult]:
    """Run temperature diversity experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 07b: TEMPERATURE DIVERSITY ANALYSIS")
    print("="*60)
    print(f"  Temperatures: {TEMPERATURES}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    results = []

    for i, temp in enumerate(TEMPERATURES):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(TEMPERATURES)}] Testing temperature = {temp}")
        print(f"{'─'*60}")

        # Generate with diversity metrics
        gen_result = generate_with_diversity_metrics(
            embeddings, labels, texts, cache, temp
        )

        X_synth = gen_result["X_synth"]
        y_synth = gen_result["y_synth"]
        metrics = gen_result["metrics"]

        print(f"  Generated: {metrics['n_generated']} -> Accepted: {metrics['n_accepted']}")
        print(f"  Rejection rate: {metrics['rejection_rate']*100:.1f}%")
        print(f"  Text diversity: {metrics['text_diversity']:.4f}")
        print(f"  Vocab diversity: {metrics['vocab_diversity']:.3f}")
        print(f"  Semantic drift: {metrics['semantic_drift']:.4f}")

        # Evaluate classification
        eval_result = evaluate_classification(embeddings, labels, X_synth, y_synth)

        result = TemperatureDiversityResult(
            temperature=temp,
            n_generated=int(metrics["n_generated"]),
            n_synthetic=int(metrics["n_accepted"]),
            rejection_rate=float(metrics["rejection_rate"]),
            text_diversity=float(metrics["text_diversity"]),
            vocab_diversity=float(metrics["vocab_diversity"]),
            semantic_drift=float(metrics["semantic_drift"]),
            avg_quality=float(metrics["avg_quality"]),
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            baseline_f1=float(eval_result["baseline_mean"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"])
        )
        results.append(result)

        # Save individual result
        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"temp_{temp}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        sig = "*" if result.significant else ""
        print(f"  F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}")

    return results


def generate_latex_table(results: List[TemperatureDiversityResult]) -> str:
    """Generate LaTeX table for temperature diversity experiment."""
    latex = r"""
% Tabla: Analisis de diversidad por temperatura del LLM
% Experimento 07b: Diversidad por Temperatura
\begin{table}[h]
\centering
\caption{Analisis de diversidad por temperatura del LLM}
\label{tab:temperature_diversity}
\begin{tabular}{ccccccc}
\hline
$\tau$ & N Gen & Rechazo & Div Texto & Div Vocab & Drift & $\Delta$ F1 \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sig_marker = "*" if r.significant else ""

        if i == best_idx:
            latex += f"\\textbf{{{r.temperature}}} & \\textbf{{{r.n_generated}}} & \\textbf{{{r.rejection_rate*100:.0f}\\%}} & \\textbf{{{r.text_diversity:.3f}}} & \\textbf{{{r.vocab_diversity:.3f}}} & \\textbf{{{r.semantic_drift:.3f}}} & \\textbf{{{r.delta_pct:+.2f}\\%{sig_marker}}} \\\\\n"
        else:
            latex += f"{r.temperature} & {r.n_generated} & {r.rejection_rate*100:.0f}\\% & {r.text_diversity:.3f} & {r.vocab_diversity:.3f} & {r.semantic_drift:.3f} & {r.delta_pct:+.2f}\\%{sig_marker} \\\\\n"

    latex += r"""
\hline
\multicolumn{7}{l}{\footnotesize *Estadisticamente significativo (p $<$ 0.05)} \\
\multicolumn{7}{l}{\footnotesize Div Texto: diversidad embedding, Div Vocab: palabras unicas/total, Drift: distancia al anchor} \\
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_temperature_diversity_experiment()

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_temperature_diversity.tex", 'w') as f:
        f.write(latex_table)

    # Save all results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Print summary
    print("\n" + "="*85)
    print("  SUMMARY: Temperature Diversity Analysis Results")
    print("="*85)
    print(f"{'Temp':>5} | {'Gen':>5} | {'Accept':>6} | {'Reject':>7} | {'TextDiv':>8} | {'VocabDiv':>8} | {'Drift':>7} | {'Delta':>8} | {'Sig':>4}")
    print("-"*85)
    for r in results:
        sig_str = "*" if r.significant else ""
        print(f"{r.temperature:>5.1f} | {r.n_generated:>5} | {r.n_synthetic:>6} | {r.rejection_rate*100:>6.1f}% | {r.text_diversity:>8.4f} | {r.vocab_diversity:>8.3f} | {r.semantic_drift:>7.4f} | {r.delta_pct:>+7.2f}% | {sig_str:>4}")

    # Find best
    best_idx = np.argmax([r.delta_pct for r in results])
    best = results[best_idx]
    print("-"*85)
    print(f"  BEST: Temperature={best.temperature} with Delta F1={best.delta_pct:+.2f}%")

    # Analyze trade-offs
    print("\n  DIVERSITY vs PERFORMANCE TRADE-OFF:")
    for r in results:
        print(f"    T={r.temperature}: Higher diversity ({r.text_diversity:.3f}) -> "
              f"{'Higher' if r.semantic_drift > results[0].semantic_drift else 'Lower'} drift ({r.semantic_drift:.3f}), "
              f"Delta={r.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
