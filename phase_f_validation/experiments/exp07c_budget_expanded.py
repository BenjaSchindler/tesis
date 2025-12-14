#!/usr/bin/env python3
"""
Experiment 07c Expanded: Budget Validation - Extended Range

Tests budget percentages: [5%, 8%, 10%, 12%, 15%, 18%, 20%, 25%, 30%]
Focus: Find saturation point and potential degradation at higher budgets

Metrics: Macro F1, Delta%, N Synthetic, Quality Score

Output: tab:budget_validation_expanded for Metodologia.tex
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
from typing import List, Tuple, Dict

from base_config import (
    RESULTS_DIR, LATEX_DIR, LLM_MODEL, MAX_TOKENS, EXPERIMENT_PARAMS, TEMPERATURE
)
from validation_runner import load_data, EmbeddingCache, print_summary
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats


# Use expanded budget values
BUDGET_VALUES = EXPERIMENT_PARAMS["budget"]["BUDGET_VALUES_EXPANDED"]
EXPERIMENT_NAME = "budget_expanded"


@dataclass
class BudgetResult:
    budget_pct: float
    n_synthetic: int
    avg_quality: float
    macro_f1: float
    delta_pct: float
    baseline_f1: float
    p_value: float
    significant: bool
    win_rate: float


class FlexibleKFoldEvaluator:
    """K-fold evaluator with detailed metrics."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 3):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.kfold = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )

    def evaluate(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_synthetic: np.ndarray,
        y_synthetic: np.ndarray,
        synthetic_weight: float = 0.5
    ) -> Dict:
        """Evaluate with detailed metrics."""
        baseline_scores = []
        augmented_scores = []

        for train_idx, test_idx in self.kfold.split(X_original, y_original):
            X_train, X_test = X_original[train_idx], X_original[test_idx]
            y_train, y_test = y_original[train_idx], y_original[test_idx]

            # Baseline
            clf_base = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf_base.fit(X_train, y_train)
            y_pred_base = clf_base.predict(X_test)
            baseline_scores.append(f1_score(y_test, y_pred_base, average='macro'))

            # Augmented with configurable weight
            if len(X_synthetic) > 0:
                X_aug = np.vstack([X_train, X_synthetic])
                y_aug = np.concatenate([y_train, y_synthetic])
                weights = np.concatenate([
                    np.ones(len(X_train)),
                    np.full(len(X_synthetic), synthetic_weight)
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

        # Win rate
        wins = sum(1 for a, b in zip(augmented_scores, baseline_scores) if a > b)
        win_rate = wins / len(baseline_scores)

        return {
            "baseline_mean": baseline_mean,
            "augmented_mean": augmented_mean,
            "delta_pct": delta_pct,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_synthetic": len(X_synthetic),
            "win_rate": win_rate
        }


def compute_full_cascade_scores(
    candidates: np.ndarray,
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str
) -> np.ndarray:
    """Compute full cascade quality scores (all 4 filters)."""
    if len(candidates) == 0:
        return np.array([])

    n_candidates = len(candidates)
    scores_dict = {}

    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)

    # Filter 1: Length/distance from anchor
    dists_to_anchor = np.linalg.norm(candidates - anchor_emb, axis=1)
    max_dist = np.max(dists_to_anchor) + 1e-6
    scores_dict["length"] = 1 - (dists_to_anchor / max_dist)

    # Filter 2: Cosine similarity
    similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()
    scores_dict["similarity"] = np.clip(similarities, 0, 1)

    # Filter 3: K-NN purity
    class_mask = all_labels == target_class
    class_embs = all_embeddings[class_mask]
    knn_scores = np.zeros(n_candidates)
    if len(class_embs) > 0:
        for i, cand in enumerate(candidates):
            dists = np.linalg.norm(class_embs - cand, axis=1)
            k = min(10, len(class_embs))
            nearest_dists = np.sort(dists)[:k]
            knn_scores[i] = 1.0 / (1.0 + nearest_dists.mean())
    scores_dict["knn"] = knn_scores

    # Filter 4: Confidence (distance to centroid)
    if len(class_embs) > 0:
        centroid = class_embs.mean(axis=0)
        dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
        max_dist = np.max(dists_to_centroid) + 1e-6
        scores_dict["confidence"] = 1 - (dists_to_centroid / max_dist)
    else:
        scores_dict["confidence"] = np.ones(n_candidates) * 0.5

    # Geometric mean
    combined = np.ones(n_candidates)
    for score_array in scores_dict.values():
        combined *= score_array
    combined = np.power(combined, 1.0 / len(scores_dict))

    return combined


def generate_with_budget(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache,
    budget_pct: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate synthetic samples with specific budget percentage."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_qualities = []

    # Calculate target per class based on budget
    unique_classes = np.unique(labels)

    for target_class in unique_classes:
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Budget determines how many synthetic samples per class
        target_for_class = max(3, int(len(class_embeddings) * budget_pct))

        n_clusters = min(5, max(1, len(class_embeddings) // 30))
        target_per_cluster = max(2, target_for_class // n_clusters)

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

            # Generate more candidates for higher budgets
            n_candidates = max(20, target_per_cluster * 3)

            prompt = f"""Generate {n_candidates} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_candidates} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

            try:
                # Cap max_tokens to model limit (16384 for gpt-4o-mini)
                max_tokens_request = min(MAX_TOKENS * n_candidates, 16000)
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens_request,
                )

                generated_text = response.choices[0].message.content.strip()
                samples = [s.strip() for s in generated_text.split('\n')
                          if s.strip() and len(s.strip()) > 10]

                if not samples:
                    continue

                candidate_embeddings = cache.embed_synthetic(samples)

                # Full cascade scoring
                scores = compute_full_cascade_scores(
                    candidate_embeddings, anchor_emb,
                    embeddings, labels, target_class
                )

                # Select top by ranking
                n_select = min(target_per_cluster, len(candidate_embeddings))
                top_idx = np.argsort(scores)[-n_select:]

                selected = candidate_embeddings[top_idx]
                avg_quality = scores[top_idx].mean()
                all_qualities.append(avg_quality)

                for emb in selected:
                    synthetic_embeddings.append(emb)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class}: {e}", flush=True)
                continue

    avg_quality = np.mean(all_qualities) if all_qualities else 0.0

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), avg_quality

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), avg_quality


def run_budget_experiment() -> List[BudgetResult]:
    """Run expanded budget validation experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 07c EXPANDED: BUDGET VALIDATION")
    print("="*60)
    print(f"  Budget values: {[f'{b*100:.0f}%' for b in BUDGET_VALUES]}")
    print(f"  Total configurations: {len(BUDGET_VALUES)}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = FlexibleKFoldEvaluator()
    results = []

    for i, budget in enumerate(BUDGET_VALUES):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(BUDGET_VALUES)}] Testing budget = {budget*100:.0f}%")
        print(f"{'─'*60}")

        X_synth, y_synth, avg_quality = generate_with_budget(
            embeddings, labels, texts, cache, budget
        )

        print(f"  Generated {len(X_synth)} samples, quality={avg_quality:.3f}")

        eval_result = evaluator.evaluate(
            embeddings, labels, X_synth, y_synth,
            synthetic_weight=0.5
        )

        result = BudgetResult(
            budget_pct=float(budget),
            n_synthetic=int(len(X_synth)),
            avg_quality=float(avg_quality),
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            baseline_f1=float(eval_result["baseline_mean"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"]),
            win_rate=float(eval_result["win_rate"])
        )
        results.append(result)

        # Save individual result
        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"budget_{int(budget*100)}pct_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        sig = "*" if result.significant else ""
        print(f"  F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}, p={result.p_value:.4f}")

    return results


def generate_latex_table(results: List[BudgetResult]) -> str:
    """Generate LaTeX table for expanded budget validation."""
    latex = r"""
% Tabla: Validacion expandida de presupuesto de generacion
% Experimento 07c: Presupuesto de Generacion (5%-30%)
\begin{table}[h]
\centering
\caption{Impacto expandido del presupuesto de generacion sintetica}
\label{tab:budget_validation_expanded}
\begin{tabular}{ccccc}
\hline
Presupuesto (\%) & N Sinteticas & Calidad & Macro F1 & $\Delta$ F1 \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sig_marker = "*" if r.significant else ""
        budget_str = f"{r.budget_pct*100:.0f}\\%"

        if i == best_idx:
            latex += f"\\textbf{{{budget_str}}} & \\textbf{{{r.n_synthetic}}} & \\textbf{{{r.avg_quality:.3f}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.delta_pct:+.2f}\\%{sig_marker}}} \\\\\n"
        else:
            latex += f"{budget_str} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig_marker} \\\\\n"

    latex += r"""
\hline
\multicolumn{5}{l}{\footnotesize *Estadisticamente significativo (p $<$ 0.05)} \\
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_budget_experiment()

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_budget_validation_expanded.tex", 'w') as f:
        f.write(latex_table)

    # Save all results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY: Expanded Budget Validation Results")
    print("="*70)
    print(f"{'Budget':>8} | {'N Synth':>8} | {'Quality':>8} | {'F1':>8} | {'Delta':>10} | {'Sig':>4}")
    print("-"*70)
    for r in results:
        sig_str = "*" if r.significant else ""
        print(f"{r.budget_pct*100:>7.0f}% | {r.n_synthetic:>8} | {r.avg_quality:>8.3f} | {r.macro_f1:>.4f} | {r.delta_pct:>+9.2f}% | {sig_str:>4}")

    # Find best and highlight saturation
    best_idx = np.argmax([r.delta_pct for r in results])
    best = results[best_idx]
    print("-"*70)
    print(f"  BEST: Budget={best.budget_pct*100:.0f}% with Delta F1={best.delta_pct:+.2f}%")

    # Check for saturation/degradation
    if len(results) > 3:
        deltas = [r.delta_pct for r in results]
        peak_idx = np.argmax(deltas)
        if peak_idx < len(results) - 1:
            degradation = deltas[peak_idx] - deltas[-1]
            if degradation > 0.5:
                print(f"  WARNING: Degradation detected after {results[peak_idx].budget_pct*100:.0f}% "
                      f"(lost {degradation:.2f}% improvement)")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
