#!/usr/bin/env python3
"""
Experiment 07: Comprehensive Parameter Validation

Tests multiple parameters using our BEST configuration:
- Base: full_cascade filter + adaptive ranking + purity-based thresholds
- K_max=12, Medoid anchors

Parameters to test:
1. Synthetic Weight: w = {0.3, 0.5, 0.7, 1.0}
2. Temperature: τ = {0.3, 0.5, 0.7, 0.9}
3. Budget Multiplier: {5%, 8%, 12%, 15%}

Each test uses ~100 synthetic samples for fair comparison.
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

from base_config import RESULTS_DIR, LATEX_DIR, LLM_MODEL, MAX_TOKENS
from validation_runner import load_data, EmbeddingCache, print_summary
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats


EXPERIMENT_NAME = "comprehensive"
TARGET_SYNTH_PER_CLASS = 7  # ~112 total


@dataclass
class ComprehensiveResult:
    experiment: str
    config: str
    param_value: float
    n_synthetic: int
    avg_quality: float
    macro_f1: float
    delta_pct: float
    p_value: float
    significant: bool


class FlexibleKFoldEvaluator:
    """K-fold evaluator with configurable synthetic weight."""

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
        """Evaluate with configurable synthetic weight."""
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

        return {
            "baseline_mean": baseline_mean,
            "augmented_mean": augmented_mean,
            "delta_pct": delta_pct,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_synthetic": len(X_synthetic)
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


def generate_with_temperature(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache,
    temperature: float = 0.7,
    budget_pct: float = 0.08
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate synthetic samples with specific temperature and budget."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_qualities = []

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        # Budget based on class size
        target_for_class = max(3, int(len(class_embeddings) * budget_pct))
        target_for_class = min(target_for_class, TARGET_SYNTH_PER_CLASS)

        n_clusters = min(3, len(class_embeddings) // 30)
        if n_clusters < 1:
            n_clusters = 1

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
                    temperature=temperature,  # Variable temperature
                    max_tokens=MAX_TOKENS * n_candidates,
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


def run_weight_experiment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    avg_quality: float
) -> List[ComprehensiveResult]:
    """Test different synthetic weights using pre-generated data."""
    print("\n" + "="*60)
    print("  PART 1: SYNTHETIC WEIGHT EXPERIMENT")
    print("="*60)

    weights = [0.3, 0.5, 0.7, 1.0]
    results = []
    evaluator = FlexibleKFoldEvaluator()

    for w in weights:
        print(f"\n  Testing weight = {w}")

        eval_result = evaluator.evaluate(
            embeddings, labels, X_synth, y_synth,
            synthetic_weight=w
        )

        result = ComprehensiveResult(
            experiment="weight",
            config=f"w={w}",
            param_value=float(w),
            n_synthetic=int(eval_result["n_synthetic"]),
            avg_quality=float(avg_quality),
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"])
        )
        results.append(result)

        sig = "*" if result.significant else ""
        print(f"    F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}, p={result.p_value:.4f}")

    return results


def run_temperature_experiment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache
) -> List[ComprehensiveResult]:
    """Test different temperatures (requires new API calls)."""
    print("\n" + "="*60)
    print("  PART 2: TEMPERATURE EXPERIMENT")
    print("="*60)

    temperatures = [0.3, 0.5, 0.7, 0.9]
    results = []
    evaluator = FlexibleKFoldEvaluator()

    for temp in temperatures:
        print(f"\n  Testing temperature = {temp}")

        X_synth, y_synth, avg_quality = generate_with_temperature(
            embeddings, labels, texts, cache,
            temperature=temp
        )

        print(f"    Generated {len(X_synth)} samples, quality={avg_quality:.3f}")

        eval_result = evaluator.evaluate(
            embeddings, labels, X_synth, y_synth,
            synthetic_weight=0.5  # Use default weight
        )

        result = ComprehensiveResult(
            experiment="temperature",
            config=f"τ={temp}",
            param_value=float(temp),
            n_synthetic=int(len(X_synth)),
            avg_quality=float(avg_quality),
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"])
        )
        results.append(result)

        sig = "*" if result.significant else ""
        print(f"    F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}, p={result.p_value:.4f}")

    return results


def run_budget_experiment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache
) -> List[ComprehensiveResult]:
    """Test different budget multipliers."""
    print("\n" + "="*60)
    print("  PART 3: BUDGET MULTIPLIER EXPERIMENT")
    print("="*60)

    budgets = [0.05, 0.08, 0.12, 0.15]  # 5%, 8%, 12%, 15%
    results = []
    evaluator = FlexibleKFoldEvaluator()

    for budget in budgets:
        print(f"\n  Testing budget = {budget*100:.0f}%")

        X_synth, y_synth, avg_quality = generate_with_temperature(
            embeddings, labels, texts, cache,
            temperature=0.7,  # Use best temperature
            budget_pct=budget
        )

        print(f"    Generated {len(X_synth)} samples, quality={avg_quality:.3f}")

        eval_result = evaluator.evaluate(
            embeddings, labels, X_synth, y_synth,
            synthetic_weight=0.5
        )

        result = ComprehensiveResult(
            experiment="budget",
            config=f"budget={budget*100:.0f}%",
            param_value=float(budget),
            n_synthetic=int(len(X_synth)),
            avg_quality=float(avg_quality),
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"])
        )
        results.append(result)

        sig = "*" if result.significant else ""
        print(f"    F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}, p={result.p_value:.4f}")

    return results


def generate_latex_tables(all_results: Dict[str, List[ComprehensiveResult]]) -> str:
    """Generate LaTeX tables for all experiments."""
    latex = ""

    # Weight table
    latex += r"""
\begin{table}[h]
\centering
\caption{Impacto del peso de muestras sintéticas}
\label{tab:synthetic_weight}
\begin{tabular}{lccccc}
\hline
Peso & N Synth & Quality & Macro F1 & $\Delta$ & p-value \\
\hline
"""
    for r in all_results.get("weight", []):
        sig = "*" if r.significant else ""
        latex += f"{r.param_value} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig} & {r.p_value:.4f} \\\\\n"
    latex += r"""\hline
\end{tabular}
\end{table}
"""

    # Temperature table
    latex += r"""
\begin{table}[h]
\centering
\caption{Impacto de la temperatura de generación}
\label{tab:temperature}
\begin{tabular}{lccccc}
\hline
$\tau$ & N Synth & Quality & Macro F1 & $\Delta$ & p-value \\
\hline
"""
    for r in all_results.get("temperature", []):
        sig = "*" if r.significant else ""
        latex += f"{r.param_value} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig} & {r.p_value:.4f} \\\\\n"
    latex += r"""\hline
\end{tabular}
\end{table}
"""

    # Budget table
    latex += r"""
\begin{table}[h]
\centering
\caption{Impacto del multiplicador de presupuesto}
\label{tab:budget_multiplier}
\begin{tabular}{lccccc}
\hline
Budget & N Synth & Quality & Macro F1 & $\Delta$ & p-value \\
\hline
"""
    for r in all_results.get("budget", []):
        sig = "*" if r.significant else ""
        latex += f"{r.config} & {r.n_synthetic} & {r.avg_quality:.3f} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig} & {r.p_value:.4f} \\\\\n"
    latex += r"""\hline
\end{tabular}
\end{table}
"""

    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60)
    print("  EXPERIMENT 07: COMPREHENSIVE PARAMETER VALIDATION")
    print("="*60)
    print("  Base config: full_cascade + adaptive ranking")
    print("  Tests: weight, temperature, budget")

    # Load data
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    all_results = {}

    # First, generate base synthetic data for weight experiment
    print("\n  Generating base synthetic data (τ=0.7, full_cascade)...")
    X_synth_base, y_synth_base, base_quality = generate_with_temperature(
        embeddings, labels, texts, cache,
        temperature=0.7, budget_pct=0.08
    )
    print(f"  Base: {len(X_synth_base)} samples, quality={base_quality:.3f}")

    # Part 1: Weight experiment (no new API calls)
    weight_results = run_weight_experiment(
        embeddings, labels, X_synth_base, y_synth_base, base_quality
    )
    all_results["weight"] = weight_results

    # Part 2: Temperature experiment (new API calls)
    temp_results = run_temperature_experiment(
        embeddings, labels, texts, cache
    )
    all_results["temperature"] = temp_results

    # Part 3: Budget experiment (new API calls)
    budget_results = run_budget_experiment(
        embeddings, labels, texts, cache
    )
    all_results["budget"] = budget_results

    # Save results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, results in all_results.items():
        for r in results:
            with open(output_dir / f"{exp_name}_{r.config.replace('=', '_').replace('%', 'pct')}_result.json", 'w') as f:
                json.dump(asdict(r), f, indent=2)

    # Generate LaTeX
    latex = generate_latex_tables(all_results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_comprehensive.tex", 'w') as f:
        f.write(latex)

    # Print summary
    print("\n" + "="*60)
    print("  COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)

    for exp_name, results in all_results.items():
        print(f"\n  {exp_name.upper()}:")
        print(f"  {'Config':<15} | {'N Synth':>8} | {'Delta':>8} | {'p-value':>8} | Sig")
        print("  " + "-"*55)
        for r in results:
            sig = "*" if r.significant else " "
            print(f"  {r.config:<15} | {r.n_synthetic:>8} | {r.delta_pct:>+7.2f}% | {r.p_value:>8.4f} | {sig}")

    # Find best config per experiment
    print("\n  BEST CONFIGS:")
    for exp_name, results in all_results.items():
        best = max(results, key=lambda r: r.delta_pct)
        print(f"    {exp_name}: {best.config} -> {best.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
