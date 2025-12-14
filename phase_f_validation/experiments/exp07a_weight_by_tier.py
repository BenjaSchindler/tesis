#!/usr/bin/env python3
"""
Experiment 07a: Weight by Performance Tier

Tests different synthetic sample weights based on class performance tier:
- LOW tier: Classes with baseline F1 < 0.20
- MID tier: Classes with baseline F1 between 0.20 and 0.45
- HIGH tier: Classes with baseline F1 > 0.45

Configurations:
1. uniform_1.0: All tiers get weight 1.0
2. tier_boost_low: LOW=1.5, MID=1.0, HIGH=0.5
3. tier_boost_extreme: LOW=2.0, MID=0.8, HIGH=0.3
4. only_low: LOW=1.0, MID=0.0, HIGH=0.0

Metrics: Macro F1, Delta%, Per-tier F1

Output: tab:weight_by_tier for Metodologia.tex
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
from validation_runner import load_data, EmbeddingCache
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, classification_report
from scipy import stats


WEIGHT_CONFIGS = EXPERIMENT_PARAMS["weight_by_tier"]["CONFIGS"]
TIER_BOUNDS = EXPERIMENT_PARAMS["tier_impact"]["TIERS"]
EXPERIMENT_NAME = "weight_by_tier"


@dataclass
class WeightTierResult:
    config_name: str
    weight_low: float
    weight_mid: float
    weight_high: float
    n_synthetic: int
    macro_f1: float
    delta_pct: float
    baseline_f1: float
    low_tier_f1: float
    mid_tier_f1: float
    high_tier_f1: float
    p_value: float
    significant: bool
    win_rate: float


def get_class_tier(baseline_f1: float) -> str:
    """Assign tier based on baseline F1 performance."""
    if baseline_f1 < TIER_BOUNDS["LOW"]["max"]:
        return "LOW"
    elif baseline_f1 < TIER_BOUNDS["MID"]["max"]:
        return "MID"
    else:
        return "HIGH"


def compute_per_class_baseline_f1(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute baseline F1 per class using a single CV fold."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_scores = {c: [] for c in np.unique(labels)}

    for train_idx, test_idx in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        for c in np.unique(labels):
            c_mask = y_test == c
            if c_mask.sum() > 0:
                c_f1 = f1_score(y_test[c_mask] == c, y_pred[c_mask] == c)
                class_scores[c].append(c_f1)

    return {c: np.mean(scores) if scores else 0.0 for c, scores in class_scores.items()}


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


def generate_synthetic_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic samples with class labels preserved."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    unique_classes = np.unique(labels)

    for target_class in unique_classes:
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

        target_for_class = max(5, int(len(class_embeddings) * 0.10))

        n_clusters = min(3, max(1, len(class_embeddings) // 40))
        target_per_cluster = max(3, target_for_class // n_clusters)

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
                    temperature=TEMPERATURE,
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

                for idx in top_idx:
                    synthetic_embeddings.append(candidate_embeddings[idx])
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class}: {e}", flush=True)
                continue

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), np.array([])

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), np.array(synthetic_labels_list)


class TierWeightedEvaluator:
    """K-fold evaluator with per-tier synthetic weights."""

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
        class_tiers: Dict[str, str],
        tier_weights: Dict[str, float]
    ) -> Dict:
        """Evaluate with per-tier weights."""
        baseline_scores = []
        augmented_scores = []

        # Track per-tier performance
        tier_class_results = {"LOW": [], "MID": [], "HIGH": []}

        for train_idx, test_idx in self.kfold.split(X_original, y_original):
            X_train, X_test = X_original[train_idx], X_original[test_idx]
            y_train, y_test = y_original[train_idx], y_original[test_idx]

            # Baseline
            clf_base = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf_base.fit(X_train, y_train)
            y_pred_base = clf_base.predict(X_test)
            baseline_scores.append(f1_score(y_test, y_pred_base, average='macro'))

            # Augmented with tier-based weights
            if len(X_synthetic) > 0:
                X_aug = np.vstack([X_train, X_synthetic])
                y_aug = np.concatenate([y_train, y_synthetic])

                # Compute weights: original samples get 1.0, synthetic get tier-based weight
                synth_weights = []
                for label in y_synthetic:
                    tier = class_tiers.get(str(label), "MID")
                    synth_weights.append(tier_weights.get(tier, 1.0))

                weights = np.concatenate([
                    np.ones(len(X_train)),
                    np.array(synth_weights)
                ])

                clf_aug = LogisticRegression(max_iter=2000, solver='lbfgs')
                clf_aug.fit(X_aug, y_aug, sample_weight=weights)
                y_pred_aug = clf_aug.predict(X_test)
                augmented_scores.append(f1_score(y_test, y_pred_aug, average='macro'))

                # Track per-tier performance
                for tier in ["LOW", "MID", "HIGH"]:
                    tier_classes = [c for c, t in class_tiers.items() if t == tier]
                    if tier_classes:
                        tier_mask = np.isin(y_test, tier_classes)
                        if tier_mask.sum() > 0:
                            tier_f1 = f1_score(
                                y_test[tier_mask], y_pred_aug[tier_mask],
                                average='macro', labels=tier_classes, zero_division=0
                            )
                            tier_class_results[tier].append(tier_f1)
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

        # Compute tier F1 averages
        tier_f1_avg = {
            tier: np.mean(scores) if scores else 0.0
            for tier, scores in tier_class_results.items()
        }

        return {
            "baseline_mean": baseline_mean,
            "augmented_mean": augmented_mean,
            "delta_pct": delta_pct,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_synthetic": len(X_synthetic),
            "win_rate": win_rate,
            "tier_f1": tier_f1_avg
        }


def run_weight_tier_experiment() -> List[WeightTierResult]:
    """Run weight by tier experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 07a: WEIGHT BY PERFORMANCE TIER")
    print("="*60)
    print(f"  Configurations: {[c['name'] for c in WEIGHT_CONFIGS]}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    # Compute per-class baseline F1 and assign tiers
    print("\n  Computing per-class baseline F1...")
    class_baseline_f1 = compute_per_class_baseline_f1(embeddings, labels)

    class_tiers = {}
    tier_counts = {"LOW": 0, "MID": 0, "HIGH": 0}
    print("\n  Class tier assignments:")
    for cls, f1 in sorted(class_baseline_f1.items(), key=lambda x: x[1]):
        tier = get_class_tier(f1)
        class_tiers[cls] = tier
        tier_counts[tier] += 1
        print(f"    {cls}: F1={f1:.3f} -> {tier}")

    print(f"\n  Tier distribution: LOW={tier_counts['LOW']}, MID={tier_counts['MID']}, HIGH={tier_counts['HIGH']}")

    # Generate synthetic data once (will reuse with different weights)
    print("\n  Generating synthetic data...")
    X_synth, y_synth, _ = generate_synthetic_data(embeddings, labels, texts, cache)
    print(f"  Generated {len(X_synth)} synthetic samples")

    evaluator = TierWeightedEvaluator()
    results = []

    for i, config in enumerate(WEIGHT_CONFIGS):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(WEIGHT_CONFIGS)}] Testing {config['name']}")
        print(f"  Weights: LOW={config['LOW']}, MID={config['MID']}, HIGH={config['HIGH']}")
        print(f"{'─'*60}")

        tier_weights = {
            "LOW": config["LOW"],
            "MID": config["MID"],
            "HIGH": config["HIGH"]
        }

        eval_result = evaluator.evaluate(
            embeddings, labels, X_synth, y_synth,
            class_tiers, tier_weights
        )

        result = WeightTierResult(
            config_name=config["name"],
            weight_low=config["LOW"],
            weight_mid=config["MID"],
            weight_high=config["HIGH"],
            n_synthetic=eval_result["n_synthetic"],
            macro_f1=float(eval_result["augmented_mean"]),
            delta_pct=float(eval_result["delta_pct"]),
            baseline_f1=float(eval_result["baseline_mean"]),
            low_tier_f1=float(eval_result["tier_f1"]["LOW"]),
            mid_tier_f1=float(eval_result["tier_f1"]["MID"]),
            high_tier_f1=float(eval_result["tier_f1"]["HIGH"]),
            p_value=float(eval_result["p_value"]),
            significant=bool(eval_result["significant"]),
            win_rate=float(eval_result["win_rate"])
        )
        results.append(result)

        # Save individual result
        output_dir = RESULTS_DIR / EXPERIMENT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{config['name']}_result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

        sig = "*" if result.significant else ""
        print(f"  Macro F1={result.macro_f1:.4f}, Delta={result.delta_pct:+.2f}%{sig}")
        print(f"  Tier F1s: LOW={result.low_tier_f1:.3f}, MID={result.mid_tier_f1:.3f}, HIGH={result.high_tier_f1:.3f}")

    return results


def generate_latex_table(results: List[WeightTierResult]) -> str:
    """Generate LaTeX table for weight by tier experiment."""
    latex = r"""
% Tabla: Validacion de peso por tier de rendimiento
% Experimento 07a: Peso por Tier
\begin{table}[h]
\centering
\caption{Impacto de pesos diferenciados por tier de rendimiento}
\label{tab:weight_by_tier}
\begin{tabular}{lccccccc}
\hline
Configuracion & $w_{LOW}$ & $w_{MID}$ & $w_{HIGH}$ & Macro F1 & $\Delta$ F1 & F1 LOW & F1 HIGH \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        sig_marker = "*" if r.significant else ""
        name = r.config_name.replace("_", " ")

        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{r.weight_low}}} & \\textbf{{{r.weight_mid}}} & \\textbf{{{r.weight_high}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.delta_pct:+.2f}\\%{sig_marker}}} & \\textbf{{{r.low_tier_f1:.3f}}} & \\textbf{{{r.high_tier_f1:.3f}}} \\\\\n"
        else:
            latex += f"{name} & {r.weight_low} & {r.weight_mid} & {r.weight_high} & {r.macro_f1:.4f} & {r.delta_pct:+.2f}\\%{sig_marker} & {r.low_tier_f1:.3f} & {r.high_tier_f1:.3f} \\\\\n"

    latex += r"""
\hline
\multicolumn{8}{l}{\footnotesize *Estadisticamente significativo (p $<$ 0.05)} \\
\multicolumn{8}{l}{\footnotesize LOW: F1 baseline $<$ 0.20, MID: 0.20-0.45, HIGH: $>$ 0.45} \\
\end{tabular}
\end{table}
"""
    return latex


def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_weight_tier_experiment()

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEX_DIR / "tab_weight_by_tier.tex", 'w') as f:
        f.write(latex_table)

    # Save all results
    output_dir = RESULTS_DIR / EXPERIMENT_NAME
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY: Weight by Tier Experiment Results")
    print("="*80)
    print(f"{'Config':<20} | {'wL':>4} | {'wM':>4} | {'wH':>4} | {'F1':>7} | {'Delta':>8} | {'LOW F1':>7} | {'HIGH F1':>7} | {'Sig':>4}")
    print("-"*80)
    for r in results:
        sig_str = "*" if r.significant else ""
        print(f"{r.config_name:<20} | {r.weight_low:>4.1f} | {r.weight_mid:>4.1f} | {r.weight_high:>4.1f} | {r.macro_f1:>.4f} | {r.delta_pct:>+7.2f}% | {r.low_tier_f1:>7.3f} | {r.high_tier_f1:>7.3f} | {sig_str:>4}")

    # Find best
    best_idx = np.argmax([r.delta_pct for r in results])
    best = results[best_idx]
    print("-"*80)
    print(f"  BEST: {best.config_name} with Delta F1={best.delta_pct:+.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
