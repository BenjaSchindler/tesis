#!/usr/bin/env python3
"""
Experiment 05: Adaptive Thresholds Validation

Tests fixed vs adaptive thresholds:
- fixed_permissive (0.60), fixed_medium (0.70), fixed_strict (0.90), adaptive

Metrics: Macro F1, Acceptance%, Quality, Contamination%

Output: tab:adaptive_validation for Metodologia.tex
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

from base_config import RESULTS_DIR, LATEX_DIR, BASE_PARAMS, KFOLD_CONFIG, EXPERIMENT_PARAMS, LLM_MODEL, TEMPERATURE, MAX_TOKENS
from validation_runner import load_data, EmbeddingCache, KFoldEvaluator, print_summary
from typing import List, Tuple


THRESHOLD_CONFIGS = EXPERIMENT_PARAMS["adaptive_thresholds"]["CONFIGS"]
EXPERIMENT_NAME = "adaptive_thresholds"


@dataclass
class ThresholdResult:
    config: str
    macro_f1: float
    acceptance_rate: float
    quality: float
    contamination: float
    delta_pct: float
    n_synthetic: int
    p_value: float
    significant: bool


def get_adaptive_thresholds(purity: float) -> dict:
    """Get adaptive thresholds based on anchor purity."""
    if purity < 0.30:
        return {"similarity": 0.90, "knn": 0.52, "confidence": 0.20, "risk": "Critico"}
    elif purity < 0.45:
        return {"similarity": 0.72, "knn": 0.45, "confidence": 0.15, "risk": "Alto"}
    elif purity < 0.60:
        return {"similarity": 0.63, "knn": 0.40, "confidence": 0.12, "risk": "Medio"}
    else:
        return {"similarity": 0.60, "knn": 0.35, "confidence": 0.10, "risk": "Bajo"}


def compute_purity(
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str
) -> float:
    """Compute K-NN purity for an anchor."""
    dists = np.linalg.norm(all_embeddings - anchor_emb, axis=1)
    k = min(15, len(all_embeddings))
    nearest = np.argsort(dists)[:k]
    return (all_labels[nearest] == target_class).mean()


def apply_threshold_filter(
    candidates: np.ndarray,
    anchor_emb: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    target_class: str,
    config: dict
) -> tuple:
    """Apply threshold-based filtering."""
    if len(candidates) == 0:
        return np.array([]), 0.0, 0.0, 0.0

    if config["adaptive"]:
        # Compute purity and get adaptive thresholds
        purity = compute_purity(anchor_emb, all_embeddings, all_labels, target_class)
        thresholds = get_adaptive_thresholds(purity)
        sim_threshold = thresholds["similarity"]
    else:
        sim_threshold = config["threshold"]
        purity = 0.5  # Default

    # Apply similarity filter
    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)
    similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()
    accepted_mask = similarities >= sim_threshold

    accepted = candidates[accepted_mask]
    acceptance_rate = accepted_mask.sum() / len(accepted_mask)

    # Quality: purity of accepted samples
    if len(accepted) > 0:
        qualities = []
        for emb in accepted[:30]:
            dists = np.linalg.norm(all_embeddings - emb, axis=1)
            k = min(10, len(all_embeddings))
            nearest = np.argsort(dists)[:k]
            q = (all_labels[nearest] == target_class).mean()
            qualities.append(q)
        quality = np.mean(qualities)
    else:
        quality = 0.0

    # Contamination: estimated false positives
    contamination = max(0, 1.0 - quality) * 0.5 * (1.0 - sim_threshold)

    return accepted, acceptance_rate, quality, contamination


def generate_synthetic_with_threshold(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    config: dict,
    cache: EmbeddingCache
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Generate synthetic samples with LLM and apply threshold filter."""
    from openai import OpenAI

    client = OpenAI()
    texts_array = np.array(texts)

    synthetic_embeddings = []
    synthetic_labels_list = []
    all_acceptance = []
    all_quality = []
    all_contamination = []

    for target_class in np.unique(labels):
        class_mask = labels == target_class
        class_embeddings = embeddings[class_mask]
        class_texts = texts_array[class_mask]

        if len(class_embeddings) < 10:
            continue

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

            anchor_emb = kmeans.cluster_centers_[c_id]

            # Get example texts near anchor
            dists = np.linalg.norm(c_points - anchor_emb, axis=1)
            nearest_idx = np.argsort(dists)[:5]
            example_texts = [c_texts[i] for i in nearest_idx]

            # Generate candidates using LLM
            examples_text = "\n".join([
                f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
                for ex in example_texts
            ])
            n_candidates = BASE_PARAMS["samples_per_prompt"] * 2

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
                candidates = cache.embed_synthetic(samples)

                # Apply threshold filter
                accepted, acc_rate, quality, contam = apply_threshold_filter(
                    candidates, anchor_emb, embeddings, labels, target_class, config
                )

                all_acceptance.append(acc_rate)
                all_quality.append(quality)
                all_contamination.append(contam)

                for emb in accepted:
                    synthetic_embeddings.append(emb)
                    synthetic_labels_list.append(target_class)

            except Exception as e:
                print(f"    API error for {target_class} cluster {c_id}: {e}", flush=True)
                continue

    avg_acceptance = np.mean(all_acceptance) if all_acceptance else 0.0
    avg_quality = np.mean(all_quality) if all_quality else 0.0
    avg_contamination = np.mean(all_contamination) if all_contamination else 0.0

    if not synthetic_embeddings:
        return np.array([]).reshape(0, embeddings.shape[1]), np.array([]), avg_acceptance, avg_quality, avg_contamination

    return np.array(synthetic_embeddings), np.array(synthetic_labels_list), avg_acceptance, avg_quality, avg_contamination


def run_threshold_experiment() -> list:
    """Run adaptive thresholds validation experiment."""
    print("\n" + "="*60)
    print("  EXPERIMENT 05: ADAPTIVE THRESHOLDS VALIDATION")
    print("="*60)

    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    evaluator = KFoldEvaluator()
    results = []

    for config in THRESHOLD_CONFIGS:
        print(f"\n{'─'*60}")
        print(f"  Testing: {config['name']}")
        print(f"{'─'*60}")

        X_synth, y_synth, acceptance, quality, contamination = generate_synthetic_with_threshold(
            embeddings, labels, texts, config, cache
        )

        print(f"  Acceptance: {acceptance*100:.1f}%")
        print(f"  Quality: {quality:.2f}")
        print(f"  Contamination: {contamination*100:.1f}%")
        print(f"  Generated {len(X_synth)} synthetic samples")

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
            acceptance_rate=float(acceptance),
            quality=float(quality),
            contamination=float(contamination),
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
    """Generate LaTeX table for adaptive thresholds."""
    config_names = {
        "fixed_permissive": "Fijo permisivo (0.60)",
        "fixed_medium": "Fijo medio (0.70)",
        "fixed_strict": "Fijo estricto (0.90)",
        "adaptive": "Adaptativo"
    }

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparacion de estrategias de umbralizacion}
\label{tab:adaptive_validation}
\begin{tabular}{lcccc}
\hline
Estrategia & Macro F1 & Acceptance & Quality & Contam. \\
\hline
"""
    best_idx = np.argmax([r.delta_pct for r in results])

    for i, r in enumerate(results):
        name = config_names.get(r.config, r.config)
        if i == best_idx:
            latex += f"\\textbf{{{name}}} & \\textbf{{{r.macro_f1:.4f}}} & \\textbf{{{r.acceptance_rate*100:.0f}\\%}} & \\textbf{{{r.quality:.2f}}} & \\textbf{{{r.contamination*100:.1f}\\%}} \\\\\n"
        else:
            latex += f"{name} & {r.macro_f1:.4f} & {r.acceptance_rate*100:.0f}\\% & {r.quality:.2f} & {r.contamination*100:.1f}\\% \\\\\n"

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
    with open(LATEX_DIR / "tab_adaptive_validation.tex", 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("  SUMMARY: Adaptive Thresholds Results")
    print("="*60)
    for r in results:
        print(f"{r.config:>20} | F1={r.macro_f1:.4f} | Acc={r.acceptance_rate*100:.0f}% | Q={r.quality:.2f} | Contam={r.contamination*100:.1f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
