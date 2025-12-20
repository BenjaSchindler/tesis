#!/usr/bin/env python3
"""
Experiment 02: Anchor Strategies validation with PyTorch MLP

Tests: random, nearest_neighbor, medoid, quality_gated, diverse, ensemble
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from validation_runner import load_data, EmbeddingCache, AsyncLLMGenerator
from pytorch_mlp_classifier import run_pytorch_kfold, print_result_summary, DEVICE
from base_config import RESULTS_DIR, TEMPERATURE, MAX_CONCURRENT_API_CALLS

print(f"Using device: {DEVICE}")

# Anchor strategies to test
ANCHOR_STRATEGIES = ["random", "nearest_neighbor", "medoid", "quality_gated", "diverse", "ensemble"]

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 3,
    "prompts_per_cluster": 2,
    "samples_per_prompt": 3,
    "k_neighbors": 15,
}


def select_anchors(texts, embeddings, strategy, n_anchors=10):
    """Select anchor examples using specified strategy."""
    if len(texts) <= n_anchors:
        return list(range(len(texts)))

    if strategy == "random":
        return np.random.choice(len(texts), n_anchors, replace=False).tolist()

    elif strategy == "nearest_neighbor":
        # Select examples closest to centroid
        centroid = np.mean(embeddings, axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        return np.argsort(dists)[:n_anchors].tolist()

    elif strategy == "medoid":
        # Select true medoid
        sims = cosine_similarity(embeddings)
        avg_sims = np.mean(sims, axis=1)
        medoid_idx = np.argmax(avg_sims)
        # Then select nearest to medoid
        dists = 1 - sims[medoid_idx]
        return np.argsort(dists)[:n_anchors].tolist()

    elif strategy == "quality_gated":
        # Select based on average similarity (proxy for quality)
        sims = cosine_similarity(embeddings)
        np.fill_diagonal(sims, 0)
        avg_sims = np.mean(sims, axis=1)
        return np.argsort(avg_sims)[-n_anchors:].tolist()

    elif strategy == "diverse":
        # K-means clustering and select one from each cluster
        k = min(n_anchors, len(embeddings))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        selected = []
        for c in range(k):
            c_mask = cluster_labels == c
            c_idx = np.where(c_mask)[0]
            if len(c_idx) > 0:
                # Select medoid from cluster
                c_emb = embeddings[c_mask]
                centroid = kmeans.cluster_centers_[c]
                dists = np.linalg.norm(c_emb - centroid, axis=1)
                selected.append(c_idx[np.argmin(dists)])
        return selected

    elif strategy == "ensemble":
        # Combine medoid + diverse
        medoid_idx = select_anchors(texts, embeddings, "medoid", n_anchors // 2)
        diverse_idx = select_anchors(texts, embeddings, "diverse", n_anchors // 2)
        combined = list(set(medoid_idx + diverse_idx))
        return combined[:n_anchors]

    return list(range(min(n_anchors, len(texts))))


class StrategyGenerator:
    """Generate synthetics using specific anchor strategy."""

    def __init__(self, cache, params, strategy):
        self.cache = cache
        self.params = params
        self.strategy = strategy
        self.llm = AsyncLLMGenerator(
            temperature=params.get("temperature", TEMPERATURE),
            max_concurrent=MAX_CONCURRENT_API_CALLS
        )

    def create_prompt(self, examples, target_class, n_samples=5):
        n_shot = self.params.get("n_shot", 5)
        examples_to_use = examples[:n_shot]
        examples_text = "\n".join([
            f"- {ex[:200]}..." if len(ex) > 200 else f"- {ex}"
            for ex in examples_to_use
        ])

        return f"""Generate {n_samples} new social media posts that sound like they were written by someone with {target_class} personality type.

Here are examples of posts from this personality type:
{examples_text}

Generate {n_samples} new, unique posts in a similar style. Each post should be 1-3 sentences.
Output ONLY the posts, one per line, no numbering or prefixes."""

    def generate_for_class(self, class_texts, class_embeddings, target_class):
        n_anchors = self.params.get("n_shot", 60)
        max_clusters = self.params.get("max_clusters", 3)
        prompts_per_cluster = self.params.get("prompts_per_cluster", 2)
        samples_per_prompt = self.params.get("samples_per_prompt", 3)

        if len(class_texts) < 3:
            return [], []

        # Select anchors using strategy
        anchor_idx = select_anchors(class_texts, class_embeddings, self.strategy, n_anchors)
        anchor_texts = [class_texts[i] for i in anchor_idx]

        # Simple clustering for prompt diversity
        k_actual = min(max_clusters, max(1, len(class_embeddings) // 30))

        all_prompts = []
        for _ in range(k_actual):
            for _ in range(prompts_per_cluster):
                prompt = self.create_prompt(anchor_texts, str(target_class), samples_per_prompt)
                all_prompts.append(prompt)

        if not all_prompts:
            return [], []

        # Generate
        responses = self.llm.generate_sync(all_prompts)

        synthetic_texts = []
        for response in responses:
            if not response:
                continue
            samples = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) > 10]
            synthetic_texts.extend(samples[:samples_per_prompt])

        return synthetic_texts, [target_class] * len(synthetic_texts)


def main():
    print("=" * 70)
    print("Experiment 02: Anchor Strategies Validation (PyTorch MLP)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    results = []

    for strategy in ANCHOR_STRATEGIES:
        print(f"\n{'='*70}")
        print(f"Testing anchor strategy: {strategy}")
        print("=" * 70)

        params = BASE_CONFIG.copy()
        generator = StrategyGenerator(cache, params, strategy)

        all_synth_emb = []
        all_synth_labels = []

        for label in unique_labels:
            class_mask = np.array(labels) == label
            class_texts = texts[class_mask]
            class_emb = embeddings[class_mask]

            try:
                synth_texts, _ = generator.generate_for_class(class_texts, class_emb, label)
                if synth_texts:
                    synth_emb = cache.embed_synthetic(synth_texts)
                    all_synth_emb.append(synth_emb)
                    all_synth_labels.extend([label] * len(synth_emb))
                    print(f"  {label}: +{len(synth_emb)} synthetic")
            except Exception as e:
                print(f"  {label}: Error - {e}")

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        print(f"Total synthetic: {len(X_synth)}")

        result = run_pytorch_kfold(
            embeddings, labels,
            X_synth, y_synth,
            unique_labels,
            config_name="anchor_strategy",
            config_value=strategy,
            dropout=0.2
        )
        result.extra_metrics = {"anchor_strategy": strategy}
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "anchor_strategies",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "base_config": BASE_CONFIG,
        "strategies_tested": ANCHOR_STRATEGIES,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp02_anchor.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Anchor Strategies Validation (PyTorch MLP)}",
        r"\label{tab:pytorch_anchor}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Strategy & Baseline & Augmented & $\Delta$ (pp) & p-value & Synth \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r.significant else ""
        latex_lines.append(
            f"{r.config_value} & {r.baseline_mean:.4f} & {r.augmented_mean:.4f} & "
            f"{r.delta_pp:+.2f}{sig} & {r.p_value:.4f} & {r.n_synthetic} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_anchor.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"\n{'='*70}")
    print("SUMMARY: Anchor Strategies Experiment")
    print("=" * 70)
    print(f"{'Strategy':<18} {'Δ (pp)':<10} {'p-value':<10} {'Sig':<6}")
    print("-" * 46)
    for r in results:
        sig = "YES" if r.significant else "no"
        print(f"{r.config_value:<18} {r.delta_pp:+.2f}      {r.p_value:.4f}     {sig}")

    print(f"\nResults saved to {output_dir / 'exp02_anchor.json'}")


if __name__ == "__main__":
    main()
