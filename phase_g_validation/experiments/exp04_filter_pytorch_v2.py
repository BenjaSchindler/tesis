#!/usr/bin/env python3
"""
Experiment 04 v2: Filter Cascade validation with PyTorch MLP
Using ADAPTIVE thresholds to control synthetic count (~125 per config)

Tests:
- none: No filters
- length_only: Length filter (percentile-based)
- length_similarity: + Similarity filter (adaptive)
- three_filters: + K-NN purity (adaptive)
- full_cascade: + Anchor coherence (adaptive)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime
from dataclasses import asdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from validation_runner import load_data, EmbeddingCache, SyntheticGenerator
from pytorch_mlp_classifier import run_pytorch_kfold, print_result_summary, DEVICE
from base_config import RESULTS_DIR

print(f"Using device: {DEVICE}")

# Target number of synthetics per config (for fair comparison)
TARGET_SYNTHETICS = 125

# Base config
BASE_CONFIG = {
    "n_shot": 60,
    "temperature": 0.2,
    "max_clusters": 6,  # More clusters to generate more candidates
    "prompts_per_cluster": 3,
    "samples_per_prompt": 5,
    "k_neighbors": 15,
}


def apply_adaptive_filters(
    synth_texts,
    synth_embs,
    class_embs,
    anchor_emb,
    filter_config,
    target_count=None
):
    """
    Apply filter cascade with ADAPTIVE thresholds.
    Adjusts thresholds to keep approximately target_count synthetics.
    """
    if not synth_texts or len(synth_embs) == 0:
        return [], np.array([]), {}

    n_candidates = len(synth_embs)
    metrics = {"total_candidates": n_candidates}

    # Compute all metrics upfront
    # 1. Distance to anchor (for length proxy)
    if anchor_emb.ndim == 1:
        anchor_emb = anchor_emb.reshape(1, -1)
    dists_to_anchor = np.linalg.norm(synth_embs - anchor_emb, axis=1)

    # 2. Similarity to anchor
    similarities_to_anchor = 1 - cdist(synth_embs, anchor_emb, metric='cosine').flatten()

    # 3. K-NN purity (avg similarity to top-5 class neighbors)
    sims_to_class = cosine_similarity(synth_embs, class_embs)
    knn_purity = np.mean(np.sort(sims_to_class, axis=1)[:, -5:], axis=1)

    # 4. Anchor coherence (similarity to class centroid)
    centroid = np.mean(class_embs, axis=0, keepdims=True)
    coherence = 1 - cdist(synth_embs, centroid, metric='cosine').flatten()

    # Start with all accepted
    accepted_mask = np.ones(n_candidates, dtype=bool)

    if filter_config == "none":
        # No filtering
        pass

    elif filter_config == "length_only":
        # Filter 1: Keep samples not too far from anchor (90th percentile)
        threshold = np.percentile(dists_to_anchor, 90)
        accepted_mask &= (dists_to_anchor < threshold)
        metrics["length_threshold"] = float(threshold)

    elif filter_config == "length_similarity":
        # Filter 1: Length
        threshold1 = np.percentile(dists_to_anchor, 90)
        accepted_mask &= (dists_to_anchor < threshold1)
        metrics["length_threshold"] = float(threshold1)

        # Filter 2: Similarity (adaptive - find threshold that keeps ~target)
        remaining = np.sum(accepted_mask)
        if remaining > 0 and target_count:
            target_after_filter = max(target_count, remaining // 2)
            valid_sims = similarities_to_anchor[accepted_mask]
            if len(valid_sims) > target_after_filter:
                threshold2 = np.percentile(valid_sims, 100 * (1 - target_after_filter / len(valid_sims)))
            else:
                threshold2 = np.min(valid_sims) - 0.01
            accepted_mask &= (similarities_to_anchor >= threshold2)
            metrics["similarity_threshold"] = float(threshold2)

    elif filter_config == "three_filters":
        # Filter 1: Length
        threshold1 = np.percentile(dists_to_anchor, 90)
        accepted_mask &= (dists_to_anchor < threshold1)

        # Filter 2: Similarity (keep 70%)
        remaining = np.sum(accepted_mask)
        if remaining > 0:
            valid_sims = similarities_to_anchor[accepted_mask]
            threshold2 = np.percentile(valid_sims, 30)  # Keep top 70%
            accepted_mask &= (similarities_to_anchor >= threshold2)

        # Filter 3: K-NN purity (adaptive to reach target)
        remaining = np.sum(accepted_mask)
        if remaining > 0 and target_count:
            valid_purity = knn_purity[accepted_mask]
            if remaining > target_count:
                threshold3 = np.percentile(valid_purity, 100 * (1 - target_count / remaining))
            else:
                threshold3 = np.min(valid_purity) - 0.01
            accepted_mask &= (knn_purity >= threshold3)
            metrics["knn_threshold"] = float(threshold3)

    elif filter_config == "full_cascade":
        # Filter 1: Length
        threshold1 = np.percentile(dists_to_anchor, 90)
        accepted_mask &= (dists_to_anchor < threshold1)

        # Filter 2: Similarity (keep 80%)
        remaining = np.sum(accepted_mask)
        if remaining > 0:
            valid_sims = similarities_to_anchor[accepted_mask]
            threshold2 = np.percentile(valid_sims, 20)
            accepted_mask &= (similarities_to_anchor >= threshold2)

        # Filter 3: K-NN purity (keep 80%)
        remaining = np.sum(accepted_mask)
        if remaining > 0:
            valid_purity = knn_purity[accepted_mask]
            threshold3 = np.percentile(valid_purity, 20)
            accepted_mask &= (knn_purity >= threshold3)

        # Filter 4: Coherence (adaptive to reach target)
        remaining = np.sum(accepted_mask)
        if remaining > 0 and target_count:
            valid_coh = coherence[accepted_mask]
            if remaining > target_count:
                threshold4 = np.percentile(valid_coh, 100 * (1 - target_count / remaining))
            else:
                threshold4 = np.min(valid_coh) - 0.01
            accepted_mask &= (coherence >= threshold4)
            metrics["coherence_threshold"] = float(threshold4)

    # Apply mask
    accepted_texts = [t for t, m in zip(synth_texts, accepted_mask) if m]
    accepted_embs = synth_embs[accepted_mask]

    metrics["accepted"] = len(accepted_texts)
    metrics["acceptance_rate"] = len(accepted_texts) / n_candidates if n_candidates > 0 else 0

    # Compute quality metrics for accepted samples
    if len(accepted_embs) > 0:
        acc_sims = similarities_to_anchor[accepted_mask]
        acc_purity = knn_purity[accepted_mask]
        acc_coh = coherence[accepted_mask]
        metrics["avg_similarity"] = float(np.mean(acc_sims))
        metrics["avg_knn_purity"] = float(np.mean(acc_purity))
        metrics["avg_coherence"] = float(np.mean(acc_coh))

    return accepted_texts, accepted_embs, metrics


def main():
    print("=" * 70)
    print("Experiment 04 v2: Filter Cascade (Adaptive) - PyTorch MLP")
    print("=" * 70)
    print(f"Target synthetics per config: ~{TARGET_SYNTHETICS}")

    # Load data
    print("\nLoading data...")
    texts, labels = load_data()
    texts = np.array(texts)

    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)
    unique_labels = sorted(set(labels))

    print(f"Samples: {len(texts)}, Classes: {len(unique_labels)}")

    # Generate MORE candidates (we'll filter down)
    print("\nGenerating candidate synthetics (more than needed)...")
    generator = SyntheticGenerator(cache, BASE_CONFIG)

    # Store candidates per class
    candidates_by_class = {}

    for label in unique_labels:
        class_mask = np.array(labels) == label
        class_texts = texts[class_mask]
        class_emb = embeddings[class_mask]

        try:
            synth_texts, _ = generator.generate_for_class(class_texts, class_emb, label)
            if synth_texts:
                synth_emb = cache.embed_synthetic(synth_texts)
                # Compute anchor (medoid)
                sims = cosine_similarity(class_emb)
                medoid_idx = np.argmax(np.mean(sims, axis=1))
                anchor = class_emb[medoid_idx]

                candidates_by_class[label] = {
                    "texts": synth_texts,
                    "embs": synth_emb,
                    "class_embs": class_emb,
                    "anchor": anchor
                }
                print(f"  {label}: {len(synth_texts)} candidates")
        except Exception as e:
            print(f"  {label}: Error - {e}")

    total_candidates = sum(len(c["texts"]) for c in candidates_by_class.values())
    print(f"Total candidates: {total_candidates}")

    # Target per class
    target_per_class = TARGET_SYNTHETICS // len(candidates_by_class)
    print(f"Target per class: ~{target_per_class}")

    # Test each filter config
    filter_configs = ["none", "length_only", "length_similarity", "three_filters", "full_cascade"]
    results = []

    for filter_name in filter_configs:
        print(f"\n{'='*70}")
        print(f"Testing filter: {filter_name}")
        print("=" * 70)

        all_synth_emb = []
        all_synth_labels = []
        all_metrics = []

        for label, cands in candidates_by_class.items():
            filtered_texts, filtered_embs, metrics = apply_adaptive_filters(
                cands["texts"],
                cands["embs"],
                cands["class_embs"],
                cands["anchor"],
                filter_name,
                target_count=target_per_class
            )

            if len(filtered_embs) > 0:
                all_synth_emb.append(filtered_embs)
                all_synth_labels.extend([label] * len(filtered_embs))

            all_metrics.append(metrics)
            print(f"  {label}: {metrics['total_candidates']} -> {metrics['accepted']} "
                  f"({metrics['acceptance_rate']*100:.0f}%)")

        if all_synth_emb:
            X_synth = np.vstack(all_synth_emb)
            y_synth = np.array(all_synth_labels)
        else:
            X_synth = np.array([]).reshape(0, embeddings.shape[1])
            y_synth = np.array([])

        total_accepted = len(X_synth)
        avg_acceptance = np.mean([m["acceptance_rate"] for m in all_metrics])
        avg_quality = np.mean([m.get("avg_similarity", 0) for m in all_metrics if "avg_similarity" in m])

        print(f"\nTotal accepted: {total_accepted} (target: {TARGET_SYNTHETICS})")
        print(f"Avg acceptance rate: {avg_acceptance*100:.1f}%")
        print(f"Avg quality (similarity): {avg_quality:.3f}")

        # Run K-fold CV
        result = run_pytorch_kfold(
            embeddings, labels,
            X_synth, y_synth,
            unique_labels,
            config_name="filter",
            config_value=filter_name,
            dropout=0.2
        )
        result.extra_metrics = {
            "filter_config": filter_name,
            "total_candidates": total_candidates,
            "total_accepted": total_accepted,
            "acceptance_rate": avg_acceptance,
            "avg_quality": avg_quality
        }
        results.append(result)
        print_result_summary(result)

    # Save results
    output_dir = RESULTS_DIR / "pytorch_phase_f"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "filter_cascade_v2_adaptive",
        "classifier": "pytorch_mlp",
        "dropout": 0.2,
        "target_synthetics": TARGET_SYNTHETICS,
        "base_config": BASE_CONFIG,
        "filter_configs": filter_configs,
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "exp04_filter_v2.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Filter Cascade - Adaptive (PyTorch MLP)}",
        r"\label{tab:pytorch_filter_v2}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Filter & Synth & Quality & $\Delta$ (pp) & p-value & Sig \\",
        r"\midrule",
    ]

    for r in results:
        sig = r"$^{*}$" if r.significant else ""
        em = r.extra_metrics
        latex_lines.append(
            f"{r.config_value} & {em['total_accepted']} & {em['avg_quality']:.2f} & "
            f"{r.delta_pp:+.2f}{sig} & {r.p_value:.4f} & {'Yes' if r.significant else 'No'} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_path = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/Escrito_Tesis/Tables/tab_pytorch_filter_v2.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: Filter Cascade (Adaptive) - PyTorch MLP")
    print("=" * 70)
    print(f"{'Filter':<18} {'Synth':<8} {'Quality':<8} {'Δ (pp)':<10} {'p-value':<10} {'Sig':<6}")
    print("-" * 62)
    for r in results:
        sig = "YES" if r.significant else "no"
        em = r.extra_metrics
        print(f"{r.config_value:<18} {em['total_accepted']:<8} {em['avg_quality']:.2f}     "
              f"{r.delta_pp:+.2f}      {r.p_value:.4f}     {sig}")

    print(f"\nResults saved to {output_dir / 'exp04_filter_v2.json'}")
    print(f"LaTeX saved to {latex_path}")


if __name__ == "__main__":
    main()
