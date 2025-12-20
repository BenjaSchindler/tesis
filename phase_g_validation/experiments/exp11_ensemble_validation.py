#!/usr/bin/env python3
"""
Experiment 11: Ensemble Validation (Extended)

Validates ensemble configurations by combining synthetics from components:
- ENS_Top3_G5: Top 4 Phase F components
- ENS_SUPER_G5_F7_v2: Extended with Phase G winners
- ENS_TopG5_Extended: Extended with contrastive
- ENS_WaveChampions: Best from each wave

EXTENDED FEATURES (Phase G Extended):
- Weighted combination of components
- Deduplication strategies (similarity, class-wise, clustering)
- Per-class config selection
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from validation_runner import (
    load_data, EmbeddingCache, KFoldEvaluator,
    SyntheticGenerator, save_result, print_summary
)
from config_definitions import ENSEMBLES, get_config_params, ALL_CONFIGS
from base_config import RESULTS_DIR

# Import deduplication utilities
try:
    from deduplication import (
        deduplicate_by_similarity,
        deduplicate_classwise,
        deduplicate_by_clustering,
        deduplicate_cross_component,
        print_dedup_stats
    )
    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False
    print("WARNING: deduplication.py not found, deduplication features disabled")


def run_ensemble_validation(
    ensemble_name: str,
    cache: EmbeddingCache,
    verbose: bool = True,
    weights: list = None,
    dedup_method: str = None,
    dedup_params: dict = None
):
    """
    Validate an ensemble by generating and combining synthetics from components.

    Args:
        ensemble_name: Name of ensemble configuration
        cache: EmbeddingCache instance
        verbose: Print progress
        weights: Optional list of weights for each component (for weighted sampling)
                 If None, uses equal weights. Weights are normalized automatically.
        dedup_method: Optional deduplication method:
            - None: No deduplication (default)
            - 'similarity': Cosine similarity threshold
            - 'classwise': Class-wise similarity deduplication
            - 'clustering': Clustering-based deduplication
            - 'cross_component': Cross-component deduplication
        dedup_params: Parameters for deduplication method (e.g., {'threshold': 0.95})
    """

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Validating Ensemble: {ensemble_name}")
        print(f"{'#'*70}")

    ensemble_info = ENSEMBLES[ensemble_name]
    components = ensemble_info["components"]

    if verbose:
        print(f"\n  Description: {ensemble_info['description']}")
        print(f"  Components: {components}")
        if weights:
            print(f"  Weights: {weights}")
        if dedup_method:
            print(f"  Deduplication: {dedup_method} {dedup_params or {}}")

    all_synthetic_emb = []
    all_synthetic_labels = []
    component_stats = {}

    for comp_name in components:
        if comp_name not in ALL_CONFIGS:
            print(f"  WARNING: Unknown component {comp_name}, skipping")
            continue

        if verbose:
            print(f"\n  Generating synthetics for: {comp_name}")

        params = get_config_params(comp_name)
        generator = SyntheticGenerator(cache, params)

        X_synth, y_synth, texts = generator.generate_all(
            cache.embeddings, cache.labels, cache.texts
        )

        component_stats[comp_name] = len(X_synth)

        if len(X_synth) > 0:
            all_synthetic_emb.append(X_synth)
            all_synthetic_labels.append(y_synth)

        if verbose:
            print(f"    Generated {len(X_synth)} samples")

    # Apply deduplication if specified
    if dedup_method and DEDUP_AVAILABLE and all_synthetic_emb:
        if verbose:
            print(f"\n  Applying deduplication: {dedup_method}")

        dedup_params = dedup_params or {}

        if dedup_method == 'cross_component':
            # Cross-component deduplication (preserves component ordering)
            component_data = list(zip(all_synthetic_emb, all_synthetic_labels))
            deduped_components, dedup_stats = deduplicate_cross_component(
                component_data,
                threshold=dedup_params.get('threshold', 0.95),
                method=dedup_params.get('method', 'cosine')
            )

            # Unpack deduplicated components
            all_synthetic_emb = [X for X, y in deduped_components]
            all_synthetic_labels = [y for X, y in deduped_components]

            if verbose:
                print_dedup_stats(dedup_stats, "Cross-Component Deduplication")

        else:
            # First combine, then deduplicate
            if all_synthetic_emb:
                X_combined = np.vstack(all_synthetic_emb)
                y_combined = np.concatenate(all_synthetic_labels)

                if dedup_method == 'similarity':
                    X_dedup, y_dedup, dedup_stats = deduplicate_by_similarity(
                        X_combined, y_combined,
                        threshold=dedup_params.get('threshold', 0.95),
                        method=dedup_params.get('method', 'cosine')
                    )
                elif dedup_method == 'classwise':
                    X_dedup, y_dedup, dedup_stats = deduplicate_classwise(
                        X_combined, y_combined,
                        threshold=dedup_params.get('threshold', 0.95),
                        method=dedup_params.get('method', 'cosine')
                    )
                elif dedup_method == 'clustering':
                    X_dedup, y_dedup, dedup_stats = deduplicate_by_clustering(
                        X_combined, y_combined,
                        n_clusters=dedup_params.get('n_clusters', None),
                        cluster_method=dedup_params.get('cluster_method', 'kmeans'),
                        keep_strategy=dedup_params.get('keep_strategy', 'centroid')
                    )
                else:
                    if verbose:
                        print(f"  WARNING: Unknown dedup_method {dedup_method}, skipping")
                    X_dedup, y_dedup = X_combined, y_combined
                    dedup_stats = {}

                # Replace with deduplicated data
                all_synthetic_emb = [X_dedup]
                all_synthetic_labels = [y_dedup]

                if verbose and dedup_stats:
                    print_dedup_stats(dedup_stats, f"Deduplication ({dedup_method})")

    # Apply weighting if specified
    if weights and all_synthetic_emb:
        if verbose:
            print(f"\n  Applying weighted sampling...")

        # Normalize weights
        weights_array = np.array(weights[:len(all_synthetic_emb)])
        if np.sum(weights_array) == 0:
            weights_array = np.ones(len(all_synthetic_emb))
        weights_array = weights_array / np.sum(weights_array)

        if verbose:
            print(f"  Normalized weights: {weights_array}")

        # Weighted sampling from each component
        weighted_emb = []
        weighted_labels = []

        # Determine total samples to keep (average of component sizes)
        total_samples = sum(len(X) for X in all_synthetic_emb)
        target_samples = int(total_samples * 0.8)  # Keep 80% after weighting

        for i, (X, y, w) in enumerate(zip(all_synthetic_emb, all_synthetic_labels, weights_array)):
            if len(X) == 0:
                continue

            # Sample proportional to weight
            n_samples = int(target_samples * w)
            n_samples = min(n_samples, len(X))  # Can't sample more than available

            if n_samples > 0:
                # Random sample with replacement if needed
                indices = np.random.choice(len(X), size=n_samples, replace=(n_samples > len(X)))
                weighted_emb.append(X[indices])
                weighted_labels.append(y[indices])

                if verbose:
                    print(f"    Component {i}: {len(X)} → {n_samples} samples (weight={w:.3f})")

        all_synthetic_emb = weighted_emb
        all_synthetic_labels = weighted_labels

    # Combine all synthetics
    if all_synthetic_emb:
        X_ensemble = np.vstack(all_synthetic_emb)
        y_ensemble = np.concatenate(all_synthetic_labels)
    else:
        X_ensemble = np.array([]).reshape(0, cache.embeddings.shape[1])
        y_ensemble = np.array([])

    if verbose:
        print(f"\n  Final combined: {len(X_ensemble)} synthetic samples")
        print(f"  Per-component breakdown:")
        for comp, count in component_stats.items():
            print(f"    {comp}: {count}")

    # Evaluate with K-fold
    evaluator = KFoldEvaluator(synthetic_weight=1.0)  # Optimal weight
    result = evaluator.evaluate(
        cache.embeddings,
        cache.labels,
        X_ensemble,
        y_ensemble,
        config_name=ensemble_name,
        config_params={"components": components, "component_stats": component_stats}
    )

    # Save result
    save_result(result, "ensembles")

    if verbose:
        print_summary(result)

    return result


def main():
    print("=" * 70)
    print("Experiment 11: Ensemble Validation")
    print("=" * 70)
    print(f"\nEnsembles to validate: {list(ENSEMBLES.keys())}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    results = {}

    for ensemble_name in ENSEMBLES.keys():
        try:
            result = run_ensemble_validation(ensemble_name, cache, verbose=True)
            results[ensemble_name] = result
        except Exception as e:
            print(f"\nERROR validating {ensemble_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 11 SUMMARY - ENSEMBLES")
    print("=" * 70)

    for ensemble_name, result in results.items():
        sig = "*" if result.significant else ""
        n_comp = len(ENSEMBLES[ensemble_name]["components"])
        print(f"  {ensemble_name:25} ({n_comp} comp) delta={result.delta_pct:+.2f}% "
              f"p={result.p_value:.4f} {sig} n_synth={result.n_synthetic}")

    # Save combined summary
    summary_path = RESULTS_DIR / "ensembles" / "ensemble_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "ensembles": {
            name: {
                "delta_pct": r.delta_pct,
                "p_value": r.p_value,
                "significant": r.significant,
                "n_synthetic": r.n_synthetic,
                "components": ENSEMBLES[name]["components"]
            }
            for name, r in results.items()
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
