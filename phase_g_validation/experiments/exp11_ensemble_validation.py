#!/usr/bin/env python3
"""
Experiment 11: Ensemble Validation

Validates ensemble configurations by combining synthetics from components:
- ENS_Top3_G5: Top 4 Phase F components
- ENS_SUPER_G5_F7_v2: Extended with Phase G winners
- ENS_TopG5_Extended: Extended with contrastive
- ENS_WaveChampions: Best from each wave
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


def run_ensemble_validation(
    ensemble_name: str,
    cache: EmbeddingCache,
    verbose: bool = True
):
    """Validate an ensemble by generating and combining synthetics from components."""

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Validating Ensemble: {ensemble_name}")
        print(f"{'#'*70}")

    ensemble_info = ENSEMBLES[ensemble_name]
    components = ensemble_info["components"]

    if verbose:
        print(f"\n  Description: {ensemble_info['description']}")
        print(f"  Components: {components}")

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

    # Combine all synthetics
    if all_synthetic_emb:
        X_ensemble = np.vstack(all_synthetic_emb)
        y_ensemble = np.concatenate(all_synthetic_labels)
    else:
        X_ensemble = np.array([]).reshape(0, cache.embeddings.shape[1])
        y_ensemble = np.array([])

    if verbose:
        print(f"\n  Combined {len(X_ensemble)} synthetic samples")
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
