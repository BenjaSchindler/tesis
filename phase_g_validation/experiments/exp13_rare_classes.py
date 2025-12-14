#!/usr/bin/env python3
"""
Experiment 13: Rare Class Experiments

Focus on improving ESFJ, ESFP, ESTJ which have <50 samples each.
These classes have not improved in any previous experiment.

Configs:
- RARE_massive_oversample: Generate 5x more, min 100 per class
- RARE_yolo_extreme: No filtering at all
- RARE_few_shot_expert: Use all examples in prompts
- RARE_high_temperature: High diversity generation
- RARE_contrastive_transfer: Use similar classes as reference
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import asdict
from config_definitions import RARE_CLASS_EXPERIMENTS, get_config_params
from validation_runner import (
    load_data, EmbeddingCache, SyntheticGenerator, KFoldEvaluator
)
from base_config import RESULTS_DIR, PROBLEM_CLASSES
import json

CONFIGS = list(RARE_CLASS_EXPERIMENTS.keys())

def main():
    print("=" * 70)
    print("Experiment 13: Rare Class Experiments")
    print("=" * 70)
    print(f"\nTarget classes: ESFJ (42), ESFP (48), ESTJ (39)")
    print(f"Goal: Generate enough synthetic samples to reach ~250+ total per class")
    print(f"\nConfigs to validate: {CONFIGS}")

    # Load data
    texts, labels = load_data()
    cache = EmbeddingCache()
    embeddings = cache.load_or_compute(texts, labels)

    results = {}

    for config_name in CONFIGS:
        print(f"\n{'#' * 70}")
        print(f"# Validating: {config_name}")
        print(f"{'#' * 70}")

        config = RARE_CLASS_EXPERIMENTS[config_name]
        params = get_config_params(config_name)

        print(f"\n  Description: {config['description']}")
        print(f"  Wave: {config['wave']}")
        print(f"  Crucial params: {config['crucial_params']}")

        # Generate synthetic data
        print(f"\n  Generating synthetic data for rare classes...")
        generator = SyntheticGenerator(cache, params)
        synth_emb, synth_labels, synth_texts = generator.generate_all(
            embeddings, labels, texts
        )

        print(f"  Generated {len(synth_texts)} synthetic samples")

        # Show distribution
        from collections import Counter
        synth_dist = Counter(synth_labels)
        print(f"\n  Synthetic distribution:")
        for cls in ["ESFJ", "ESFP", "ESTJ"]:
            orig_count = sum(1 for l in labels if l == cls)
            synth_count = synth_dist.get(cls, 0)
            total = orig_count + synth_count
            print(f"    {cls}: {orig_count} orig + {synth_count} synth = {total} total")

        if len(synth_texts) == 0:
            print(f"\n  WARNING: No synthetic samples generated!")
            continue

        # Run K-fold evaluation
        print(f"\n  Running 5-fold x 3 repeats = 15 folds")
        synthetic_weight = params.get("synthetic_weight", 1.0)
        print(f"  Synthetic weight: {synthetic_weight}")

        evaluator = KFoldEvaluator(synthetic_weight=synthetic_weight)
        fold_results = evaluator.evaluate(
            embeddings, labels, synth_emb, synth_labels
        )

        # Save results
        results_dir = RESULTS_DIR / "rare_class"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict
        fold_dict = asdict(fold_results)

        result_data = {
            "config_name": config_name,
            "description": config['description'],
            "n_synthetic": len(synth_texts),
            "synth_distribution": dict(synth_dist),
            **fold_dict
        }

        result_file = results_dir / f"{config_name}_kfold.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"  Saved results to {result_file}")

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"  Config: {config_name}")
        print(f"  Folds: {fold_results.n_folds}")
        print(f"  Baseline:  {fold_results.baseline_mean:.4f} +/- {fold_results.baseline_std:.4f}")
        print(f"  Augmented: {fold_results.augmented_mean:.4f} +/- {fold_results.augmented_std:.4f}")
        print(f"  Delta:     {fold_results.delta_mean:+.4f} ({fold_results.delta_pct:+.2f}%)")
        print(f"  95% CI:    [{fold_results.ci_95_lower:+.4f}, {fold_results.ci_95_upper:+.4f}]")

        sig_marker = "*" if fold_results.significant else ""
        print(f"  p-value:   {fold_results.p_value:.6f} {sig_marker}")
        print(f"  Win rate:  {fold_results.win_rate*100:.1f}%")
        print(f"  Synthetics: {len(synth_texts)}")

        # Problem class deltas
        print(f"\n  Problem class deltas:")
        per_class = fold_results.per_class_delta or {}
        for cls in ["ESFJ", "ESFP", "ESTJ"]:
            delta = per_class.get(cls, 0)
            print(f"    {cls}: {delta:+.4f}")
        print("=" * 70)

        results[config_name] = result_data

    # Print final summary
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 13 SUMMARY - RARE CLASSES")
    print("=" * 70)

    for config_name, data in results.items():
        sig = "*" if data.get('significant', False) else ""
        delta_pct = data.get('delta_pct', 0)
        n_synth = data.get('n_synthetic', 0)

        # Get rare class improvement
        per_class = data.get('per_class_delta', {}) or {}
        esfj = per_class.get('ESFJ', 0)
        esfp = per_class.get('ESFP', 0)
        estj = per_class.get('ESTJ', 0)

        print(f"  {config_name:25} delta={delta_pct:+.2f}% synth={n_synth:4} ESFJ={esfj:+.3f} ESFP={esfp:+.3f} ESTJ={estj:+.3f} {sig}")

    print("=" * 70)


if __name__ == "__main__":
    main()
