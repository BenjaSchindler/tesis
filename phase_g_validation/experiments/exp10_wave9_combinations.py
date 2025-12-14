#!/usr/bin/env python3
"""
Experiment 10: Wave 9 - Combination Experiments

Validates combination/advanced configurations:
- W9_contrastive: Contrastive prompting
- W9_best_combo: Best combination from previous phases
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE9_COMBINATIONS

CONFIGS = list(WAVE9_COMBINATIONS.keys())


def main():
    print("=" * 70)
    print("Experiment 10: Wave 9 - Combination Experiments")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    results = {}

    for config_name in CONFIGS:
        try:
            result = run_config_validation(config_name, cache, verbose=True)
            results[config_name] = result
        except Exception as e:
            print(f"\nERROR validating {config_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXPERIMENT 10 SUMMARY - WAVE 9 COMBINATIONS")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
