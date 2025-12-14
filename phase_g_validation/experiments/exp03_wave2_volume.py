#!/usr/bin/env python3
"""
Experiment 03: Wave 2 - Volume Experiments

Validates volume-related configurations:
- W2_ultra_vol: Ultra high volume (10x15x10=1500 candidates)
- W2_mega_vol: Mega volume (12x20x12=2880 candidates)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE2_VOLUME

CONFIGS = list(WAVE2_VOLUME.keys())


def main():
    print("=" * 70)
    print("Experiment 03: Wave 2 - Volume Experiments")
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
    print("EXPERIMENT 03 SUMMARY - WAVE 2 VOLUME")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
