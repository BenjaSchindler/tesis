#!/usr/bin/env python3
"""
Experiment 08: Wave 7 - YOLO Experiments

Validates YOLO (no filters) configurations:
- W7_yolo: ALL filters disabled, maximum volume
- W7_yolo_force: YOLO mode + force problem classes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE7_YOLO

CONFIGS = list(WAVE7_YOLO.keys())


def main():
    print("=" * 70)
    print("Experiment 08: Wave 7 - YOLO Experiments")
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
    print("EXPERIMENT 08 SUMMARY - WAVE 7 YOLO")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
